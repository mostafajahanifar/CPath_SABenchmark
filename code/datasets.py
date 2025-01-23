import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def get_msi_dataset(root_dir, case_list_path, label_dict=None, filter_small_bags=None):
    # Load the case list
    case_df = pd.read_csv(case_list_path)
    if label_dict is not None:
        case_to_label = {case: label_dict[label] for case, label in zip(case_df['slide'], case_df['msi_status'])}
    else:
        case_to_label = dict(zip(case_df['slide'], case_df['msi_status']))
    bag_names = [d for d in case_to_label.keys()]
    labels = [case_to_label[case] for case in bag_names]
    bag_folders = [os.path.join(root_dir, f, d) for d, f in zip(case_df['slide'], case_df['cancer_type'])]

    # Filtering bags with low number of instances
    bags_to_neglect = []
    for bi, bag_folder in tqdm(enumerate(bag_folders), total=len(bag_folders), leave=False):
        all_instances = np.load(os.path.join(bag_folder,"paths.npy"))
        all_instances = [Path(path).stem for path in all_instances]
        if filter_small_bags and len(all_instances) < filter_small_bags:
            # flag this bag to be removed
            bags_to_neglect.append(bi)
            continue
    # delete the bags with not enough instances
    if  len(bags_to_neglect)>0:
        bags_to_neglect.sort(reverse=True)
        for bi in bags_to_neglect:
            del bag_names[bi]
            del bag_folders[bi]
            del labels[bi]

    df = pd.DataFrame({'slide': bag_names,
                       'target': labels,
                       'tensor_path': bag_folders})
    
    return slide_dataset_classification(df)

def get_datasets(mccv=0, data='', encoder='', method=''):
    # Load slide data
    df = pd.read_csv(os.path.join('/sc/arion/projects/comppath_500k/SAbenchmarks/data', data, 'slide_data.csv'))
    df['tensor_path'] = [os.path.join(x.tensor_root, encoder, x.tensor_name) for _, x in df.iterrows()]
    spatial_root = Path(df.loc[0,"tensor_root"]).parent / "spatial"
    if method == 'GTP': # Graph with adjacency matrix
        df['method_tensor_path'] = [spatial_root / 'GTP' / x.tensor_name for _, x in df.iterrows()]
    elif method in ['PatchGCN', 'DeepGraphConv']: # Graph with edge_index
        df['method_tensor_path'] = [spatial_root / 'PatchGCN' / x.tensor_name for _, x in df.iterrows()]
    elif method in ['MIL_Cluster_FC']: # cluster
        df['method_tensor_path'] = [spatial_root / 'Cluster' / x.tensor_name for _, x in df.iterrows()]
    elif method in ['ViT_MIL','DTMIL']: # Positional Encoded Transformer
        df['method_tensor_path'] = [spatial_root / 'DT-MIL' / x.tensor_name for _, x in df.iterrows()]
    else:
        df['method_tensor_path'] = [os.path.join(x.tensor_root, encoder, x.tensor_name) for _, x in df.iterrows()]
    # Select mccv and clean
    df = df.rename(columns={'mccv{}'.format(mccv): 'mccvsplit'})[['slide', 'target', 'mccvsplit', 'tensor_path','method_tensor_path']]
    df['mccvsplit'] = df['mccvsplit'].fillna('test')
    # Split into train and val
    df_train = df[df.mccvsplit == 'train'].reset_index(drop=True).drop(columns=['mccvsplit'])
    df_val = df[df.mccvsplit == 'val'].reset_index(drop=True).drop(columns=['mccvsplit'])
    df_test = None
    if data in ['camelyon16']:
        df_test = df[df.mccvsplit == 'test'].reset_index(drop=True).drop(columns=['mccvsplit'])
    # Create my loader objects
    if method in ['GTP','PatchGCN', 'DeepGraphConv' ,'MIL_Cluster_FC','MIL_Sum_FC','ViT_MIL','DTMIL']:
        dset_train = slide_dataset_classification_graph(df_train)
        dset_val = slide_dataset_classification_graph(df_val)
        dset_test = slide_dataset_classification_graph(df_test) if df_test is not None else None
    else:
        dset_train = slide_dataset_classification(df_train)
        dset_val = slide_dataset_classification(df_val)
        dset_test = slide_dataset_classification(df_test) if df_test is not None else None
    
    return dset_train, dset_val, dset_test

class slide_dataset_classification(torch.utils.data.Dataset):
    '''
    Slide level dataset which returns for each slide the feature matrix (h) and the target
    '''
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        data = np.load(row.tensor_path + "/embeddings.npy")
        # data = torch.load(row.tensor_path)  # feature matrix and possibly other data
        try:
            feat = data['features']
        except:
            feat = torch.tensor(data, dtype=torch.float32)
        return {'features': feat, 'target': row.target}

class slide_dataset_classification_graph(slide_dataset_classification):
    def __init__(self, df):
        super(slide_dataset_classification_graph, self).__init__(df)

    def __getitem__(self, index):
        # Load data using the parent class method
        item = super(slide_dataset_classification_graph, self).__getitem__(index)
        # Additional graph-specific data extraction
        data = torch.load(self.df.iloc[index].method_tensor_path)
        if 'adj_mtx' in data:
            item['adj_mtx'] = data['adj_mtx']
            item['mask'] = data['mask']
        if 'edge_latent' in data.keys():
            item['edge_index'] = data['edge_index']
            item['edge_latent'] = data['edge_latent']
            item['centroid'] = data['centroid']
        if 'feat_map' in data:
            item['feat_map'] = data['feat_map']
            item['mask'] = data['mask']
        return item
