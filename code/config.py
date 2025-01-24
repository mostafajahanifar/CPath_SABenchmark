# config.py
import argparse

def get_config():
    parser = argparse.ArgumentParser()

    #I/O PARAMS
    parser.add_argument('--output_root', type=str, default='.', help='output directory')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--experiment_list_root', type=str, default="where_experiment_lists_are")
    parser.add_argument('--additional_desc', type=str, default="3cancers-newTrans-idars-s25x0.1-e50x0.3-valAUROC")
    parser.add_argument('-f','--folds', nargs='+', type=int, help='Experiment folds to be processed', default=[1, 2, 3])
    parser.add_argument('-d','--design', type=str, help='Experiment design to be processed', default="keep-1-domains-in" )
    parser.add_argument('--label_dict', type=dict, default=None)

    parser.add_argument('--train_list_path', type=str)
    parser.add_argument('--val_list_path', type=str)
    parser.add_argument('--test_list_paths', type=list, default=[])
    parser.add_argument('--log', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    # Method PARAMS
    parser.add_argument('--method', type=str, default='', choices=[
        'AB-MIL',
        'CLAM_SB',
        'CLAM_MB',
        'transMIL',
        'DS-MIL',
        'VarMIL',
        'GTP',
        'PatchGCN',
        'DeepGraphConv',
        'MIL_Cluster_FC',
        'AB-MIL_FC',
        'MIL_Sum_FC',
        'ViT_MIL',
        'DTMIL'
    ], help='which aggregation method to use')
    parser.add_argument('--encoder', type=str, default='', choices=[
        'tres50_imagenet',
        'dinosmall',
        'dinobase',
        'uni'
    ], help='which encoder to use')

    parser.add_argument('--mccv', default=1, type=int, choices=list(range(1,22)), help='which seed (default: 1/20)')
    parser.add_argument('--ndim', default=None, type=int, help='output dimension of feature extractor')

    #OPTIMIZATION PARAMS
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of linear warmup (highest LR used during training). The learning rate is linearly scaled with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument('--lr_end', type=float, default=1e-6, help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")
    parser.add_argument('--nepochs', type=int, default=40, help='number of epochs (default: 40)')
    parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 10)')
    parser.add_argument('--random_seed', default=0, type=int, help='random seed')

    # Weight and Bias Config
    parser.add_argument('--wandb_project', type=str, help='name of project in wandb')
    parser.add_argument('--wandb_note', type=str, help='note of project in wandb')
    parser.add_argument('--sweep_config', type=str, help='Path to the sweep configuration YAML file')
    parser.add_argument('--parameter_path', type=str, help='Read hyperparameters after tuning')


    args = parser.parse_args()
    return args
