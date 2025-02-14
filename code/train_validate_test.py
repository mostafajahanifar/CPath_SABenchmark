import datasets
import modules
import os
import argparse
import pandas as pd
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import time
import wandb
from evaluation import evaluate, log_results


def set_random_seed(seed_value):
    print(f"Set random seed to: {seed_value}")
    random.seed(seed_value)  # Python random module.
    np.random.seed(seed_value)  # Numpy module.
    torch.manual_seed(seed_value)  # Sets the seed for generating random numbers for CPU.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)  # Sets the seed for generating random numbers on all GPUs.
        torch.cuda.manual_seed_all(seed_value)  # Sets the seed for generating random numbers on all GPUs.
        torch.backends.cudnn.deterministic = True  # Makes CUDA operations deterministic.
        torch.backends.cudnn.benchmark = False  # If True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

def calculate_model_size(model):
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    return size_model

def main(args):
    # Initialize wandb
    wandb.login()
    wandb.init(project=args.wandb_project, group=args.wandb_group, name=args.wandb_note)

    # if args.sweep_config:
    # Now update args with wandb.config
    for key, value in wandb.config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    args_dict = vars(args) if isinstance(args, argparse.Namespace) else args
    # Update wandb.config with new arguments from args
    for key, value in args_dict.items():
        if key not in wandb.config:
            wandb.config[key] = value

    wandb.config.update(args_dict, allow_val_change=True)

    if args.random_seed:
        set_random_seed(args.random_seed)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Set datasets
    train_dset = datasets.get_msi_dataset(root_dir=args.data_root, case_list_path=args.train_list_path, label_dict=args.label_dict)
    val_dset = datasets.get_msi_dataset(root_dir=args.data_root, case_list_path=args.val_list_path, label_dict=args.label_dict)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=args.workers)

    # Get model
    model = modules.get_aggregator(method=args.method, ndim=args.ndim) # model = modules.get_aggregator(method='AB-MIL', ndim=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set loss
    criterion = nn.CrossEntropyLoss().to(device)

    # Set optimizer
    params_groups = get_params_groups(model)
    optimizer = optim.AdamW(params_groups)

    # Set schedulers
    lr_schedule = cosine_scheduler(
        args.lr,
        args.lr_end,
        args.nepochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.nepochs,
        len(train_loader),
    )
    cudnn.benchmark = True

    best_auc = 0.0
    best_thresh = None
    # Main training loop
    for epoch in range(1, args.nepochs+1):
        print(f"********************** Epoch {epoch} / {args.nepochs} **********************")
        # Regular training and validation logic
        train_start_time = time.time()
        loss = train(epoch, train_loader, model, criterion, optimizer, lr_schedule, wd_schedule, device, args)
        train_end_time = time.time()
        train_runtime =  train_end_time - train_start_time
        memory_gpu = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        # Get the current learning rate from the first parameter group
        current_lr = optimizer.param_groups[0]['lr']
        current_wd = optimizer.param_groups[0]['weight_decay']

        if (epoch+1)%2 == 0:  # Special case for testing feature extractor
            # Validation logic for feature extractor testing
            inference_start_time = time.time()
            probs = test(epoch, val_loader, model, device, args)
            inference_end_time = time.time()
            inference_runtime = inference_end_time - inference_start_time

            val_metrics = evaluate(val_loader.dataset.df.target, probs)
            val_metrics_prefixed = {f"val_{key}": value for key, value in val_metrics.items()}

            # Regular AUC logging
            print(f"Eepoch {epoch} -- Train Loss={loss}")
            print(val_metrics_prefixed)
            wandb.log({
                    "epoch": epoch,
                    "train_loss": loss,
                    "lr_step": current_lr,
                    "wd_step": current_wd,
                    "train_runtime": train_runtime,
                    "inference_runtime": inference_runtime,
                    "memory_gpu": memory_gpu,
                    **val_metrics_prefixed  # Log prefixed validation metrics
                })

            # Check if the current model is the best one
            if val_metrics["AUROC"] > best_auc:
                print(f"New best model found")
                best_auc = val_metrics["AUROC"]
                best_thresh = val_metrics["Treshold"]
                # Log this event to wandb
                wandb.run.summary["best_auc"] = val_metrics["AUROC"]
                wandb.run.summary["best_epoch"] = epoch

                # save the best model
                best_model_filename = os.path.join(args.output,'best_auc_model.pth')
                obj = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'auc': val_metrics["AUROC"],
                    'optimizer': optimizer.state_dict()
                }
                torch.save(obj, best_model_filename)
                best_val_metrics = val_metrics.copy()
        else:
            print(f"Eepoch {epoch} -- Train Loss={loss}")
            wandb.log({"epoch": epoch, "train_loss": loss , 'lr_step': current_lr, 'wd_step': current_wd,
                        "train_runtime": train_runtime, "memory_gpu": memory_gpu})

    model_filename = os.path.join(args.output,'final_model.pth')
    run_num = args.wandb_note.split("_")[-1]
    log_results(best_val_metrics,  f"{args.wandb_group}_validation", os.path.join(args.output_root, args.wandb_project, f"validation_results_run{run_num}.csv"))
    # only save the last model to artifact
    ### Model saving logic
    obj = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(obj, model_filename)

    # Create a wandb Artifact and add the file to it
    model_artifact = wandb.Artifact('final_model_checkpoint', type='model')
    model_artifact.add_file(model_filename)
    wandb.log_artifact(model_artifact)

    print(f"Saved final model at epoch {epoch}")

    print("Start testing using the best validatino model")
    # save the prediction and attention scores for patches
    model.load_state_dict(torch.load(best_model_filename)['state_dict'])
    for test_list_path in args.test_list_paths:
        # creating the test data loader
        domain_name = test_list_path.stem.split('_')[-1]  # Extracting the domain name from the filename
        test_dset = datasets.get_msi_dataset(root_dir=args.data_root, case_list_path=test_list_path, label_dict=args.label_dict)
        test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=True, num_workers=args.workers)

        # Running the model on the test set
        inference_start_time = time.time()
        probs = test(epoch, test_loader, model, device, args)
        inference_end_time = time.time()
        inference_runtime = inference_end_time - inference_start_time
        # evaluate
        test_metrics = evaluate(test_loader.dataset.df.target, probs, best_thresh)
        test_metrics_prefixed = {f"{domain_name}_{key}": value for key, value in test_metrics.items()}
        # Log each domain's test result
        wandb.log({
            "run_id": domain_name,
            **test_metrics_prefixed
        })
        print(domain_name, test_metrics)
        log_results(best_val_metrics,  f"{args.wandb_group}_{domain_name}", os.path.join(args.output_root, args.wandb_project, f"test_results_run{run_num}.csv"))

    # Create a wandb Artifact for the best model and add the file to it
    model_artifact = wandb.Artifact('best_model_checkpoint', type='model')
    model_artifact.add_file(model_filename)
    wandb.log_artifact(model_artifact)
    wandb.finish()

def test(run, loader, model, device, args):
    # Set model in test mode
    model.eval()
    # Initialize probability vector
    probs = torch.FloatTensor(len(loader)).cuda()
    # Loop through batches
    with torch.no_grad():
        for i, input in enumerate(loader):#
            ## Copy batch to GPU
            torch.cuda.reset_peak_memory_stats(device) 
            step_start_time = time.time()
            if args.method in ['ViT_MIL','DTMIL']:
                feat = input['feat_map'].float().permute(0, 3, 1, 2).cuda()
            else:
                ## Copy to GPU
                feat = input['features'].squeeze(0).cuda()
            
            ## Forward pass
            if args.method in ['GTP']:
                adj = input['adj_mtx'].float().cuda()
                mask = input['mask'].float().cuda()
                results_dict = model(feat, adj, mask)
            elif args.method in ['PatchGCN','DeepGraphConv']:
                edge_index = input['edge_index'].squeeze(0).cuda()
                edge_latent = input['edge_latent'].squeeze(0).cuda()
                # centroid = input['centroid'].squeeze(0).cuda()
                results_dict = model(feat=feat, edge_index=edge_index, edge_latent=edge_latent)
            elif args.method in ['MIL_Cluster_FC']:
                edge_index = input['edge_index'].squeeze(0).cuda()
                edge_latent = input['edge_latent'].squeeze(0).cuda()
                centroid = input['centroid'].squeeze(0).cuda()
                results_dict = model(feat=feat, edge_index=edge_index, edge_latent=edge_latent, centroid=centroid)
            # elif args.method in ['DTMIL']:
            #     mask = input['mask'].bool().cuda()
            #     tensors = NestedTensor(feat, mask)
            #     results_dict = model(tensors)
            else:
                results_dict = model(feat)
            
            logits, Y_prob, Y_hat = (results_dict[key] for key in ['logits', 'Y_prob', 'Y_hat'])
            ## Clone output to output vector
            probs[i] = Y_prob.detach()[:,1].item()


            # Calculate step runtime and memory usage
            step_end_time = time.time()
            step_runtime = step_end_time - step_start_time
            memory_gpu = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # Convert bytes to megabytes
            
            # Log the runtime and GPU memory to wandb
            wandb.log({
                'val/patches':input['features'].shape[1],
                'val/step_runtime': step_runtime,
                'val/memory_gpu': memory_gpu,
            })
            torch.cuda.reset_peak_memory_stats(device)  # Reset memory stats for the next step

    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer, lr_schedule, wd_schedule, device, args):
    # Set model in training mode
    model.train()
    # Initialize loss
    running_loss = 0.
    # Loop through batches
    for i, input in enumerate(loader):#
        torch.cuda.reset_peak_memory_stats(device) 
        step_start_time = time.time()
        
        ## Update weight decay and learning rate according to their schedule
        it = len(loader) * (run-1) + i # global training iteration
        for j, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if j == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        
        target = input['target'].long().cuda()

        if args.method in ['ViT_MIL','DTMIL']:
            feat = input['feat_map'].float().permute(0, 3, 1, 2).cuda()
        else:
            ## Copy to GPU
            feat = input['features'].squeeze(0).cuda()
            
        ## Forward pass
        if args.method == 'GTP':
            adj = input['adj_mtx'].float().cuda()
            mask = input['mask'].float().cuda()
            results_dict = model(feat, adj, mask)
            logits = results_dict['logits']
            mc1 = results_dict['mc1']
            o1 = results_dict['o1']
            ## Calculate loss
            loss = criterion(logits, target)
            loss = loss + mc1 + o1
            
        elif args.method in ['PatchGCN', 'DeepGraphConv']:
            edge_index = input['edge_index'].squeeze(0).cuda()
            edge_latent = input['edge_latent'].squeeze(0).cuda()
            results_dict = model(feat=feat, edge_index=edge_index, edge_latent=edge_latent)
            logits = results_dict['logits']
            ## Calculate loss
            loss = criterion(logits, target)
            
        elif args.method in ['MIL_Cluster_FC']:
            edge_index = input['edge_index'].squeeze(0).cuda()
            edge_latent = input['edge_latent'].squeeze(0).cuda()
            centroid = input['centroid'].squeeze(0).cuda()
            results_dict = model(feat=feat, edge_index=edge_index, edge_latent=edge_latent, centroid=centroid)
            logits = results_dict['logits']
            ## Calculate loss
            loss = criterion(logits, target)

        # elif args.method in ['DTMIL']:
        #     mask = input['mask'].bool().cuda()       
        #     tensors = NestedTensor(feat, mask)
        #     results_dict = model(tensors)
            
        else:
            results_dict = model(feat)
            logits = results_dict['logits']
            ## Calculate loss
            loss = criterion(logits, target)

        ## Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ## Store loss
        running_loss += loss.item()

        # Calculate step runtime and memory usage
        step_end_time = time.time()
        step_runtime = step_end_time - step_start_time

        # Here, use torch.cuda.max_memory_allocated() for peak memory usage
        peak_memory_usage = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # Convert bytes to megabytes
        torch.cuda.reset_peak_memory_stats(device)  # Ensure to reset for the next measurement

        # Log the runtime and GPU memory to wandb
        wandb.log({
            'train/patches':input['features'].shape[1],
            'train/step_runtime': step_runtime,
            'train/memory_gpu': peak_memory_usage,
        })

    return running_loss / len(loader)

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

# Define the function to read best hyperparameters
def find_best_hyperparameters(project_path, data_filter, encoder_filter, method_filter, metric='val_auc', config_interest=None):
    # Initialize the API and get the runs
    api = wandb.Api()
    runs = api.runs(project_path)

    # Create a list of dictionaries for each run with config and the metric
    runs_data = []
    for run in runs:
        # Flatten the config and include the metric of interest
        run_data = {f'config.{k}': v for k, v in run.config.items()}
        run_data[metric] = run.summary.get(metric)
        runs_data.append(run_data)
    
    # Create a DataFrame from the list of dictionaries
    runs_df = pd.DataFrame(runs_data)

    # Apply filters to the DataFrame
    filtered_df = runs_df[
        (runs_df['config.data'] == data_filter) &
        (runs_df['config.encoder'] == encoder_filter) &
        (runs_df['config.method'] == method_filter)
    ]

    # Check if the filtered DataFrame is empty
    if filtered_df.empty:
        raise ValueError("No runs found with the specified filters.")

    # Sort by the specified metric to find the best run
    best_run = filtered_df.sort_values(by=metric, ascending=False).iloc[0]

    # Extract the config of the best run
    # Check if the key starts with 'config.' and is in the list of interest
    if config_interest:
        best_config = {
            key.split('config.')[1]: value for key, value in best_run.items()
            if key.startswith('config.') and key.split('config.')[1] in config_interest
        }
    else:
        best_config = {
            key.split('config.')[1]: value for key, value in best_run.items()
            if key.startswith('config.')
        }
    
    return best_config

# Function to update the config with best hyperparameters
def update_args_with_best_hyperparameters(args):
    print(f"Reading fine-tuned hyperparameters from wandb {args.parameter_path}")
    best_hyperparameters = find_best_hyperparameters(args.parameter_path, args.data, args.encoder, args.method,
                                                     metric='val_auc', config_interest=['lr', 'weight_decay'])
    
    # Update the args namespace with the best hyperparameters
    for param, value in best_hyperparameters.items():
        setattr(args, param, value)
    
    print(f"Hyperparameters {best_hyperparameters} from wandb {args.parameter_path} updated!")
    
# Function to update the config with selected hyperparameters
def update_args_with_selected_hyperparameters(args):
    print(f"Reading selected hyperparameters from {args.parameter_path}")
    
    # Load hyperparameters from CSV
    df_hyperparameters = pd.read_csv(args.parameter_path)
    
    # Selecting hyperparameters based on specific criteria
    selected_row = df_hyperparameters[
        (df_hyperparameters['data'] == args.data) &
        (df_hyperparameters['encoder'] == args.encoder) &
        (df_hyperparameters['method'] == args.method)
    ].iloc[0]

    # Update the args namespace with the selected hyperparameters
    for param in ['lr', 'weight_decay', 'data', 'encoder', 'method']:  # Add other hyperparameters as needed
        if param in selected_row:
            setattr(args, param, selected_row[param])
    
    print(f"Hyperparameters {selected_row[['lr', 'weight_decay']]} from {args.parameter_path} updated!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #I/O PARAMS
    parser.add_argument('--output', type=str, default='.', help='output directory')
    parser.add_argument('--log', type=str, default='convergence.csv', help='name of log file')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--train_list_path', type=str)
    parser.add_argument('--val_list_path', type=str)
    parser.add_argument('--test_list_paths', type=list, default=[])
    parser.add_argument('--label_dict', type=dict, default=None)

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
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 10)')
    parser.add_argument('--random_seed', default=0, type=int, help='random seed')

    # Weight and Bias Config
    parser.add_argument('--wandb_project', type=str, help='name of project in wandb')
    parser.add_argument('--wandb_note', type=str, help='note of project in wandb')
    parser.add_argument('--sweep_config', type=str, help='Path to the sweep configuration YAML file')
    parser.add_argument('--parameter_path', type=str, help='Read hyperparameters after tuning')


    args = parser.parse_args()
    
    # Dim of features
    if args.ndim is None:
        if args.encoder.startswith('tres50'):
            args.ndim = 1024
        elif args.encoder.startswith('dinosmall'):
            args.ndim = 384
        elif args.encoder.startswith('dinobase'):
            args.ndim = 768
        elif args.encoder.startswith('uni'):
            args.ndim = 1024

    main(args)