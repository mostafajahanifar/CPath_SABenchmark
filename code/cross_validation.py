import os
from pathlib import Path
from train_validate_test import main
import pandas as pd
import json
from config import get_config

configs = get_config()

exp_name = (
    f"PanMSI__{configs.design}__{configs.encoder}"
)
if isinstance(configs.additional_desc, str):
    add_desc = configs.additional_desc
    exp_name = exp_name + "__" + add_desc



for ex in configs.folds:
    # setting saving and logging directories directory for the training
    log_dir = Path(os.path.join(configs.output_root, exp_name, 'logs'))
    os.makedirs(log_dir, exist_ok=True)
    configs.log = os.path.join(log_dir, f"experiment_{ex}_run_{run}.log")

    # writing the configs
    os.makedirs(os.path.join(configs.output_root, exp_name, 'configs'), exist_ok=True)
    with open(os.path.join(configs.output_root, exp_name, 'configs', f"experiment_{ex}.json"), 'w') as f:
        json.dump(vars(configs), f, indent=4)

    # perform 3 independant runs
    for run in range(3):
        save_dir = Path(
            os.path.join(configs.dir_checkpoint, exp_name, f"experiment_{ex}", f"run_{run}")
        )
        os.makedirs(save_dir, exist_ok=True)
        configs.output = save_dir
        # setting the logger
        print(f"==== Experiment name: {exp_name} ====")
        print(f"== Start experiment {ex} - Run {run}: Training & Validation ==")

        # Getting the list for trianing, validation, and test cases
        configs.train_list_path = list(Path(configs.experiment_list_root+configs.design).glob(f"train_{ex}_{run}_*.csv"))[0]
        configs.val_list_path = list(Path(configs.experiment_list_root+configs.design).glob(f"val_{ex}_{run}_*.csv"))[0]
        configs.test_list_paths = list(Path(configs.experiment_list_root+configs.design).glob(f"test_{ex}_*.csv"))

        # Dim of features
        if configs.ndim is None:
            if configs.encoder.startswith('tres50'):
                configs.ndim = 1024
            elif configs.encoder.startswith('dinosmall'):
                configs.ndim = 384
            elif configs.encoder.startswith('dinobase'):
                configs.ndim = 768
            elif configs.encoder.startswith('uni'):
                configs.ndim = 1024

        configs.mccv = run+1
        configs.wandb_project = exp_name
        configs.wandb_note = f"run_{run+1}"

        main(configs)
