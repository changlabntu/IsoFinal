from __future__ import print_function
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import os, shutil, time, sys, subprocess
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from utils.make_config import load_json, save_json
import json
import requests
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from dataloader.data_multi import PairedImageDataset as Dataset
from utils.get_args import get_args

os.environ['OPENBLAS_NUM_THREADS'] = '1'


def prepare_log(args, checkpoint_dir):
    """Finalize arguments and save config + model source to the checkpoint directory.

    Args:
        args: Parsed argument namespace containing all training configuration.
        checkpoint_dir: Path to the timestamped checkpoint directory.

    Returns:
        The args namespace with ``not_tracking_hparams`` populated.
    """
    args.not_tracking_hparams = ['mode', 'port', 'host', 'preload', 'test_batch_size']
    save_json(args, os.path.join(checkpoint_dir, 'config.json'))
    shutil.copy(os.path.join('models', args.models + '.py'),
                os.path.join(checkpoint_dir, args.models + '.py'))
    shutil.copy(os.path.join('cfg', args.yaml + '.yaml'),
                os.path.join(checkpoint_dir, args.yaml + '.yaml'))
    return args


if __name__ == '__main__':
    parser = get_args()

    # 1) Do a preliminary parse to get --yaml and any CLI overrides
    prelim_args = parser.parse_known_args()[0]

    # 2) Load YAML config early to obtain defaults (including models)
    with open('cfg/' + prelim_args.yaml + '.yaml', 'rt') as f:
        json_args = argparse.Namespace()
        json_args.__dict__.update(yaml.safe_load(f)['train'])

    # 3) Resolve model name: CLI --models overrides YAML; otherwise use YAML
    models_name = getattr(prelim_args, 'models', None) or getattr(json_args, 'models', None)
    if models_name is None:
        raise ValueError("Model name not specified. Provide --models on the CLI or set 'models' in the YAML under 'train'.")

    # 4) Import model and augment parser with model-specific args
    GAN = getattr(__import__('models.' + models_name), models_name).GAN
    parser = GAN.add_model_specific_args(parser)

    # 5) Final parse with YAML defaults as the namespace, allowing CLI to override
    args = parser.parse_args(namespace=json_args)

    # environment file
    with open('cfg/env.json', 'r') as f:
        configs = json.load(f)[args.env]

    os.environ.setdefault('DATASET', configs['DATASET'])
    os.environ.setdefault('LOGS', configs['LOGS'])

    if args.env is not None:
        load_dotenv('cfg/.' + args.env)
    else:
        load_dotenv('cfg/.t09')

    # Validate required arguments
    if args.prj is None:
        raise ValueError("--prj must be specified (either via CLI or in YAML config)")

    # Finalize Arguments and create files for logging
    args.bash = ' '.join(sys.argv)

    def get_git_hash():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"]
            ).decode().strip()
        except Exception:
            return "unknown"
    args.git_hash = get_git_hash()

    logs_root = os.environ.get('LOGS')
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints = os.path.join(logs_root, args.dataset, args.prj, 'checkpoints', run_timestamp)
    os.makedirs(checkpoints, exist_ok=True)

    args = prepare_log(args, checkpoints)
    print(args)

    # Load Dataset and DataLoader
    train_set = Dataset(root=os.path.join(os.environ.get('DATASET'), args.dataset, 'train'),
                        path=args.direction,
                        config=args, mode='train')#, index=None, filenames=True)

    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)

    try:
        eval_set = Dataset(root=os.path.join(os.environ.get('DATASET'), args.dataset, 'val'),
                           path=args.direction,
                           config=args, mode='test')#, index=None, filenames=True)
        eval_loader = DataLoader(dataset=eval_set, num_workers=1, batch_size=1, shuffle=False,
                                 pin_memory=True)
    except (FileNotFoundError, ValueError) as e:
        eval_loader = None
        print(f'No validation set: {e}')

    # preload
    if args.preload:
        tini = time.time()
        print('Preloading...')
        for i, x in enumerate(tqdm(train_loader)):
            pass
        if eval_loader is not None:
            for i, x in enumerate(tqdm(eval_loader)):
                pass
        print('Preloading time: ' + str(time.time() - tini))

    # Resolve MLflow tracking: an http:// server (CLI/env) OR a local bare NAME.
    # In local mode --tracking_uri is just a NAME -> $LOGS/NAME.db + $LOGS/NAME/
    # (defaults to 'mlflow' when unset). A raw sqlite:/// path is NOT accepted anymore.
    http_uri = args.tracking_uri or configs.get('TRACKING_URI')
    artifact_location = None  # server governs artifacts; only set for local SQLite

    if http_uri and http_uri.startswith("http"):
        try:
            requests.get(f"{http_uri}/health", timeout=2)
            tracking_uri = http_uri
            print(f"MLflow: using server at {tracking_uri}")
        except requests.RequestException:
            raise RuntimeError(
                f"MLflow server {http_uri} unreachable. "
                "Start the server or remove TRACKING_URI from cfg/env.json to use local SQLite."
            )
    else:
        os.makedirs(logs_root, exist_ok=True)
        name = args.tracking_uri or 'mlflow'
        local_db = os.path.join(logs_root, f'{name}.db')
        artifact_dir = os.path.join(logs_root, name)
        tracking_uri = f"sqlite:///{local_db}"
        artifact_location = f"file:{artifact_dir}"
        print(f"MLflow: using local SQLite at {local_db}")
        print(f"MLflow: artifacts under {artifact_dir}")

    mlf_logger = MLFlowLogger(
        experiment_name=args.dataset,
        run_name=args.prj.strip('/').replace('/', '·') + '_' + run_timestamp,
        tracking_uri=tracking_uri,
        artifact_location=artifact_location,
        tags={
            'env': args.env,
            'yaml_config': args.yaml,
            'git_hash': args.git_hash,
            'models': args.models,
            'prj': args.prj,
        }
    )

    net = GAN(hparams=args, train_loader=train_loader, eval_loader=eval_loader, checkpoints=checkpoints)

    "Please use `Trainer(accelerator='gpu', devices=-1)` instead."
    trainer = pl.Trainer(gpus=-1, strategy='ddp',
                         max_epochs=args.n_epochs, # progress_bar_refresh_rate=20
                         logger=mlf_logger,
                         enable_checkpointing=False, log_every_n_steps=100,
                         check_val_every_n_epoch=1, accumulate_grad_batches=2)
    try:
        if eval_loader is not None:
            trainer.fit(net, train_loader, eval_loader)
        else:
            trainer.fit(net, train_loader)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except Exception as e:
        if 'mlflow' in str(e).lower():
            print("\nERROR: MLflow Tracking Server disconnected during training.")
            print("Checkpoints are saved up to the last epoch_save interval.")
        raise

