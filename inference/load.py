"""Checkpoint loader for the inference package.

A deliberate self-contained twin of test/load.py (the package must not depend
on testing code); only shared utils are imported. Semantics are identical:

Every training run snapshots into $LOGS/{dataset}/{prj}/checkpoints/{timestamp}/:
  - config.json                     full hparams of the run
  - {models}.py                     source snapshot of the model file
  - {name}_model_epoch_{N}.pth      whole pickled nn.Module per component

The loader rebuilds the GAN from the snapshot + config.json, then swaps in the
pickled component modules discovered from the filenames on disk — so it works
for the entire lineage (old single-VQ names like quant_conv/quantize as well as
the multi-scale quantizers/quant_convs/post_quant_convs/inject_convs).
"""

import os
import re
import sys
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
from utils.model_utils import import_model, read_json_to_args  # noqa: E402


def resolve_checkpoint_dir(path):
    """Resolve a user-supplied path to a timestamped checkpoint directory.

    Accepts (a) the timestamped dir itself, (b) a `.../checkpoints` parent, or
    (c) an experiment root `$LOGS/{dataset}/{prj}` — (b) and (c) pick the
    newest timestamp that actually contains .pth files.
    """
    path = os.path.abspath(path)
    if os.path.isfile(os.path.join(path, 'config.json')):
        return path
    base = path if os.path.basename(path) == 'checkpoints' else os.path.join(path, 'checkpoints')
    if not os.path.isdir(base):
        raise FileNotFoundError(f'No checkpoint directory found at {path}')
    candidates = [d for d in sorted(glob.glob(os.path.join(base, '*')), reverse=True)
                  if glob.glob(os.path.join(d, '*_model_epoch_*.pth'))]
    if not candidates:
        raise FileNotFoundError(f'No timestamped run with .pth files under {base}')
    return candidates[0]


def available_epochs(ckpt_dir):
    """Epochs for which every component of the run has a .pth file, sorted."""
    by_epoch = {}
    names = set()
    for f in glob.glob(os.path.join(ckpt_dir, '*_model_epoch_*.pth')):
        m = re.match(r'(.+)_model_epoch_(\d+)\.pth$', os.path.basename(f))
        if m:
            names.add(m.group(1))
            by_epoch.setdefault(int(m.group(2)), set()).add(m.group(1))
    return sorted(e for e, comps in by_epoch.items() if comps == names)


def load_model(ckpt_dir, epoch=None, device=None, model_file=None):
    """Rebuild the GAN of a run and load its component weights.

    Args:
        ckpt_dir: any path accepted by resolve_checkpoint_dir.
        epoch: checkpoint epoch to load (default: latest complete one).
        device: 'cuda' / 'cpu' (default: cuda if available).
        model_file: path to a model .py whose GAN class is used INSTEAD of the
            run's source snapshot — for running a refactored models/*.py
            against weights trained with the original code. The class must keep
            the run's component module names (encoder, quantizers, net_g, ...).

    Returns:
        (gan, args): the model in eval mode on `device`, and the run's config.
    """
    ckpt_dir = resolve_checkpoint_dir(ckpt_dir)
    args = read_json_to_args(os.path.join(ckpt_dir, 'config.json'))

    if epoch is None:
        epochs = available_epochs(ckpt_dir)
        if not epochs:
            raise FileNotFoundError(f'No complete epoch checkpoints in {ckpt_dir}')
        epoch = epochs[-1]

    # GAN.__init__ reads ldm/{ldmyaml}.yaml relative to cwd
    os.chdir(REPO_ROOT)

    # Prefer the snapshot so the code matches the weights even if models/ moved on
    snapshot = os.path.join(ckpt_dir, args.models + '.py')
    if model_file is not None:
        model_file = os.path.abspath(model_file)
        print(f'Model source override: {model_file}')
        GAN = import_model(os.path.dirname(model_file),
                           os.path.splitext(os.path.basename(model_file))[0]).GAN
    elif os.path.isfile(snapshot):
        GAN = import_model(ckpt_dir, args.models).GAN
    else:
        GAN = getattr(__import__('models.' + args.models), args.models).GAN

    gan = GAN(hparams=args, train_loader=None, eval_loader=None, checkpoints=ckpt_dir)

    components = sorted(
        re.match(r'(.+)_model_epoch_', os.path.basename(f)).group(1)
        for f in glob.glob(os.path.join(ckpt_dir, f'*_model_epoch_{epoch}.pth'))
    )
    if not components:
        raise FileNotFoundError(f'No .pth files for epoch {epoch} in {ckpt_dir}')
    all_names = set()
    for f in glob.glob(os.path.join(ckpt_dir, '*_model_epoch_*.pth')):
        m = re.match(r'(.+)_model_epoch_\d+\.pth$', os.path.basename(f))
        if m:
            all_names.add(m.group(1))
    missing = all_names - set(components)
    if missing:
        raise FileNotFoundError(
            f'Epoch {epoch} in {ckpt_dir} is missing component(s) '
            f'{", ".join(sorted(missing))} — loading it would leave them at random '
            f'init. Complete epochs: {available_epochs(ckpt_dir)}')
    for name in components:
        path = os.path.join(ckpt_dir, f'{name}_model_epoch_{epoch}.pth')
        # whole pickled nn.Modules (this repo's own artifacts), so the
        # weights_only=True default of torch>=2.6 cannot load them
        setattr(gan, name, torch.load(path, map_location='cpu', weights_only=False))
    print(f'Loaded epoch {epoch} components: {", ".join(components)}')

    if device is not None:
        gan = gan.eval().to(device)
    else:
        # cuda.is_available() can be True while init still fails (e.g. driver
        # older than the torch build) — fall back to CPU instead of dying
        try:
            gan = gan.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
        except RuntimeError as e:
            print(f'CUDA unusable ({str(e).splitlines()[0]}); falling back to CPU')
            gan = gan.eval().to('cpu')
    return gan, args
