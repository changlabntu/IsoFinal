"""Generic staged inference API for this repo's models.

    from inference import Engine
    eng = Engine.from_checkpoint('$LOGS/{dataset}/{prj}')
    scale_latents, indices = eng.encode(x)      # x: (B, C, Y, X, Z), normalized
    iso = eng.decode(scale_latents)             # (B, 1, Y', X', Z')
    iso = eng.full(x)                           # encode + decode in one call
    recon = eng.reconstruction(scale_latents)   # 2D VQ-head recon, no Z upsampling
    scale_latents = eng.latents_from_indices(indices)  # codec round-trip

Pure tensor API: no CLI, no file I/O, no normalization — callers own those,
plus device/dtype placement. Independent of test/ (see test/README.md for the
scenario tooling this does not replace).
"""

from inference.engine import Engine
from inference.load import load_model, resolve_checkpoint_dir, available_epochs
from inference.normalize import normalize, invert_gamma
from inference.registry import MODELS, available, get, register

__all__ = ['Engine', 'load_model', 'resolve_checkpoint_dir', 'available_epochs',
           'normalize', 'invert_gamma', 'MODELS', 'available', 'get', 'register']
