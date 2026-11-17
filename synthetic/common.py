"""Shared paths, array conventions and I/O for the synthetic/ tools.

Array convention inside synthetic/: volumes are (Z, Y, X), like the training
TIFFs. The THX10 zarr is native (z, x, y); read_block() does the transpose.
Lengths are in XY pixels unless a name says "slices"; one Z slice of the
THX10 crop is ZSTEP = 8 XY pixels.
"""
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(REPO, 'synthetic')
OUT = os.path.join(HERE, 'out')            # figures / caches, git-ignored
CONFIG = os.path.join(HERE, 'config')
THX10_ZARR = os.environ.get('THX10_ZARR', '/media/cheese/Ghc_data3/THX10/thx10cropzarr')
THX10_TIFS = os.environ.get('THX10_TIFS', '/home/cheese/workspace/Data/thx10')
ZSTEP = 8                                  # XY pixels per Z slice
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))    # Gaussian sigma -> FWHM


def out_dir(*parts):
    d = os.path.join(OUT, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def open_level(level=0, store=None):
    import sys
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from inference.zarr_io import open_zarr
    return open_zarr(store or THX10_ZARR, level)


def read_block(z0, z1, y0, y1, x0, x1, level=0, store=None, arr=None):
    """Raw counts of a region as float32 (Z, Y, X)."""
    a = arr if arr is not None else open_level(level, store)
    v = a[z0:z1, x0:x1, y0:y1].read().result()
    return np.ascontiguousarray(v.transpose(0, 2, 1)).astype(np.float32)


def load_json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=conv)
    return path
