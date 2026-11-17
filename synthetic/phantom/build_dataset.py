"""Write the synthetic dataset: degraded stacks for training, and held-out
blocks with their isotropic truth for scoring.

One block = one rendered phantom of (384 + margin)^3, degraded with the THX10
transfer (its axial width scaled per block), cropped to the central 384^3,
every 8th slice kept, camera noise and in-plane resampling applied, windowed
to [-1, 1]. Slice i of a
degraded stack sits at truth plane 8 i.

Layout under $DATASET/<name>/ (loader-compatible, truth outside train/val):

    train/deg/*.tif          (48, 384, 384) float32, width drawn from the range
    val/deg/*.tif            held-out blocks at the nominal width
    val_narrow/deg/*.tif     the same blocks at the narrow end of the range
    val_wide/deg/*.tif       the same blocks at the wide end
    gt/val/*.tif             (384, 384, 384) float16, counts above offset
    gt/val/*.npz             fibre centrelines, radii, brightness, blobs
    manifest.json            per-block seed, variant, width, scene parameters

    python -m synthetic.phantom.build_dataset --name synthA --n_train 400 --n_val 24
    python -m synthetic.phantom.build_dataset --preview 24 --tag r1    # calibration set
"""
import argparse
import json
import os
import time

import numpy as np

from synthetic import common
from synthetic.common import CONFIG, OUT, ZSTEP, load_json, save_json

SHAPE = (48, 384, 384)          # degraded stack (Z, Y, X)
MARGIN_Z, MARGIN_XY = 96, 48    # rendered margin on each side (XY px)
VARIANTS = ('A', 'B', 'C')
DEFAULT_WIDTH = dict(nominal=1.1, lo=1.0, hi=1.3)


def width_range(deg):
    return deg.get('width') or DEFAULT_WIDTH


def render(seed, variant, overrides=None):
    """Truth with margins: ((Z, Y, X) float32, meta)."""
    from synthetic.phantom.generate import make_block, sample_scene
    N = (SHAPE[0] * ZSTEP + 2 * MARGIN_Z, SHAPE[1] + 2 * MARGIN_XY, SHAPE[2] + 2 * MARGIN_XY)
    return make_block(N, seed, sample_scene(seed, variant, overrides))


def crop(v):
    return v[MARGIN_Z:v.shape[0] - MARGIN_Z, MARGIN_XY:v.shape[1] - MARGIN_XY, MARGIN_XY:v.shape[2] - MARGIN_XY]


def degrade_block(truth_full, deg, width_scale, seed, with_noise=True):
    """Rendered truth (with margins) -> degraded stack in [-1, 1], (48, 384, 384)."""
    from synthetic.phantom.degrade import acquire, apply_transfer, scale_transfer, to_window
    rng = np.random.default_rng([int(seed), 15485863])
    blurred = apply_transfer(truth_full, scale_transfer(deg['transfer'], width_scale), 0, 0)
    z, y, x = truth_full.shape
    # one spare pixel in Y and X for the in-plane resampling
    low = np.clip(blurred[MARGIN_Z:z - MARGIN_Z:ZSTEP, MARGIN_XY:y - MARGIN_XY + 1, MARGIN_XY:x - MARGIN_XY + 1], 0, None)
    raw = acquire(low, deg['noise'], rng) if with_noise else low[:, :-1, :-1] + deg['noise']['offset']
    return to_window(raw, deg['window']['lo'], deg['window']['hi'])


def _init(threads):
    common.FFT_WORKERS = threads
    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        pass


def _job(job):
    import tifffile
    deg = load_json(os.path.join(CONFIG, 'degradation.json'))
    t = time.time()
    truth_full, meta = render(job['seed'], job['variant'], job.get('overrides'))
    for path, w in job['outputs']:
        tifffile.imwrite(path, degrade_block(truth_full, deg, w, job['seed']))
    if job.get('gt'):
        truth = np.ascontiguousarray(crop(truth_full))
        tifffile.imwrite(job['gt'] + '.tif', truth.astype(np.float16))
        off = np.array([MARGIN_Z, MARGIN_XY, MARGIN_XY])
        lines = [l - off for l in meta['centrelines']]
        np.savez_compressed(job['gt'] + '.npz',
                            points=np.concatenate(lines).astype(np.float32) if lines else np.zeros((0, 3), np.float32),
                            line_len=np.array([len(l) for l in lines]),
                            fibre_radius=np.array(meta['fibre_radius'], np.float32),
                            fibre_brightness=np.array(meta['fibre_brightness'], np.float32),
                            blobs=np.array([[*(c - off), r, b] for c, r, b in meta['blobs']], np.float32).reshape(-1, 5))
    P = {k: v for k, v in meta['params'].items() if not isinstance(v, (list, dict))}
    return dict(name=job['name'], seed=job['seed'], variant=job['variant'], widths=[w for _, w in job['outputs']],
                n_fibres=len(meta['centrelines']), n_blobs=len(meta['blobs']), params=P, seconds=time.time() - t)


def run_jobs(jobs, workers=6, threads=4):
    import multiprocessing as mp
    out = []
    with mp.get_context('spawn').Pool(workers, initializer=_init, initargs=(threads,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            out.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print('  %d / %d blocks (last %.0f s)' % (i + 1, len(jobs), r['seconds']), flush=True)
    return sorted(out, key=lambda r: r['name'])


def train_jobs(root, n, seed0, deg, variants=VARIANTS):
    w = width_range(deg)
    rng = np.random.default_rng(seed0)
    d = os.path.join(root, 'train', 'deg')
    os.makedirs(d, exist_ok=True)
    jobs = []
    for i in range(n):
        name = 's%05d' % i
        jobs.append(dict(name=name, seed=seed0 + i, variant=variants[i % len(variants)],
                         outputs=[(os.path.join(d, name + '.tif'), float(rng.uniform(w['lo'], w['hi'])))]))
    return jobs


def val_jobs(root, n, seed0, deg, variants=VARIANTS):
    w = width_range(deg)
    for sub in ('val/deg', 'val_narrow/deg', 'val_wide/deg', 'gt/val'):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    jobs = []
    for i in range(n):
        name = 'v%05d' % i
        jobs.append(dict(name=name, seed=seed0 + i, variant=variants[i % len(variants)],
                         overrides=dict(empty_prob=0.0),          # an empty block cannot be scored
                         outputs=[(os.path.join(root, 'val', 'deg', name + '.tif'), w['nominal']),
                                  (os.path.join(root, 'val_narrow', 'deg', name + '.tif'), w['lo']),
                                  (os.path.join(root, 'val_wide', 'deg', name + '.tif'), w['hi'])],
                         gt=os.path.join(root, 'gt', 'val', name)))
    return jobs


def preview(n, tag, variants=VARIANTS, workers=6, seed0=500000, overrides=None):
    """Small calibration set in synthetic/out/phantom/preview_<tag>/."""
    deg = load_json(os.path.join(CONFIG, 'degradation.json'))
    d = os.path.join(OUT, 'phantom', 'preview_%s' % tag)
    os.makedirs(d, exist_ok=True)
    w = width_range(deg)['nominal']
    jobs = [dict(name='p%04d' % i, seed=seed0 + i, variant=variants[i % len(variants)], overrides=overrides,
                 outputs=[(os.path.join(d, 'p%04d_%s.tif' % (i, variants[i % len(variants)])), w)]) for i in range(n)]
    res = run_jobs(jobs, workers)
    save_json(os.path.join(d, 'manifest.json'), res)
    return d


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', type=str, default='synthA')
    ap.add_argument('--n_train', type=int, default=400)
    ap.add_argument('--n_val', type=int, default=24)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--variants', type=str, default=','.join(VARIANTS))
    ap.add_argument('--preview', type=int, default=0, help='write this many calibration stacks instead')
    ap.add_argument('--tag', type=str, default='r0')
    a = ap.parse_args()
    variants = tuple(a.variants.split(','))
    if a.preview:
        print(preview(a.preview, a.tag, variants, a.workers))
    else:
        deg = load_json(os.path.join(CONFIG, 'degradation.json'))
        env = load_json(os.path.join(common.REPO, 'cfg', 'env.json'))
        root = os.path.join(os.environ.get('DATASET') or env['cheese']['DATASET'], a.name)
        os.makedirs(root, exist_ok=True)
        jobs = val_jobs(root, a.n_val, 900000, deg, variants) + train_jobs(root, a.n_train, 100000, deg, variants)
        print('%d blocks -> %s' % (len(jobs), root), flush=True)
        res = run_jobs(jobs, a.workers)
        save_json(os.path.join(root, 'manifest.json'), dict(width=width_range(deg), transfer=deg['transfer'],
                                                            noise=deg['noise'], window=deg['window'], blocks=res))
        print('done')
