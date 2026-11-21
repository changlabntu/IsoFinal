"""Score an isotropic reconstruction of the held-out synthF stacks against the truth.

    python -m synthetic.evaluate run --method cubic                 # interpolation only: the floor
    python -m synthetic.evaluate run --method truth                 # sanity: must score perfectly
    python -m synthetic.evaluate run --method skipU_qd              # a model of inference/registry.py
    python -m synthetic.evaluate run --method MScleanQD/base_r1 --epoch 50,100
                                                                    # a training run under $LOGS/synthF/synthF/
    python -m synthetic.evaluate table [--set val]

The dataset is read from $DATASET/synthF (or --root) and runs from $LOGS; if
unset, both come from the cfg/env.json entry named by $ENV_NAME (default: cheese).

Each held-out block gives four 24 x 192 x 192 crops of the degraded stack. A
method turns a crop into a 192^3 volume; only the crop interior is scored.

Alignment. The degraded stacks were resampled in-plane at a sub-pixel shift, so
the truth is resampled to the same shift (replayed from the block seed). Along
Z a method may place slice i at any plane 8 i + s; s is estimated per method
and set by cross-correlation, reported, and removed before scoring.

Intensities are in window units ([-1, 1] = 99..673 counts). Truth and output
are clipped to [FLOOR, 1]: FLOOR is the baseline's input floor (--gamma_lo,
156 counts), below which the model is shown nothing; 1 is the window ceiling.

Numbers per method and set:
    psnr, corr      fidelity to the clipped truth (PSNR for a range of 1.8;
                    Pearson correlation); corr_u is against truth that is not
                    clipped at the ceiling
    fz50, z_res     axial resolution: correlation between output and truth as
                    a function of Z frequency (in-plane frequencies up to 0.25
                    cycles/px); fz50 is where it falls below 0.5, z_res the
                    half period there, in px (8 = the slice spacing)
    false           share of bright output voxels (above BRIGHT) with no truth
                    structure (above SUPPORT) within REACH px
    recall          share of bright truth voxels with output within REACH px
"""
import argparse
import glob
import os
import sys

import numpy as np

from synthetic.common import CONFIG, REPO, ZSTEP, load_json, out_dir, save_json

CROP_Z, CROP_XY = 24, 192
CROPS = ((0, 0, 0), (24, 0, 192), (24, 192, 0), (0, 192, 192))     # (z0 in slices, y0, x0)
MARGIN_Z, MARGIN_XY = 32, 16          # planes / px left out of the score at every crop face
PAD_Z, PAD_XY = 16, 8                 # truth read around a crop, for the shifts
FLOOR = -0.8
BRIGHT, SUPPORT, REACH = -0.6, -0.7, 3
FR_MAX = 0.25
LAGS = np.arange(-40, 41)            # planes searched for a method's Z offset (PAD_Z + MARGIN_Z allows up to 48)
SETS = ('val', 'val_narrow', 'val_wide')
NZ = CROP_Z * ZSTEP
INNER = (slice(MARGIN_Z, NZ - MARGIN_Z), slice(MARGIN_XY, CROP_XY - MARGIN_XY), slice(MARGIN_XY, CROP_XY - MARGIN_XY))
SUMS = ('n', 'so', 'st', 'soo', 'stt', 'sot', 'se', 'su', 'suu', 'sou', 'n_obright', 'n_false', 'n_tbright', 'n_found')


def degradation():
    return load_json(os.path.join(CONFIG, 'degradation.json'))


def blocks(root, split):
    seeds = {b['name']: b['seed'] for b in load_json(os.path.join(root, 'manifest.json'))['blocks']}
    out = []
    for f in sorted(glob.glob(os.path.join(root, split, 'deg', '*.tif'))):
        name = os.path.splitext(os.path.basename(f))[0]
        out.append(dict(name=name, deg=f, truth=os.path.join(root, 'gt', 'val', name + '.tif'), seed=seeds[name]))
    return out


def resample_shift(seed, deg):
    """(wx, wy) of a block: replays the draws of build_dataset.degrade_block /
    degrade.acquire up to the resampling fractions."""
    from synthetic.phantom.degrade import draw_resample, slice_artefacts
    rng = np.random.default_rng([int(seed), 15485863])
    slice_artefacts(np.zeros((48, 2, 2), np.float32), deg['noise'], rng)
    return draw_resample(deg['noise'], rng)


# ---------------------------------------------------------------- truth

_TRUTH = {}


def _truth_block(path, deg):
    """Whole truth block in window units, not clipped at the ceiling (cached per process)."""
    import tifffile
    if path not in _TRUTH:
        _TRUTH.clear()
        w, off = deg['window'], deg['noise']['offset']
        t = tifffile.imread(path).astype(np.float32)
        _TRUTH[path] = (t + (off - w['lo'])) / (w['hi'] - w['lo']) * 2 - 1
    return _TRUTH[path]


def truth_region(path, crop, deg):
    """Truth around a crop with PAD_Z / PAD_XY margins (edge-padded at the block faces)."""
    t = _truth_block(path, deg)
    z0, y0, x0 = crop[0] * ZSTEP, crop[1], crop[2]
    idx = [np.clip(np.arange(a - p, a + n + p), 0, s - 1)
           for a, n, p, s in ((z0, NZ, PAD_Z, t.shape[0]), (y0, CROP_XY, PAD_XY, t.shape[1]), (x0, CROP_XY, PAD_XY, t.shape[2]))]
    return t[np.ix_(*idx)]


def truth_aligned(path, crop, deg, s, wx, wy):
    """Truth on the grid of a method's output: T[k, y, x] = truth(8 z0 + k - s, y0 + y + wy, x0 + x + wx)."""
    from scipy import ndimage as ndi
    reg = truth_region(path, crop, deg)
    if s or wx or wy:
        reg = ndi.shift(reg, (s, -wy, -wx), order=3, mode='nearest')
    return reg[PAD_Z:PAD_Z + NZ, PAD_XY:PAD_XY + CROP_XY, PAD_XY:PAD_XY + CROP_XY]


# ---------------------------------------------------------------- methods

def cubic(v):
    """Cubic interpolation along Z: plane k of the output is slice k / 8."""
    from scipy.interpolate import CubicSpline
    z = np.arange(v.shape[0])
    return np.clip(CubicSpline(z, v, axis=0)(np.arange(v.shape[0] * ZSTEP) / ZSTEP), -1, 1).astype(np.float32)


class Model:
    """A registry model or a training run, called on one (Z, Y, X) crop in [-1, 1]."""

    def __init__(self, name, epoch=None):
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        cwd = os.getcwd()
        os.chdir(REPO)
        try:
            from inference import Engine, registry
            if name in registry.available():
                self.eng = Engine.from_registry(name, epoch=epoch)
            else:
                self.eng = Engine.from_checkpoint(run_dir(name), epoch=epoch)
        finally:
            os.chdir(cwd)

    def __call__(self, v):
        import torch
        x = torch.from_numpy(np.ascontiguousarray(v.transpose(1, 2, 0)))[None, None].float()
        with torch.no_grad():
            out = self.eng.denormalize(self.eng.full(self.eng.normalize(x).to(self.eng.device)))
        return np.ascontiguousarray(out[0, 0].float().cpu().numpy().transpose(2, 0, 1))


def env_path(key):
    """$DATASET or $LOGS; if unset, from the cfg/env.json entry named by $ENV_NAME (default: cheese)."""
    return os.environ.get(key) or load_json(os.path.join(REPO, 'cfg', 'env.json'))[os.environ.get('ENV_NAME', 'cheese')][key]


def run_dir(name):
    if os.path.isdir(name):
        return name
    return os.path.join(env_path('LOGS'), 'synthF', 'synthF', name)


# ---------------------------------------------------------------- per-crop work (runs in a process pool)

def _lag_task(task):
    """Correlation of the output with the truth displaced by each lag in LAGS."""
    out, path, crop, deg = task
    reg = np.clip(truth_region(path, crop, deg), FLOOR, 1.0)[:, PAD_XY:PAD_XY + CROP_XY, PAD_XY:PAD_XY + CROP_XY]
    sub = (INNER[0], slice(MARGIN_XY, CROP_XY - MARGIN_XY, 2), slice(MARGIN_XY, CROP_XY - MARGIN_XY, 2))
    o = np.clip(out[sub], FLOOR, 1.0).astype(np.float64)
    o -= o.mean()
    c = []
    for j in LAGS:
        t = reg[PAD_Z + MARGIN_Z - j:PAD_Z + NZ - MARGIN_Z - j, sub[1], sub[2]].astype(np.float64)
        t -= t.mean()
        c.append((o * t).sum() / max(np.sqrt((o * o).sum() * (t * t).sum()), 1e-12))
    return np.array(c)


def z_offset(curves):
    """Peak of the mean lag curve, refined with a parabola."""
    c = np.mean(curves, 0)
    i = int(np.argmax(c))
    if 0 < i < len(c) - 1:
        d = c[i - 1] - 2 * c[i] + c[i + 1]
        return float(LAGS[i] + (0.5 * (c[i - 1] - c[i + 1]) / d if d < 0 else 0.0))
    return float(LAGS[i])


def _score_task(task):
    """Sums from which every number of a set is formed (so that blocks can be pooled and resampled)."""
    from scipy import ndimage as ndi
    import scipy.fft as sfft
    out, path, crop, deg, s, wx, wy = task
    T = truth_aligned(path, crop, deg, s, wx, wy)
    tu_full = np.maximum(T, FLOOR)
    tc_full = np.minimum(tu_full, 1.0)
    oc_full = np.clip(out, FLOOR, 1.0)
    tu, tc, oc = (a[INNER].astype(np.float64) for a in (tu_full, tc_full, oc_full))
    r = dict(n=tc.size, so=oc.sum(), st=tc.sum(), soo=(oc * oc).sum(), stt=(tc * tc).sum(), sot=(oc * tc).sum(),
             se=((oc - tc) ** 2).sum(), su=tu.sum(), suu=(tu * tu).sum(), sou=(oc * tu).sum())
    # structure with no counterpart
    size = 2 * REACH + 1
    ob, tb = oc_full[INNER] > BRIGHT, tc_full[INNER] > BRIGHT
    t_near = ndi.maximum_filter(tc_full, size=size)[INNER] > SUPPORT
    o_near = ndi.maximum_filter(oc_full, size=size)[INNER] > SUPPORT
    r.update(n_obright=int(ob.sum()), n_false=int((ob & ~t_near).sum()), n_tbright=int(tb.sum()), n_found=int((tb & o_near).sum()))
    # correlation per Z frequency
    win = np.hanning(tc.shape[0])[:, None, None] * np.hanning(tc.shape[1])[None, :, None] * np.hanning(tc.shape[2])[None, None, :]
    fo = sfft.rfftn((oc - oc.mean()) * win, workers=1)
    ft = sfft.rfftn((tc - tc.mean()) * win, workers=1)
    keep = np.hypot(np.fft.fftfreq(tc.shape[1])[:, None], np.fft.rfftfreq(tc.shape[2])[None, :]) <= FR_MAX
    kz = np.minimum(np.arange(tc.shape[0]), tc.shape[0] - np.arange(tc.shape[0]))
    nb = tc.shape[0] // 2 + 1
    for key, v in (('fc_num', (fo * np.conj(ft)).real), ('fc_po', np.abs(fo) ** 2), ('fc_pt', np.abs(ft) ** 2)):
        r[key] = np.bincount(kz, weights=(v * keep[None]).sum((1, 2)), minlength=nb)
    return r


# ---------------------------------------------------------------- numbers from sums

def add(rows):
    out = {k: sum(r[k] for r in rows) for k in SUMS}
    for k in ('fc_num', 'fc_po', 'fc_pt'):
        out[k] = np.sum([np.asarray(r[k]) for r in rows], 0)
    return out


def numbers(s):
    """The reported numbers from summed statistics."""
    n = s['n']

    def corr(sa, saa, sb, sbb, sab):
        return (sab - sa * sb / n) / max(np.sqrt((saa - sa * sa / n) * (sbb - sb * sb / n)), 1e-12)
    fc = np.asarray(s['fc_num']) / np.maximum(np.sqrt(np.asarray(s['fc_po']) * np.asarray(s['fc_pt'])), 1e-30)
    fz = np.arange(len(fc)) / (2.0 * (len(fc) - 1))
    below = np.nonzero(fc[1:] < 0.5)[0]
    if len(below) == 0:
        fz50 = fz[-1]
    elif below[0] == 0:
        fz50 = fz[1]
    else:
        i = below[0] + 1
        fz50 = fz[i - 1] + (fc[i - 1] - 0.5) / (fc[i - 1] - fc[i]) * (fz[i] - fz[i - 1])
    mse = s['se'] / n
    return dict(psnr=float(10 * np.log10((1 - FLOOR) ** 2 / max(mse, 1e-12))),
                corr=float(corr(s['so'], s['soo'], s['st'], s['stt'], s['sot'])),
                corr_u=float(corr(s['so'], s['soo'], s['su'], s['suu'], s['sou'])),
                fz50=float(fz50), z_res=float(0.5 / fz50),
                false=float(s['n_false'] / s['n_obright']) if s['n_obright'] else float('nan'),
                recall=float(s['n_found'] / s['n_tbright']) if s['n_tbright'] else float('nan'),
                bright_out=float(s['n_obright'] / n), bright_truth=float(s['n_tbright'] / n))


def spread(per_block, n_boot=200, seed=0):
    """Standard error of each number over blocks (bootstrap)."""
    rng = np.random.default_rng(seed)
    draws = [numbers(add([per_block[i] for i in rng.integers(len(per_block), size=len(per_block))])) for _ in range(n_boot)]
    return {k: float(np.nanstd([d[k] for d in draws])) for k in draws[0]}


# ---------------------------------------------------------------- driver

def score_set(fn, root, split, deg, pool, truth_shift=None, limit=None):
    """`fn(crop) -> (192, 192, 192)` on every crop of a set. With `truth_shift`
    (planes) the output is the truth itself displaced by that much."""
    import tifffile
    bl = blocks(root, split)[:limit]
    tasks = []
    for b in bl:
        v = tifffile.imread(b['deg'])
        wx, wy = resample_shift(b['seed'], deg)
        for crop in CROPS:
            z0, y0, x0 = crop
            if truth_shift is None:
                out = fn(v[z0:z0 + CROP_Z, y0:y0 + CROP_XY, x0:x0 + CROP_XY])
            else:
                out = np.clip(truth_aligned(b['truth'], crop, deg, truth_shift, wx, wy), -1, 1)
            assert out.shape == (NZ, CROP_XY, CROP_XY), out.shape
            tasks.append((out.astype(np.float32), b['truth'], crop, deg, wx, wy, b['name']))
    s = z_offset(pool.map(_lag_task, [t[:4] for t in tasks], chunksize=1))
    if abs(s) >= LAGS[-1]:
        print('WARNING: Z offset at the edge of the search range (%+.1f planes)' % s, flush=True)
    rows = pool.map(_score_task, [t[:4] + (s, t[4], t[5]) for t in tasks], chunksize=1)
    per_block = [add([r for r, t in zip(rows, tasks) if t[6] == b['name']]) for b in bl]
    out = dict(set=split, n_blocks=len(bl), n_crops=len(tasks), z_offset=s, **numbers(add(per_block)))
    out['spread'] = spread(per_block)
    out['per_block'] = [dict(name=b['name'], **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()})
                        for b, p in zip(bl, per_block)]
    return out


def run(method, root, sets=SETS, epoch=None, tag=None, workers=8, limit=None):
    import multiprocessing as mp
    deg = degradation()
    truth_shift = None
    if method == 'cubic':
        fn = cubic
    elif method.startswith('truth'):
        fn, truth_shift = None, float(method[5:] or 0)
    else:
        fn = Model(method, epoch)
    tag = tag or (method.strip('/').replace('/', '_') + ('' if epoch is None else '_e%03d' % epoch))
    results = []
    with mp.get_context('spawn').Pool(workers) as pool:
        for split in sets:
            if isinstance(fn, Model):
                import torch
                torch.manual_seed(0)                      # one fixed draw of the model's dropout per crop
            r = dict(method=method, epoch=epoch, tag=tag, **score_set(fn, root, split, deg, pool, truth_shift, limit))
            save_json(os.path.join(out_dir('eval'), '%s__%s.json' % (tag, split)), r)
            results.append(r)
            print(row(r), flush=True)
    return results


HEAD = '%-28s %-10s %6s %6s %6s %7s %6s %6s %6s %6s' % ('tag', 'set', 'psnr', 'corr', 'corr_u', 'z_res', 'false', 'recall', 'bright', 'z_off')


def row(r):
    return '%-28s %-10s %6.2f %6.3f %6.3f %7.1f %6.3f %6.3f %6.4f %6.2f' % (
        r['tag'], r['set'], r['psnr'], r['corr'], r['corr_u'], r['z_res'], r['false'], r['recall'], r['bright_out'], r['z_offset'])


def table(split=None):
    rs = [load_json(f) for f in sorted(glob.glob(os.path.join(out_dir('eval'), '*__*.json')))]
    rs = [r for r in rs if split is None or r['set'] == split]
    print(HEAD)
    for r in sorted(rs, key=lambda r: (SETS.index(r['set']) if r['set'] in SETS else 9, r['tag'])):
        print(row(r))
        print('%-28s %-10s %6.2f %6.3f %6.3f %7.1f %6.3f %6.3f' % (
            '', '  +- (blocks)', *(r['spread'][k] for k in ('psnr', 'corr', 'corr_u', 'z_res', 'false', 'recall'))))
    return rs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'table'])
    ap.add_argument('--method', type=str, default='cubic', help='cubic | truth | truth+<planes> | registry name | run name or directory')
    ap.add_argument('--epoch', type=str, default=None, help='checkpoint epoch(s), comma-separated (default: the latest)')
    ap.add_argument('--root', type=str, default=None, help='dataset folder (default: $DATASET/synthF)')
    ap.add_argument('--sets', type=str, default=','.join(SETS))
    ap.add_argument('--set', type=str, default=None, help='table: only this set')
    ap.add_argument('--tag', type=str, default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None, help='only the first N blocks (for tests)')
    a = ap.parse_args()
    if a.cmd == 'table':
        table(a.set)
    else:
        print(HEAD)
        root = a.root or os.path.join(env_path('DATASET'), 'synthF')
        for e in ([int(x) for x in a.epoch.split(',')] if a.epoch else [None]):
            run(a.method, root, a.sets.split(','), e, a.tag, a.workers, a.limit)
