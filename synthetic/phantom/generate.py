"""Procedural isotropic phantom blocks (the ground truth).

A block is a noise-free isotropic volume, in camera counts above the offset,
built from the kinds of structure seen in THX10 XY planes:

  * tissue background with a slow envelope;
  * cells: a Voronoi foam with dark interiors and brighter, patchy walls;
  * tortuous fibres of varied thickness, branching, with varicosities;
  * bright compact bodies;
  * a fine multiplicative grain on all of it (granular labelling).

Everything is rendered on an isotropic grid and blurred by one isotropic
Gaussian (the lateral resolution), so every direction is as sharp as XY is in
the real data. Parameters and their per-block priors live in
config/phantom.yaml.

    from synthetic.phantom.generate import make_block, sample_scene
    truth, meta = make_block((576, 480, 480), seed=0, params=sample_scene(0, 'A'))

    python -m synthetic.phantom.generate --seed 0 --variant A     # preview figure
"""
import argparse
import os

import numpy as np
import scipy.fft as sfft
from scipy.spatial import cKDTree

from synthetic import common
from synthetic.common import CONFIG

_CFG = None


def config():
    global _CFG
    if _CFG is None:
        import yaml
        with open(os.path.join(CONFIG, 'phantom.yaml')) as f:
            _CFG = yaml.safe_load(f)
    return _CFG


def sample_scene(seed, variant='A', overrides=None):
    """Draw the per-block parameters from the priors of a variant."""
    cfg = config()
    spec = dict(cfg['base'])
    spec.update(cfg['variants'].get(variant) or {})
    spec.update(overrides or {})
    rng = np.random.default_rng([int(seed), 7919])
    P = {}
    for k, v in spec.items():
        if isinstance(v, dict) and 'lognormal' in v:
            P[k] = float(v['lognormal'][0] * np.exp(rng.normal() * v['lognormal'][1]))
        elif isinstance(v, dict) and 'uniform' in v:
            P[k] = float(rng.uniform(*v['uniform']))
        else:
            P[k] = v
    P['empty'] = bool(rng.random() < P['empty_prob'])
    P['variant'] = variant
    return P


class _Grid:
    """Frequency grid of one block, shared by all the Fourier filters."""

    def __init__(self, N):
        self.N = N
        fz = np.fft.fftfreq(N[0]).astype(np.float32)[:, None, None]
        fy = np.fft.fftfreq(N[1]).astype(np.float32)[None, :, None]
        fx = np.fft.rfftfreq(N[2]).astype(np.float32)[None, None, :]
        self.f2 = fz ** 2 + fy ** 2 + fx ** 2

    def gauss(self, sigma):
        return np.exp((-2 * np.pi ** 2 * sigma ** 2) * self.f2)

    def fft(self, v):
        return sfft.rfftn(v, workers=common.FFT_WORKERS)

    def ifft(self, F):
        return sfft.irfftn(F, s=self.N, workers=common.FFT_WORKERS).astype(np.float32)

    def random_field(self, rng, corr, slope):
        """Unit-variance Gaussian random field, power ~ (f^2 + 1/corr^2)^(-slope/2)."""
        amp = (self.f2 + np.float32((1.0 / (2 * np.pi * corr)) ** 2)) ** np.float32(-slope / 4.0)
        amp[0, 0, 0] = 0.0
        F = self.fft(rng.standard_normal(self.N, dtype=np.float32))
        g = self.ifft(F * amp)
        return g / g.std()


def _deposit(N, pts, w):
    """Trilinear deposit of weights w at float points (n, 3) -> float32 volume."""
    n = np.array(N)
    p0 = np.floor(pts).astype(np.int64)
    fr = (pts - p0).astype(np.float32)
    idx_all, w_all = [], []
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                idx = p0 + (dz, dy, dx)
                ww = w * (fr[:, 0] if dz else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * \
                    (fr[:, 2] if dx else 1 - fr[:, 2])
                ok = np.all((idx >= 0) & (idx < n), axis=1)
                idx_all.append(np.ravel_multi_index(idx[ok].T, N))
                w_all.append(ww[ok])
    if not idx_all:
        return np.zeros(N, np.float32)
    flat = np.bincount(np.concatenate(idx_all), weights=np.concatenate(w_all), minlength=int(np.prod(N)))
    return flat.astype(np.float32).reshape(N)


def _walk(rng, starts, lengths, persistence, step):
    """Persistent random walks for many fibres at once. Returns a list of (n_i, 3)."""
    n = len(starts)
    nmax = int(lengths.max() / step)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    p = np.array(starts, dtype=np.float64)
    sig = np.sqrt(2 * step / persistence)[:, None]
    pts = np.empty((nmax, n, 3), np.float32)
    for i in range(nmax):
        d += rng.normal(size=(n, 3)) * sig
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        p += step * d
        pts[i] = p
    return [pts[:max(int(L / step), 2), j].astype(np.float64) for j, L in enumerate(lengths)]


def _fibres(rng, N, P, grid, acc, step=0.5):
    """Render fibres into the Fourier accumulator. Returns per-fibre metadata."""
    nvox = float(np.prod(N))
    n_fib = int(rng.poisson(P['fibre_density'] * nvox))
    if n_fib == 0:
        return [], [], []
    n_child = int(round(n_fib * P['fibre_branch']))
    n_par = max(n_fib - n_child, 1)

    def draw(n):
        L = np.exp(rng.uniform(np.log(P['fibre_length'][0]), np.log(P['fibre_length'][1]), n))
        per = np.exp(rng.uniform(np.log(P['fibre_persistence'][0]), np.log(P['fibre_persistence'][1]), n))
        return L, per
    L, per = draw(n_par)
    lines = _walk(rng, rng.uniform(0, 1, (n_par, 3)) * N, L, per, step)
    if n_child > 0:
        par = rng.integers(0, n_par, n_child)
        starts = np.array([lines[j][rng.integers(len(lines[j]))] for j in par])
        L, per = draw(n_child)
        lines += _walk(rng, starts, 0.6 * L, per, step)
    n_fib = len(lines)
    radii = np.minimum(P['fibre_radius'][0] * np.exp(rng.normal(size=n_fib) * P['fibre_radius'][1]),
                       P['fibre_radius_max'] - 1e-3)
    # children are thinner than their parents on average
    radii[n_par:] *= 0.8
    # a minority of thick processes
    thick = rng.random(n_fib) < P['fibre_thick_fraction']
    radii[thick] = P['fibre_thick_radius'][0] * np.exp(rng.normal(size=int(thick.sum())) * P['fibre_thick_radius'][1])
    radii = np.minimum(radii, P['fibre_radius_max'] - 1e-3)
    bri = P['brightness'] * P['fibre_brightness'][0] * np.exp(rng.normal(size=n_fib) * P['fibre_brightness'][1])
    bri[thick] *= P['fibre_thick_dim']
    edges = np.array([0.0, 0.9, 1.3, 1.8, 2.5, 3.5, 5.0, P['fibre_radius_max']])
    which = np.clip(np.searchsorted(edges, radii, side='right') - 1, 0, len(edges) - 2)
    bead_pts, bead_w = [], []
    for k in range(len(edges) - 1):
        sel = np.where(which == k)[0]
        if len(sel) == 0:
            continue
        rc = max(0.5 * (edges[k] + edges[k + 1]), 0.6)
        pts_k, w_k = [], []
        for j in sel:
            pts = lines[j]
            # slow brightness variation along the fibre
            m = len(pts)
            t = np.cumsum(rng.normal(size=m)) * np.sqrt(step / 30.0)
            t -= np.linspace(t[0], t[-1], m)
            wob = np.exp(P['fibre_wobble'] * t / max(t.std(), 1e-6))
            # a line of unit linear density blurred by sigma rc peaks at 1 / (2 pi rc^2)
            pts_k.append(pts)
            w_k.append((bri[j] * 2 * np.pi * rc ** 2 * step) * wob)
            if rng.random() < P['bead_fraction']:
                s = 0.0
                while True:
                    s += rng.uniform(*P['bead_spacing'])
                    i = int(s / step)
                    if i >= m:
                        break
                    g = rng.uniform(*P['bead_gain'])
                    rb = max(P['bead_radius'], 1.3 * radii[j])
                    bead_pts.append(pts[i])
                    bead_w.append((bri[j] * wob[i] * (g - 1), rb))
        layer = _deposit(N, np.concatenate(pts_k), np.concatenate(w_k).astype(np.float32))
        acc += grid.fft(layer) * grid.gauss(rc)
    if bead_pts:
        bp = np.array(bead_pts)
        amp = np.array([b[0] for b in bead_w])
        rb = np.array([b[1] for b in bead_w])
        for lo, hi in ((0, 3.0), (3.0, 5.0), (5.0, 99.0)):
            s = (rb >= lo) & (rb < hi)
            if s.any():
                rr = float(np.median(rb[s]))
                # a point of weight w blurred by sigma peaks at w / ((2 pi)^1.5 sigma^3)
                layer = _deposit(N, bp[s], (amp[s] * (2 * np.pi) ** 1.5 * rr ** 3).astype(np.float32))
                acc += grid.fft(layer) * grid.gauss(rr)
    return lines, radii, bri


def _cells(rng, N, P, coarse=3):
    """Voronoi foam: returns (interior darkening in [0,1], wall field in [0,1])."""
    import torch
    import torch.nn.functional as F
    Nc = tuple(int(np.ceil(q / coarse)) for q in N)
    n_seed = max(int(np.prod(N) / P['cell_size'] ** 3), 8)
    seeds = rng.uniform(0, 1, (n_seed, 3)) * N
    zz, yy, xx = np.meshgrid(*[(np.arange(q) + 0.5) * coarse for q in Nc], indexing='ij')
    q = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], 1)
    # bend the walls: smooth random displacement of the query points
    from scipy import ndimage as ndi
    warp = np.stack([ndi.gaussian_filter(rng.standard_normal(Nc, dtype=np.float32), P['cell_warp_corr'] / coarse).ravel()
                     for _ in range(3)], 1)
    warp *= P['cell_warp'] / max(float(warp.std()), 1e-9)
    d, idx = cKDTree(seeds).query(q + warp, k=2, workers=common.FFT_WORKERS)
    sep = np.linalg.norm(seeds[idx[:, 0]] - seeds[idx[:, 1]], axis=1)
    face = ((d[:, 1] ** 2 - d[:, 0] ** 2) / (2 * np.maximum(sep, 1e-6))).astype(np.float32)   # distance to the wall
    cell_dark = rng.uniform(0.5, 1.0, n_seed).astype(np.float32)[idx[:, 0]]

    def up(a):
        t = torch.from_numpy(a.reshape(Nc))[None, None]
        return F.interpolate(t, size=N, mode='trilinear', align_corners=False)[0, 0].numpy()
    face_f = up(face)
    dark = up(cell_dark) * (1.0 - np.exp(-(face_f / (P['void_edge'] * 0.5 * P['cell_size'])) ** 2))
    wall = np.exp(-0.5 * (face_f / P['wall_width']) ** 2)
    return dark, wall


def _blobs(rng, N, P, vol):
    """Bright ellipsoidal bodies with soft edges, added to `vol` in place."""
    n = int(rng.poisson(P['blob_density'] * float(np.prod(N))))
    out = []
    for _ in range(n):
        if out and rng.random() < P['blob_cluster']:
            c = out[rng.integers(len(out))][0] + rng.normal(scale=P['blob_cluster_spread'], size=3)
        else:
            c = rng.uniform(0, 1, 3) * N
        r = min(P['blob_radius'][0] * np.exp(rng.normal() * P['blob_radius'][1]), P['blob_radius_max'])
        ax = r * np.array([1.0, rng.uniform(*P['blob_aspect']) ** 0.5, 1.0 / rng.uniform(*P['blob_aspect']) ** 0.5])
        b = P['brightness'] * P['blob_brightness'][0] * np.exp(rng.normal() * P['blob_brightness'][1])
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))            # random orientation
        h = int(ax.max() + 4)
        lo = np.maximum(np.floor(c).astype(int) - h, 0)
        hi = np.minimum(np.floor(c).astype(int) + h + 1, np.array(N))
        if np.any(hi <= lo):
            continue
        zz, yy, xx = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        u = (np.stack([zz - c[0], yy - c[1], xx - c[2]], -1) @ Q) / ax
        rho = np.sqrt((u ** 2).sum(-1))
        vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += (b / (1.0 + np.exp(np.clip((rho - 1.0) * r / P['blob_edge'], -30, 30)))).astype(np.float32)
        out.append((c, r, b))
    return out


def make_block(size=384, seed=0, params=None):
    """Return (truth float32 (Z, Y, X) in counts above offset, meta dict).

    `size` is an int (cube) or (Z, Y, X); the block is periodic-safe only in
    the sense that structures are cut at its faces, so callers that blur it
    further should render a margin and crop. `params` comes from
    sample_scene(); by default the scene of variant A for this seed.

    meta: params, centrelines (list of (n, 3) arrays, z y x), fibre_radius,
    fibre_brightness, blobs.
    """
    P = params if params is not None else sample_scene(seed, 'A')
    rng = np.random.default_rng([int(seed), 104729])
    N = (size,) * 3 if np.isscalar(size) else tuple(int(q) for q in size)
    grid = _Grid(N)
    meta = dict(seed=int(seed), size=list(N), params=P, centrelines=[], fibre_radius=[], fibre_brightness=[], blobs=[])
    if P['empty']:
        tex = grid.random_field(rng, P['env_corr'], 3.0)
        return (P['empty_level'] * np.exp(0.15 * tex)).astype(np.float32), meta

    # ---- background: level x envelope x cell interiors, plus walls and bright bodies
    env = np.exp(P['env_contrast'] * grid.random_field(rng, P['env_corr'], 3.5))
    dark, wall = _cells(rng, N, P)
    patch = np.exp(P['wall_patch_contrast'] * grid.random_field(rng, P['wall_patch_corr'], 3.5))
    vol = P['level'] * env * (1.0 - P['void_depth'] * dark)
    vol += (P['brightness'] * P['wall_gain'] * P['level']) * env * wall * patch
    del env, dark, wall, patch
    meta['blobs'] = _blobs(rng, N, P, vol)
    acc = grid.fft(vol.astype(np.float32))
    del vol

    # ---- fibres, each radius class blurred once by its own radius
    lines, radii, bri = _fibres(rng, N, P, grid, acc)
    vol = grid.ifft(acc)
    del acc

    # ---- granular labelling: fine multiplicative texture on everything, then
    #      the isotropic resolution of the truth
    c = P['tex_contrast']
    vol *= np.exp(c * grid.random_field(rng, P['tex_corr'], P['tex_slope']) - 0.5 * c * c)
    truth = np.clip(grid.ifft(grid.fft(vol) * grid.gauss(P['psf_sigma'])), 0, None)
    del vol
    meta.update(centrelines=lines, fibre_radius=list(map(float, radii)), fibre_brightness=list(map(float, bri)))
    return truth, meta


if __name__ == '__main__':
    import time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from synthetic.common import out_dir
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variant', type=str, default='A')
    a = ap.parse_args()
    t = time.time()
    v, m = make_block(a.size, a.seed, sample_scene(a.seed, a.variant))
    print('block %s in %.1f s, %d fibres, %d blobs; percentiles 1/50/99/99.9: %s' % (
        v.shape, time.time() - t, len(m['centrelines']), len(m['blobs']),
        np.round(np.percentile(v, [1, 50, 99, 99.9]), 1)))
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    for i, (name, sl) in enumerate((('XY', v[a.size // 2]), ('XZ', v[:, a.size // 2]), ('YZ', v[:, :, a.size // 2]))):
        ax[i].imshow(sl, cmap='gray', vmin=0, vmax=345); ax[i].set_title('truth ' + name)
    p = os.path.join(out_dir('phantom'), 'truth_preview_%s_seed%d.png' % (a.variant, a.seed))
    fig.tight_layout(); fig.savefig(p, dpi=70); print(p)
