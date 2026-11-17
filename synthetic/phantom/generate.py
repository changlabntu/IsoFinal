"""Procedural isotropic phantom blocks (the ground truth).

A block is a noise-free isotropic volume, in camera counts above the offset,
of the kinds of structure seen in THX10: sparse curved fibres with
varicosities, hollow cell bodies, puncta, and a dim textured background. It is
rendered on an isotropic grid and blurred by the same (lateral) Gaussian in all
three axes, so every direction is as sharp as XY is in the real data.

    from synthetic.phantom.generate import make_block, DEFAULTS
    truth, meta = make_block(384, seed=0)

    python -m synthetic.phantom.generate --size 256 --seed 0   # preview figure
"""
import argparse
import os

import numpy as np
import scipy.fft as sfft

DEFAULTS = dict(
    pad=24,                       # rendered margin, cropped away
    psf_sigma=1.6,                # isotropic blur of the truth (px), the lateral resolution
    # fibres
    fibre_density=1.6e-6,         # fibres started per voxel
    fibre_length=(150.0, 900.0),  # px, log-uniform
    fibre_radius=(1.2, 0.45),     # lognormal: median px, sigma of log
    fibre_radius_max=6.0,
    fibre_persistence=60.0,       # px over which direction decorrelates
    fibre_brightness=(140.0, 0.6),  # lognormal peak counts: median, sigma of log
    fibre_cluster=0.6,            # fraction of fibres started near another fibre (bundles / arbours)
    varicosity_spacing=(18.0, 60.0),
    varicosity_gain=(1.5, 3.5),
    varicosity_radius=2.2,        # extra Gaussian radius of a varicosity (px)
    # cell bodies (hollow shells)
    soma_density=2.0e-8,
    soma_radius=(14.0, 26.0),
    soma_shell=2.0,
    soma_brightness=(40.0, 110.0),
    # puncta
    punctum_density=1.5e-6,
    punctum_brightness=(15.0, 0.7),
    punctum_sigma=1.2,
    # background texture: Gaussian random field with a power-law spectrum
    bg_level=24.0,                # mean counts above offset
    bg_contrast=0.35,             # std / mean of the texture
    bg_slope=2.6,                 # 3D power spectrum ~ f^-slope
    bg_corr=80.0,                 # px, flattening scale at low frequency
)


def _gauss_filter_fft(v, sigma):
    """Isotropic Gaussian blur by FFT (periodic; callers keep a margin)."""
    if sigma <= 0:
        return v
    F = sfft.rfftn(v, workers=-1)
    fz = np.fft.fftfreq(v.shape[0])[:, None, None]
    fy = np.fft.fftfreq(v.shape[1])[None, :, None]
    fx = np.fft.rfftfreq(v.shape[2])[None, None, :]
    F *= np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fz ** 2 + fy ** 2 + fx ** 2))
    return sfft.irfftn(F, s=v.shape, workers=-1).astype(np.float32)


def _splat(vol, pts, w):
    """Trilinear deposit of weights w at float points pts (n, 3) = (z, y, x)."""
    n = np.array(vol.shape)
    p0 = np.floor(pts).astype(np.int64)
    fr = pts - p0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                idx = p0 + (dz, dy, dx)
                ww = w * (fr[:, 0] if dz else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * \
                    (fr[:, 2] if dx else 1 - fr[:, 2])
                ok = np.all((idx >= 0) & (idx < n), axis=1)
                np.add.at(vol, (idx[ok, 0], idx[ok, 1], idx[ok, 2]), ww[ok])


def _fibre_path(rng, start, length, persistence, step=0.5):
    """Persistent random walk in 3D with isotropic initial direction."""
    n = int(length / step)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    # Ornstein-Uhlenbeck-like turning: small random rotations each step
    kicks = rng.normal(size=(n, 3)) * np.sqrt(2 * step / persistence)
    pts = np.empty((n, 3))
    p = np.array(start, dtype=float)
    for i in range(n):
        d = d + kicks[i]
        d /= np.linalg.norm(d)
        p = p + step * d
        pts[i] = p
    return pts


def make_block(size=384, seed=0, params=None, return_parts=False):
    """Return (truth float32, meta dict). `size` is an int (cube) or (Z, Y, X).

    meta['centrelines'] is a list of (n, 3) float arrays (z, y, x) in block
    coordinates, with meta['fibre_radius'] / ['fibre_brightness'] per fibre.
    """
    P = dict(DEFAULTS)
    P.update(params or {})
    rng = np.random.default_rng(seed)
    pad = int(P['pad'])
    size3 = (size,) * 3 if np.isscalar(size) else tuple(int(q) for q in size)
    N = tuple(q + 2 * pad for q in size3)           # rendered (Z, Y, X) shape
    nvox = float(np.prod(N))
    step = 0.5

    # ---- fibres, binned by radius so each bin is blurred once
    n_fib = rng.poisson(P['fibre_density'] * nvox)
    radius_bins = np.array([0.0, 0.8, 1.2, 1.7, 2.4, 3.3, 4.5, P['fibre_radius_max']])
    layers = [np.zeros(N, np.float32) for _ in range(len(radius_bins) - 1)]
    vari = np.zeros(N, np.float32)
    lines, radii, brights = [], [], []
    starts = []
    for _ in range(n_fib):
        if starts and rng.random() < P['fibre_cluster']:
            ref = lines[rng.integers(len(lines))]
            start = ref[rng.integers(len(ref))] + rng.normal(scale=6.0, size=3)
        else:
            start = rng.uniform(0, 1, size=3) * N
        starts.append(start)
        length = np.exp(rng.uniform(np.log(P['fibre_length'][0]), np.log(P['fibre_length'][1])))
        pts = _fibre_path(rng, start, length, P['fibre_persistence'], step)
        r = min(P['fibre_radius'][0] * np.exp(rng.normal() * P['fibre_radius'][1]), P['fibre_radius_max'] - 1e-3)
        bri = P['fibre_brightness'][0] * np.exp(rng.normal() * P['fibre_brightness'][1])
        # a line of unit linear density blurred by sigma r peaks at 1 / (2 pi r^2)
        lin = bri * 2 * np.pi * max(r, 0.6) ** 2
        k = int(np.searchsorted(radius_bins, r, side='right') - 1)
        _splat(layers[k], pts, np.full(len(pts), lin * step, np.float32))
        # varicosities
        s = 0.0
        while True:
            s += rng.uniform(*P['varicosity_spacing'])
            i = int(s / step)
            if i >= len(pts):
                break
            g = rng.uniform(*P['varicosity_gain'])
            # a point of weight w blurred by sigma peaks at w / ((2 pi)^1.5 sigma^3)
            w = bri * (g - 1) * (2 * np.pi) ** 1.5 * P['varicosity_radius'] ** 3
            _splat(vari, pts[i:i + 1], np.array([w], np.float32))
        lines.append(pts)
        radii.append(r)
        brights.append(bri)
    vol = np.zeros(N, np.float32)
    for k, layer in enumerate(layers):
        if layer.any():
            rc = max(0.5 * (radius_bins[k] + radius_bins[k + 1]), 0.6)
            vol += _gauss_filter_fft(layer, rc)
    if vari.any():
        vol += _gauss_filter_fft(vari, P['varicosity_radius'])
    fibres = vol.copy() if return_parts else None

    # ---- puncta
    n_p = rng.poisson(P['punctum_density'] * nvox)
    if n_p:
        pv = np.zeros(N, np.float32)
        pts = rng.uniform(0, 1, size=(n_p, 3)) * (np.array(N) - 1)
        b = P['punctum_brightness'][0] * np.exp(rng.normal(size=n_p) * P['punctum_brightness'][1])
        _splat(pv, pts, (b * (2 * np.pi) ** 1.5 * P['punctum_sigma'] ** 3).astype(np.float32))
        vol += _gauss_filter_fft(pv, P['punctum_sigma'])

    # ---- cell bodies: hollow shells
    n_s = rng.poisson(P['soma_density'] * nvox)
    somas = []
    for _ in range(n_s):
        c = rng.uniform(0, 1, size=3) * N
        R = rng.uniform(*P['soma_radius'])
        b = rng.uniform(*P['soma_brightness'])
        h = int(R + 4 * P['soma_shell'] + 2)
        lo = np.maximum(np.floor(c).astype(int) - h, 0)
        hi = np.minimum(np.floor(c).astype(int) + h + 1, np.array(N))
        zz, yy, xx = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        dist = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
        vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += (b * np.exp(-0.5 * ((dist - R) / P['soma_shell']) ** 2)).astype(np.float32)
        somas.append((c - pad, R, b))

    # ---- background texture
    if P['bg_level'] > 0:
        fz = np.fft.fftfreq(N[0])[:, None, None]
        fy = np.fft.fftfreq(N[1])[None, :, None]
        fx = np.fft.rfftfreq(N[2])[None, None, :]
        f2 = fz ** 2 + fy ** 2 + fx ** 2
        amp = (f2 + (1.0 / P['bg_corr']) ** 2) ** (-P['bg_slope'] / 4.0)
        amp[0, 0, 0] = 0.0
        white = sfft.rfftn(rng.normal(size=N).astype(np.float32), workers=-1)
        g = sfft.irfftn(white * amp, s=N, workers=-1)
        g = g / g.std()
        s = np.sqrt(np.log(1 + P['bg_contrast'] ** 2))       # log-normal: positive, right-skewed
        vol += (P['bg_level'] * np.exp(s * g - 0.5 * s * s)).astype(np.float32)

    # ---- isotropic resolution of the truth
    vol = _gauss_filter_fft(vol, P['psf_sigma'])
    sl = tuple(slice(pad, pad + q) for q in size3)
    truth = np.ascontiguousarray(np.clip(vol[sl], 0, None))
    meta = dict(seed=int(seed), size=list(size3), params=P,
                centrelines=[l - pad for l in lines], fibre_radius=radii, fibre_brightness=brights,
                somas=somas)
    if return_parts:
        meta['fibres_only'] = _gauss_filter_fft(fibres, P['psf_sigma'])[sl]
    return truth, meta


if __name__ == '__main__':
    import time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from synthetic.common import out_dir
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    t = time.time()
    v, m = make_block(a.size, a.seed)
    print('block %s in %.1f s, %d fibres, %d somas; percentiles 1/50/99/99.9: %s' % (
        v.shape, time.time() - t, len(m['centrelines']), len(m['somas']),
        np.round(np.percentile(v, [1, 50, 99, 99.9]), 1)))
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    for i, (name, sl) in enumerate((('XY', v[a.size // 2]), ('XZ', v[:, a.size // 2]), ('YZ', v[:, :, a.size // 2]))):
        ax[i].imshow(sl, cmap='gray', vmin=0, vmax=np.percentile(v, 99.9)); ax[i].set_title('truth ' + name)
    p = os.path.join(out_dir('phantom'), 'truth_preview_seed%d.png' % a.seed)
    fig.tight_layout(); fig.savefig(p, dpi=70); print(p)
