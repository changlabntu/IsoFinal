"""Relative axial kernel from individual fibres (estimator 2, independent of
the isotropy assumption used by spectra.py).

Bright, isolated fibre segments that are elongated in the XY plane are
located. For each one two profiles through its centre are taken:

  * across the fibre in XY (along its minor axis)  -> lateral line spread
    function convolved with the fibre's own thickness;
  * along Z at the same XY position                -> axial line spread
    function convolved with the same thickness.

So "axial = lateral (*) k" with k the relative axial kernel, whatever the fibre
thickness. Because the axial blur is several slices wide, the Z profile is
adequately sampled at the 8-px slice spacing, and pooling many fibres at their
fitted sub-slice centres gives a super-sampled axial profile. The Z profile at
a fixed XY position is only broadened noticeably for fibres steeper than about
65 degrees out of plane; those look round in XY and are rejected by the
aspect-ratio test.

    python -m synthetic.estimate.lsf
"""
import argparse
import os

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares

from synthetic.common import FWHM, ZSTEP, open_level, out_dir, read_block, save_json
from synthetic.estimate.spectra import kernel_profile

HALF_Z = 12          # slices on each side of the peak
HALF_XY = 24         # px on each side of the peak
LAT_GRID = np.arange(-20.0, 20.01, 0.5)                 # px, pooled lateral profile
AX_GRID = np.arange(-HALF_Z * ZSTEP, HALF_Z * ZSTEP + 0.01, 2.0)   # px, pooled axial profile


def _gauss2d(p, yy, xx):
    a, y0, x0, s_maj, s_min, th, b = p
    c, s = np.cos(th), np.sin(th)
    u = (xx - x0) * c + (yy - y0) * s          # along the fibre
    w = -(xx - x0) * s + (yy - y0) * c         # across the fibre
    return a * np.exp(-0.5 * (u / s_maj) ** 2 - 0.5 * (w / s_min) ** 2) + b


def fit_fibre_2d(patch):
    """Elliptical Gaussian fit of a (2H+1)^2 patch centred on a peak.
    Returns dict(a, y0, x0, s_maj, s_min, theta, b, r2) or None."""
    n = patch.shape[0]
    h = n // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float64)
    b0 = float(np.percentile(patch, 10))
    a0 = float(patch[h, h] - b0)
    if a0 <= 0:
        return None
    # initial orientation from second moments of the bright part
    w = np.clip(patch - b0 - 0.3 * a0, 0, None)
    w = w * (yy ** 2 + xx ** 2 <= (h * 0.8) ** 2)
    if w.sum() <= 0:
        return None
    cyy = (w * yy * yy).sum() / w.sum()
    cxx = (w * xx * xx).sum() / w.sum()
    cxy = (w * xx * yy).sum() / w.sum()
    th0 = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    best = None
    for s_maj0 in (8.0, 20.0):
        try:
            r = least_squares(lambda p: (_gauss2d(p, yy, xx) - patch).ravel(),
                              x0=[a0, 0, 0, s_maj0, 2.5, th0, b0],
                              bounds=([0, -4, -4, 1.0, 0.5, th0 - np.pi, -np.inf],
                                      [np.inf, 4, 4, 200.0, 15.0, th0 + np.pi, np.inf]))
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    a, y0, x0, s_maj, s_min, th, b = best.x
    if s_min > s_maj:
        s_maj, s_min, th = s_min, s_maj, th + np.pi / 2
    ss_res = float((best.fun ** 2).sum())
    ss_tot = float(((patch - patch.mean()) ** 2).sum())
    return dict(a=a, y0=y0, x0=x0, s_maj=s_maj, s_min=s_min, theta=th, b=b,
                r2=1 - ss_res / max(ss_tot, 1e-12))


def fit_gauss1d_core(z, prof, b, core=0.3):
    """Gaussian fit to the core of a profile (points above `core` of the peak,
    after removing baseline b). Returns (amp, mu, sigma) or None."""
    q = prof - b
    pk = q.max()
    if pk <= 0:
        return None
    i0 = int(np.argmax(q))
    lo = i0
    while lo > 0 and q[lo - 1] > core * pk:
        lo -= 1
    hi = i0
    while hi < len(q) - 1 and q[hi + 1] > core * pk:
        hi += 1
    lo, hi = max(lo - 1, 0), min(hi + 1, len(q) - 1)
    if hi - lo < 3:
        return None
    zz, qq = z[lo:hi + 1], q[lo:hi + 1]
    r = least_squares(lambda p: p[0] * np.exp(-0.5 * ((zz - p[1]) / p[2]) ** 2) - qq,
                      x0=[pk, z[i0], 1.5 * (z[1] - z[0])],
                      bounds=([0, z[lo], 0.2 * (z[1] - z[0])], [np.inf, z[hi], 100 * (z[1] - z[0])]))
    return float(r.x[0]), float(r.x[1]), float(r.x[2])


def find_objects(v, max_objects=600, pct=99.5):
    """3D local maxima of the XY-smoothed block, brightest first."""
    s = ndi.gaussian_filter(v, (0, 1.5, 1.5))
    thr = np.percentile(s[::2, ::4, ::4], pct)
    mx = ndi.maximum_filter(s, size=(5, 25, 25))
    zz, yy, xx = np.nonzero((s == mx) & (s > thr))
    keep = ((zz >= HALF_Z) & (zz < v.shape[0] - HALF_Z) &
            (yy >= HALF_XY + 2) & (yy < v.shape[1] - HALF_XY - 2) &
            (xx >= HALF_XY + 2) & (xx < v.shape[2] - HALF_XY - 2))
    zz, yy, xx = zz[keep], yy[keep], xx[keep]
    order = np.argsort(-s[zz, yy, xx])[:max_objects]
    return zz[order], yy[order], xx[order]


def measure_block(v, zstep=ZSTEP, min_aspect=2.5, min_r2=0.8):
    """Measure every usable fibre in a (Z, Y, X) block.

    Returns a list of dicts with the Gaussian widths and the two normalised
    profiles resampled on LAT_GRID / AX_GRID (NaN where not covered).
    """
    out = []
    zs, ys, xs = find_objects(v)
    for z0, y0, x0 in zip(zs, ys, xs):
        patch = v[z0, y0 - HALF_XY:y0 + HALF_XY + 1, x0 - HALF_XY:x0 + HALF_XY + 1].astype(np.float64)
        f = fit_fibre_2d(patch)
        if f is None or f['r2'] < min_r2 or f['s_maj'] / f['s_min'] < min_aspect:
            continue
        cy, cx = y0 + f['y0'], x0 + f['x0']
        # axial profile at the fitted centre (bilinear, mean of 3 points along the fibre)
        c, s = np.cos(f['theta']), np.sin(f['theta'])
        zi = np.arange(z0 - HALF_Z, z0 + HALF_Z + 1)
        prof = np.zeros(len(zi))
        for t in (-1.5, 0.0, 1.5):
            yy = np.full(len(zi), cy + t * s)
            xx = np.full(len(zi), cx + t * c)
            prof += ndi.map_coordinates(v, [zi.astype(float), yy, xx], order=1)
        prof /= 3.0
        b = f['b']
        zpx = (zi - z0) * float(zstep)
        g = fit_gauss1d_core(zpx, prof, b)
        if g is None:
            continue
        amp, mu, sz = g
        if abs(mu) > zstep or amp < 0.5 * f['a']:
            continue
        # isolation: the profile must come back down on both sides
        q = (prof - b) / amp
        if q[:3].mean() > 0.35 or q[-3:].mean() > 0.35:
            continue
        ax_prof = np.interp(AX_GRID, zpx - mu, q, left=np.nan, right=np.nan)
        # lateral profile across the fibre, at the peak slice
        w = LAT_GRID
        lat = ndi.map_coordinates(v[z0].astype(np.float64), [cy + w * c, cx - w * s], order=1, mode='nearest')
        lat = (lat - b) / f['a']
        out.append(dict(z=int(z0), y=float(cy), x=float(cx), s_min=float(f['s_min']),
                        s_maj=float(f['s_maj']), s_z=float(sz), amp=float(amp), r2=float(f['r2']),
                        ax_prof=ax_prof, lat_prof=lat))
    return out


def fit_kernel_to_profiles(ax_prof, lat_prof):
    """Find the stretched kernel (f0, p) such that lateral (*) k = axial.

    Both pooled profiles are peak-normalised; a free amplitude and a small
    baseline absorb the normalisation. Returns dict(f0, p, fwhm, rms)."""
    dz = 0.5
    grid, _ = kernel_profile(0.01, 2.0, dz=dz, half=300.0)      # the kernel's own z grid
    lat = np.interp(grid, LAT_GRID, np.clip(lat_prof, 0, None), left=0.0, right=0.0)
    lat = lat / lat.sum()
    ok = np.isfinite(ax_prof)

    def model(q):
        f0, p, a, b = q
        z, k = kernel_profile(f0, p, dz=dz, half=300.0)
        conv = np.convolve(lat, k, mode='same') * dz
        conv = conv / conv.max()
        return a * np.interp(AX_GRID[ok], z, conv) + b
    best = None
    for p0 in (1.0, 1.5, 2.0):
        r = least_squares(lambda q: model(q) - ax_prof[ok], x0=[0.008, p0, 1.0, 0.0],
                          bounds=([1e-3, 0.5, 0.5, -0.2], [0.2, 4.0, 1.5, 0.2]))
        if best is None or r.cost < best.cost:
            best = r
    f0, p, a, b = best.x
    z, k = kernel_profile(f0, p)
    above = z[k >= 0.5 * k.max()]
    return dict(f0=float(f0), p=float(p), amp=float(a), base=float(b),
                fwhm=float(above.max() - above.min() + (z[1] - z[0])),
                rms=float(np.sqrt(np.mean(best.fun ** 2))), model=model(best.x), ok=ok)


def summarise(objs):
    s_min = np.array([o['s_min'] for o in objs])
    s_z = np.array([o['s_z'] for o in objs])
    sk2 = s_z ** 2 - s_min ** 2
    sk = np.sqrt(np.clip(sk2, 0, None))
    AX = np.array([o['ax_prof'] for o in objs])
    LAT = np.array([o['lat_prof'] for o in objs])
    with np.errstate(all='ignore'):
        ax_med = np.nanmedian(AX, 0)
        lat_med = np.nanmedian(LAT, 0)
    kf = fit_kernel_to_profiles(ax_med, lat_med)
    return dict(method='lsf', n_objects=len(objs),
                sigma_lat=float(np.median(s_min)), sigma_lat_iqr=[float(np.percentile(s_min, 25)), float(np.percentile(s_min, 75))],
                sigma_z=float(np.median(s_z)), sigma_z_iqr=[float(np.percentile(s_z, 25)), float(np.percentile(s_z, 75))],
                sigma_k=float(np.median(sk)), sigma_k_iqr=[float(np.percentile(sk, 25)), float(np.percentile(sk, 75))],
                fwhm_k_gauss=float(np.median(sk) * FWHM),
                stretched=dict(f0=kf['f0'], p=kf['p'], fwhm=kf['fwhm'], rms=kf['rms']),
                per_fibre_sigma_lat=s_min, per_fibre_sigma_z=s_z,
                ax_grid=AX_GRID, ax_profile=ax_med, ax_model=np.where(kf['ok'], np.nan, np.nan),
                lat_grid=LAT_GRID, lat_profile=lat_med), kf


def plot(objs, summary, kf, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    s_min = np.array([o['s_min'] for o in objs]); s_z = np.array([o['s_z'] for o in objs])
    ax[0].hist(s_min * FWHM, bins=40, alpha=0.7, label='lateral FWHM (median %.1f px)' % (summary['sigma_lat'] * FWHM))
    ax[0].hist(s_z * FWHM, bins=40, alpha=0.7, label='axial FWHM (median %.1f px)' % (summary['sigma_z'] * FWHM))
    ax[0].set_xlabel('FWHM of Gaussian core fit (XY px)'); ax[0].legend(); ax[0].set_title('%d fibres' % len(objs))
    ax[1].plot(LAT_GRID, summary['lat_profile'], 'C0', label='lateral (across fibre)')
    ax[1].plot(AX_GRID, summary['ax_profile'], 'C1o-', ms=3, label='axial (pooled, super-sampled)')
    ax[1].plot(AX_GRID[kf['ok']], kf['model'], 'k--', label='lateral (*) k: p=%.2f, FWHM %.1f px' % (kf['p'], kf['fwhm']))
    ax[1].set_xlabel('offset (XY px)'); ax[1].legend(); ax[1].set_title('Pooled profiles')
    zc = np.array([o['z'] for o in objs])
    ax[2].plot(zc, s_z * FWHM, '.', alpha=0.4)
    ax[2].set_xlabel('slice index inside block'); ax[2].set_ylabel('axial FWHM (px)'); ax[2].set_title('Per-fibre axial width')
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def run(blocks, tag='thx10'):
    arr = open_level(0)
    objs = []
    per_band = {}
    for i, b in enumerate(blocks):
        v = read_block(b['z0'], b['z1'], b['y0'], b['y1'], b['x0'], b['x1'], arr=arr)
        o = measure_block(v)
        for q in o:
            q['band'] = b.get('band', -1)
            q['zabs'] = q['z'] + b['z0']
        objs += o
        per_band.setdefault(b.get('band', -1), []).extend(o)
        if o:
            print('block %2d band %d: %3d fibres, lateral FWHM %.1f, axial FWHM %.1f px' % (
                i, b.get('band', -1), len(o), np.median([q['s_min'] for q in o]) * FWHM,
                np.median([q['s_z'] for q in o]) * FWHM), flush=True)
    s, kf = summarise(objs)
    s.pop('ax_model')
    s['per_band'] = {str(k): dict(n=len(vv), sigma_z=float(np.median([q['s_z'] for q in vv])),
                                  sigma_lat=float(np.median([q['s_min'] for q in vv])))
                     for k, vv in per_band.items() if vv}
    d = out_dir('estimate')
    plot(objs, s, kf, os.path.join(d, 'lsf_%s.png' % tag))
    save_json(os.path.join(d, 'lsf_%s.json' % tag), s)
    return s


if __name__ == '__main__':
    from synthetic.estimate.blocks import get_blocks
    p = argparse.ArgumentParser()
    p.add_argument('--max_blocks', type=int, default=0)
    a = p.parse_args()
    bl = get_blocks()
    if a.max_blocks:
        bl = bl[:a.max_blocks]
    s = run(bl)
    print('lateral FWHM %.2f px, axial FWHM %.1f px (%.2f slices); Gaussian sigma_k %.2f (FWHM %.1f); '
          'stretched k: p %.2f FWHM %.1f px' % (s['sigma_lat'] * FWHM, s['sigma_z'] * FWHM,
                                                s['sigma_z'] * FWHM / ZSTEP, s['sigma_k'], s['fwhm_k_gauss'],
                                                s['stretched']['p'], s['stretched']['fwhm']))
    print('per band:', s['per_band'])
