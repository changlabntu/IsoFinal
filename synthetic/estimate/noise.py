"""Camera noise and per-slice artefacts of THX10 (the non-blur part of the
degradation).

  * offset, read noise, lateral noise correlation: from a block outside the
    tissue (pure camera background);
  * gain (variance per count of signal): from the mean-variance relation of a
    high-pass residual inside the tissue, on the smoothest pixels only so that
    real structure does not masquerade as noise;
  * flicker: slice-to-slice intensity fluctuation around the smooth Z trend;
  * jitter: slice-to-slice lateral shift from phase correlation.

    python -m synthetic.estimate.noise
"""
import argparse
import os

import numpy as np
from scipy import ndimage as ndi

from synthetic.common import open_level, out_dir, read_block, save_json


def find_background_block(size_xy=768, size_z=96):
    """Darkest, flattest level-0 block (outside the tissue), located on level 3."""
    a3 = open_level(3)
    Z, X3, Y3 = a3.shape
    z0 = Z // 2 - size_z // 2
    sl = a3[z0:z0 + size_z:8].read().result().astype(np.float32)
    s3 = size_xy // 8
    best = None
    for xi in range(X3 // s3):
        for yi in range(Y3 // s3):
            c = sl[:, xi * s3:(xi + 1) * s3, yi * s3:(yi + 1) * s3]
            if c.min() <= 0:
                continue                      # padded / empty area
            score = c.mean() + 5 * c.std()
            if best is None or score < best[0]:
                best = (score, xi, yi)
    _, xi, yi = best
    return dict(z0=z0, z1=z0 + size_z, x0=xi * size_xy, x1=(xi + 1) * size_xy,
                y0=yi * size_xy, y1=(yi + 1) * size_xy)


def _residual(v):
    """High-pass residual along X: r = x - mean(left, right). For white noise
    var(r) = 1.5 sigma^2."""
    return v[..., 1:-1] - 0.5 * (v[..., :-2] + v[..., 2:])


def background_stats(v):
    """Offset, read noise and lateral noise autocorrelation from pure background."""
    off = float(np.median(v))
    d = v - ndi.uniform_filter(v, (1, 31, 31))
    var = float(d.var())
    ac = {}
    for lag in (1, 2, 3):
        ac['x%d' % lag] = float((d[..., lag:] * d[..., :-lag]).mean() / var)
        ac['y%d' % lag] = float((d[:, lag:] * d[:, :-lag]).mean() / var)
    acz = float((d[1:] * d[:-1]).mean() / var)
    # Gaussian-smoothed white noise has autocorrelation exp(-lag^2 / (4 s^2))
    r1 = max(0.5 * (ac['x1'] + ac['y1']), 1e-6)
    corr_sigma = float(np.sqrt(-1.0 / (4 * np.log(r1)))) if r1 < 1 else 0.0
    return dict(offset=off, read_std=float(np.sqrt(var)), autocorr=ac, autocorr_z=acz,
                noise_corr_sigma=corr_sigma if r1 > 0.02 else 0.0)


def mean_variance(v, offset, corr_sigma=0.0, n_bins=24):
    """Noise variance vs signal on the smoothest pixels. Returns (mean, var)."""
    m = ndi.gaussian_filter(v, (0, 4, 4))
    g = np.hypot(ndi.sobel(m, 1), ndi.sobel(m, 2))
    r = _residual(v)
    m, g = m[..., 1:-1], g[..., 1:-1]
    # var(r) / sigma^2 for noise with lag-1 / lag-2 correlation rho1, rho2
    rho1 = np.exp(-1.0 / (4 * corr_sigma ** 2)) if corr_sigma > 0 else 0.0
    rho2 = np.exp(-4.0 / (4 * corr_sigma ** 2)) if corr_sigma > 0 else 0.0
    k = 1.5 - 2 * rho1 + 0.5 * rho2
    sig = (m - offset).ravel()
    edges = np.unique(np.percentile(sig[::17], np.linspace(1, 99.5, n_bins + 1)))
    means, vars_ = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (m > lo + offset) & (m <= hi + offset)
        if sel.sum() < 2000:
            continue
        gsel = g[sel]
        flat = gsel <= np.percentile(gsel, 30)
        rr = r[sel][flat]
        mad = np.median(np.abs(rr - np.median(rr)))
        means.append(float(sig.reshape(m.shape)[sel][flat].mean()))
        vars_.append(float((1.4826 * mad) ** 2 / k))
    return np.array(means), np.array(vars_)


def fit_gain(means, vars_, read_var):
    """var = gain * mean + read_var, read_var fixed from the background block."""
    ok = means > 0
    gain = float(((vars_[ok] - read_var) * means[ok]).sum() / (means[ok] ** 2).sum())
    return max(gain, 0.0)


def slice_series(level=2, crop=1024):
    """Per-slice robust intensity and adjacent-slice shifts over a central crop
    of `crop` px at pyramid `level`. Returns dict of arrays."""
    from skimage.registration import phase_cross_correlation
    a = open_level(level)
    Z, X, Y = a.shape
    x0, y0 = X // 2 - crop // 2, Y // 2 - crop // 2
    vol = a[:, x0:x0 + crop, y0:y0 + crop].read().result().astype(np.float32)
    flat = vol.reshape(Z, -1)
    med = flat.mean(1)                               # mean: the median of integer data is quantised
    thr = np.percentile(flat[:, ::7], 99, axis=1)
    hi = np.array([f[f >= t].mean() for f, t in zip(flat, thr)])   # mean of the brightest 1 %
    shifts = np.zeros((Z - 1, 2))
    for i in range(Z - 1):
        s, _, _ = phase_cross_correlation(vol[i], vol[i + 1], upsample_factor=100, normalization=None)
        shifts[i] = s * (2 ** level)            # level-0 pixels
    return dict(median=med, p99=hi, shifts=shifts)


def _detrended_std(x, width=9):
    """Robust std of x around its running median (the smooth Z trend)."""
    t = ndi.median_filter(x, size=width, mode='nearest')
    r = x - t
    lo, hi = np.percentile(r, [5, 95])               # trimmed: ignore the rare large jumps
    core = r[(r >= lo) & (r <= hi)]
    return float(core.std() / 0.79)                  # 0.79 = std of a Gaussian trimmed at 5/95 %


def run(blocks, tag='thx10', max_blocks=6):
    arr = open_level(0)
    bgb = find_background_block()
    bg = read_block(bgb['z0'], bgb['z1'], bgb['y0'], bgb['y1'], bgb['x0'], bgb['x1'], arr=arr)
    bs = background_stats(bg)
    print('background block %s: offset %.1f, read std %.2f, autocorr %s, corr sigma %.2f' % (
        bgb, bs['offset'], bs['read_std'], {k: round(v, 3) for k, v in bs['autocorr'].items()},
        bs['noise_corr_sigma']), flush=True)
    M, V = [], []
    for b in blocks[:max_blocks]:
        v = read_block(b['z0'], b['z0'] + 48, b['y0'], b['y1'], b['x0'], b['x1'], arr=arr)
        m, va = mean_variance(v, bs['offset'], bs['noise_corr_sigma'])
        M.append(m); V.append(va)
    M, V = np.concatenate(M), np.concatenate(V)
    gain = fit_gain(M, V, bs['read_std'] ** 2)
    print('gain %.3f counts of variance per count of signal' % gain, flush=True)
    ss = slice_series()
    base = ss['median'] - bs['offset']
    flick = _detrended_std(ss['p99'] - bs['offset']) / float(np.median(ss['p99'] - bs['offset']))
    flick_med = _detrended_std(base) / float(np.median(base))
    # a shift series x[i+1]-x[i] of independent per-slice jitter e has std sqrt(2) * std(e)
    jit = [_detrended_std(ss['shifts'][:, k]) / np.sqrt(2) for k in (0, 1)]
    # correlation between adjacent background slices = pattern common to all slices
    fpn = bs['read_std'] * float(np.sqrt(max(bs['autocorr_z'], 0.0)))
    out = dict(method='noise', offset=bs['offset'], read_std=bs['read_std'], gain=gain, fixed_pattern_std=fpn,
               noise_corr_sigma=bs['noise_corr_sigma'], autocorr=bs['autocorr'], autocorr_z=bs['autocorr_z'],
               flicker_std=float(min(flick, flick_med)), flicker_std_p99=float(flick), flicker_std_median=float(flick_med),
               jitter_std_px=float(np.mean(jit)), jitter_std_xy=[float(j) for j in jit],
               shift_abs_median=[float(np.median(np.abs(ss['shifts'][:, k]))) for k in (0, 1)],
               background_block=bgb, mv_mean=M, mv_var=V)
    d = out_dir('estimate')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    ax[0].plot(M, V, '.'); xx = np.linspace(0, M.max(), 50)
    ax[0].plot(xx, gain * xx + bs['read_std'] ** 2, 'k--', label='var = %.2f * signal + %.1f^2' % (gain, bs['read_std']))
    ax[0].set_xlabel('signal (counts above offset %.0f)' % bs['offset']); ax[0].set_ylabel('noise variance'); ax[0].legend()
    ax[1].plot(ss['median'], label='slice mean'); ax[1].plot(ss['p99'], label='mean of brightest 1 %')
    ax[1].set_xlabel('slice'); ax[1].legend(); ax[1].set_title('flicker: %.2f%% (p99), %.2f%% (median)' % (100 * flick, 100 * flick_med))
    ax[2].plot(ss['shifts'][:, 0], label='axis 0'); ax[2].plot(ss['shifts'][:, 1], label='axis 1')
    ax[2].set_xlabel('slice pair'); ax[2].set_ylabel('shift (level-0 px)'); ax[2].legend()
    ax[2].set_title('jitter std %.2f px per slice' % out['jitter_std_px'])
    fig.tight_layout(); fig.savefig(os.path.join(d, 'noise_%s.png' % tag), dpi=90); plt.close(fig)
    save_json(os.path.join(d, 'noise_%s.json' % tag), out)
    return out


if __name__ == '__main__':
    from synthetic.estimate.blocks import get_blocks
    o = run(get_blocks())
    for k in ('offset', 'read_std', 'fixed_pattern_std', 'gain', 'noise_corr_sigma', 'autocorr_z', 'flicker_std_p99',
              'flicker_std_median', 'jitter_std_xy', 'shift_abs_median'):
        print(k, o[k])
