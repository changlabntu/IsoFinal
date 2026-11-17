"""Relative axial kernel from the 3D power spectrum (estimator 1).

Model. For a statistically isotropic object with spectrum S(|f|), imaged with a
lateral transfer H_lat and an axial transfer H_z, the 3D power spectrum is

    P(fx, fy, fz) = S(|f|) |H_lat(fx, fy)|^2 |H_z(fz)|^2  + noise floor.

With a near-Gaussian H_lat inside the band used here (|f| below the Z Nyquist
of 1/16 cycles per XY pixel), on a shell of constant |f| = rho

    log(P - noise) = g(rho) + h(|fz|),      h = log |K(fz)|^2,  K = H_z / H_lat.

K is the *relative* axial kernel: the extra blur along Z beyond the lateral
blur. g and h are fitted as a two-way additive table (no shape assumed), then
a Gaussian is fitted to h. Running the same fit with X (or Y) in the role of Z
is the control: for an isotropic object it must return a flat h (sigma ~ 0).

All frequencies are in cycles per XY pixel, all lengths in XY pixels.

    python -m synthetic.estimate.spectra            # all selected blocks
"""
import argparse
import os

import numpy as np
import scipy.fft as sfft
from scipy.optimize import least_squares

from synthetic.common import FWHM, ZSTEP, open_level, out_dir, read_block, save_json


def power_spectrum_3d(v, zstep=ZSTEP):
    """Hann-windowed 3D power spectrum of (Z, Y, X) -> P, (fz, fy, fx) 1D axes."""
    v = v.astype(np.float32)
    v = v - v.mean()
    for ax, n in enumerate(v.shape):
        w = np.hanning(n).astype(np.float32)
        shape = [1, 1, 1]
        shape[ax] = n
        v = v * w.reshape(shape)
    F = sfft.fftn(v, workers=-1)
    P = (F.real ** 2 + F.imag ** 2).astype(np.float64)
    fz = np.fft.fftfreq(v.shape[0], d=zstep)
    fy = np.fft.fftfreq(v.shape[1], d=1.0)
    fx = np.fft.fftfreq(v.shape[2], d=1.0)
    return P, (fz, fy, fx)


def noise_floor(P, freqs, lo=0.4):
    """White-noise level: mean power where both lateral frequencies exceed `lo`
    (far beyond the lateral cut-off, so no signal is left there)."""
    fz, fy, fx = freqs
    my = np.abs(fy) > lo
    mx = np.abs(fx) > lo
    return float(P[:, my][:, :, mx].mean())


def shell_table(P, freqs, axis, noise, rho_max=None, n_rho=40, n_ax=48, rho_min=None):
    """Mean noise-subtracted power in bins of (|f|, |f_axis|).

    axis: 0 (z), 1 (y) or 2 (x): the direction whose transfer is being probed.
    For the in-plane controls (axis 1 or 2) only the fz ~ 0 slab is used, so the
    axial blur cannot leak into the comparison: the control then measures X
    against Y only.
    Returns table (n_rho, n_ax), counts, rho centres, axis-frequency centres.
    """
    fz, fy, fx = freqs
    f_nyq_z = np.abs(fz).max()
    rho_max = rho_max or f_nyq_z
    rho_min = rho_min or 4.0 / (len(fx))               # skip the DC lobe of the window
    # only the cube |f_i| <= rho_max matters
    iz = np.where(np.abs(fz) <= (rho_max if axis == 0 else 1.5 * abs(fz[1])))[0]
    iy = np.where(np.abs(fy) <= rho_max)[0]
    ix = np.where(np.abs(fx) <= rho_max)[0]
    sub = P[np.ix_(iz, iy, ix)]
    FZ, FY, FX = np.meshgrid(fz[iz], fy[iy], fx[ix], indexing='ij')
    rho = np.sqrt((FZ ** 2 if axis == 0 else 0.0) + FY ** 2 + FX ** 2)
    fa = np.abs((FZ, FY, FX)[axis])
    keep = (rho >= rho_min) & (rho <= rho_max)
    rho_edges = np.linspace(rho_min, rho_max, n_rho + 1)
    ax_edges = np.linspace(0, rho_max, n_ax + 1)
    ri = np.clip(np.digitize(rho[keep], rho_edges) - 1, 0, n_rho - 1)
    ai = np.clip(np.digitize(fa[keep], ax_edges) - 1, 0, n_ax - 1)
    flat = ri * n_ax + ai
    cnt = np.bincount(flat, minlength=n_rho * n_ax).reshape(n_rho, n_ax)
    tot = np.bincount(flat, weights=sub[keep], minlength=n_rho * n_ax).reshape(n_rho, n_ax)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = tot / cnt - noise
    return mean, cnt, 0.5 * (rho_edges[1:] + rho_edges[:-1]), 0.5 * (ax_edges[1:] + ax_edges[:-1])


def additive_fit(table, cnt, min_count=8, floor=None):
    """Weighted least squares for log(table) = g[i] + h[j], h[0] = 0.

    Cells with too few voxels or non-positive / sub-floor power are dropped.
    Returns g, h, h_valid (bool per column), residual rms.
    """
    ok = (cnt >= min_count) & np.isfinite(table) & (table > (floor or 0.0))
    n_r, n_a = table.shape
    L = np.where(ok, np.log(np.where(ok, table, 1.0)), 0.0)
    W = np.where(ok, np.sqrt(cnt), 0.0)
    g = np.zeros(n_r)
    h = np.zeros(n_a)
    for _ in range(200):
        sw = (W ** 2).sum(1)
        g_new = np.where(sw > 0, ((W ** 2) * (L - h[None])).sum(1) / np.maximum(sw, 1e-30), 0.0)
        sw = (W ** 2).sum(0)
        h_new = np.where(sw > 0, ((W ** 2) * (L - g_new[:, None])).sum(0) / np.maximum(sw, 1e-30), 0.0)
        shift = h_new[0]
        h_new = h_new - shift
        g_new = g_new + shift
        done = np.abs(h_new - h).max() < 1e-7 and np.abs(g_new - g).max() < 1e-7
        g, h = g_new, h_new
        if done:
            break
    h_valid = (ok.sum(0) >= 3)
    res = (W ** 2 * (L - g[:, None] - h[None]) ** 2).sum() / max((W ** 2).sum(), 1e-30)
    return g, h, h_valid, float(np.sqrt(res)), ok


def fit_gaussian_h(f, h, valid, h_min=-7.0):
    """h(f) = -(2 pi sigma f)^2 over the columns that are valid and above h_min
    (below that the estimate sits in the noise). Returns sigma (XY pixels)."""
    m = valid & (h > h_min)
    m[0] = True
    if m.sum() < 4:
        return float('nan'), m
    A = -(2 * np.pi * f[m]) ** 2
    s2 = float((A * h[m]).sum() / (A * A).sum())
    return float(np.sign(s2) * np.sqrt(abs(s2))), m      # signed: negative = sharper than average


def stretched_K(f, f0, p):
    """Amplitude transfer K(f) = exp(-0.5 (|f| / f0)^p). p = 2 is a Gaussian with
    sigma = 1 / (2 pi f0); p = 1 is a Lorentzian (heavy-tailed) kernel."""
    return np.exp(-0.5 * (np.abs(f) / f0) ** p)


def fit_stretched(f, h, valid):
    """h(f) = log(K(f)^2 + c): stretched-exponential transfer plus a floor c.

    The floor is power that does not fall with fz: slice-to-slice artefacts
    (intensity flicker, lateral jitter) that are independent between slices.
    """
    m = valid.copy()
    m[0] = True
    if m.sum() < 6:
        return None

    def model(q, ff):
        f0, p, logc = q
        return np.log(stretched_K(ff, f0, p) ** 2 + np.exp(logc))
    best = None
    for p0 in (1.0, 1.5, 2.0):
        r = least_squares(lambda q: model(q, f[m]) - h[m], x0=[0.01, p0, -7.0],
                          bounds=([1e-4, 0.5, -20.0], [1.0, 4.0, 0.0]))
        if best is None or r.cost < best.cost:
            best = r
    f0, p, logc = best.x
    return dict(f0=float(f0), p=float(p), floor=float(np.exp(logc)),
                rms=float(np.sqrt(np.mean(best.fun ** 2))),
                fwhm=float(kernel_fwhm(f0, p)))


def kernel_profile(f0, p, dz=0.25, half=400.0):
    """Real-space kernel k(z) of the stretched transfer, sampled every dz px,
    normalised to unit sum * dz (unit area). Returns z, k."""
    n = int(2 * half / dz)
    ff = np.fft.rfftfreq(n, d=dz)
    k = np.fft.irfft(stretched_K(ff, f0, p), n)
    k = np.fft.fftshift(k)
    z = (np.arange(n) - n // 2) * dz
    k = np.maximum(k, 0)
    return z, k / (k.sum() * dz)


def kernel_fwhm(f0, p):
    z, k = kernel_profile(f0, p)
    above = z[k >= 0.5 * k.max()]
    return above.max() - above.min() + (z[1] - z[0])


def analyse_volume(v, zstep=ZSTEP, n_rho=40, n_ax=48):
    """Full estimate for one (Z, Y, X) block. Returns a dict (per-axis curves + fits)."""
    P, freqs = power_spectrum_3d(v, zstep)
    noise = noise_floor(P, freqs)
    out = dict(noise_floor=noise, axes={})
    for axis, name in ((0, 'z'), (1, 'y'), (2, 'x')):
        table, cnt, rho_c, ax_c = shell_table(P, freqs, axis, noise, n_rho=n_rho, n_ax=n_ax)
        g, h, valid, rms, ok = additive_fit(table, cnt)
        sigma, used = fit_gaussian_h(ax_c, h, valid)
        d = dict(f=ax_c, h=h, valid=valid, sigma=sigma, fit_rms=rms, used=used,
                 n_cells=int(ok.sum()))
        if name == 'z':
            d['stretched'] = fit_stretched(ax_c, h, valid)
            out['g'] = g
            out['rho'] = rho_c
        out['axes'][name] = d
    return out


def summarise(results):
    """Pool per-block results -> dict with medians, spread and the pooled curve."""
    def col(name, key):
        return np.array([r['axes'][name][key] for r in results], dtype=float)
    sz, sy, sx = col('z', 'sigma'), col('y', 'sigma'), col('x', 'sigma')
    f = results[0]['axes']['z']['f']
    H = np.array([np.where(r['axes']['z']['valid'], r['axes']['z']['h'], np.nan) for r in results])
    with np.errstate(all='ignore'):
        h_med = np.nanmedian(H, 0)
        h_lo = np.nanpercentile(H, 16, 0)
        h_hi = np.nanpercentile(H, 84, 0)
    pooled_sigma, _ = fit_gaussian_h(f, np.nan_to_num(h_med, nan=-99.0), np.isfinite(h_med))
    st = fit_stretched(f, np.nan_to_num(h_med, nan=-99.0), np.isfinite(h_med))
    per = [r['axes']['z'].get('stretched') or {} for r in results]
    return dict(method='spectra', n_blocks=len(results),
                sigma_k=float(np.nanmedian(sz)), sigma_k_iqr=[float(np.nanpercentile(sz, 25)),
                                                             float(np.nanpercentile(sz, 75))],
                sigma_k_pooled_curve=pooled_sigma, fwhm_k=float(np.nanmedian(sz) * FWHM),
                control_sigma_x=float(np.nanmedian(sx)), control_sigma_y=float(np.nanmedian(sy)),
                stretched=st, per_block_fwhm=[q.get('fwhm', float('nan')) for q in per],
                per_block_p=[q.get('p', float('nan')) for q in per],
                per_block_floor=[q.get('floor', float('nan')) for q in per],
                per_block_sigma_k=sz, per_block_sigma_x=sx, per_block_sigma_y=sy,
                f=f, h_median=h_med, h_p16=h_lo, h_p84=h_hi)


def plot(results, summary, path, labels=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    f = summary['f']
    for r in results:
        for name, c in (('z', 'C0'), ('x', 'C1'), ('y', 'C2')):
            d = r['axes'][name]
            ax[0].plot(d['f'][d['valid']], d['h'][d['valid']], c, alpha=0.25, lw=1)
    s = summary['sigma_k']
    ax[0].plot(f, -(2 * np.pi * s * f) ** 2, 'k--', lw=2, label='Gaussian sigma_k = %.1f px' % s)
    ax[0].plot([], [], 'C0', label='Z'); ax[0].plot([], [], 'C1', label='X (control)'); ax[0].plot([], [], 'C2', label='Y (control)')
    ax[0].set_ylim(-12, 2); ax[0].set_xlabel('frequency along the probed axis (cycles / XY px)')
    ax[0].set_ylabel('h = log |K|^2'); ax[0].legend(); ax[0].set_title('Directional transfer, per block')
    ax[1].plot(f, summary['h_median'], 'C0', lw=2, label='Z, median over blocks')
    ax[1].fill_between(f, summary['h_p16'], summary['h_p84'], color='C0', alpha=0.2)
    ax[1].plot(f, -(2 * np.pi * s * f) ** 2, 'k--', label='Gaussian')
    t = summary['stretched']
    if t:
        ax[1].plot(f, np.log(stretched_K(f, t['f0'], t['p']) ** 2 + t['floor']), 'r:', lw=2,
                   label='stretched p=%.2f, FWHM %.1f px, floor %.1e' % (t['p'], t['fwhm'], t['floor']))
    ax[1].set_ylim(-12, 2); ax[1].legend(); ax[1].set_xlabel('fz (cycles / XY px)'); ax[1].set_title('Pooled')
    n = len(results)
    ax[2].plot(range(n), summary['per_block_sigma_k'], 'o', label='Z (Gaussian sigma)')
    ax[2].plot(range(n), np.array(summary['per_block_fwhm']) / FWHM, 'kx', label='Z (stretched FWHM / 2.355)')
    ax[2].plot(range(n), summary['per_block_sigma_x'], 's', label='X control')
    ax[2].plot(range(n), summary['per_block_sigma_y'], '^', label='Y control')
    ax[2].set_xlabel('block'); ax[2].set_ylabel('sigma (XY px)'); ax[2].legend(); ax[2].set_title('Per-block Gaussian fit')
    if labels:
        ax[2].set_xticks(range(n)); ax[2].set_xticklabels(labels, rotation=90, fontsize=6)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def run(blocks, tag='thx10'):
    arr = open_level(0)
    results = []
    for i, b in enumerate(blocks):
        v = read_block(b['z0'], b['z1'], b['y0'], b['y1'], b['x0'], b['x1'], arr=arr)
        r = analyse_volume(v)
        r['block'] = b
        results.append(r)
        st = r['axes']['z'].get('stretched') or {}
        print('block %2d band %d  gauss sigma_k %.2f | stretched FWHM %.1f p %.2f floor %.1e | control x %+.2f y %+.2f' % (
            i, b.get('band', -1), r['axes']['z']['sigma'], st.get('fwhm', float('nan')),
            st.get('p', float('nan')), st.get('floor', float('nan')),
            r['axes']['x']['sigma'], r['axes']['y']['sigma']), flush=True)
    s = summarise(results)
    s['per_block_band'] = [b.get('band', -1) for b in blocks]
    d = out_dir('estimate')
    plot(results, s, os.path.join(d, 'spectra_%s.png' % tag), labels=['b%d' % b.get('band', -1) for b in blocks])
    save_json(os.path.join(d, 'spectra_%s.json' % tag), s)
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
    print('sigma_k = %.2f px (IQR %.2f-%.2f), FWHM %.1f px = %.2f slices; controls x %.2f y %.2f' % (
        s['sigma_k'], s['sigma_k_iqr'][0], s['sigma_k_iqr'][1], s['fwhm_k'], s['fwhm_k'] / ZSTEP,
        s['control_sigma_x'], s['control_sigma_y']))
    print('stretched model:', s['stretched'])
