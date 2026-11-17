"""Non-separable relative transfer R(f_r, fz) of the THX10 degradation
(the main degradation model; spectra.py is its shell-averaged 1D summary).

For a statistically isotropic object the 3D power spectrum of the data is
P(f_r, fz) = S(rho) |H(f_r, fz)|^2 with rho^2 = f_r^2 + fz^2. The fz = 0 plane
gives the in-plane spectrum T(rho) = S(rho) |H(rho, 0)|^2, which is exactly the
spectrum an isotropic volume "as sharp as XY in every direction" would have.
Hence, with no assumption on the shape of the blur,

    R(f_r, fz) = P(f_r, fz) / T(sqrt(f_r^2 + fz^2))

is the power transfer that turns that isotropic target into the data: the
phantom is degraded by filtering with sqrt(R) and keeping every 8th slice.

R is measured as a table, then summarised by

    R = exp(-(|fz| / f0(f_r))^p) + floor(f_r),     f0(f_r) = a + b f_r.

f0 growing with f_r means fine lateral detail is sectioned better in Z than
coarse detail (a cone-shaped transfer, as optics predicts), i.e. the blur is
not a 1D kernel along Z. The floor is power that is independent from slice to
slice (noise with lateral correlation, flicker, jitter, aliasing); it belongs
to the noise model, not to the blur.

Frequencies in cycles per XY pixel.

    python -m synthetic.estimate.transfer
"""
import argparse
import os

import numpy as np
from scipy.optimize import least_squares

from synthetic.common import ZSTEP, open_level, out_dir, read_block, save_json
from synthetic.estimate.spectra import noise_floor, power_spectrum_3d

FR_EDGES = np.concatenate([[0.0], np.geomspace(0.004, 0.30, 22)])
FIT_FR = (0.006, 0.15)       # lateral band where R is trusted (see module docstring)


def relative_transfer(v, zstep=ZSTEP, fr_edges=FR_EDGES):
    """R table of one (Z, Y, X) block. Returns dict(logR (n_fr, n_fz), fr, fz,
    slice_noise (n_fr,) = absolute power of the slice-independent part)."""
    P, (fz, fy, fx) = power_spectrum_3d(v, zstep)
    N = noise_floor(P, (fz, fy, fx))
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    fr = np.sqrt(FY ** 2 + FX ** 2).ravel()
    # in-plane reference T(rho) from the fz = 0 plane
    red = np.linspace(0, 0.5, 385)
    ri = np.clip(np.digitize(fr, red) - 1, 0, 383)
    cnt = np.maximum(np.bincount(ri, minlength=384), 1)
    T = np.bincount(ri, weights=P[0].ravel(), minlength=384) / cnt - N
    rc = 0.5 * (red[1:] + red[:-1])
    nz = len(fz) // 2
    nb = len(fr_edges) - 1
    bi = np.digitize(fr, fr_edges) - 1
    inb = (bi >= 0) & (bi < nb)
    bcnt = np.maximum(np.bincount(bi[inb], minlength=nb), 1)
    fr_mean = np.bincount(bi[inb], weights=fr[inb], minlength=nb) / bcnt
    Pm = np.zeros((nb, nz))
    for k in range(nz):
        Pk = (0.5 * (P[k] + P[-k]) if k > 0 else P[0]).ravel()
        Pm[:, k] = np.bincount(bi[inb], weights=Pk[inb], minlength=nb) / bcnt - N
    rho = np.sqrt(fr_mean[:, None] ** 2 + np.abs(fz[:nz])[None] ** 2)
    den = np.interp(rho, rc, T)
    with np.errstate(invalid='ignore', divide='ignore'):
        logR = np.where((Pm > 0) & (den > 0), np.log(Pm / den), np.nan)
    return dict(logR=logR, fr=fr_mean, fz=np.abs(fz[:nz]), T=T, rho=rc, noise=N,
                slice_noise=Pm[:, int(0.75 * nz):].mean(1))


def f0_of(fr, a, b):
    return a + b * np.asarray(fr, dtype=float)


def transfer_power(fr, fz, a, b, p):
    """Model power transfer K^2(f_r, fz), without the floor."""
    return np.exp(-(np.abs(fz) / f0_of(fr, a, b)) ** p)


def transfer_amplitude(fr, fz, params):
    """Amplitude filter sqrt(K^2) applied to a phantom. `params`: dict(a, b, p)."""
    return np.sqrt(transfer_power(fr, fz, params['a'], params['b'], params['p']))


def fit_transfer(logR, fr, fz, fit_fr=FIT_FR):
    """Fit (a, b, p) and one floor per lateral band to the pooled table."""
    rows = np.where((fr >= fit_fr[0]) & (fr <= fit_fr[1]))[0]
    L = logR[rows][:, 1:]
    F = fz[1:]
    ok = np.isfinite(L)
    n = len(rows)

    def model(q):
        a, b, p = q[:3]
        K2 = transfer_power(fr[rows][:, None], F[None], a, b, p)
        return np.log(K2 + np.exp(q[3:3 + n])[:, None])
    q0 = np.concatenate([[0.008, 0.1, 1.5], np.full(n, -6.0)])
    lo = np.concatenate([[1e-4, 0.0, 0.5], np.full(n, -25.0)])
    hi = np.concatenate([[0.1, 2.0, 4.0], np.full(n, 0.0)])
    r = least_squares(lambda q: (model(q) - np.where(ok, L, 0.0))[ok], q0, bounds=(lo, hi))
    a, b, p = r.x[:3]
    return dict(a=float(a), b=float(b), p=float(p), rms=float(np.sqrt(np.mean(r.fun ** 2))),
                fit_fr=list(fit_fr), floor_fr=fr[rows], floor=np.exp(r.x[3:3 + n])), rows


def predict_line_profile(params, sigma_line, z):
    """Axial profile through an in-plane line whose lateral profile is a
    Gaussian of width sigma_line (PSF and thickness together): the line
    carries no frequency along itself, so its Z profile integrates the
    amplitude transfer over the across-fibre frequency only."""
    fy = np.linspace(-0.25, 0.25, 2001)
    W = np.exp(-0.5 * (2 * np.pi * sigma_line * fy) ** 2)
    dz = float(z[1] - z[0])
    n = 4096
    fzz = np.fft.rfftfreq(n, d=dz)
    A = transfer_amplitude(np.abs(fy)[:, None], fzz[None], params)        # (nfy, nfz)
    # the fibre's own thickness also blurs it along Z
    spec = (W[:, None] * A).sum(0) * np.exp(-0.5 * (2 * np.pi * sigma_line * fzz) ** 2)
    prof = np.fft.fftshift(np.fft.irfft(spec, n))
    zz = (np.arange(n) - n // 2) * dz
    prof = prof / prof.max()
    return np.interp(z, zz, prof)


def plot(pooled, fit, rows, fr, fz, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    cm = plt.get_cmap('viridis')
    for n_, j in enumerate(rows):
        c = cm(n_ / max(len(rows) - 1, 1))
        ax[0].plot(fz, pooled[j], color=c, lw=1.2, label='f_r %.3f' % fr[j] if n_ % 3 == 0 else None)
        K2 = transfer_power(fr[j], fz, fit['a'], fit['b'], fit['p']) + fit['floor'][n_]
        ax[0].plot(fz, np.log(K2), '--', color=c, lw=0.8)
    ax[0].set_ylim(-10, 1); ax[0].set_xlabel('fz (cycles / XY px)'); ax[0].set_ylabel('log R')
    ax[0].set_title('Measured (solid) and model (dashed), by lateral frequency'); ax[0].legend(fontsize=7)
    im = ax[1].pcolormesh(fz, np.arange(len(fr) + 1), pooled, vmin=-9, vmax=0, shading='flat') \
        if False else ax[1].imshow(pooled, aspect='auto', origin='lower', vmin=-9, vmax=0,
                                   extent=[0, fz.max(), 0, len(fr)])
    ax[1].set_yticks(np.arange(len(fr))[::3] + 0.5); ax[1].set_yticklabels(['%.3f' % f for f in fr[::3]])
    ax[1].set_xlabel('fz'); ax[1].set_ylabel('f_r'); ax[1].set_title('log R(f_r, fz), median over blocks')
    fig.colorbar(im, ax=ax[1])
    ff = np.linspace(0, 0.3, 100)
    ax[2].plot(ff, f0_of(ff, fit['a'], fit['b']), 'k')
    ax[2].axvspan(*fit['fit_fr'], color='C0', alpha=0.1, label='fitted band')
    ax[2].set_xlabel('f_r'); ax[2].set_ylabel('f0 (axial cut-off scale)')
    ax[2].set_title('f0 = %.4f + %.3f f_r,  p = %.2f,  rms %.2f' % (fit['a'], fit['b'], fit['p'], fit['rms']))
    ax[2].legend()
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def pooled_fit(tables):
    with np.errstate(all='ignore'):
        pooled = np.nanmedian(np.array([t['logR'] for t in tables]), 0)
    fr, fz = tables[0]['fr'], tables[0]['fz']
    fit, rows = fit_transfer(pooled, fr, fz)
    return pooled, fit, rows, fr, fz


def run(blocks, tag='thx10'):
    arr = open_level(0)
    tables = []
    for i, b in enumerate(blocks):
        v = read_block(b['z0'], b['z1'], b['y0'], b['y1'], b['x0'], b['x1'], arr=arr)
        t = relative_transfer(v)
        f, _ = fit_transfer(t['logR'], t['fr'], t['fz'])
        t['fit'] = f
        t['band'] = b.get('band', -1)
        tables.append(t)
        print('block %2d band %d  a %.4f  b %.3f  p %.2f  rms %.2f' % (
            i, t['band'], f['a'], f['b'], f['p'], f['rms']), flush=True)
    pooled, fit, rows, fr, fz = pooled_fit(tables)
    d = out_dir('estimate')
    plot(pooled, fit, rows, fr, fz, os.path.join(d, 'transfer_%s.png' % tag))
    out = dict(method='transfer', n_blocks=len(tables), a=fit['a'], b=fit['b'], p=fit['p'], rms=fit['rms'],
               fit_fr=fit['fit_fr'], floor_fr=fit['floor_fr'], floor=fit['floor'],
               per_block=[dict(band=t['band'], a=t['fit']['a'], b=t['fit']['b'], p=t['fit']['p']) for t in tables],
               fr=fr, fz=fz, logR=np.where(np.isfinite(pooled), pooled, -99.0))
    by_band = {}
    for t in tables:
        by_band.setdefault(t['band'], []).append(t)
    out['per_band'] = {}
    for k, ts in by_band.items():
        _, f, _, _, _ = pooled_fit(ts)
        out['per_band'][str(k)] = dict(a=f['a'], b=f['b'], p=f['p'], n=len(ts))
    save_json(os.path.join(d, 'transfer_%s.json' % tag), out)
    return out


if __name__ == '__main__':
    from synthetic.estimate.blocks import get_blocks
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_blocks', type=int, default=0)
    a = ap.parse_args()
    bl = get_blocks()
    if a.max_blocks:
        bl = bl[:a.max_blocks]
    o = run(bl)
    print('pooled: f0 = %.4f + %.3f f_r, p = %.2f, rms %.2f' % (o['a'], o['b'], o['p'], o['rms']))
    print('per band:', o['per_band'])
