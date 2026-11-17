"""Noise of THX10 as the training stacks actually carry it.

Measured on the training TIFFs (raw counts recovered through the known
window). Two facts that estimate/noise.py did not account for:

1. The volume was resampled in-plane with linear interpolation, with a
   sub-pixel shift (wx, wy) that differs from region to region. Interpolating
   at fraction w along one axis multiplies the noise spectrum by
       H(f; w) = 1 - 2 w (1 - w) (1 - cos 2 pi f)
   so the noise is smoothed by a different amount along X and Y in every
   stack (none for w = 0, a two-tap average for w = 0.5) and the per-pixel
   variance is lower than the camera's.

2. The noise that is independent from slice to slice can be read without any
   assumption about the tissue: structure is strongly blurred along Z, so at
   high Z frequency (>= 0.3 cycles / slice) only noise is left. Its in-plane
   spectrum gives, per stack, the camera variance V0 and the two fractions:
       P(fx, fy) = V0 H(fx; wx) H(fy; wy)
   V0 against the mean signal of the stack gives gain and read noise of the
   slice-independent part:
       V0 = read_std^2 + gain * S

A pattern fixed to the pixel grid (the same in every slice) is measured from
the cross-spectrum of two groups of slices far apart in Z.

    python -m synthetic.estimate.noise_shape                 # real stacks -> out/estimate/noise_shape_thx10.json
    python -m synthetic.estimate.noise_shape --synth DIR     # the same estimator on synthetic stacks
"""
import argparse
import glob
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.special import digamma

from synthetic.common import CONFIG, load_json, out_dir, save_json

F_LO = 0.15          # in-plane band used for the fits (below it fine structure leaks in)
FZ_MIN = 0.3         # cycles / slice above which only slice-independent noise is left


def H(f, u):
    """Power gain of linear interpolation; u = 2 w (1 - w) in [0, 0.5]."""
    return 1.0 - u * (1.0 - np.cos(2 * np.pi * f))


def stack_noise(raw):
    """Fit one (Z, Y, X) stack. Returns V0, ux, uy (camera variance and the
    interpolation strengths), the fixed-pattern variance, and the fit rms."""
    Z, Y, X = raw.shape
    v = raw - raw.mean()
    F = np.fft.fftn(v)
    fz = np.abs(np.fft.fftfreq(Z))
    n_fz = int((fz >= FZ_MIN).sum())
    Pz = (np.abs(F[fz >= FZ_MIN]) ** 2).mean(0) / v.size
    fy = np.abs(np.fft.fftfreq(Y))[:, None]
    fx = np.abs(np.fft.fftfreq(X))[None, :]
    m = (fy >= F_LO) & (fx >= F_LO)
    FY, FX = np.broadcast_to(fy, Pz.shape)[m], np.broadcast_to(fx, Pz.shape)[m]
    L = np.log(Pz[m])
    # coarse bins keep the fit fast and robust
    by, bx = np.minimum((FY - F_LO) / (0.5 - F_LO) * 7, 6.999).astype(int), np.minimum((FX - F_LO) / (0.5 - F_LO) * 7, 6.999).astype(int)
    k = by * 7 + bx
    cnt = np.bincount(k, minlength=49)
    Lb = np.bincount(k, weights=L, minlength=49) / cnt
    fyb = np.bincount(k, weights=FY, minlength=49) / cnt
    fxb = np.bincount(k, weights=FX, minlength=49) / cnt
    r = least_squares(lambda q: q[0] + np.log(H(fxb, q[1])) + np.log(H(fyb, q[2])) - Lb,
                      x0=[Lb.max(), 0.2, 0.2], bounds=([-10, 0, 0], [20, 0.4999, 0.4999]))
    # each Pz value averages n_fz exponential variates: E[log] = digamma(n) - log(n)
    V0, ux, uy = float(np.exp(r.x[0] - (digamma(n_fz) - np.log(n_fz)))), float(r.x[1]), float(r.x[2])
    # fixed pattern: cross-spectrum of the means of two slice groups a third of the stack apart
    n = Z // 3
    A, B = np.fft.fft2(raw[:n].mean(0) - raw[:n].mean()), np.fft.fft2(raw[-n:].mean(0) - raw[-n:].mean())
    cross = (A * np.conj(B)).real / (Y * X)
    Vf = float(np.mean(cross[m] / (H(FX, ux) * H(FY, uy))))
    return dict(V0=V0, ux=ux, uy=uy, V_fixed=Vf, rms=float(np.sqrt(np.mean(r.fun ** 2))))


def fit_camera(S, V0, Vf):
    """read_std, gain from V0(S); saturating-linear fixed pattern from Vf(S)."""
    r = least_squares(lambda q: (q[0] + q[1] * S - V0) / np.maximum(V0, 1.0), x0=[1.5, 1.0], bounds=([0, 0], [20, 10]),
                      loss='soft_l1', f_scale=0.2)
    f = least_squares(lambda q: q[0] + q[1] * np.minimum(S, q[2]) - Vf, x0=[0.3, 0.05, 45.0],
                      bounds=([0, 0, 10], [10, 2, 200]), loss='soft_l1', f_scale=1.0)
    return dict(read_std=float(np.sqrt(r.x[0])), gain=float(r.x[1]),
                fixed_pattern=dict(var0=float(f.x[0]), slope=float(f.x[1]), sat=float(f.x[2])))


def measure(files, lo, hi, offset, verbose=False):
    import tifffile
    from synthetic import realism
    rows = []
    for i, f in enumerate(files):
        raw = realism.from_window(tifffile.imread(f), lo, hi)
        s = stack_noise(raw)
        s['signal'] = float(raw.mean() - offset)
        s['clipped'] = float((raw >= hi - 0.5).mean())
        rows.append(s)
        if verbose and (i + 1) % 40 == 0:
            print('  %d / %d' % (i + 1, len(files)), flush=True)
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def w_of(u):
    """Interpolation fraction in [0, 0.5] from u = 2 w (1 - w)."""
    return 0.5 * (1.0 - np.sqrt(np.clip(1.0 - 2.0 * u, 0, 1)))


def summarise(d):
    wx, wy = w_of(d['ux']), w_of(d['uy'])
    return dict(n=len(d['signal']), w_x_quartiles=np.percentile(wx, [10, 25, 50, 75, 90]),
                w_y_quartiles=np.percentile(wy, [10, 25, 50, 75, 90]), fit_rms_median=float(np.median(d['rms'])))


def run(n=200, tag='thx10'):
    from synthetic import realism
    deg = load_json(os.path.join(CONFIG, 'degradation.json'))
    lo, hi, off = deg['window']['lo'], deg['window']['hi'], deg['noise']['offset']
    d = measure(realism.real_files(n=n), lo, hi, off, verbose=True)
    ok = d['clipped'] < 0.002                      # heavy clipping distorts the spectrum
    fit = fit_camera(d['signal'][ok], d['V0'][ok], d['V_fixed'][ok])
    out = dict(per_stack=d, fit=fit, summary=summarise(d), n_used=int(ok.sum()),
               resample=dict(model='linear interpolation, per-stack fraction w uniform in [0, 0.5] on each in-plane axis',
                             w_max=0.5))
    save_json(os.path.join(out_dir('estimate'), 'noise_shape_%s.json' % tag), out)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    ax[0].plot(d['signal'], d['V0'], '.'); xx = np.linspace(0, d['signal'].max(), 50)
    ax[0].plot(xx, fit['read_std'] ** 2 + fit['gain'] * xx, 'k--', label='%.2f^2 + %.2f S' % (fit['read_std'], fit['gain']))
    ax[0].set_xlabel('mean signal of the stack (counts above offset)'); ax[0].set_ylabel('camera variance V0'); ax[0].legend()
    ax[1].plot(w_of(d['ux']), w_of(d['uy']), '.'); ax[1].set_xlabel('interpolation fraction along X'); ax[1].set_ylabel('along Y')
    ax[1].set_title('Per-stack in-plane resampling')
    fp = fit['fixed_pattern']
    ax[2].plot(d['signal'], d['V_fixed'], '.'); ax[2].plot(xx, fp['var0'] + fp['slope'] * np.minimum(xx, fp['sat']), 'k--')
    ax[2].set_ylim(0, np.percentile(d['V_fixed'], 97) * 1.5); ax[2].set_xlabel('mean signal'); ax[2].set_ylabel('fixed-pattern variance')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir('estimate'), 'noise_shape_%s.png' % tag), dpi=90); plt.close(fig)
    return out


def check_synthetic(folder, n=60):
    deg = load_json(os.path.join(CONFIG, 'degradation.json'))
    lo, hi, off = deg['window']['lo'], deg['window']['hi'], deg['noise']['offset']
    d = measure(sorted(glob.glob(os.path.join(folder, '*.tif')))[:n], lo, hi, off)
    ok = d['clipped'] < 0.002
    return dict(fit=fit_camera(d['signal'][ok], d['V0'][ok], d['V_fixed'][ok]), summary=summarise(d))


def _print(name, fit, s):
    fp = fit['fixed_pattern']
    print('%s: var = %.2f^2 + %.3f * signal | fixed pattern %.2f + %.3f * min(signal, %.0f)' % (
        name, fit['read_std'], fit['gain'], fp['var0'], fp['slope'], fp['sat']))
    print('   interpolation fraction, percentiles 10/25/50/75/90: X %s  Y %s  (fit rms %.2f)' % (
        np.round(s['w_x_quartiles'], 2), np.round(s['w_y_quartiles'], 2), s['fit_rms_median']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth', type=str, default=None, help='run the estimator on this synthetic folder instead')
    a = ap.parse_args()
    if a.synth:
        real = load_json(os.path.join(out_dir('estimate'), 'noise_shape_thx10.json'))
        _print('real ', real['fit'], real['summary'])
        c = check_synthetic(a.synth)
        _print('synth', c['fit'], c['summary'])
    else:
        o = run()
        _print('real', o['fit'], o['summary'])
