"""How wide is the THX10 axial blur? Settle it by simulation.

Stage 1 left two estimates that disagree by 10-25 %: the spectral transfer fit
and the widths of individual fibres. Each estimator has its own bias (the
spectral fit reads narrow, fibres read wide). Here both are run, unchanged, on
calibrated phantom blocks degraded at a series of known width scales (the
fitted transfer with f0 divided by the scale). The scale at which an estimator
returns what it returned on the real data is that estimator's bias-corrected
answer:

    spectral  kernel FWHM implied by the fitted (a, b, p), at three lateral
              frequencies, relative to the real fit
    fibres    median of measured / predicted axial width, the prediction made
              with the real fit (the Stage 1 cross-check; real value 1.24)

If the two corrected answers agree, the width is known. Where they do not,
the dataset is built over the range that covers both (each widened by its
own standard error), centred between them.

    python -m synthetic.estimate.calibrate_width --n_blocks 6
    python -m synthetic.estimate.calibrate_width --summarise     # refit from the cache
"""
import argparse
import os
import time

import numpy as np

from synthetic import common
from synthetic.common import CONFIG, FWHM, OUT, ZSTEP, load_json, out_dir, save_json
from synthetic.estimate import lsf, report, transfer

SCALES = (0.85, 1.0, 1.15, 1.3, 1.5)
BLOCK = (192, 768, 768)             # slices, y, x: the size the real estimators ran on
MARGIN_Z, MARGIN_XY = 96, 48
# real blocks were picked for their fibre content, so the phantom scenes are dense ones
DENSE = dict(empty_prob=0.0, fibre_density={'lognormal': [1.0e-5, 0.5]}, brightness={'lognormal': [1.7, 0.3]})
F_R = (0.01, 0.03, 0.10)


def _kernel_fwhm(tr, f_r):
    z = np.arange(-300.0, 300.01, 0.5)
    return report._fwhm(z, report._kernel_at(tr, f_r, z))


def simulate_block(seed, variant, deg, scales=SCALES):
    """One phantom block degraded at every scale -> list of (192, 768, 768) raw stacks."""
    from synthetic.phantom.degrade import acquire, apply_transfers, scale_transfer
    from synthetic.phantom.generate import make_block, sample_scene
    N = (BLOCK[0] * ZSTEP + 2 * MARGIN_Z, BLOCK[1] + 2 * MARGIN_XY, BLOCK[2] + 2 * MARGIN_XY)
    truth, _ = make_block(N, seed, sample_scene(seed, variant, DENSE))
    out = []
    for i, blurred in enumerate(apply_transfers(truth, [scale_transfer(deg['transfer'], s) for s in scales], 0, 0)):
        low = np.clip(blurred[MARGIN_Z:-MARGIN_Z:ZSTEP, MARGIN_XY:-MARGIN_XY + 1, MARGIN_XY:-MARGIN_XY + 1], 0, None)
        out.append(acquire(low, deg['noise'], np.random.default_rng([int(seed), i])))
    return out


def measure(raw):
    """Both estimators on one raw stack."""
    t = transfer.relative_transfer(raw)
    objs = lsf.measure_block(raw)
    return dict(logR=t['logR'], fr=t['fr'], fz=t['fz'],
                s_lat=np.array([o['s_min'] for o in objs]), s_z=np.array([o['s_z'] for o in objs]))


def run(n_blocks=6, seed0=700000, variants=('A', 'B', 'C'), blocks=None):
    """Simulate and measure blocks (cached one file per block; `blocks` picks a
    subset so several processes can share the work), then summarise."""
    deg = load_json(os.path.join(CONFIG, 'degradation.json'))
    d = out_dir('estimate', 'width_cache')
    for i in (range(n_blocks) if blocks is None else blocks):
        path = os.path.join(d, 'block%02d.npz' % i)
        if os.path.isfile(path):
            continue
        t0 = time.time()
        stacks = simulate_block(seed0 + i, variants[i % len(variants)], deg)
        res = [measure(r) for r in stacks]
        np.savez(path, scales=np.array(SCALES), fr=res[0]['fr'], fz=res[0]['fz'],
                 logR=np.array([r['logR'] for r in res]),
                 **{'s_lat_%d' % k: r['s_lat'] for k, r in enumerate(res)},
                 **{'s_z_%d' % k: r['s_z'] for k, r in enumerate(res)})
        print('block %d (%s): %.0f s, fibres per scale %s' % (i, variants[i % len(variants)], time.time() - t0,
                                                              [len(r['s_lat']) for r in res]), flush=True)
    return summarise()


def _solve(x, y, target):
    """x at which the (monotonic) curve y(x) crosses target, by interpolation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(y)
    return float(np.interp(target, y[o], x[o]))


def summarise(tag='thx10'):
    import glob
    d = os.path.join(OUT, 'estimate')
    real_tr = load_json(os.path.join(d, 'transfer_thx10.json'))
    real_ls = load_json(os.path.join(d, 'lsf_thx10.json'))
    real_cc, real_ratio, _ = report.cross_check(real_tr, real_ls)
    files = sorted(glob.glob(os.path.join(d, 'width_cache', 'block*.npz')))
    blocks = [np.load(f) for f in files]
    scales = blocks[0]['scales']
    fr, fz = blocks[0]['fr'], blocks[0]['fz']
    z = np.arange(-200.0, 200.01, 1.0)
    sig = np.linspace(1.5, 14.0, 26)
    pred = np.array([report._fwhm(z, transfer.predict_line_profile(real_tr, s, z)) for s in sig])
    real_fwhm = np.array([_kernel_fwhm(real_tr, f) for f in F_R])

    rows, ratios = [], []
    for k, s in enumerate(scales):
        with np.errstate(all='ignore'):
            pooled = np.nanmedian(np.array([b['logR'][k] for b in blocks]), 0)
        fit, _ = transfer.fit_transfer(pooled, fr, fz)
        per_block = [transfer.fit_transfer(b['logR'][k], fr, fz)[0] for b in blocks]
        fw = np.array([_kernel_fwhm(fit, f) for f in F_R])
        fw_blocks = np.array([[_kernel_fwhm(f_, f) for f in F_R] for f_ in per_block])
        s_lat = np.concatenate([b['s_lat_%d' % k] for b in blocks])
        s_z = np.concatenate([b['s_z_%d' % k] for b in blocks])
        ratio = s_z * FWHM / np.interp(s_lat, sig, pred) if len(s_lat) else np.array([np.nan])
        ratios.append(ratio)
        rows.append(dict(scale=float(s), a=fit['a'], b=fit['b'], p=fit['p'],
                         spectral_width=float(np.mean(fw / real_fwhm)),
                         spectral_width_blocks=(fw_blocks / real_fwhm).mean(1),
                         n_fibres=int(len(s_lat)), lateral_fwhm=float(np.median(s_lat) * FWHM) if len(s_lat) else None,
                         axial_fwhm=float(np.median(s_z) * FWHM) if len(s_lat) else None,
                         fibre_ratio=float(np.median(ratio)), fibre_ratio_mode=float(report._mode(ratio)) if len(ratio) > 30 else None,
                         fibre_ratio_iqr=[float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))]))
    sc = [r['scale'] for r in rows]
    s_spec = _solve(sc, [r['spectral_width'] for r in rows], 1.0)
    s_fib = _solve(sc, [r['fibre_ratio'] for r in rows], real_cc['ratio_median'])
    # spread of the spectral answer from block to block (each block solved on its own)
    per_block = [_solve(sc, [r['spectral_width_blocks'][j] for r in rows], 1.0) for j in range(len(blocks))]
    se_spec = float(np.std(per_block))
    # bootstrap the fibre answer over fibres (phantom and real)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(300):
        med = [np.median(rng.choice(r, len(r))) for r in ratios]
        boot.append(_solve(sc, med, np.median(rng.choice(real_ratio, len(real_ratio)))))
    se_fib = float(np.std(boot))
    nominal = 0.5 * (s_spec + s_fib)
    lo = min(s_spec - se_spec, s_fib - se_fib)
    hi = max(s_spec + se_spec, s_fib + se_fib)
    out = dict(n_blocks=len(blocks), scales=sc, rows=rows, real=dict(a=real_tr['a'], b=real_tr['b'], p=real_tr['p'],
               fibre_ratio=real_cc['ratio_median'], fibre_ratio_mode=real_cc['ratio_mode'],
               lateral_fwhm=real_ls['sigma_lat'] * FWHM, axial_fwhm=real_ls['sigma_z'] * FWHM, n_fibres=real_cc['n_fibres']),
               scale_from_spectral=s_spec, scale_from_fibres=s_fib, se_spectral=se_spec, se_fibres=se_fib,
               scale_from_spectral_per_block=per_block,
               nominal=round(nominal, 2), lo=round(lo, 2), hi=round(hi, 2),
               source='calibrate_width: %d phantom blocks; spectral %.2f +- %.2f, fibres %.2f +- %.2f' % (
                   len(blocks), s_spec, se_spec, s_fib, se_fib))
    save_json(os.path.join(d, 'width_%s.json' % tag), out)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(sc, [r['spectral_width'] for r in rows], 'o-', label='phantom')
    for j in range(len(blocks)):
        ax[0].plot(sc, [r['spectral_width_blocks'][j] for r in rows], color='C0', alpha=0.2, lw=0.8)
    ax[0].axhline(1.0, color='k', ls='--', label='real data'); ax[0].axvline(s_spec, color='r', label='match at %.2f' % s_spec)
    ax[0].set_xlabel('true width scale'); ax[0].set_ylabel('kernel width from spectral fit / real fit'); ax[0].legend()
    ax[0].set_title('Spectral estimator')
    ax[1].plot(sc, [r['fibre_ratio'] for r in rows], 'o-', label='phantom (median)')
    ax[1].fill_between(sc, [r['fibre_ratio_iqr'][0] for r in rows], [r['fibre_ratio_iqr'][1] for r in rows], alpha=0.2)
    ax[1].axhline(real_cc['ratio_median'], color='k', ls='--', label='real data (%.2f)' % real_cc['ratio_median'])
    ax[1].axvline(s_fib, color='r', label='match at %.2f' % s_fib)
    ax[1].set_xlabel('true width scale'); ax[1].set_ylabel('fibre axial width: measured / predicted by real fit'); ax[1].legend()
    ax[1].set_title('Fibre estimator')
    fig.tight_layout(); fig.savefig(os.path.join(d, 'width_%s.png' % tag), dpi=90); plt.close(fig)
    return out


def print_summary(o):
    print('real: a %.4f b %.3f p %.2f | fibres: lateral %.1f px, axial %.1f px, ratio %.2f (n %d)' % (
        o['real']['a'], o['real']['b'], o['real']['p'], o['real']['lateral_fwhm'], o['real']['axial_fwhm'],
        o['real']['fibre_ratio'], o['real']['n_fibres']))
    for r in o['rows']:
        print('scale %.2f: a %.4f b %.3f p %.2f  spectral width %.2f | fibres n %d lateral %s axial %s ratio %.2f' % (
            r['scale'], r['a'], r['b'], r['p'], r['spectral_width'], r['n_fibres'],
            '%.1f' % r['lateral_fwhm'] if r['lateral_fwhm'] else '-', '%.1f' % r['axial_fwhm'] if r['axial_fwhm'] else '-',
            r['fibre_ratio']))
    print('width scale that reproduces the real readings: spectral %.2f +- %.2f, fibres %.2f +- %.2f' % (
        o['scale_from_spectral'], o['se_spectral'], o['scale_from_fibres'], o['se_fibres']))
    print('per-block spectral:', np.round(o['scale_from_spectral_per_block'], 2))
    print('-> nominal %.2f, range %.2f - %.2f' % (o['nominal'], o['lo'], o['hi']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_blocks', type=int, default=6)
    ap.add_argument('--blocks', type=str, default=None, help='comma-separated block indices (to split the work)')
    ap.add_argument('--summarise', action='store_true')
    a = ap.parse_args()
    bl = [int(q) for q in a.blocks.split(',')] if a.blocks else None
    print_summary(summarise() if a.summarise else run(a.n_blocks, blocks=bl))
