"""Stage 1 summary: run (or reuse) the estimators, cross-check them, and write
config/degradation.json, the single file the phantom pipeline reads.

Cross-check (the gate). The transfer model comes from power spectra and
assumes the tissue is statistically isotropic. The fibre measurement assumes
nothing of the kind. For every measured fibre the model predicts the axial
width it should have given its lateral width; the ratio measured / predicted
should be close to 1.

    python -m synthetic.estimate.report            # reuse cached estimator outputs
    python -m synthetic.estimate.report --rerun    # recompute everything
"""
import argparse
import os

import numpy as np

from synthetic.common import CONFIG, FWHM, OUT, ZSTEP, load_json, out_dir, save_json
from synthetic.estimate import transfer


def _get(name, runner, rerun):
    path = os.path.join(OUT, 'estimate', '%s_thx10.json' % name)
    if rerun or not os.path.isfile(path):
        from synthetic.estimate.blocks import get_blocks
        runner(get_blocks())
    return load_json(path)


def _fwhm(z, p):
    p = np.nan_to_num(np.asarray(p, float))
    p = p / p.max()
    zz = np.linspace(z.min(), z.max(), 4001)
    a = zz[np.interp(zz, z, p) >= 0.5]
    return float(a.max() - a.min())


def cross_check(tr, ls):
    """Per-fibre measured / predicted axial FWHM."""
    z = np.arange(-200.0, 200.01, 1.0)
    sig = np.linspace(1.5, 14.0, 26)
    pred = np.array([_fwhm(z, transfer.predict_line_profile(tr, s, z)) for s in sig])
    s_lat = np.asarray(ls['per_fibre_sigma_lat'], float)
    s_z = np.asarray(ls['per_fibre_sigma_z'], float)
    ratio = s_z * FWHM / np.interp(s_lat, sig, pred)
    grid = np.asarray(ls['ax_grid'], float)
    prof = np.array([np.nan if q is None else q for q in ls['ax_profile']], float)
    pooled_pred = transfer.predict_line_profile(tr, ls['sigma_lat'], grid)
    return dict(n_fibres=int(len(ratio)), ratio_median=float(np.median(ratio)),
                ratio_p25=float(np.percentile(ratio, 25)), ratio_p75=float(np.percentile(ratio, 75)),
                ratio_mode=float(_mode(ratio)),
                pooled_axial_fwhm_measured=_fwhm(grid, prof),
                pooled_axial_fwhm_predicted=_fwhm(grid, pooled_pred),
                sigma_grid=sig, predicted_fwhm=pred), ratio, (grid, prof, pooled_pred)


def _mode(x, bins=30):
    h, e = np.histogram(x, bins=bins, range=(np.percentile(x, 1), np.percentile(x, 95)))
    i = int(np.argmax(np.convolve(h, [1, 2, 1], mode='same')))
    return 0.5 * (e[i] + e[i + 1])


def main(rerun=False):
    from synthetic.estimate import lsf, noise, spectra
    tr = _get('transfer', transfer.run, rerun)
    ls = _get('lsf', lsf.run, rerun)
    no = _get('noise', noise.run, rerun)
    sp = _get('spectra', spectra.run, rerun)
    win = load_json(os.path.join(OUT, 'estimate', 'window_thx10.json'))
    shape = load_json(os.path.join(OUT, 'estimate', 'noise_shape_thx10.json'))      # estimate/noise_shape.py
    width = load_json(os.path.join(OUT, 'estimate', 'width_thx10.json'))            # estimate/calibrate_width.py
    cc, ratio, (grid, prof, pooled_pred) = cross_check(tr, ls)

    # equivalent 1D kernel widths, for orientation only
    z = np.arange(-200.0, 200.01, 0.5)
    widths = {('%.2f' % f): _fwhm(z, _kernel_at(tr, f, z)) for f in (0.01, 0.03, 0.10)}

    deg = dict(
        source='THX10 crop zarr, %d blocks of 192 x 768 x 768' % tr['n_blocks'],
        zstep=ZSTEP,
        transfer=dict(a=tr['a'], b=tr['b'], p=tr['p'], model='R = exp(-(|fz| / (a + b f_r))^p)',
                      fitted_band_fr=tr['fit_fr'], fit_rms_log=tr['rms'], per_band=tr['per_band']),
        noise=dict(offset=no['offset'], gain=no['gain'], read_std=no['read_std'],
                   fixed_pattern_std=no['fixed_pattern_std'], noise_corr_sigma=no['noise_corr_sigma'],
                   flicker_std=no['flicker_std_p99'], jitter_std_px=no['jitter_std_px']),
        window=(dict(lo=float(np.median([w['lo'] for w in win])), hi=float(np.median([w['hi'] for w in win])),
                     source='fitted on %d training TIFFs located in the zarr' % len(win))
                if win else dict(lo=113.0, hi=588.0, source='inference window (training window not resolved)')),
        summary=dict(axial_kernel_fwhm_px_by_lateral_frequency=widths,
                     shell_average_fwhm_px=(sp.get('stretched') or {}).get('fwhm'),
                     lateral_fibre_fwhm_px=ls['sigma_lat'] * FWHM,
                     axial_fibre_fwhm_px=ls['sigma_z'] * FWHM),
        cross_check={k: v for k, v in cc.items() if k not in ('sigma_grid', 'predicted_fwhm')},
    )
    if shape:
        from synthetic.estimate.noise_shape import w_of
        wx, wy = w_of(np.asarray(shape['per_stack']['ux'])), w_of(np.asarray(shape['per_stack']['uy']))
        w = np.concatenate([wx, wy])
        deg['noise']['gain_first_estimate'] = deg['noise']['gain']
        deg['noise']['gain'] = shape['fit']['gain']
        deg['noise']['read_std'] = shape['fit']['read_std']
        deg['noise']['fixed_pattern'] = shape['fit']['fixed_pattern']
        # X and Y fractions belong together: one pair per tile of the real volume
        deg['noise']['resample'] = dict(w_pairs=np.round(np.stack([wx, wy], 1), 3),
                                        w_quantiles=np.percentile(w, np.linspace(0, 100, 21)),
                                        note='in-plane linear interpolation fraction per block: (x, y) pairs measured '
                                             'on real stacks, and 21 quantiles of both axes pooled')
        for k in ('fixed_pattern_std', 'noise_corr_sigma'):      # superseded by the entries above
            deg['noise'].pop(k, None)
    if width:
        deg['width'] = {k: width[k] for k in ('nominal', 'lo', 'hi', 'source')}
    path = save_json(os.path.join(CONFIG, 'degradation.json'), deg)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    ax[0].hist(ratio, bins=40, range=(0.5, 2.5))
    ax[0].axvline(1, color='k', ls='--'); ax[0].axvline(cc['ratio_median'], color='r', label='median %.2f' % cc['ratio_median'])
    ax[0].axvline(cc['ratio_mode'], color='g', label='mode %.2f' % cc['ratio_mode'])
    ax[0].set_xlabel('axial width: measured on fibre / predicted by transfer model'); ax[0].legend()
    ax[0].set_title('%d fibres' % cc['n_fibres'])
    ax[1].plot(grid, prof, 'C1o-', ms=3, label='measured, pooled (FWHM %.0f px)' % cc['pooled_axial_fwhm_measured'])
    ax[1].plot(grid, pooled_pred, 'k--', label='predicted (FWHM %.0f px)' % cc['pooled_axial_fwhm_predicted'])
    ax[1].set_xlabel('Z offset (XY px)'); ax[1].legend(); ax[1].set_title('Axial profile through an in-plane fibre')
    for f in (0.01, 0.03, 0.10):
        ax[2].plot(z, _kernel_at(tr, f, z), label='f_r = %.2f: FWHM %.0f px' % (f, widths['%.2f' % f]))
    ax[2].set_xlim(-120, 120); ax[2].set_xlabel('Z offset (XY px)'); ax[2].legend()
    ax[2].set_title('Axial kernel depends on lateral frequency')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir('estimate'), 'report_thx10.png'), dpi=90); plt.close(fig)
    return deg, path


def _kernel_at(tr, f_r, z):
    """Real-space axial kernel of the model at one lateral frequency."""
    dz = float(z[1] - z[0])
    n = 8192
    ff = np.fft.rfftfreq(n, d=dz)
    k = np.fft.fftshift(np.fft.irfft(transfer.transfer_amplitude(f_r, ff, tr), n))
    zz = (np.arange(n) - n // 2) * dz
    k = np.interp(z, zz, k)
    return k / k.max()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rerun', action='store_true')
    a = ap.parse_args()
    deg, path = main(a.rerun)
    import json
    deg = load_json(path)
    print(json.dumps({k: deg[k] for k in ('transfer', 'noise', 'window', 'summary', 'cross_check')}, indent=1))
    print('wrote', path)
