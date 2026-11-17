"""Self-test of the degradation estimators on a volume whose degradation is
known: build an isotropic phantom, blur it with a chosen non-separable
transfer, keep every 8th slice, add noise, then run the same estimators that
are run on THX10 and compare with what was put in.

    python -m synthetic.estimate.selftest
"""
import argparse
import os

import numpy as np

from synthetic.common import FWHM, out_dir, save_json
from synthetic.estimate import lsf, spectra, transfer
from synthetic.phantom.degrade import degrade
from synthetic.phantom.generate import make_block

TRUE = dict(transfer=dict(a=0.0060, b=0.043, p=1.39), zstep=8,
            noise=dict(offset=100.0, gain=1.0, read_std=3.0, flicker_std=0.0, jitter_std_px=0.0),
            window=dict(lo=100.0, hi=600.0))


def run(n_blocks=3, shape=(1536, 512, 512), params=None, true=TRUE, tag='selftest'):
    tabs, objs, spec = [], [], []
    for i in range(n_blocks):
        truth, meta = make_block(shape, seed=100 + i, params=params)
        v = degrade(truth, true, seed=i)['raw']
        del truth
        t = transfer.relative_transfer(v)
        tabs.append(t)
        objs += lsf.measure_block(v)
        spec.append(spectra.analyse_volume(v))
        f, _ = transfer.fit_transfer(t['logR'], t['fr'], t['fz'])
        print('block %d: a %.4f b %.3f p %.2f | %d fibres so far' % (i, f['a'], f['b'], f['p'], len(objs)), flush=True)
    pooled, fit, rows, fr, fz = transfer.pooled_fit(tabs)
    d = out_dir('estimate')
    transfer.plot(pooled, fit, rows, fr, fz, os.path.join(d, 'transfer_%s.png' % tag))
    out = dict(true=true['transfer'], est=dict(a=fit['a'], b=fit['b'], p=fit['p'], rms=fit['rms']))
    # compare the two transfers where it matters: the fz at which power falls to 1/e
    for f_r in (0.01, 0.03, 0.1):
        out['f0_true_%.2f' % f_r] = true['transfer']['a'] + true['transfer']['b'] * f_r
        out['f0_est_%.2f' % f_r] = fit['a'] + fit['b'] * f_r
    if len(objs) >= 10:
        s, kf = lsf.summarise(objs)
        lsf.plot(objs, s, kf, os.path.join(d, 'lsf_%s.png' % tag))
        grid = np.asarray(s['ax_grid'], float)
        pred_true = transfer.predict_line_profile(true['transfer'], s['sigma_lat'], grid)
        pred_est = transfer.predict_line_profile(fit, s['sigma_lat'], grid)
        out['lsf'] = dict(n=len(objs), sigma_lat=s['sigma_lat'], axial_fwhm_measured=float(s['sigma_z'] * FWHM),
                          axial_fwhm_pooled=_fwhm(grid, np.asarray(s['ax_profile'], float)),
                          axial_fwhm_predicted_true=_fwhm(grid, pred_true),
                          axial_fwhm_predicted_est=_fwhm(grid, pred_est))
    sp = spectra.summarise(spec)
    out['shell_fwhm'] = sp['stretched']['fwhm'] if sp['stretched'] else None
    save_json(os.path.join(d, '%s.json' % tag), out)
    return out


def _fwhm(z, p):
    p = np.nan_to_num(p / np.nanmax(p))
    zz = np.linspace(z.min(), z.max(), 4001)
    q = np.interp(zz, z, p)
    a = zz[q >= 0.5]
    return float(a.max() - a.min())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_blocks', type=int, default=3)
    a = ap.parse_args()
    o = run(a.n_blocks)
    for k, v in o.items():
        print(k, v)
