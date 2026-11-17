"""Real vs synthetic: are degraded phantom stacks statistically like THX10?

Everything here works on stacks in the format the model trains on:
(Z, Y, X) float32 in [-1, 1], one Z slice = 8 XY px. The same statistics are
computed on real training TIFFs and on synthetic ones, then compared.

    python -m synthetic.realism stats  --real           # cache real statistics
    python -m synthetic.realism compare --synth DIR     # table + figure
    python -m synthetic.realism classify --synth DIR    # real-vs-synthetic classifier
    python -m synthetic.realism headroom --root ROOT    # truth brightness vs the window

Statistics per stack (all in raw camera counts, i.e. the window undone):
  scalars  mean, median, percentiles, fraction above 150 / 250 / 400 counts,
           fraction clipped at each end of the window
  hist     intensity histogram
  xy_spec  radially averaged power spectrum of XY slices
  z_corr   correlation between slices as a function of lag
  logR     relative axial transfer table (estimate/transfer.py)
"""
import argparse
import glob
import os

import numpy as np

from synthetic.common import CONFIG, THX10_TIFS, ZSTEP, load_json, out_dir, save_json
from synthetic.estimate import transfer

SPEC_EDGES = np.geomspace(1.0 / 384, 0.5, 33)
SPEC_F = np.sqrt(SPEC_EDGES[1:] * SPEC_EDGES[:-1])
LAGS = np.arange(1, 9)
SCALARS = ('mean', 'med', 'p1', 'p90', 'p99', 'p999', 'f150', 'f250', 'f400', 'fhi', 'flo', 'std')


def degradation():
    return load_json(os.path.join(CONFIG, 'degradation.json'))


def from_window(v, lo, hi):
    """[-1, 1] -> raw counts (values clipped by the window stay clipped)."""
    return (np.asarray(v, np.float32) + 1.0) * 0.5 * (hi - lo) + lo


def xy_spectrum(raw, n_slices=8):
    """Radial power spectrum of XY slices (Hann window), counts^2 per mode."""
    Z, Y, X = raw.shape
    w = np.outer(np.hanning(Y), np.hanning(X)).astype(np.float32)
    fy = np.fft.fftfreq(Y)[:, None]
    fx = np.fft.rfftfreq(X)[None, :]
    fr = np.sqrt(fy ** 2 + fx ** 2).ravel()
    bi = np.digitize(fr, SPEC_EDGES) - 1
    ok = (bi >= 0) & (bi < len(SPEC_F))
    cnt = np.maximum(np.bincount(bi[ok], minlength=len(SPEC_F)), 1)
    acc = np.zeros(len(SPEC_F))
    zs = np.linspace(0, Z - 1, n_slices).round().astype(int)
    for z in zs:
        s = raw[z] - raw[z].mean()
        P = np.abs(np.fft.rfft2(s * w)) ** 2 / (w ** 2).sum()
        acc += np.bincount(bi[ok], weights=P.ravel()[ok], minlength=len(SPEC_F)) / cnt
    return acc / len(zs)


def z_correlation(raw):
    """Mean Pearson correlation between slices `lag` apart."""
    v = raw - raw.mean(axis=(1, 2), keepdims=True)
    sd = np.sqrt((v ** 2).mean(axis=(1, 2)))
    out = []
    for lag in LAGS:
        num = (v[:-lag] * v[lag:]).mean(axis=(1, 2))
        den = sd[:-lag] * sd[lag:]
        out.append(float(np.mean(num / np.maximum(den, 1e-9))))
    return np.array(out)


def stack_stats(v, deg=None):
    """Statistics of one (Z, Y, X) stack in [-1, 1]."""
    deg = deg or degradation()
    lo, hi = deg['window']['lo'], deg['window']['hi']
    raw = from_window(v, lo, hi)
    pc = np.percentile(raw[:, ::2, ::2], [1, 50, 90, 99, 99.9])
    s = dict(mean=float(raw.mean()), med=float(pc[1]), p1=float(pc[0]), p90=float(pc[2]), p99=float(pc[3]),
             p999=float(pc[4]), std=float(raw.std()),
             f150=float((raw > 150).mean()), f250=float((raw > 250).mean()), f400=float((raw > 400).mean()),
             fhi=float((v >= 1).mean()), flo=float((v <= -1).mean()))
    s['hist'] = np.bincount(np.rint(raw).astype(np.int64).ravel() - int(round(lo)), minlength=int(round(hi - lo)) + 1) / raw.size
    s['xy_spec'] = xy_spectrum(raw)
    s['z_corr'] = z_correlation(raw)
    t = transfer.relative_transfer(raw, ZSTEP)
    s['logR'], s['fr'], s['fz'] = t['logR'], t['fr'], t['fz']
    return s


def collect(files, deg=None, verbose=True):
    import tifffile
    out = []
    for i, f in enumerate(files):
        out.append(stack_stats(tifffile.imread(f), deg))
        if verbose and (i + 1) % 20 == 0:
            print('  %d / %d' % (i + 1, len(files)), flush=True)
    return pack(out)


def pack(stats):
    d = {k: np.array([s[k] for s in stats]) for k in SCALARS + ('hist', 'xy_spec', 'z_corr', 'logR')}
    d['fr'], d['fz'] = stats[0]['fr'], stats[0]['fz']
    return d


def pooled_transfer(d, tissue_only=True):
    """Fit (a, b, p) to the median logR of the stacks that hold tissue."""
    keep = d['mean'] > 115 if tissue_only else np.ones(len(d['mean']), bool)
    with np.errstate(all='ignore'):
        pooled = np.nanmedian(d['logR'][keep], 0)
    fit, _ = transfer.fit_transfer(pooled, d['fr'], d['fz'])
    return fit


def real_files(split='train', n=200, seed=0):
    files = sorted(glob.glob(os.path.join(THX10_TIFS, split, 'roiD', '*.tif')))
    rng = np.random.default_rng(seed)
    return [files[i] for i in sorted(rng.choice(len(files), min(n, len(files)), replace=False))]


def real_stats(n=200, refresh=False):
    path = os.path.join(out_dir('realism'), 'real_stats_%d.npz' % n)
    if os.path.isfile(path) and not refresh:
        return dict(np.load(path))
    d = collect(real_files(n=n))
    np.savez(path, **d)
    return d


def ks(a, b):
    """Two-sample Kolmogorov-Smirnov distance."""
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, side='right') / len(a) -
                               np.searchsorted(b, x, side='right') / len(b))))


def compare(real, synth, tag='synth', title=None):
    """Table (returned) and figure of real vs synthetic statistics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    q = (5, 25, 50, 75, 95)
    table = {}
    for k in SCALARS:
        table[k] = dict(real=np.percentile(real[k], q), synth=np.percentile(synth[k], q), ks=ks(real[k], synth[k]))
    tr, ts = real['mean'] > 115, synth['mean'] > 115
    with np.errstate(all='ignore'):
        spec_r = np.exp(np.nanmean(np.log(real['xy_spec'][tr]), 0))
        spec_s = np.exp(np.nanmean(np.log(synth['xy_spec'][ts]), 0))
    okf = np.isfinite(spec_r) & np.isfinite(spec_s) & (spec_r > 0) & (spec_s > 0)
    table['xy_spec_ratio'] = np.where(okf, spec_s / np.where(okf, spec_r, 1.0), np.nan)
    table['xy_spec_log_ratio_rms'] = float(np.sqrt(np.mean(np.log(table['xy_spec_ratio'][okf]) ** 2)))
    zr, zs = np.median(real['z_corr'][tr], 0), np.median(synth['z_corr'][ts], 0)
    table['z_corr_real'], table['z_corr_synth'] = zr, zs
    fr_, fs_ = pooled_transfer(real), pooled_transfer(synth)
    table['transfer_real'] = dict(a=fr_['a'], b=fr_['b'], p=fr_['p'])
    table['transfer_synth'] = dict(a=fs_['a'], b=fs_['b'], p=fs_['p'])

    fig, ax = plt.subplots(2, 4, figsize=(24, 10))
    x = np.arange(real['hist'].shape[1]) + 99
    ax[0, 0].semilogy(x, real['hist'].mean(0), label='real'); ax[0, 0].semilogy(x, synth['hist'].mean(0), label='synthetic')
    ax[0, 0].set_xlabel('raw counts'); ax[0, 0].set_title('Intensity histogram (mean over stacks)'); ax[0, 0].legend()
    for lab, d, sel, c in (('real', real, tr, 'C0'), ('synthetic', synth, ts, 'C1')):
        lg = np.log10(d['xy_spec'][sel])
        ax[0, 1].fill_between(SPEC_F, 10 ** np.percentile(lg, 10, 0), 10 ** np.percentile(lg, 90, 0), color=c, alpha=0.2)
        ax[0, 1].loglog(SPEC_F, 10 ** np.median(lg, 0), color=c, label=lab)
    ax[0, 1].set_xlabel('in-plane frequency (cycles / px)'); ax[0, 1].set_title('XY power spectrum (median, 10-90 %)'); ax[0, 1].legend()
    ax[0, 2].plot(LAGS, zr, 'o-', label='real'); ax[0, 2].plot(LAGS, zs, 'o-', label='synthetic')
    for lab, d, sel, c in (('real', real, tr, 'C0'), ('synthetic', synth, ts, 'C1')):
        ax[0, 2].fill_between(LAGS, np.percentile(d['z_corr'][sel], 10, 0), np.percentile(d['z_corr'][sel], 90, 0), color=c, alpha=0.2)
    ax[0, 2].set_xlabel('lag (slices)'); ax[0, 2].set_title('Slice-to-slice correlation'); ax[0, 2].legend()
    with np.errstate(all='ignore'):
        Lr, Ls = np.nanmedian(real['logR'][tr], 0), np.nanmedian(synth['logR'][ts], 0)
    cm = plt.get_cmap('viridis')
    rows = np.where((real['fr'] >= 0.006) & (real['fr'] <= 0.15))[0][::3]
    for n_, j in enumerate(rows):
        c = cm(n_ / max(len(rows) - 1, 1))
        ax[0, 3].plot(real['fz'], Lr[j], color=c, label='f_r %.3f' % real['fr'][j])
        ax[0, 3].plot(synth['fz'], Ls[j], '--', color=c)
    ax[0, 3].set_ylim(-10, 1); ax[0, 3].set_xlabel('fz'); ax[0, 3].legend(fontsize=7)
    ax[0, 3].set_title('Axial transfer log R: real (solid), synthetic (dashed)')
    for a_, k in zip(ax[1], ('mean', 'p99', 'f150', 'f400')):
        lo_, hi_ = min(real[k].min(), synth[k].min()), max(real[k].max(), synth[k].max())
        bins = np.linspace(lo_, hi_, 30)
        a_.hist(real[k], bins=bins, alpha=0.6, density=True, label='real')
        a_.hist(synth[k], bins=bins, alpha=0.6, density=True, label='synthetic')
        a_.set_title('per-stack %s  (KS %.2f)' % (k, table[k]['ks'])); a_.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    p = os.path.join(out_dir('realism'), 'compare_%s.png' % tag)
    fig.savefig(p, dpi=70); plt.close(fig)
    save_json(os.path.join(out_dir('realism'), 'compare_%s.json' % tag), table)
    return table, p


def montage(real_files_, synth_files, tag='synth', n=6, seed=0):
    """Real (top two rows) and synthetic (bottom two) XY slices and XZ sections,
    stacks ordered by mean brightness so like is compared with like."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import tifffile

    def pick(files):
        rng = np.random.default_rng(seed)
        files = [files[i] for i in rng.choice(len(files), min(len(files), 4 * n), replace=False)]
        vols = [tifffile.imread(f) for f in files]
        order = np.argsort([v.mean() for v in vols])
        return [vols[order[int(q * (len(order) - 1))]] for q in np.linspace(0.1, 0.95, n)]
    fig, ax = plt.subplots(4, n, figsize=(5 * n, 20))
    for r, (lab, vols) in enumerate((('real', pick(real_files_)), ('synthetic', pick(synth_files)))):
        for k, v in enumerate(vols):
            ax[2 * r, k].imshow(v[v.shape[0] // 2], cmap='gray', vmin=-1, vmax=0.1, interpolation='nearest')
            ax[2 * r, k].set_title('%s XY' % lab)
            ax[2 * r + 1, k].imshow(np.repeat(v[:, v.shape[1] // 2], ZSTEP, axis=0), cmap='gray', vmin=-1, vmax=0.1,
                                    interpolation='nearest')
            ax[2 * r + 1, k].set_title('%s XZ (slices repeated 8x)' % lab)
    for a_ in ax.ravel():
        a_.axis('off')
    fig.tight_layout(pad=0.3)
    p = os.path.join(out_dir('realism'), 'montage_%s.png' % tag)
    fig.savefig(p, dpi=40); plt.close(fig)
    return p


def headroom(root, floor_window=-0.8, tag='synth'):
    """How far the sharp truth exceeds the intensity window of the data.

    The blur spreads a thin structure over many planes, so its peak in the
    data is several times lower than in the isotropic truth. The window (and
    the model's [-1, 1] output range) was set on the data. Per held-out block:
    truth and degraded percentiles in raw counts, the fraction of truth voxels
    above the window ceiling, the same among voxels above the model's input
    floor (`floor_window`, the --gamma_lo of the baseline), and the share of
    above-floor signal that a clip at the ceiling removes."""
    import tifffile
    deg = degradation()
    lo, hi, off = deg['window']['lo'], deg['window']['hi'], deg['noise']['offset']
    floor = (floor_window + 1) * 0.5 * (hi - lo) + lo
    rows = []
    for f in sorted(glob.glob(os.path.join(root, 'gt', 'val', '*.tif'))):
        t = tifffile.imread(f).astype(np.float32) + off
        d = from_window(tifffile.imread(os.path.join(root, 'val', 'deg', os.path.basename(f))), lo, hi)
        tp, dp = np.percentile(t[::2, ::2, ::2], [99, 99.9, 99.99]), np.percentile(d, [99, 99.9, 99.99])
        bright = t > floor
        rows.append(dict(truth_p99=tp[0], truth_p999=tp[1], truth_p9999=tp[2], truth_max=float(t.max()),
                         data_p99=dp[0], data_p999=dp[1], data_p9999=dp[2],
                         frac_above_ceiling=float((t > hi).mean()),
                         frac_bright=float(bright.mean()),
                         frac_bright_above_ceiling=float((t[bright] > hi).mean()) if bright.any() else 0.0,
                         signal_clipped=float(np.clip(t - hi, 0, None).sum() / max(np.clip(t - floor, 0, None).sum(), 1e-9)),
                         data_frac_clipped=float((d >= hi).mean())))
    out = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    out.update(n_blocks=len(rows), window=[lo, hi], floor_counts=float(floor),
               peak_ratio_p999=float(np.median([(r['truth_p999'] - off) / max(r['data_p999'] - off, 1e-6) for r in rows])),
               per_block=rows)
    save_json(os.path.join(out_dir('realism'), 'headroom_%s.json' % tag), out)
    return out


def print_table(table):
    print('%-6s %-9s %s' % ('stat', 'KS', 'percentiles 5/25/50/75/95: real | synthetic'))
    for k in SCALARS:
        t = table[k]
        fmt = '%.4f' if k.startswith('f') else '%.0f'
        print('%-6s %-9.2f %s | %s' % (k, t['ks'], ' '.join(fmt % x for x in t['real']), ' '.join(fmt % x for x in t['synth'])))
    print('XY spectrum, rms log ratio: %.2f; synth/real by frequency:' % table['xy_spec_log_ratio_rms'])
    print('   ', ' '.join('%.3f:%.2f' % (f, r) for f, r in zip(SPEC_F[::3], np.asarray(table['xy_spec_ratio'])[::3])))
    print('z corr real :', np.round(table['z_corr_real'], 3))
    print('z corr synth:', np.round(table['z_corr_synth'], 3))
    print('transfer real :', {k: round(v, 4) for k, v in table['transfer_real'].items()})
    print('transfer synth:', {k: round(v, 4) for k, v in table['transfer_synth'].items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['stats', 'compare', 'classify', 'headroom'])
    ap.add_argument('--root', type=str, default=None, help='dataset root (for headroom)')
    ap.add_argument('--real', action='store_true')
    ap.add_argument('--synth', type=str, default=None, help='folder of synthetic (48,384,384) TIFFs')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--tag', type=str, default='synth')
    ap.add_argument('--model_domain', action='store_true', help='classify: apply the baseline input mapping first')
    a = ap.parse_args()
    if a.cmd == 'stats':
        d = real_stats(a.n, refresh=True)
        f = pooled_transfer(d)
        print('real: %d stacks; pooled transfer a %.4f b %.3f p %.2f' % (len(d['mean']), f['a'], f['b'], f['p']))
    elif a.cmd == 'compare':
        files = sorted(glob.glob(os.path.join(a.synth, '*.tif')))[:a.n]
        t, p = compare(real_stats(a.n), collect(files), a.tag)
        print_table(t); print(p)
        print(montage(real_files(n=a.n), files, a.tag))
    elif a.cmd == 'headroom':
        h = headroom(a.root, tag=a.tag)
        for k, v in h.items():
            if k != 'per_block':
                print('%-28s %s' % (k, np.round(v, 4)))
    else:
        from synthetic.classify import run as run_classifier
        print(run_classifier(a.synth, n=a.n, tag=a.tag, model_domain=(0.7, -0.8) if a.model_domain else None))
