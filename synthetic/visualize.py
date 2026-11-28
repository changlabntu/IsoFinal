"""Side views (XZ) of raw / interpolated / model output, overlaid on the truth.

    python -m synthetic.visualize --method MScleanQDPSF/bg085_r1 --epoch 100
    python -m synthetic.visualize --method MScleanQDPSF/bg085_r1 --epoch 100 \
        --set val --block 0 --crop 0 --ys 48,96,144

One column per requested Y plane. Rows: the truth in grayscale, then for each
method a grayscale row and a green/magenta overlay row against the aligned
truth — truth-only structure shows green, method-only magenta, agreement white.
"Raw" is the degraded stack with each slice repeated ZSTEP times (the staircase
the model starts from). Alignment (per-method Z offset, in-plane resample
shift) follows synthetic/evaluate.py. The figure lands in synthetic/out/vis/.

Display maps [FLOOR, 1] (window units) to [0, 1]; --gamma < 1 lifts dim
structure. The methods always include raw and cubic; --method/--epoch adds a
training run (or registry model), and may repeat.
"""
import argparse
import os

import numpy as np

from synthetic.common import ZSTEP, out_dir
from synthetic import evaluate as ev


def nearest(v):
    """Each degraded slice repeated ZSTEP times: the raw stack at output scale."""
    return np.repeat(v, ZSTEP, axis=0)


def method_outputs(args, crops):
    """[(label, fn)] applied to each degraded crop -> {label: [volumes]}."""
    fns = [('raw', nearest), ('cubic', ev.cubic)]
    for name, epoch in zip(args.method, args.epoch):
        label = name.strip('/').replace('/', '_') + ('' if epoch is None else '_e%03d' % epoch)
        fns.append((label, ev.Model(name, epoch)))
    return {label: [np.clip(fn(v), -1, 1).astype(np.float32) for v in crops] for label, fn in fns}


def aligned_truth(outs, block, croplist, deg, wx, wy):
    """Truth on each method's grid (per-method Z offset from the lag search)."""
    truths, offsets = {}, {}
    for label, vols in outs.items():
        curves = [ev._lag_task((o, block['truth'], crop, deg)) for o, crop in zip(vols, croplist)]
        s = ev.z_offset(curves)
        offsets[label] = s
        truths[label] = [ev.truth_aligned(block['truth'], crop, deg, s, wx, wy) for crop in croplist]
    return truths, offsets


def disp(a, gamma, pnorm):
    """[FLOOR, 1] -> [0, 1]; pnorm > 0 rescales the volume so that percentile is 1,
    putting both overlay channels on a comparable footing (blur dims the raw peaks)."""
    d = (np.clip(a, ev.FLOOR, 1.0) - ev.FLOOR) / (1.0 - ev.FLOOR)
    if pnorm:
        d = np.clip(d / max(np.percentile(d, pnorm), 1e-6), 0.0, 1.0)
    return d ** gamma if gamma != 1.0 else d


def figure(outs, truths, offsets, ys, gamma, pnorm, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = list(outs)
    nrows, ncols = 1 + 2 * len(labels), len(ys)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False)
    t0 = disp(truths[labels[0]][0], gamma, pnorm)
    for j, y in enumerate(ys):
        axes[0][j].imshow(t0[:, y, :], cmap='gray', vmin=0, vmax=1)
        axes[0][j].set_title('y = %d' % y, fontsize=10)
    axes[0][0].set_ylabel('truth', fontsize=10)
    for i, label in enumerate(labels):
        m = disp(outs[label][0], gamma, pnorm)
        t = disp(truths[label][0], gamma, pnorm)
        for j, y in enumerate(ys):
            axes[1 + 2 * i][j].imshow(m[:, y, :], cmap='gray', vmin=0, vmax=1)
            axes[2 + 2 * i][j].imshow(np.stack([m[:, y, :], t[:, y, :], m[:, y, :]], -1))
        axes[1 + 2 * i][0].set_ylabel(label, fontsize=9)
        axes[2 + 2 * i][0].set_ylabel('overlay (truth=G)', fontsize=9)
    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title + '   (Z vertical, X horizontal; green = truth only, magenta = method only)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def spectrum(tags, split, path, fmax=0.16):
    """The axial band-correlation curves behind z_res: output-truth spectral
    correlation per Z frequency, summed over blocks, from the eval JSONs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from synthetic.common import load_json
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag in tags:
        r = load_json(os.path.join(out_dir('eval'), '%s__%s.json' % (tag, split)))
        if r is None:
            print('no eval record for %s on %s, skipping' % (tag, split))
            continue
        num, po, pt = (np.sum([np.asarray(b[k]) for b in r['per_block']], 0) for k in ('fc_num', 'fc_po', 'fc_pt'))
        fc = num / np.maximum(np.sqrt(po * pt), 1e-30)
        fz = np.arange(len(fc)) / (2.0 * (len(fc) - 1))
        ax.plot(fz[1:], fc[1:], label='%s (z_res %.1f)' % (tag, r['z_res']))
    ax.axhline(0.5, color='gray', lw=0.8, ls='--')
    ax.axvline(0.5 / ZSTEP / 2, color='gray', lw=0.8, ls=':')
    ax.text(0.5 / ZSTEP / 2, 1.02, 'sampling Nyquist (16 px)', fontsize=8, ha='center')
    top = ax.secondary_xaxis('top', functions=(lambda f: 0.5 / np.maximum(f, 1e-3), lambda p: 0.5 / np.maximum(p, 1e-3)))
    top.set_xticks([0.5 / p for p in (48, 24, 16, 12, 8, 6, 4)])
    top.set_xticklabels(['48', '24', '16', '12', '8', '6', '4'])
    top.set_xlabel('half period (px)', fontsize=9)
    ax.set_xlabel('axial frequency (cycles/px)')
    ax.set_ylabel('output-truth band correlation')
    ax.set_xlim(0, fmax)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8)
    ax.set_title('set %s: where each method resolves (z_res = 0.5 crossing)' % split, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fplane_sums(out, truth, nb_r=25):
    """2D spectral-correlation sums binned by (axial, lateral) frequency."""
    import scipy.fft as sfft
    oc = np.clip(out, ev.FLOOR, 1.0)[ev.INNER].astype(np.float64)
    tc = np.clip(truth, ev.FLOOR, 1.0)[ev.INNER].astype(np.float64)
    win = (np.hanning(oc.shape[0])[:, None, None] * np.hanning(oc.shape[1])[None, :, None]
           * np.hanning(oc.shape[2])[None, None, :])
    fo = sfft.rfftn((oc - oc.mean()) * win, workers=1)
    ft = sfft.rfftn((tc - tc.mean()) * win, workers=1)
    nz = oc.shape[0]
    kz = np.minimum(np.arange(nz), nz - np.arange(nz))                     # axial bin per plane
    fr = np.hypot(np.fft.fftfreq(oc.shape[1])[:, None], np.fft.rfftfreq(oc.shape[2])[None, :])
    kr = np.minimum((fr / 0.5 * nb_r).astype(int), nb_r - 1)               # lateral bin per (ky, kx)
    idx = (kz[:, None, None] * nb_r + kr[None]).ravel()
    n = (nz // 2 + 1) * nb_r
    sums = {}
    for key, v in (('num', (fo * np.conj(ft)).real), ('po', np.abs(fo) ** 2), ('pt', np.abs(ft) ** 2)):
        sums[key] = np.bincount(idx, weights=v.ravel(), minlength=n)[:n].reshape(nz // 2 + 1, nb_r)
    return sums


def fplane(outs, truths, title, path, nb_r=25):
    """Heatmaps of output-truth correlation over the (lateral, axial) frequency plane."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = list(outs)
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 4.2), squeeze=False)
    nz = ev.NZ - 2 * ev.MARGIN_Z
    for ax, label in zip(axes[0], labels):
        ss = [_fplane_sums(o, t, nb_r) for o, t in zip(outs[label], truths[label])]
        num, po, pt = (np.sum([s[k] for s in ss], 0) for k in ('num', 'po', 'pt'))
        fc = num / np.maximum(np.sqrt(po * pt), 1e-30)
        im = ax.imshow(np.clip(fc, 0, 1), origin='lower', aspect='auto', vmin=0, vmax=1, cmap='magma',
                       extent=(0, 0.5, 0, (nz // 2) / nz))
        ax.contour(np.clip(fc, 0, 1), levels=[0.5], colors='cyan', linewidths=1.0,
                   extent=(0, 0.5, 0, (nz // 2) / nz))
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('lateral frequency (cycles/px)')
        ax.set_xlim(0, 0.25)
        ax.set_ylim(0, 0.25)
    axes[0][0].set_ylabel('axial frequency (cycles/px)')
    fig.colorbar(im, ax=axes[0], shrink=0.85, label='output-truth correlation')
    fig.suptitle(title + '   (cyan: 0.5 contour)', fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', action='append', default=[], help='run name (under $LOGS/synthF/synthF/) or registry model; may repeat')
    ap.add_argument('--epoch', action='append', type=int, default=[], help='checkpoint epoch, one per --method (default: latest)')
    ap.add_argument('--root', type=str, default=None, help='dataset folder (default: $DATASET/synthF)')
    ap.add_argument('--set', type=str, default='val', choices=ev.SETS)
    ap.add_argument('--block', type=int, default=0, help='index into the set\'s blocks')
    ap.add_argument('--crop', type=int, default=0, help='index into evaluate.CROPS')
    ap.add_argument('--ys', type=str, default='48,96,144', help='Y planes of the side views')
    ap.add_argument('--gamma', type=float, default=0.8, help='display gamma on [FLOOR, 1]')
    ap.add_argument('--pnorm', type=float, default=99.5, help='normalize each volume so this percentile is white; 0 = absolute intensities')
    ap.add_argument('--out', type=str, default=None, help='output PNG (default: synthetic/out/vis/...)')
    ap.add_argument('--spectrum', type=str, default=None, help='comma-separated eval tags: plot their axial band-correlation curves instead (no GPU)')
    ap.add_argument('--fplane', action='store_true', help='plot the 2D (lateral x axial) frequency-plane correlation maps instead, over all four crops of the block')
    a = ap.parse_args()
    if a.spectrum:
        tags = a.spectrum.split(',')
        path = a.out or os.path.join(out_dir('vis'), 'spectrum_%s.png' % a.set)
        spectrum(tags, a.set, path)
        print('wrote', path)
        return
    a.epoch += [None] * (len(a.method) - len(a.epoch))

    import tifffile
    root = a.root or os.path.join(ev.env_path('DATASET'), 'synthF')
    deg = ev.degradation()
    block = ev.blocks(root, a.set)[a.block]
    crop = ev.CROPS[a.crop]
    wx, wy = ev.resample_shift(block['seed'], deg)
    v = tifffile.imread(block['deg'])
    croplist = list(ev.CROPS) if a.fplane else [crop]
    crops = [v[z0:z0 + ev.CROP_Z, y0:y0 + ev.CROP_XY, x0:x0 + ev.CROP_XY] for z0, y0, x0 in croplist]

    outs = method_outputs(a, crops)
    truths, offsets = aligned_truth(outs, block, croplist, deg, wx, wy)
    for label, s in offsets.items():
        print('%-28s z offset %+.2f planes' % (label, s))

    models = ', '.join(list(outs)[2:]) or 'interpolation'
    stem = '_'.join(list(outs)[2:]) or 'interp'
    if a.fplane:
        title = '%s  %s  block %s (4 crops)' % (a.set, models, block['name'])
        path = a.out or os.path.join(out_dir('vis'), 'fplane_%s_b%d_%s.png' % (a.set, a.block, stem))
        fplane(outs, truths, title, path)
    else:
        ys = [int(y) for y in a.ys.split(',')]
        title = '%s  %s  block %s  crop %d' % (a.set, models, block['name'], a.crop)
        path = a.out or os.path.join(out_dir('vis'), 'side_%s_b%d_c%d_%s.png' % (a.set, a.block, a.crop, stem))
        figure(outs, truths, offsets, ys, a.gamma, a.pnorm, title, path)
    print('wrote', path)


if __name__ == '__main__':
    main()
