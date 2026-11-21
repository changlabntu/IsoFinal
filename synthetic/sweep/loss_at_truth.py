"""Which Z-consistency loss is correct?

Each candidate loss is evaluated on the held-out synthF blocks with a known
volume in the place of the generator output:

    truth     the true volume, limited to what the model can output (input
              floor to window ceiling)
    blurred   the noise-free stack interpolated along Z: no deblurring at all

A correct loss is near zero at the truth when compared with the noise-free
stack (near the noise level when compared with the stored, noisy stack), has
its minimum with the truth at its own position (shift 0), and prefers the
truth to the blurred volume.

    python -m synthetic.sweep.loss_at_truth [--limit N] [--workers 6]
    python -m synthetic.sweep.loss_at_truth --model MScleanQD/base_r1 --epoch 100

With --model the old and the new loss are compared head to head on three
volumes: the truth, the blurred volume and the output of that trained model
(the evaluate.py crops, 24 x 192 x 192, interior only). It also reports the new
loss at the truth on training-size crops (16 x 128 x 128), by the number of
border slices left out, i.e. how much of a crop the loss can use.

Everything is computed as in training: L1 in the model's input units (window
units through the floor and gamma of the baseline, --gamma_lo -0.8 --gamma 0.7),
slice i against output planes around 8 i. A shift s means the volume is placed
s planes deeper (the content of slice i at output plane 8 i + s). Only the block
interior is used (slices 8-39, 32 px in from the faces), because the blur of the
stored truth is reflect-padded here instead of using the rendered margin.
"""
import argparse
import os

import numpy as np

from synthetic import common
from synthetic import evaluate as ev
from synthetic.common import ZSTEP, out_dir, save_json

LO, GAMMA = -0.8, 0.7                 # the baseline's input mapping
Z0, Z1 = 8, 40                        # interior slices
XY = slice(32, 352)
SHIFTS = np.arange(-24, 25)
WRONG = 1.3                           # deliberately wrong blur width, relative to the nominal one
BG_FIXED = -0.9                       # one background level for all stacks (128 counts)


def norm(v, lo=LO):
    """Window units -> model units (floor, gamma), as inference/normalize.py '11g'."""
    return (np.clip((v - lo) / (1 - lo), 0, 1) ** GAMMA * 2 - 1).astype(np.float32)


def background(v, b, how):
    """A volume limited to [LO, 1] with its floor read as the background level b:
    'hard' moves only the voxels at the floor, 'affine' maps [LO, 1] onto [b, 1]."""
    if how == 'hard':
        return np.where(v > LO, v, np.float32(b)).astype(np.float32)
    return (b + (v - LO) * ((1 - b) / (1 - LO))).astype(np.float32)


def on_stack_grid(v, wx, wy):
    """(Z, Y, X) on the truth grid -> the stack's in-plane grid, with the
    acquisition's own linear resampling; the last row and column are repeated."""
    from synthetic.phantom.degrade import resample_xy
    return np.pad(resample_xy(v, wx, wy), ((0, 0), (0, 1), (0, 1)), mode='edge')


def curves(out_n, blurred, targets, Z0=Z0, Z1=Z1, XY=XY):
    """Loss of every candidate at every shift. `out_n`: the volume in model
    units (384, Y, X); `blurred`: {name: (the volume after a blur, kind)} with
    kind 'model' (already in model units), 'linear' (window units) or 'nofloor'
    (window units, scored with the floor removed from the mapping); `targets`:
    {kind: [stored stack, noise-free stack]} in model units.
    Returns {loss: (shifts, 2) sums of |difference|}, {loss: n}."""
    sums, counts = {}, {}

    def put(name, sel, pred_of_shift, kind='linear'):
        tg = [t[sel][:, XY, XY] for t in targets[kind]]
        sums[name] = np.array([[np.abs(p - t).sum(dtype=np.float64) for t in tg]
                               for p in (pred_of_shift(s) for s in SHIFTS)])
        counts[name] = tg[0].size

    o = out_n[:, XY, XY]
    for how, depth, skip in (('max', 32, 4), ('max', 8, 1), ('mean', 32, 4), ('mean', 8, 1)):
        sel = np.arange(Z0, Z1, skip)
        idx = (ZSTEP * sel)[:, None] + np.arange(depth)[None]
        red = np.max if how == 'max' else np.mean
        put('%s over %d planes, every %s' % (how, depth, '4th slice' if skip == 4 else 'slice'),
            sel, lambda s, idx=idx, red=red: red(o[idx - s], axis=1))
    sel = np.arange(Z0, Z1)
    put('one plane (dsp), every slice', sel, lambda s: o[ZSTEP * sel + ZSTEP // 2 - s])
    for name, (vol, kind) in blurred.items():
        v = vol[:, XY, XY]
        fn = {'model': lambda p: np.clip(p, -1, 1), 'linear': norm, 'nofloor': lambda p: norm(p, -1.0)}[kind]
        put(name, sel, lambda s, v=v, fn=fn: fn(v[ZSTEP * sel - s]), kind)
    return sums, counts


def block_task(b):
    import tifffile
    from synthetic.phantom.degrade import apply_transfers, scale_transfer
    deg = ev.degradation()
    w, off = deg['window'], deg['noise']['offset']
    w0 = (off - w['lo']) / (w['hi'] - w['lo']) * 2 - 1                  # zero signal, in window units
    nominal = deg['width']['nominal']
    tr = lambda width: scale_transfer(deg['transfer'], width)
    blur = lambda v, widths: list(apply_transfers(v, [tr(x) for x in widths]))

    x = tifffile.imread(b['deg'])
    wx, wy = ev.resample_shift(b['seed'], deg)
    t = on_stack_grid(ev._truth_block(b['truth'], deg), wx, wy)        # truth, unclipped, on the stack grid
    ev._TRUTH.clear()

    exact, = blur(t, [nominal])
    nf = np.clip(np.maximum(exact[::ZSTEP], w0), -1, 1)                 # the stack without noise
    targets = {'linear': [norm(x), norm(nf)], 'nofloor': [norm(x, -1.0), norm(nf, -1.0)]}
    targets['model'] = targets['linear']
    bg = float(np.median(x[x < LO]))                                    # background level measured on the input stack

    def candidates(v):
        """Blur-and-sample losses for an output v limited to [LO, 1] (window units)."""
        one = lambda u: blur(u, [nominal])[0]
        return {'measured blur, in model units': (one(norm(v)), 'model'),
                'measured blur, in linear units': (one(v), 'linear'),
                '  floor read as zero signal, hard': (one(background(v, w0, 'hard')), 'linear'),
                '  floor read as stack background, hard': (one(background(v, bg, 'hard')), 'linear'),
                '  floor read as stack background, affine': (one(background(v, bg, 'affine')), 'linear'),
                '  floor read as %.2f, affine' % BG_FIXED: (one(background(v, BG_FIXED, 'affine')), 'linear')}

    res = {}
    # --- the truth as output
    a = np.clip(t, LO, 1)                                               # what the model can output
    c = candidates(a)
    ceil, narrow, wide, wrong = blur(np.minimum(t, 1), [nominal, deg['width']['lo'], deg['width']['hi'], nominal * WRONG])
    c['output with no floor, ceiling kept'] = (ceil, 'linear')
    c['  blur width %.2f (data: %.2f)' % (deg['width']['lo'], nominal)] = (narrow, 'linear')
    c['  blur width %.2f' % deg['width']['hi']] = (wide, 'linear')
    c['  blur width %.2f (wrong by %.1fx)' % (nominal * WRONG, WRONG)] = (wrong, 'linear')
    c['same, floor also removed from the mapping'] = (ceil, 'nofloor')
    c['exact: output not limited at all'] = (exact, 'linear')
    c['exact, floor removed from the mapping'] = (exact, 'nofloor')
    res['truth'] = curves(norm(a), c, targets)
    del c, ceil, narrow, wide, wrong, exact, a, t
    # --- the blurred volume as output
    bv = ev.cubic(nf)
    c = candidates(np.clip(bv, LO, 1))
    ceil, = blur(bv, [nominal])
    c['output with no floor, ceiling kept'] = (ceil, 'linear')
    c['same, floor also removed from the mapping'] = (ceil, 'nofloor')
    res['blurred'] = curves(norm(bv), c, targets)
    return b['name'], res, bg


# ---------------------------------------------------------------- head to head on crops, with a trained model

C_Z0, C_Z1, C_XY = 8, 18, slice(32, 160)          # scored part of a 24 x 192 x 192 crop
OLD = 'max over 32 planes, every 4th slice'
NEW = ('new, floor as it is', 'new, floor read as crop background', 'new, floor read as %.2f' % BG_FIXED)
SMALL = ((8, 64, 64), (24, 192, 192), (16, 128, 200), (20, 220, 80))    # training-size crops (z0 in slices, y0, x0)


def block_volumes(b):
    """A held-out block: the stored stack, the truth on the stack grid (not
    limited), the stack without noise, and the nominal blur as a function."""
    import tifffile
    from synthetic.phantom.degrade import apply_transfers, scale_transfer
    deg = ev.degradation()
    w, off = deg['window'], deg['noise']['offset']
    w0 = (off - w['lo']) / (w['hi'] - w['lo']) * 2 - 1
    tr = scale_transfer(deg['transfer'], deg['width']['nominal'])
    blur = lambda v, zpad=64, xypad=32: next(apply_transfers(v, [tr], zpad, xypad))
    x = tifffile.imread(b['deg'])
    wx, wy = ev.resample_shift(b['seed'], deg)
    t = on_stack_grid(ev._truth_block(b['truth'], deg), wx, wy)
    ev._TRUTH.clear()
    nf = np.clip(np.maximum(blur(t)[::ZSTEP], w0), -1, 1)               # from the whole block, as in block_task
    return x, t, nf, blur


def crops_task(job):
    """Every loss on the crops of one block, for the truth, the blurred volume
    and a model's output (`outs`: one 192^3 volume per evaluate.CROPS entry)."""
    b, outs = job
    level = lambda xc: float(np.median(xc[xc < LO])) if (xc < LO).any() else BG_FIXED
    x, t, nf, blur = block_volumes(b)
    t = np.clip(t, LO, 1)

    res = {}
    for (z0, y0, x0), out in zip(ev.CROPS, outs):
        cz, cy, cx = slice(z0, z0 + ev.CROP_Z), slice(y0, y0 + ev.CROP_XY), slice(x0, x0 + ev.CROP_XY)
        xc, nfc = x[cz, cy, cx], nf[cz, cy, cx]
        targets = {'linear': [norm(xc), norm(nfc)]}
        bg = level(xc)
        vols = {'truth': t[ZSTEP * z0:ZSTEP * (z0 + ev.CROP_Z), cy, cx],
                'blurred volume': np.clip(ev.cubic(nfc), LO, 1),
                'model output': np.clip(out, LO, 1)}
        for name, v in vols.items():
            c = {NEW[0]: (blur(v), 'linear'),
                 NEW[1]: (blur(background(v, bg, 'affine')), 'linear'),
                 NEW[2]: (blur(background(v, BG_FIXED, 'affine')), 'linear')}
            sums, counts = curves(norm(v), c, targets, C_Z0, C_Z1, C_XY)
            for k in sums:
                r = res.setdefault((name, k), [0.0, 0])
                r[0] = r[0] + sums[k]
                r[1] += counts[k]
    # the new loss at the truth on training-size crops: |difference| per slice of a 16-slice crop
    small = np.zeros(16)
    for z0, y0, x0 in SMALL:
        cz, cy, cx = slice(z0, z0 + 16), slice(y0, y0 + 128), slice(x0, x0 + 128)
        v = background(t[ZSTEP * z0:ZSTEP * (z0 + 16), cy, cx], level(x[cz, cy, cx]), 'affine')
        pred = norm(blur(v, 32, 16)[::ZSTEP])
        small += np.abs(pred - norm(nf[cz, cy, cx]))[:, 16:112, 16:112].mean((1, 2))
    return res, small / len(SMALL)


def run_model(root, name, epoch, limit=None, workers=6):
    import multiprocessing as mp
    import tifffile
    import torch
    bl = ev.blocks(root, 'val')[:limit]
    fn = ev.Model(name, epoch)
    torch.manual_seed(0)

    def jobs():
        for b in bl:
            v = tifffile.imread(b['deg'])
            yield b, [fn(v[z0:z0 + ev.CROP_Z, y0:y0 + ev.CROP_XY, x0:x0 + ev.CROP_XY]) for z0, y0, x0 in ev.CROPS]

    with mp.get_context('spawn').Pool(workers, initializer=_init, initargs=(4,)) as pool:
        per_block = list(pool.imap(crops_task, jobs(), chunksize=1))
    out = dict(model=name, epoch=epoch, n_blocks=len(bl), shifts=SHIFTS.tolist(), losses={},
               small_crop_per_slice=np.mean([r[1] for r in per_block], 0).tolist())
    for key in per_block[0][0]:
        loss = sum(r[0][key][0] for r in per_block) / sum(r[0][key][1] for r in per_block)       # (shifts, 2)
        out['losses']['%s | %s' % key] = dict(at0=loss[SHIFTS == 0][0].tolist(), best=loss.min(0).tolist(),
                                              best_shift=SHIFTS[loss.argmin(0)].tolist())
    save_json(os.path.join(out_dir('eval'), 'loss_head_to_head.json'), out)
    return out


def show_model(out):
    print('%d blocks, model %s epoch %s. L1 x 1000 in model units against the stored (noisy) stack, as in training;'
          % (out['n_blocks'], out['model'], out['epoch']))
    print('in brackets against the noise-free stack. Each volume at shift 0 and at its best shift.')
    print()
    print('%-38s | %-16s | %24s | %24s' % ('loss', 'volume', 'at shift 0', 'best (shift)'))
    for loss in (OLD,) + NEW:
        for vol in ('truth', 'blurred volume', 'model output'):
            r = out['losses']['%s | %s' % (vol, loss)]
            print('%-38s | %-16s | %10.1f (%10.1f) | %10.1f (%10.1f) %+4d' % (
                loss if vol == 'truth' else '', vol, 1000 * r['at0'][0], 1000 * r['at0'][1],
                1000 * r['best'][0], 1000 * r['best'][1], r['best_shift'][0]))
    s = 1000 * np.array(out['small_crop_per_slice'])
    print()
    print('New loss (crop background) at the truth on 16 x 128 x 128 crops, against the noise-free stack, by slice:')
    print('  ' + ' '.join('%5.1f' % v for v in s))
    print('  mean with m border slices left out at each end: ' + ', '.join('m=%d: %.1f' % (m, s[m:16 - m].mean()) for m in range(5)))


def _init(threads):
    common.FFT_WORKERS = threads


def run(root, limit=None, workers=6):
    import multiprocessing as mp
    bl = ev.blocks(root, 'val')[:limit]
    with mp.get_context('spawn').Pool(workers, initializer=_init, initargs=(4,)) as pool:
        per_block = list(pool.imap(block_task, bl, chunksize=1))
    out = {}
    for which in ('truth', 'blurred'):
        out[which] = {}
        for n in per_block[0][1][which][0]:
            loss = sum(r[1][which][0][n] for r in per_block) / sum(r[1][which][1][n] for r in per_block)   # (shifts, 2)
            best = loss.argmin(0)
            out[which][n] = dict(at0=loss[SHIFTS == 0][0].tolist(), best=loss.min(0).tolist(),
                                 best_shift=SHIFTS[best].tolist(), curve=loss.tolist())
    out.update(shifts=SHIFTS.tolist(), n_blocks=len(bl), background=[r[2] for r in per_block])
    save_json(os.path.join(out_dir('eval'), 'loss_at_truth.json'), out)
    return out


def show(out):
    tr, bv = out['truth'], out['blurred']
    print('%d blocks. L1 x 1000 in model units. Columns: the truth as output, scored against the noise-free stack'
          % out['n_blocks'])
    print('(at shift 0; at the best shift), against the stored stack; the blurred volume as output.')
    print('Noise level (exact forward model, stored stack): %.2f with the floor, %.2f with the floor removed from the mapping.'
          % (1000 * tr['exact: output not limited at all']['at0'][0], 1000 * tr['exact, floor removed from the mapping']['at0'][0]))
    print('Background measured on the input stacks (median below the floor, window units): %.3f to %.3f, median %.3f; floor %.1f'
          % (np.min(out['background']), np.max(out['background']), np.median(out['background']), LO))
    print()
    print('%-46s | %8s %8s %6s | %8s | %8s %6s' % ('loss', 'truth', 'best', 'shift', 'stored', 'blurred', 'shift'))
    for n, r in tr.items():
        line = '%-46s | %8.2f %8.2f %+6d | %8.2f' % (n, 1000 * r['at0'][1], 1000 * r['best'][1], r['best_shift'][1], 1000 * r['at0'][0])
        if n in bv:
            line += ' | %8.2f %+6d' % (1000 * bv[n]['best'][1], bv[n]['best_shift'][1])
        print(line)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default=None, help='dataset folder (default: $DATASET/synthF)')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--model', type=str, default=None, help='a registry name or a run, as in evaluate.py: head-to-head mode')
    ap.add_argument('--epoch', type=int, default=None)
    a = ap.parse_args()
    root = a.root or os.path.join(ev.env_path('DATASET'), 'synthF')
    if a.model:
        show_model(run_model(root, a.model, a.epoch, a.limit, a.workers))
    else:
        show(run(root, a.limit, a.workers))
