"""Does the existing THX10 model treat synthetic stacks like real ones?

Two checks with the trained `skipU_qd` checkpoint (nothing is trained here):

  codebook   encode real and synthetic stacks and compare which codebook
             entries are used at each scale (Jensen-Shannon divergence, with a
             real-vs-real split as the reference level);
  preview    run the full model on held-out synthetic crops and lay the input,
             the output and the truth side by side.

    python -m synthetic.modelcheck codebook --synth DIR
    python -m synthetic.modelcheck preview  --root $DATASET/synthA
"""
import argparse
import glob
import os
import sys

import numpy as np

from synthetic.common import CONFIG, REPO, ZSTEP, load_json, out_dir, save_json

CROP_XY, CROP_Z = 192, 24          # the crop the model was trained on


def load_engine(name='skipU_qd', train_mode=True):
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        from inference import Engine
        return Engine.from_registry(name, train_mode=train_mode)
    finally:
        os.chdir(cwd)


def model_input(eng, v):
    """(Z, Y, X) stack in [-1, 1] -> normalized (1, 1, Y, X, Z) tensor on the model's device."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(v.transpose(1, 2, 0)))[None, None].float()
    return eng.normalize(x).to(eng.device)


def _crops(v, n, rng):
    Z, Y, X = v.shape
    for _ in range(n):
        z, y, x = rng.integers(Z - CROP_Z + 1), rng.integers(Y - CROP_XY + 1), rng.integers(X - CROP_XY + 1)
        yield v[z:z + CROP_Z, y:y + CROP_XY, x:x + CROP_XY]


def index_histograms(eng, files, crops_per_stack=2, seed=0):
    """Per-scale histogram of codebook indices over crops of the given stacks."""
    import tifffile
    import torch
    rng = np.random.default_rng(seed)
    hists = None
    for f in files:
        v = tifffile.imread(f)
        for c in _crops(v, crops_per_stack, rng):
            with torch.no_grad():
                _, idx = eng.encode(model_input(eng, c))
            if hists is None:
                sizes = [int(q.n_embed) if hasattr(q, 'n_embed') else None for q in getattr(eng.gan, 'quantizers', [])]
                sizes = sizes if len(sizes) == len(idx) and all(sizes) else [int(i.max()) + 1 for i in idx]
                hists = [np.zeros(max(s, 1), np.int64) for s in sizes]
            for k, i in enumerate(idx):
                i = i.cpu().numpy().ravel()
                if i.max() >= len(hists[k]):
                    hists[k] = np.pad(hists[k], (0, int(i.max()) + 1 - len(hists[k])))
                hists[k] += np.bincount(i, minlength=len(hists[k]))
    return hists


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint)."""
    n = max(len(p), len(q))
    p = np.pad(p, (0, n - len(p))).astype(float)
    q = np.pad(q, (0, n - len(q))).astype(float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        ok = a > 0
        return float((a[ok] * np.log2(a[ok] / b[ok])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def perplexity(h):
    p = h[h > 0] / h.sum()
    return float(2 ** (-(p * np.log2(p)).sum()))


def codebook(synth_dir, n=40, tag='synth'):
    from synthetic import realism
    eng = load_engine()
    rf = realism.real_files(n=2 * n, seed=3)
    rng = np.random.default_rng(0)
    sf = sorted(glob.glob(os.path.join(synth_dir, '*.tif')))
    sf = [sf[i] for i in rng.choice(len(sf), min(n, len(sf)), replace=False)]
    ha, hb, hs = index_histograms(eng, rf[::2]), index_histograms(eng, rf[1::2]), index_histograms(eng, sf)
    out = dict(n_real=n, n_synth=len(sf), scales=[])
    for k in range(len(hs)):
        out['scales'].append(dict(
            scale=k, codebook_size=int(max(len(ha[k]), len(hs[k]))),
            perplexity_real=perplexity(ha[k]), perplexity_synth=perplexity(hs[k]),
            used_real=int((ha[k] > 0).sum()), used_synth=int((hs[k] > 0).sum()),
            js_real_vs_real=js_divergence(ha[k], hb[k]), js_real_vs_synth=js_divergence(ha[k], hs[k])))
    save_json(os.path.join(out_dir('realism'), 'codebook_%s.json' % tag), out)
    return out


def preview(root, n=4, tag='synth', split='val', n_real=2):
    """Input / model output / truth for central crops of held-out synthetic
    blocks (brightest first), followed by real crops (input / output only)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import tifffile
    import torch
    from synthetic import realism
    from synthetic.phantom.degrade import truth_to_window
    deg = load_json(os.path.join(CONFIG, 'degradation.json'))
    eng = load_engine()
    floor = eng.spec['gamma_lo']
    z0, y0 = (48 - CROP_Z) // 2, (384 - CROP_XY) // 2

    def central(v):
        return v[z0:z0 + CROP_Z, y0:y0 + CROP_XY, y0:y0 + CROP_XY]

    def run(v):
        with torch.no_grad():
            out = eng.denormalize(eng.full(model_input(eng, v)))[0, 0].float().cpu().numpy()     # (Y, X, 8 Z)
        return out.transpose(2, 0, 1)                                                             # (8 Z, Y, X)

    def brightest(files, k):
        vols = [(f, central(tifffile.imread(f))) for f in files]
        vols.sort(key=lambda q: -float((q[1] > floor).mean()))
        return vols[:k]
    synth = brightest(sorted(glob.glob(os.path.join(root, split, 'deg', '*.tif'))), n)
    real = brightest(realism.real_files(n=24, seed=5), n_real)
    rows = len(synth) + len(real)
    fig, ax = plt.subplots(rows, 6, figsize=(30, 5 * rows), squeeze=False)
    kw = dict(cmap='gray', vmin=floor, vmax=0.6, interpolation='nearest')
    zc, yc = CROP_Z * ZSTEP // 2, CROP_XY // 2
    for r, (f, v) in enumerate(synth + real):
        out = run(v)
        inp = np.maximum(np.repeat(v, ZSTEP, axis=0), floor)
        panels = [('input XZ (slices repeated)', inp[:, yc]), ('model output XZ', out[:, yc]), None,
                  ('input XY', inp[zc]), ('model output XY', out[zc]), None]
        if r < len(synth):
            t = tifffile.imread(os.path.join(root, 'gt', 'val', os.path.basename(f))).astype(np.float32)
            t = np.maximum(truth_to_window(t[z0 * ZSTEP:(z0 + CROP_Z) * ZSTEP, y0:y0 + CROP_XY, y0:y0 + CROP_XY], deg), floor)
            panels[2], panels[5] = ('truth XZ', t[:, yc]), ('truth XY', t[zc])
        for c, pn in enumerate(panels):
            ax[r, c].axis('off')
            if pn is not None:
                ax[r, c].imshow(pn[1], **kw)
                ax[r, c].set_title(('synthetic: ' if r < len(synth) else 'REAL: ') + pn[0], fontsize=16)
    fig.tight_layout(pad=0.3)
    p = os.path.join(out_dir('realism'), 'model_preview_%s.png' % tag)
    fig.savefig(p, dpi=45); plt.close(fig)
    return p


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['codebook', 'preview'])
    ap.add_argument('--synth', type=str, default=None)
    ap.add_argument('--root', type=str, default=None)
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--tag', type=str, default='synth')
    a = ap.parse_args()
    if a.cmd == 'codebook':
        o = codebook(a.synth, a.n, a.tag)
        for s in o['scales']:
            print('scale %d (%d codes): perplexity real %.0f synth %.0f | used real %d synth %d | JS real-real %.3f real-synth %.3f' % (
                s['scale'], s['codebook_size'], s['perplexity_real'], s['perplexity_synth'], s['used_real'], s['used_synth'],
                s['js_real_vs_real'], s['js_real_vs_synth']))
    else:
        print(preview(a.root, min(a.n, 4), a.tag))
