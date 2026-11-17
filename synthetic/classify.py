"""Real-vs-synthetic classifier: can a small CNN tell degraded phantom stacks
from THX10 training stacks?

A CNN is trained on 2D patches from one set of stacks and tested on patches
from other stacks (the split is by stack, never by patch). AUC 0.5 means the
two sources cannot be told apart; 1.0 means every patch can.

Views:
    xy      64 x 64 patches of XY slices at full resolution
    xy/2    the same after 2x2 averaging (pixel-level noise cues removed)
    xy/4    after 4x4 averaging (only structure at >= 8 px remains)
    xz      48 x 64 patches of XZ sections (48 slices x 64 px)

With --model_domain the patches first go through the baseline's input mapping
(`--nm 11g --gamma 0.7 --gamma_lo -0.8`, which flattens everything below 156
counts), so the test is on what the model is actually fed.

    python -m synthetic.realism classify --synth DIR
"""
import glob
import os

import numpy as np

from synthetic.common import out_dir, save_json

VIEWS = ('xy', 'xy/2', 'xy/4', 'xz')


MODEL_DOMAIN = None      # set to (gamma, gamma_lo) to classify what the model is fed


def _patches(vols, view, n, rng):
    """n random patches (float32, [-1, 1]) drawn evenly from a list of stacks.
    In the model domain the baseline's input mapping is applied first (values
    below gamma_lo are flattened) and patches with nothing above that floor
    are skipped, since they are identical for both sources."""
    out = []
    per = int(np.ceil(n / len(vols)))
    for v in vols:
        Z, Y, X = v.shape
        got = tries = 0
        while got < per and tries < 20 * per:
            tries += 1
            if view == 'xz':
                y, x = rng.integers(Y), rng.integers(X - 64)
                p = v[:, y, x:x + 64]
            else:
                f = {'xy': 1, 'xy/2': 2, 'xy/4': 4}[view]
                s = 64 * f
                z, y, x = rng.integers(Z), rng.integers(Y - s + 1), rng.integers(X - s + 1)
                p = v[z, y:y + s, x:x + s]
            if MODEL_DOMAIN is not None:
                g, lo = MODEL_DOMAIN
                if (p > lo).mean() < 0.05:
                    continue
                p = np.clip((p - lo) / (1 - lo), 0, 1) ** g * 2 - 1
                p = (p + 1) * 0.5 * 0.3 - 0.85          # back to the range _scale() expects
            if view != 'xz' and p.shape[0] > 64:
                f = p.shape[0] // 64
                p = p.reshape(64, f, 64, f).mean((1, 3))
            if rng.random() < 0.5:
                p = p[:, ::-1]
            if view != 'xz' and rng.random() < 0.5:
                p = p[::-1]
            out.append(np.ascontiguousarray(p, dtype=np.float32))
            got += 1
    idx = rng.permutation(len(out))[:n] if len(out) >= n else rng.integers(0, max(len(out), 1), n)
    return np.stack([out[i] for i in idx])[:, None]


def _net():
    import torch.nn as nn
    layers, c_in = [], 1
    for c in (32, 64, 128, 128):
        layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.GroupNorm(8, c), nn.ReLU(),
                   nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.ReLU(), nn.AvgPool2d(2)]
        c_in = c
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c_in, 1))


def _scale(x):
    """[-1, 1] window units -> roughly unit scale (most voxels sit near -0.87)."""
    return (x + 0.85) * 5.0


def auc(score, label):
    order = np.argsort(score)
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    n1 = label.sum()
    n0 = len(label) - n1
    return float((rank[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_view(real_tr, synth_tr, real_te, synth_te, view, steps=2000, batch=128, seed=0, device=None):
    """Returns (test AUC, AUC on fresh patches of the training stacks). The
    second number shows whether the network learned anything at all."""
    import torch
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = _net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for step in range(steps):
        if step % 100 == 0:            # fresh patches every 100 steps
            xr = _patches(real_tr, view, batch * 50, rng)
            xs = _patches(synth_tr, view, batch * 50, rng)
        i = rng.integers(0, len(xr), batch // 2)
        j = rng.integers(0, len(xs), batch // 2)
        x = torch.from_numpy(_scale(np.concatenate([xr[i], xs[j]]))).to(device)
        y = torch.cat([torch.zeros(batch // 2), torch.ones(batch // 2)]).to(device)
        opt.zero_grad()
        loss = loss_fn(net(x)[:, 0], y)
        loss.backward()
        opt.step()
    net.eval()

    def score(a, b, n=4000):
        xt = _scale(np.concatenate([_patches(a, view, n, rng), _patches(b, view, n, rng)]))
        with torch.no_grad():
            s_ = torch.cat([net(torch.from_numpy(xt[k:k + 500]).to(device))[:, 0] for k in range(0, len(xt), 500)]).cpu().numpy()
        return auc(s_, np.concatenate([np.zeros(n), np.ones(n)]))
    return score(real_te, synth_te), score(real_tr, synth_tr)


def run(synth_dir, n=60, tag='synth', real_list=None, steps=2000, model_domain=None):
    """AUC per view. `n` stacks from each source, 70 % train / 30 % test."""
    import tifffile
    from synthetic import realism
    rng = np.random.default_rng(0)
    sf = sorted(glob.glob(os.path.join(synth_dir, '*.tif')))
    sf = [sf[i] for i in rng.choice(len(sf), min(n, len(sf)), replace=False)]
    rf = real_list or realism.real_files(n=min(n, len(sf)), seed=1)
    real = [tifffile.imread(f) for f in rf]
    synth = [tifffile.imread(f) for f in sf]
    kr, ks = int(0.7 * len(real)), int(0.7 * len(synth))
    out = dict(n_real=len(real), n_synth=len(synth), model_domain=model_domain)
    global MODEL_DOMAIN
    MODEL_DOMAIN = model_domain
    for view in VIEWS:
        a, tr = train_view(real[:kr], synth[:ks], real[kr:], synth[ks:], view, steps=steps)
        out[view], out[view + ' train'] = a, tr
        print('  %-5s AUC %.3f on held-out stacks (%.3f on the training stacks)' % (view, a, tr), flush=True)
    MODEL_DOMAIN = None
    save_json(os.path.join(out_dir('realism'), 'classifier_%s%s.json' % (tag, '_model' if model_domain else '')), out)
    return out


def control(n=60, steps=2000):
    """Real vs real: the AUC the classifier reports when there is nothing to find."""
    import tifffile
    from synthetic import realism
    files = realism.real_files(n=2 * n, seed=2)
    a = [tifffile.imread(f) for f in files[::2]]
    b = [tifffile.imread(f) for f in files[1::2]]
    k = int(0.7 * len(a))
    out = {}
    for view in VIEWS:
        out[view], out[view + ' train'] = train_view(a[:k], b[:k], a[k:], b[k:], view, steps=steps)
        print('  control %-5s AUC %.3f on held-out stacks (%.3f on the training stacks)' % (view, out[view], out[view + ' train']), flush=True)
    save_json(os.path.join(out_dir('realism'), 'classifier_control.json'), out)
    return out
