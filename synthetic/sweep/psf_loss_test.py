"""Test of models/MScleanQDPSF.py against the check in loss_at_truth.py.

    python -m synthetic.sweep.psf_loss_test [--model MScleanQD/base_r1] [--epoch 100] [--blocks 6]

1. psf_slices (torch, as used in training) against the same steps in numpy
   with the dataset's own blur code: largest difference.
2. Gradients through psf_slices are finite, also for an output at the floor.
3. The training code path (GAN.get_projection + GAN.add_loss_l1 of a model
   built from models/MScleanQDPSF.py with the weights of a trained baseline) on
   training-size crops (16 x 128 x 128) for three volumes: the truth, the
   blurred volume and the baseline's own output; old loss (--l1how max
   --skipl1 4) against new (--l1how psf --skipl1 1). The truth and the blurred
   volume are placed s planes deeper, since the new loss puts slice i at plane
   8 i + 3.5 and the old one is lowest near 8 i + 15.
"""
import argparse
import os

import numpy as np

from synthetic import evaluate as ev
from synthetic.common import REPO, ZSTEP
from synthetic.sweep import loss_at_truth as lt

SHIFTS = (0, 3, 4, 15)
CZ, CXY = 16, 128


def to_t(vol, device):
    """(Z, Y, X) numpy -> (1, 1, Y, X, Z) tensor."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(vol.transpose(1, 2, 0)))[None, None].float().to(device)


def reference(v_lin, deg, bg, margin_pad=(32, 16)):
    """The steps of psf_slices in numpy: v_lin (Z, Y, X) in window units, limited to [LO, 1]."""
    from synthetic.phantom.degrade import apply_transfer, scale_transfer
    b = apply_transfer(lt.background(v_lin, bg, 'affine'), scale_transfer(deg['transfer'], deg['width']['nominal']), *margin_pad)
    return lt.norm(0.5 * (b[ZSTEP // 2 - 1::ZSTEP] + b[ZSTEP // 2::ZSTEP]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='MScleanQD/base_r1')
    ap.add_argument('--epoch', type=int, default=100)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--root', type=str, default=None)
    a = ap.parse_args()
    import torch
    from inference.load import load_model
    from models import MScleanQDPSF as M

    root = a.root or os.path.join(ev.env_path('DATASET'), 'synthF')
    deg = ev.degradation()
    transfer = (deg['transfer']['a'], deg['transfer']['b'], deg['transfer']['p'], deg['width']['nominal'])
    dev = 'cuda'
    gan, _ = load_model(ev.run_dir(a.model), a.epoch, dev, model_file=os.path.join(REPO, 'models', 'MScleanQDPSF.py'))
    fn = ev.Model(a.model, a.epoch)
    h = gan.hparams
    h.psf_a, h.psf_b, h.psf_p, h.psf_width = transfer
    h.psf_bg, h.psf_margin = lt.BG_FIXED, 2
    assert (h.gamma, h.gamma_lo) == (lt.GAMMA, lt.LO)

    # 2. gradients
    for name, x in (('random output', torch.rand(1, 1, 32, 32, 64, device=dev) * 2 - 1),
                    ('output at the floor', -torch.ones(1, 1, 32, 32, 64, device=dev))):
        x.requires_grad_(True)
        y = M.psf_slices(x, 8, 1, transfer, h.gamma, h.gamma_lo, h.psf_bg)
        (y - torch.rand_like(y)).abs().mean().backward()
        print('gradient, %-20s finite: %s, largest |g| x N: %.2f, share of voxels with a gradient: %.2f'
              % (name + ':', bool(torch.isfinite(x.grad).all()), float(x.grad.abs().max() * y.numel()), float((x.grad != 0).float().mean())))

    worst, rows = 0.0, {}
    torch.manual_seed(0)
    for b in ev.blocks(root, 'val')[:a.blocks]:
        x, t, nf, _ = lt.block_volumes(b)
        t = np.clip(t, lt.LO, 1)
        bv = np.clip(ev.cubic(nf), lt.LO, 1)
        for z0, y0, x0 in lt.SMALL:
            cy, cx = slice(y0, y0 + CXY), slice(x0, x0 + CXY)
            xc = x[z0:z0 + CZ, cy, cx]
            gan.oriX = to_t(lt.norm(xc), dev)
            vols = {'baseline output': np.clip(fn(xc), lt.LO, 1)}
            for s in SHIFTS:
                zz = slice(ZSTEP * z0 - s, ZSTEP * (z0 + CZ) - s)
                vols['truth, %+d planes' % s] = t[zz, cy, cx]
                vols['blurred volume, %+d planes' % s] = bv[zz, cy, cx]
            for name, v in vols.items():
                X = to_t(lt.norm(v), dev)
                with torch.no_grad():
                    h.skipl1 = 4
                    old = gan.add_loss_l1(gan.get_projection(X, depth=gan.uprate * 4, how='max'), gan.oriX[..., ::4])
                    assert torch.equal(gan.get_projection(X, 32, 'max'), type(gan).__mro__[1].get_projection(gan, X, 32, 'max'))   # other --l1how: MScleanQD
                    h.skipl1 = 1
                    new = gan.add_loss_l1(gan.get_projection(X, depth=gan.uprate, how='psf'), gan.oriX)
                    # 1. the same prediction and the same mean over the used slices, in numpy
                    pred = M.psf_slices(X, 8, 1, transfer, h.gamma, h.gamma_lo, h.psf_bg)[0, 0].permute(2, 0, 1).cpu().numpy()
                ref = reference(v, deg, lt.BG_FIXED)
                worst = max(worst, float(np.abs(pred - ref).max()),
                            abs(float(new) - float(np.abs(ref - lt.norm(xc))[2:-2].mean())))
                r = rows.setdefault(name, [0.0, 0.0, 0])
                r[0] += float(old); r[1] += float(new); r[2] += 1
    print('largest difference between the torch loss and the numpy reference: %.2g (model units)' % worst)
    print()
    print('%-28s %10s %10s      (L1 x 1000, %d crops of %d x %d x %d)' % ('volume as output', 'old loss', 'new loss', r[2], CZ, CXY, CXY))
    for name, (o, n, k) in rows.items():
        print('%-28s %10.1f %10.1f' % (name, 1000 * o / k, 1000 * n / k))
