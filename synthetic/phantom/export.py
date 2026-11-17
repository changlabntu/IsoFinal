"""Write one phantom volume as TIFFs for inspection (Fiji / napari).

    <tag>_truth_iso.tif      isotropic truth, uint16 raw counts (offset included)
    <tag>_degraded_z8.tif    the same volume as THX10 would record it: blurred,
                             every 8th plane, noise; uint16 raw counts, Z spacing 8
    <tag>_blurred_iso.tif    (--blurred) blurred but not yet sampled or noisy

Counts are in the units of the THX10 zarr, so the training window (99-673) is
a sensible display range. Slice i of the degraded stack sits at truth plane 8 i.

    python -m synthetic.phantom.export --size 512 768 768 --seed 3 --variant A
"""
import argparse
import os

import numpy as np

from synthetic.common import CONFIG, ZSTEP, load_json, out_dir, save_json

MARGIN_Z, MARGIN_XY = 96, 48


def export(size=(512, 768, 768), seed=0, variant='A', width_scale=None, blurred=False, folder=None, overrides=None):
    import tifffile
    from synthetic.phantom.degrade import acquire, apply_transfer, scale_transfer
    from synthetic.phantom.generate import make_block, sample_scene
    deg = load_json(os.path.join(CONFIG, 'degradation.json'))
    off = deg['noise']['offset']
    w = width_scale or (deg.get('width') or {}).get('nominal', 1.0)
    Z, Y, X = (int(q) for q in size)
    assert Z % ZSTEP == 0, 'Z must be a multiple of %d' % ZSTEP
    P = sample_scene(seed, variant, dict(empty_prob=0.0, **(overrides or {})))
    full, meta = make_block((Z + 2 * MARGIN_Z, Y + 2 * MARGIN_XY, X + 2 * MARGIN_XY), seed, P)
    blur = apply_transfer(full, scale_transfer(deg['transfer'], w), 0, 0)
    # one spare pixel in Y and X for the in-plane resampling
    low = np.clip(blur[MARGIN_Z:MARGIN_Z + Z:ZSTEP, MARGIN_XY:MARGIN_XY + Y + 1, MARGIN_XY:MARGIN_XY + X + 1], 0, None)
    raw = acquire(low, deg['noise'], np.random.default_rng([int(seed), 15485863]))

    def u16(v):
        return np.clip(np.rint(v + off), 0, 65535).astype(np.uint16)

    def crop(v):
        return v[MARGIN_Z:MARGIN_Z + Z, MARGIN_XY:MARGIN_XY + Y, MARGIN_XY:MARGIN_XY + X]
    folder = folder or out_dir('inspect')
    tag = os.path.join(folder, 'phantom_%s_seed%d' % (variant, seed))
    paths = dict(truth=tag + '_truth_iso.tif', degraded=tag + '_degraded_z8.tif')
    tifffile.imwrite(paths['truth'], u16(crop(full)), imagej=True, metadata=dict(axes='ZYX', spacing=1.0, unit='pixel'))
    tifffile.imwrite(paths['degraded'], np.clip(raw, 0, 65535).astype(np.uint16), imagej=True,
                     metadata=dict(axes='ZYX', spacing=float(ZSTEP), unit='pixel'))
    if blurred:
        paths['blurred'] = tag + '_blurred_iso.tif'
        tifffile.imwrite(paths['blurred'], u16(crop(blur)), imagej=True, metadata=dict(axes='ZYX', spacing=1.0, unit='pixel'))
    info = dict(size=[Z, Y, X], seed=seed, variant=variant, width_scale=w, offset=off, window=deg['window'],
                n_fibres=len(meta['centrelines']), n_bodies=len(meta['blobs']),
                scene={k: v for k, v in P.items() if not isinstance(v, (list, dict))}, files=paths)
    save_json(tag + '.json', info)
    return info


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, nargs=3, default=[512, 768, 768], metavar=('Z', 'Y', 'X'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variant', type=str, default='A')
    ap.add_argument('--width_scale', type=float, default=None, help='axial blur width scale (default: the nominal one)')
    ap.add_argument('--blurred', action='store_true', help='also write the blurred isotropic volume')
    ap.add_argument('--out', type=str, default=None)
    a = ap.parse_args()
    o = export(a.size, a.seed, a.variant, a.width_scale, a.blurred, a.out)
    print('%d fibres, %d bodies, blur width scale %.2f' % (o['n_fibres'], o['n_bodies'], o['width_scale']))
    for k, p in o['files'].items():
        print('%-9s %s' % (k, p))
