"""Pick the THX10 blocks the estimators run on.

Blocks are scored on pyramid level 3 (8x8 in-plane block mean) by how much
bright fibre signal they hold, then the best ones are taken per Z band so the
estimate can be compared across depth.

    python -m synthetic.estimate.blocks --per_band 6
"""
import argparse
import os

import numpy as np

from synthetic.common import OUT, load_json, open_level, out_dir, save_json

BLOCKS_JSON = os.path.join(OUT, 'blocks.json')


def select_blocks(size_z=192, size_xy=768, per_band=6, fibre_counts=250.0, seed=0):
    """Return a list of dicts {z0,z1,y0,y1,x0,x1,score,band} at level-0 coordinates."""
    a3 = open_level(3)
    Z, X3, Y3 = a3.shape
    s3 = size_xy // 8
    n_band = Z // size_z
    blocks = []
    for b in range(n_band):
        z0 = b * size_z
        # every 8th slice of the band is plenty for a score
        sl = a3[z0:z0 + size_z:8].read().result().astype(np.float32)   # (z, x, y)
        cand = []
        for xi in range(X3 // s3):
            for yi in range(Y3 // s3):
                c = sl[:, xi * s3:(xi + 1) * s3, yi * s3:(yi + 1) * s3]
                tissue = float((c > 125).mean())
                fibre = float((c > fibre_counts).mean())
                cand.append((fibre, tissue, xi, yi))
        cand = [c for c in cand if c[1] > 0.9]          # fully inside tissue
        cand.sort(reverse=True)
        for fibre, tissue, xi, yi in cand[:per_band]:
            blocks.append(dict(band=b, z0=z0, z1=z0 + size_z,
                               x0=xi * size_xy, x1=(xi + 1) * size_xy,
                               y0=yi * size_xy, y1=(yi + 1) * size_xy,
                               score=fibre, tissue=tissue))
    return blocks


def get_blocks(refresh=False, **kw):
    blocks = None if refresh else load_json(BLOCKS_JSON)
    if blocks is None:
        blocks = select_blocks(**kw)
        out_dir()
        save_json(BLOCKS_JSON, blocks)
    return blocks


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--per_band', type=int, default=6)
    p.add_argument('--size_z', type=int, default=192)
    p.add_argument('--size_xy', type=int, default=768)
    a = p.parse_args()
    bl = get_blocks(refresh=True, per_band=a.per_band, size_z=a.size_z, size_xy=a.size_xy)
    for b in bl:
        print(b)
    print('%d blocks -> %s' % (len(bl), BLOCKS_JSON))
