"""Recover the intensity window that produced the THX10 training TIFFs.

The TIFFs are float32 in [-1, 1] with no record of the raw-count window. One
TIFF is located inside the zarr by template matching on pyramid level 3, the
match is refined at full resolution, and the affine map raw -> TIFF value is
fitted on the unclipped voxels: value = (raw - lo) / (hi - lo) * 2 - 1.

    python -m synthetic.estimate.window
"""
import glob
import os

import numpy as np
import tifffile

from synthetic.common import THX10_TIFS, open_level, out_dir, read_block, save_json


def _ncc(img, tpl):
    """Normalised cross-correlation map, safe on flat regions (score 0 there)."""
    from scipy.signal import fftconvolve
    tpl = tpl - tpl.mean()
    n = tpl.size
    num = fftconvolve(img, tpl[::-1, ::-1], mode='valid')
    ones = np.ones_like(tpl)
    s1 = fftconvolve(img, ones, mode='valid')
    s2 = fftconvolve(img * img, ones, mode='valid')
    var = np.clip(s2 - s1 * s1 / n, 0, None)
    den = np.sqrt(var * float((tpl * tpl).sum()))
    ok = var > 1e-3 * n * max(float(img.var()), 1e-6)       # ignore near-flat windows
    return np.where(ok, num / np.maximum(den, 1e-12), 0.0)


def locate(tif_path, z_stride=1):
    """Best (score, axis order, z, y, x) of the TIFF's middle slice on level 3."""
    t = tifffile.imread(tif_path).astype(np.float32)           # (Z, A, B)
    a3 = open_level(3)
    vol = a3[:].read().result()                                # (z, x, y)
    mid = t.shape[0] // 2
    best = None
    for name, plane in (('zyx', t[mid]), ('zxy', t[mid].T)):   # plane as (y, x)
        small = plane.reshape(48, 8, 48, 8).mean((1, 3)).astype(np.float64)
        for z in range(0, vol.shape[0], z_stride):
            img = vol[z].T.astype(np.float64)                  # (y, x)
            if img.max() <= 0:
                continue
            r = _ncc(img, small)
            i = np.unravel_index(np.argmax(r), r.shape)
            if best is None or r[i] > best[0]:
                best = (float(r[i]), name, z, int(i[0]) * 8, int(i[1]) * 8)
    return t, best


def refine_and_fit(t, best):
    score, order, z, y, x = best
    arr = open_level(0)
    tt = t if order == 'zyx' else t.transpose(0, 2, 1)         # -> (Z, Y, X)
    mid = tt.shape[0] // 2
    m = 24
    raw = read_block(z, z + 1, max(y - m, 0), y + 384 + m, max(x - m, 0), x + 384 + m, arr=arr)[0]
    r = _ncc(raw.astype(np.float64), tt[mid].astype(np.float64))
    i = np.unravel_index(np.argmax(r), r.shape)
    y0, x0 = max(y - m, 0) + int(i[0]), max(x - m, 0) + int(i[1])
    z0 = z - mid
    blk = read_block(z0, z0 + tt.shape[0], y0, y0 + 384, x0, x0 + 384, arr=arr)
    ok = (tt > -0.999) & (tt < 0.999)
    A = np.stack([blk[ok], np.ones(ok.sum())], 1)
    (s, c), res, _, _ = np.linalg.lstsq(A, tt[ok].astype(np.float64), rcond=None)
    lo = (-1 - c) / s
    hi = (1 - c) / s
    resid = float(np.sqrt(np.mean((A @ [s, c] - tt[ok]) ** 2)))
    return dict(order=order, z0=int(z0), y0=int(y0), x0=int(x0), coarse_score=score, fine_score=float(r[i]),
                lo=float(lo), hi=float(hi), fit_rms=resid, frac_unclipped=float(ok.mean()))


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(THX10_TIFS, 'val', 'roiD', '*.tif')))
    out = []
    for f in files[:1] + files[len(files) // 2:len(files) // 2 + 1]:
        t, best = locate(f)
        print(os.path.basename(f), 'coarse match', best, flush=True)
        r = refine_and_fit(t, best)
        r['file'] = os.path.basename(f)
        print(r, flush=True)
        out.append(r)
    save_json(os.path.join(out_dir('estimate'), 'window_thx10.json'), out)
