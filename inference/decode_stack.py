"""Decode a codec grid back to volumes — the inverse of encode_stack.py.

Two outputs per codec cell, selected by --what:

  decode          the isotropic super-resolved volume (Z x uprate): the
                  enhancement; TIFFs are ZX pages (synthesized Z in-plane)
  reconstruction  the slice-wise VQ-head reconstruction at input resolution;
                  TIFFs are XY pages, overlayable on the source slices

Both TIFF and zarr outputs are uint16 restored to SOURCE units: model output
-> denormalize (gamma inversion) -> the chunk's stored window_lo/hi -> round,
clip, uint16 — directly comparable to the raw store.

Zarr store (the full zarr -> npz -> zarr round trip; needs the py38zarr env
= tensorstore):

    python inference/decode_stack.py --codec {out_base}/{exp}/codec \
        --zarr /path/to/{exp}_decode_ome.zarr --what decode [--batch 2 --half]

    pre-allocates ONE native (z, x, y) uint16 store covering every z-chunk
    dir of the codec tree ((Z*uprate, W, H) for decode, (Z, W, H) for
    reconstruction), writes each decoded patch as an aligned slab, and adds
    an OME-NGFF 0.5 group zarr.json + a decode_params.json. Reruns open the
    same store and fill more of it (--overwrite recreates from scratch).
    Stores are sparse: an ROI (--cells / --chunk) only writes its chunks. For
    VIEWING an ROI add --crop: the store is sized to the ROI bounding box
    (origin at the ROI corner), fully populated — small and napari-friendly,
    vs the full-volume shape (334 GB logical for sample0) that is single-scale
    and mostly empty. --orig writes the matched source into a sibling
    '<zarr>_original' store of the same shape (trilinear Z-upsampled for
    decode), so an ROI enhancement and its interpolated original overlay 1:1.

TIFF comparison subset (per-patch files, plus a stitched strip for
--row/--col). Model output lands in {tiff}/{what}/, originals (with --orig) in
{tiff}/original/, paired files sharing a name across the two folders:

    python inference/decode_stack.py --codec {out_base}/{exp}/codec \
        --tiff /path/to/tifs --what decode --chunk z0000-0032 --row 17

Cell selection (--cells R0 R1 C0 C1, half-open / --row R / --col C) and
--chunk apply to both sinks; the default is every cell of every chunk dir.
--codec may also point at a single z chunk dir. Model, epoch and intensity
window come from each chunk's norm_params.json; --model/--epoch override.
Decoding is DETERMINISTIC by default (running-stat BN, dropout off); --mc
instead runs the generator components in .train() — batch-stat BN + live
dropout, one MC draw per patch (the training-time test convention).
"""

import argparse
import json
import os
import re
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import tifffile as tiff  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from inference import Engine, available  # noqa: E402
from inference.inference_latent import to_zx_pages  # noqa: E402

CHUNK_RE = re.compile(r'^z(\d{4,})-(\d{4,})$')


def to_xy_pages(vol_yxz):
    """(Y, X, Z) array -> XY tif pages (Z, Y, X): page z=k, rows Y, cols X."""
    return np.ascontiguousarray(np.transpose(vol_yxz, (2, 0, 1)))


def chunk_z_range(name):
    m = CHUNK_RE.match(name)
    if not m:
        raise ValueError(f"'{name}' is not a z chunk dir name (zBBBB-EEEE)")
    return int(m.group(1)), int(m.group(2))


def resolve_chunks(codec, chunk_sel):
    """--codec (a codec/ tree or one chunk dir) -> (root, selected, all_names)."""
    codec = os.path.abspath(codec.rstrip('/'))
    if os.path.isfile(os.path.join(codec, 'norm_params.json')):
        root, names = os.path.dirname(codec), [os.path.basename(codec)]
    else:
        root = codec
        names = sorted(d for d in os.listdir(codec)
                       if CHUNK_RE.match(d)
                       and os.path.isfile(os.path.join(codec, d, 'norm_params.json')))
        if not names:
            raise FileNotFoundError(f'no z chunk dirs with norm_params.json under {codec}')
    all_names = sorted(d for d in os.listdir(root)
                       if CHUNK_RE.match(d)
                       and os.path.isfile(os.path.join(root, d, 'norm_params.json')))
    if chunk_sel is not None:
        names = [n for n in names if n == chunk_sel]
        if not names:
            raise ValueError(f"--chunk {chunk_sel} not found (have: {', '.join(all_names)})")
    return root, names, all_names


def select_cells(args, rows, cols):
    if args.cells is not None:
        r0, r1, c0, c1 = args.cells
        if not (0 <= r0 < r1 <= rows and 0 <= c0 < c1 <= cols):
            raise ValueError(f'--cells {args.cells} outside grid {rows} x {cols}')
        return [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
    if args.row is not None:
        if not 0 <= args.row < rows:
            raise ValueError(f'--row {args.row} outside grid of {rows} rows')
        return [(args.row, c) for c in range(cols)]
    if args.col is not None:
        if not 0 <= args.col < cols:
            raise ValueError(f'--col {args.col} outside grid of {cols} cols')
        return [(r, args.col) for r in range(rows)]
    return [(r, c) for r in range(rows) for c in range(cols)]


def infer_uprate(norm_params, eng):
    """Z upsample factor: recorded in norm_params (new codecs) or derived from
    the run's config as (cropsize // cropz) * dsp / usp."""
    up = norm_params.get('uprate')
    if up is None:
        cfg = eng.cfg
        up = (cfg.cropsize // cfg.cropz) * cfg.dsp / cfg.usp
    return int(round(up))


def load_indices(chunk_dir, cells):
    """Stack the cells' stored per-scale indices -> list of (B, Z, hk, wk) int32."""
    per_scale = None
    for r, c in cells:
        path = os.path.join(chunk_dir, f'{r:03d}{c:03d}.npz')
        with np.load(path) as d:
            keys = sorted((k for k in d.files if k.startswith('scale_')),
                          key=lambda k: int(k.split('_')[1]))
            arrs = [d[k] for k in keys]
        if per_scale is None:
            per_scale = [[] for _ in arrs]
        elif len(arrs) != len(per_scale):
            raise ValueError(f'{path}: {len(arrs)} scales, expected {len(per_scale)}')
        for lst, a in zip(per_scale, arrs):
            lst.append(a)
    return [np.concatenate(lst) for lst in per_scale]


def restore_uint16(eng, out_b, lo, hi):
    """One (Y, X, Z') model-output patch -> uint16 in source units."""
    w = eng.denormalize(out_b.float().cpu().numpy())        # pre-gamma [-1, 1]
    raw = (w + 1.0) * 0.5 * (hi - lo) + lo
    return np.clip(np.rint(raw), 0, 65535).astype(np.uint16)


def main():
    parser = argparse.ArgumentParser(
        description='Decode stored codec grids to uint16 TIFFs and/or a zarr store')
    parser.add_argument('--codec', required=True,
                        help='A codec/ tree of z chunk dirs, or one chunk dir')
    parser.add_argument('--what', choices=('decode', 'reconstruction'), default='decode',
                        help='decode = isotropic SR (Z x uprate); reconstruction = '
                             'VQ-head output at input resolution (default: decode)')
    parser.add_argument('--zarr', default=None, metavar='STORE',
                        help='Write a native (z, x, y) uint16 OME-Zarr store at this path')
    parser.add_argument('--tiff', default=None, metavar='DIR',
                        help='Write per-patch uint16 TIFFs (and a strip for --row/--col) here')
    parser.add_argument('--chunk', default=None, metavar='zBBBB-EEEE',
                        help='Restrict to one z chunk dir (default: all found)')
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument('--cells', type=int, nargs=4, default=None, metavar=('R0', 'R1', 'C0', 'C1'),
                     help='Grid block [R0:R1) x [C0:C1) (default: every cell)')
    sel.add_argument('--row', type=int, default=None, help='One grid row, all cols')
    sel.add_argument('--col', type=int, default=None, help='One grid col, all rows')
    parser.add_argument('--model', default=None,
                        help='Registry model name; default: the codec norm_params.json '
                             f'({", ".join(available())})')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Override the epoch (default: norm_params.json / registry)')
    parser.add_argument('--device', default=None, help='cuda / cpu (default: cuda if available)')
    parser.add_argument('--half', action='store_true', help='Run the model in fp16')
    parser.add_argument('--mc', action='store_true',
                        help='One MC draw per patch (batch-stat BN + dropout) instead of '
                             'the default deterministic running-stat inference')
    parser.add_argument('--batch', type=int, default=2,
                        help='Patches per decode forward (default 2; decode is ~64 MB '
                             'output per patch plus generator activations)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Recreate the zarr store instead of filling an existing one')
    parser.add_argument('--crop', action='store_true',
                        help='Zarr: size the store to the ROI bounding box (origin at the ROI '
                             'corner), fully populated, instead of the full-volume shape written '
                             'sparsely — a small, napari-friendly store for viewing an ROI. The '
                             'ROI offset is recorded in decode_params.json. Default (no --crop) '
                             'keeps the full-volume placement for a whole-volume decode.')
    parser.add_argument('--orig', action='store_true',
                        help="Also emit the source patch(es) from the codec's recorded source "
                             'store for side-by-side comparison, matched to --what: '
                             'reconstruction -> raw input-res; decode -> TRILINEAR Z-upsampled '
                             'to the SR size (blocky-Z vs the smooth enhancement). Goes to the '
                             "TIFF original/ subfolder and/or a sibling '<zarr>_original' store "
                             'at the same shape. Zarr source only.')
    args = parser.parse_args()
    if args.zarr is None and args.tiff is None:
        parser.error('pick at least one output: --zarr and/or --tiff')

    root, names, all_names = resolve_chunks(args.codec, args.chunk)
    with open(os.path.join(root, names[0], 'norm_params.json')) as f:
        np0 = json.load(f)
    model = args.model or np0['model']
    epoch = args.epoch if args.epoch is not None else (
        np0.get('epoch') if args.model is None else None)

    eng = Engine.from_registry(model, epoch=epoch, device=args.device, train_mode=args.mc)
    if args.half:
        eng.gan.half()
    if args.model is None:
        for field in ('nm', 'gamma', 'gamma_lo'):
            if field in np0 and np0[field] != eng.spec[field]:
                print(f'WARNING: codec was encoded with {field}={np0[field]} but the '
                      f'engine spec has {field}={eng.spec[field]}')

    patch = np0['patch']
    H, W = np0['slice_hw']
    rows, cols = np0['grid_rows_cols']
    up = infer_uprate(np0, eng) if args.what == 'decode' else 1
    cells = select_cells(args, rows, cols)
    z_full = max(chunk_z_range(n)[1] for n in all_names)
    print(f'{args.what} of {root}: grid {rows} x {cols}, patch {patch}, '
          f'z extent {z_full} (x{up}), chunks {", ".join(names)}, '
          f'{len(cells)} cell(s) each  (model {model} epoch {epoch}, '
          f'{"mc" if args.mc else "eval"}, device {eng.device})')

    # Store geometry. Default: full-volume shape, patches placed at absolute
    # coords (offset 0), written sparsely. --crop: shape = ROI bounding box in
    # (z, x=cols, y=rows), patches placed relative to the ROI corner (off_*).
    if args.crop:
        r_idx = [r for r, _ in cells]
        c_idx = [c for _, c in cells]
        za_min = min(chunk_z_range(n)[0] for n in names)
        zb_max = max(chunk_z_range(n)[1] for n in names)
        off_z, off_x, off_y = za_min * up, min(c_idx) * patch, min(r_idx) * patch
        shape = ((zb_max - za_min) * up,
                 (max(c_idx) - min(c_idx) + 1) * patch,
                 (max(r_idx) - min(r_idx) + 1) * patch)
    else:
        off_z = off_x = off_y = 0
        shape = (z_full * up, W, H)
    # z chunk = the largest chunk-dir extent (not names[0]'s, which may be the
    # short final chunk) so the store's chunking is the same no matter which
    # chunk dir a run decodes first; short chunks write partial, aligned slabs
    zc_max = max(zb - za for za, zb in (chunk_z_range(n) for n in all_names))
    store_chunks = (min(zc_max * up, shape[0]), patch, patch)

    def make_store(path, what):
        from inference.zarr_io import create_zarr, write_ome_group
        a = create_zarr(os.path.join(path, '0'), shape, store_chunks, overwrite=args.overwrite)
        write_ome_group(path, name=os.path.basename(path.rstrip('/')))
        with open(os.path.join(path, 'decode_params.json'), 'w') as f:
            json.dump(dict(what=what, codec=root, chunks=names, model=model,
                           epoch=epoch, mc=args.mc, half=args.half, uprate=up,
                           patch=patch, shape_zxy=list(shape), store_chunks=list(store_chunks),
                           crop=args.crop, roi_offset_zxy=[off_z, off_x, off_y],
                           window_lo=np0['window_lo'], window_hi=np0['window_hi']), f, indent=2)
        print(f'zarr store {path}: shape (z, x, y) = {shape}, chunks {store_chunks}, uint16'
              + (f', ROI offset (z,x,y) = ({off_z}, {off_x}, {off_y})' if args.crop else ''))
        return a

    arr = make_store(args.zarr, args.what) if args.zarr else None

    out_dir = orig_dir = None
    if args.tiff:
        out_dir = os.path.join(args.tiff, args.what)      # decode/ or reconstruction/
        orig_dir = os.path.join(args.tiff, 'original')
        os.makedirs(out_dir, exist_ok=True)

    src_arr = orig_arr = None
    if args.orig:
        if 'zarr_level' not in np0:
            parser.error('--orig supports a zarr source only (this codec was encoded '
                         f"from {np0.get('source')})")
        from inference.zarr_io import open_zarr
        src_arr = open_zarr(np0['source'], np0['zarr_level'])
        print(f'original source: {np0["source"]} level {np0["zarr_level"]}')
        if orig_dir is not None:
            os.makedirs(orig_dir, exist_ok=True)
        if args.zarr:                       # sibling '<zarr>_original' store, same shape
            base, ext = os.path.splitext(args.zarr.rstrip('/'))
            orig_arr = make_store(base + '_original' + ext, args.what + '_original')

    # Original comparison view follows --what: reconstruction -> raw input-res
    # XY pages; decode -> trilinear Z-upsampled ZX pages, matching the SR output.
    orig_pages = to_zx_pages if args.what == 'decode' else to_xy_pages

    def read_orig(r, c, za, zb, z_out):
        """Source patch as the comparison volume (Y, X, Z') matching --what."""
        v = src_arr[za:zb, c * patch:(c + 1) * patch, r * patch:(r + 1) * patch].read().result()
        o16 = np.ascontiguousarray(np.transpose(v, (2, 1, 0)))    # (Zc, x, y) -> (Y, X, Zc)
        if args.what == 'reconstruction':
            return o16
        t = torch.from_numpy(o16.astype(np.float32))[None, None]  # (1, 1, Y, X, Zc)
        upv = F.interpolate(t, size=(o16.shape[0], o16.shape[1], z_out),
                            mode='trilinear', align_corners=False)[0, 0]
        return np.clip(np.rint(upv.numpy()), 0, 65535).astype(np.uint16)  # (Y, X, Zc*up)

    futs = []
    for chunk in names:
        chunk_dir = os.path.join(root, chunk)
        with open(os.path.join(chunk_dir, 'norm_params.json')) as f:
            np_c = json.load(f)
        if np_c['slice_hw'] != [H, W] or np_c['grid_rows_cols'] != [rows, cols]:
            raise ValueError(f'{chunk}: geometry differs from {names[0]}')
        lo, hi = np_c['window_lo'], np_c['window_hi']
        za, zb = chunk_z_range(chunk)
        z_out = (zb - za) * up
        is_strip = args.tiff and (args.row is not None or args.col is not None)
        strip = [] if is_strip else None
        orig_strip = [] if (is_strip and src_arr is not None) else None

        t0 = time.perf_counter()
        for i in range(0, len(cells), args.batch):
            batch = cells[i:i + args.batch]
            idx = [torch.from_numpy(a.astype(np.int64)).to(eng.device)
                   for a in load_indices(chunk_dir, batch)]
            sl = eng.latents_from_indices(idx)
            out = eng.decode(sl) if args.what == 'decode' else eng.reconstruction(sl)
            if out.shape[2:] != (patch, patch, z_out):
                raise RuntimeError(f'{args.what} output {tuple(out.shape[2:])}, '
                                   f'expected {(patch, patch, z_out)} — check uprate')
            for b, (r, c) in enumerate(batch):
                u16 = restore_uint16(eng, out[b, 0], lo, hi)     # (Y, X, Z')
                zc0, xc0, yc0 = za * up - off_z, c * patch - off_x, r * patch - off_y
                if arr is not None:
                    slab = np.ascontiguousarray(u16.transpose(2, 1, 0))  # (z, x, y)
                    futs.append(arr[zc0:zc0 + z_out, xc0:xc0 + patch,
                                    yc0:yc0 + patch].write(slab))
                    while len(futs) > 8:
                        futs.pop(0).result()
                if args.tiff:
                    pages = to_zx_pages(u16) if args.what == 'decode' else to_xy_pages(u16)
                    tiff.imwrite(os.path.join(out_dir, f'{chunk}_r{r:03d}c{c:03d}.tif'), pages)
                if strip is not None:
                    strip.append(u16)
                if src_arr is not None:
                    ocmp = read_orig(r, c, za, zb, z_out)         # (Y, X, Z') matching --what
                    if orig_arr is not None:
                        oslab = np.ascontiguousarray(ocmp.transpose(2, 1, 0))  # (z, x, y)
                        futs.append(orig_arr[zc0:zc0 + z_out, xc0:xc0 + patch,
                                             yc0:yc0 + patch].write(oslab))
                        while len(futs) > 8:
                            futs.pop(0).result()
                    if args.tiff:       # per-patch original, symmetric with the model output
                        tiff.imwrite(os.path.join(
                            orig_dir, f'{chunk}_r{r:03d}c{c:03d}.tif'), orig_pages(ocmp))
                    if orig_strip is not None:
                        orig_strip.append(ocmp)
            done = i + len(batch)
            if done % (args.batch * 10) < args.batch or done == len(cells):
                rate = done / (time.perf_counter() - t0)
                print(f'  {chunk}: {done}/{len(cells)} patches  ({rate:.1f}/s, '
                      f'eta {(len(cells) - done) / rate:.0f}s)', flush=True)

        if strip:
            axis = 1 if args.row is not None else 0             # cols -> X, rows -> Y
            tag = (f'row{args.row:03d}' if args.row is not None else f'col{args.col:03d}')
            vol = np.concatenate(strip, axis=axis)
            pages = to_zx_pages(vol) if args.what == 'decode' else to_xy_pages(vol)
            path = os.path.join(out_dir, f'{chunk}_{tag}.tif')
            tiff.imwrite(path, pages)
            print(f'  {path}: (Y, X, Z) = {vol.shape}, '
                  f'{"ZX" if args.what == "decode" else "XY"} pages, '
                  f'range [{vol.min()}, {vol.max()}]')
            if orig_strip:
                ovol = np.concatenate(orig_strip, axis=axis)
                opath = os.path.join(orig_dir, f'{chunk}_{tag}.tif')
                tiff.imwrite(opath, orig_pages(ovol))
                print(f'  {opath}: (Y, X, Z) = {ovol.shape}, '
                      f'{"ZX (trilinear up)" if args.what == "decode" else "XY (input res)"} '
                      f'pages, range [{ovol.min()}, {ovol.max()}]')
    for f in futs:
        f.result()
    outs = [args.zarr, args.tiff]
    if orig_arr is not None:
        outs.append(os.path.splitext(args.zarr.rstrip('/'))[0] + '_original'
                    + os.path.splitext(args.zarr.rstrip('/'))[1])
    print('done -> ' + ' + '.join(p for p in outs if p))


if __name__ == '__main__':
    main()
