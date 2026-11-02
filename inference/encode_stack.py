"""Encode a stack of large 2D slices to codecs, patch by patch — no decoding.

TIFF-folder source (one z-chunk per run):

    python inference/encode_stack.py --model skipU --exp sample0 \
        --source /media/cheese/Ghc_data3/THX10/sample0_crop \
        [--range 0 32] [--half --batch 8] [--tsne]

OME-Zarr source (all z-chunks in ONE run — the default format for large
stacks; needs the py38zarr env = py38pl16 clone + tensorstore):

    python inference/encode_stack.py --model skipU --exp sample0 \
        --zarr /media/cheese/Ghc_data3/THX10/sample0_crop_ome.zarr \
        [--level 0] [--range 0 160] [--zchunk 32] [--half --batch 8] [--tsne]

The TIFF source folder holds large 2D tif slices (sorted filename order = Z
order), all the same (H, W) with H and W divisible by --patch (see the
crop_params.json convention from the cropping step). --range BEGIN END picks
slices [BEGIN:END] of the sorted list (default: all) — for a folder of
hundreds of slices, run one 32-slice chunk at a time. The selected chunk is
loaded into RAM once (kept uint16).

The zarr source is a zarr-v3 OME-NGFF group with axes (z, x, y) — each slice
stored TRANSPOSED vs the raw tif (y, x); level 0 chunks (32, 256, 256) line
up so one patch volume = one aligned chunk read. Nothing is staged in RAM:
patches stream straight from the store (async batched reads), and --range
(default: all Z) is split into --zchunk-sized chunks (default: the store's
z chunk) processed in one run — no manual --window handoff between chunks.

Either way, every (patch, patch, chunk) volume of the H/patch x W/patch grid
is:

  1. mapped to [-1, 1] with a GLOBAL robust percentile window
     (--lo_pct/--hi_pct, default 1/99.9) — one shared intensity scale for all
     patches, like the training data's global map. TIFF mode estimates it
     from the chunk with strided np.percentile; zarr mode computes the EXACT
     empirical quantiles of the full selected range via a one-pass uint16
     histogram (so all chunks of a run share one exact window).
  2. put through the model's own normalization (nm='11g' gamma for skipU)
  3. encoded to codebook indices, saved as
     {out_base}/{exp}/codec/z{BEGIN}-{END}/{rrr}{ccc}.npz
     (rrr = row / Y-grid index, ccc = col / X-grid index; the z chunk dir
     keeps successive chunks of the same experiment separate)

Codec only by default — decode later from the npz files
(inference/decode_stack.py: grid decode to uint16 TIFFs / a native zarr
store; inference/inference_latent.py is the per-file TIFF round trip);
--tsne additionally
runs inference/latent_tsne.py over each chunk's codec dir (plots + anomaly
csv land in that dir). A norm_params.json in the chunk dir records the
intensity window and all parameters needed to reproduce or invert the
encode. Chunks of one sample should share ONE window: zarr mode does this
automatically; for TIFF mode the first chunk estimates it and later chunks
pass it back via --window LO HI (printed and stored in the json).
"""

import argparse
import json
import os
import sys
import time
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import tifffile as tiff  # noqa: E402
import torch  # noqa: E402

from inference import Engine, available, get as registry_get  # noqa: E402
from inference.zarr_io import open_zarr  # noqa: E402

DEFAULT_OUT_BASE = '/home/cheese/workspace/Output'


def zarr_window(arr, z0, z1, lo_pct, hi_pct):
    """Exact global percentiles of uint16 arr[z0:z1] via a one-pass histogram.

    Streams x-slabs aligned to the 256-wide chunks; returns the exact
    empirical quantiles (integer intensities), unlike the strided
    interpolated np.percentile of the TIFF path.
    """
    assert arr.dtype.name == 'uint16', f'histogram window expects uint16, got {arr.dtype}'
    hist = np.zeros(65536, dtype=np.int64)
    _, X, _ = arr.shape
    # slabs aligned to the store's own chunks, so each chunk decompresses once
    zc, xc = (int(c) for c in arr.chunk_layout.read_chunk.shape[:2])
    t0 = time.perf_counter()
    for za in range(z0, z1, zc):
        for xa in range(0, X, xc):
            slab = arr[za:min(za + zc, z1), xa:xa + xc, :].read().result()
            hist += np.bincount(slab.ravel(), minlength=65536)
        print(f'  window scan z{za}-{min(za + zc, z1)} '
              f'({time.perf_counter() - t0:.0f}s)', flush=True)
    cum = np.cumsum(hist)
    n = cum[-1]

    def q(p):
        return float(np.searchsorted(cum, p / 100 * n))

    return q(lo_pct), q(hi_pct)


def encode_grid(eng, args, codec_dir, meta_str, rows, cols, fetch):
    """Encode every grid cell; fetch(cells) -> (B, 1, patch, patch, Zc) cpu tensor.

    Saves one npz per cell into codec_dir; returns (n_scales, seconds).
    """
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    t0 = time.perf_counter()
    n_scales = None
    for i in range(0, len(cells), args.batch):
        batch = cells[i:i + args.batch]
        x = fetch(batch).to(eng.device)
        if args.half:
            x = x.half()
        _, indices = eng.encode(x)                               # per scale: (B, Z, hk, wk)
        n_scales = len(indices)
        for b, (r, c) in enumerate(batch):
            np.savez_compressed(
                os.path.join(codec_dir, f'{r:03d}{c:03d}.npz'), meta=meta_str,
                **{f'scale_{k}': idx[b:b + 1].cpu().numpy().astype(np.int32)
                   for k, idx in enumerate(indices)})
        done = i + len(batch)
        if done % (args.batch * 25) < args.batch or done == len(cells):
            rate = done / (time.perf_counter() - t0)
            print(f'  {done}/{len(cells)} patches  ({rate:.1f}/s, '
                  f'eta {(len(cells) - done) / rate:.0f}s)', flush=True)
    return n_scales, time.perf_counter() - t0


def finish_chunk(codec_dir, n_scales, secs, n_cells, raw_bytes, tsne):
    total = sum(os.path.getsize(os.path.join(codec_dir, f))
                for f in os.listdir(codec_dir) if f.endswith('.npz'))
    print(f'done: {n_cells} codecs ({n_scales} scales) in {secs:.0f}s '
          f'-> {codec_dir}\ncodec total {total / 2**20:.1f} MiB vs raw '
          f'{raw_bytes / 2**30:.1f} GiB ({raw_bytes / total:.0f}x)')
    if tsne:
        from inference.latent_tsne import run as tsne_run
        tsne_run(codec_dir, out_dir=codec_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Global-percentile normalize + encode a 2D-slice stack to codecs')
    parser.add_argument('--model', required=True,
                        help=f'Registry model name ({", ".join(available())})')
    parser.add_argument('--exp', required=True,
                        help='Experiment name: codecs go to {out_base}/{exp}/codec/')
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--source',
                     help='Folder of large 2D .tif slices (sorted name order = Z)')
    src.add_argument('--zarr',
                     help='OME-Zarr v3 store dir (axes (z, x, y)); encodes all '
                          'z-chunks of --range in one run, streaming from disk')
    parser.add_argument('--level', default='0',
                        help='Zarr pyramid level to encode (default 0 = full res)')
    parser.add_argument('--zchunk', type=int, default=None,
                        help='Zarr mode: slices per encode chunk '
                             "(default: the store's z chunk size)")
    parser.add_argument('--range', type=int, nargs=2, default=None, metavar=('BEGIN', 'END'),
                        help='Slice selection [BEGIN:END] (default: all slices)')
    parser.add_argument('--patch', type=int, default=256, help='Patch size in Y/X (default 256)')
    parser.add_argument('--lo_pct', type=float, default=1.0,
                        help='Lower robust percentile of the global intensity window (default 1)')
    parser.add_argument('--hi_pct', type=float, default=99.9,
                        help='Upper robust percentile (default 99.9)')
    parser.add_argument('--pct_stride', type=int, default=8,
                        help='TIFF mode: pixel stride when sampling for percentiles (default 8)')
    parser.add_argument('--window', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                        help='Reuse a fixed intensity window instead of estimating one — '
                             'e.g. to match codecs encoded earlier from another source')
    parser.add_argument('--tsne', action='store_true',
                        help="Also run latent_tsne over each chunk's codec dir afterwards")
    parser.add_argument('--batch', type=int, default=8,
                        help='Patches per encode forward (default 8)')
    parser.add_argument('--epoch', type=int, default=None, help="Override the registry's epoch")
    parser.add_argument('--device', default=None, help='cuda / cpu (default: cuda if available)')
    parser.add_argument('--half', action='store_true', help='Run model and inputs in fp16')
    parser.add_argument('--out_base', default=DEFAULT_OUT_BASE,
                        help=f'Output base dir (default: {DEFAULT_OUT_BASE})')
    args = parser.parse_args()

    eng = Engine.from_registry(args.model, epoch=args.epoch, device=args.device)
    if args.half:
        eng.gan.half()
    epoch = args.epoch if args.epoch is not None else registry_get(args.model).get('epoch')
    root = os.path.join(args.out_base, args.exp)
    try:                        # Z upsample factor of decode, for decode_stack
        uprate = int(round((eng.cfg.cropsize // eng.cfg.cropz)
                           * eng.cfg.dsp / eng.cfg.usp))
    except (AttributeError, TypeError, ZeroDivisionError):
        uprate = None

    def norm_params_base(lo, hi):
        d = dict(model=args.model, epoch=epoch, **eng.spec,
                 patch=args.patch, lo_pct=args.lo_pct, hi_pct=args.hi_pct,
                 window_lo=float(lo), window_hi=float(hi))
        if uprate is not None:
            d['uprate'] = uprate
        return d

    if args.zarr:
        arr = open_zarr(args.zarr, args.level)
        Z, W, H = arr.shape                                     # axes (z, x, y)
        if H % args.patch or W % args.patch:
            raise ValueError(f'level {args.level} plane ({H}, {W}) not divisible by '
                             f'patch {args.patch}')
        rows, cols = H // args.patch, W // args.patch
        z0, z1 = args.range if args.range is not None else (0, Z)
        if not 0 <= z0 < z1 <= Z:
            raise ValueError(f'--range {z0} {z1} outside stack of {Z} slices')
        zc = args.zchunk or arr.chunk_layout.read_chunk.shape[0]
        print(f'zarr level {args.level}: (Z, X, Y) = {tuple(arr.shape)} {arr.dtype}, '
              f'grid {rows} x {cols} = {rows * cols} patches, '
              f'z {z0}-{z1} in chunks of {zc}')

        if args.window is not None:
            lo, hi = args.window
            print(f'global window (given): [{lo:.1f}, {hi:.1f}]')
        else:
            print(f'scanning z {z0}-{z1} for the exact global window ...', flush=True)
            lo, hi = zarr_window(arr, z0, z1, args.lo_pct, args.hi_pct)
            print(f'global window (exact histogram): p{args.lo_pct}={lo:.1f}, '
                  f'p{args.hi_pct}={hi:.1f} — reuse with --window {lo:.1f} {hi:.1f}')

        for za in range(z0, z1, zc):
            zb = min(za + zc, z1)
            if zb - za < zc:
                print(f'note: last chunk z{za}-{zb} is only {zb - za} slices')
            chunk = f'z{za:04d}-{zb:04d}'
            print(f'chunk {chunk}:')
            codec_dir = os.path.join(root, 'codec', chunk)
            os.makedirs(codec_dir, exist_ok=True)
            norm_params = dict(norm_params_base(lo, hi),
                               source=args.zarr, zarr_level=args.level,
                               z_range=[za, zb], n_slices=zb - za,
                               slice_hw=[H, W], grid_rows_cols=[rows, cols],
                               window_method='given' if args.window else 'exact-histogram',
                               window_over_z=[z0, z1])
            with open(os.path.join(codec_dir, 'norm_params.json'), 'w') as f:
                json.dump(norm_params, f, indent=2)
            meta_str = json.dumps(norm_params)

            def fetch(cells, za=za, zb=zb):
                futs = [arr[za:zb,
                            c * args.patch:(c + 1) * args.patch,
                            r * args.patch:(r + 1) * args.patch].read()
                        for r, c in cells]                       # (Zc, x, y) each
                vols = []
                for f in futs:
                    v = np.transpose(f.result(), (2, 1, 0)).astype(np.float32)  # (Y, X, Zc)
                    v = np.clip((v - lo) / (hi - lo), 0, 1) * 2 - 1
                    vols.append(torch.from_numpy(eng.normalize(v))[None, None])
                return torch.cat(vols)

            n_scales, secs = encode_grid(eng, args, codec_dir, meta_str, rows, cols, fetch)
            raw = (zb - za) * H * W * arr.dtype.numpy_dtype.itemsize
            finish_chunk(codec_dir, n_scales, secs, rows * cols, raw, args.tsne)
        return

    files = sorted(glob(os.path.join(args.source, '*.tif'))
                   + glob(os.path.join(args.source, '*.tiff')))
    if not files:
        raise FileNotFoundError(f'No .tif slices in {args.source}')
    z0, z1 = args.range if args.range is not None else (0, len(files))
    files = files[z0:z1]
    if not files:
        raise ValueError(f'--range {z0} {z1} selects no slices')
    chunk = f'z{z0:04d}-{z1:04d}'
    print(f'chunk {chunk}: slices {os.path.basename(files[0])} .. '
          f'{os.path.basename(files[-1])}')

    print(f'loading {len(files)} slices into RAM ...', flush=True)
    stack = np.stack([tiff.imread(f) for f in files])           # (Z, H, W), source dtype
    Z, H, W = stack.shape
    if H % args.patch or W % args.patch:
        raise ValueError(f'slice size ({H}, {W}) not divisible by patch {args.patch} — '
                         'crop first (see the sample0_crop convention)')
    rows, cols = H // args.patch, W // args.patch
    print(f'stack (Z, H, W) = {stack.shape} {stack.dtype}, '
          f'{stack.nbytes / 2**30:.1f} GiB -> grid {rows} x {cols} = {rows * cols} patches')

    if args.window is not None:
        lo, hi = args.window
        print(f'global window (given): [{lo:.1f}, {hi:.1f}]')
    else:
        sub = stack[:, ::args.pct_stride, ::args.pct_stride]
        lo, hi = np.percentile(sub, [args.lo_pct, args.hi_pct])
        print(f'global window: p{args.lo_pct}={lo:.1f}, p{args.hi_pct}={hi:.1f} '
              f'(sampled {sub.size / 1e6:.0f}M px, stride {args.pct_stride}) — '
              f'reuse with --window {lo:.1f} {hi:.1f} for the other chunks')

    codec_dir = os.path.join(root, 'codec', chunk)
    os.makedirs(codec_dir, exist_ok=True)
    norm_params = dict(norm_params_base(lo, hi),
                       source=args.source, z_range=[z0, z1], n_slices=Z,
                       slice_hw=[H, W], grid_rows_cols=[rows, cols])
    with open(os.path.join(codec_dir, 'norm_params.json'), 'w') as f:
        json.dump(norm_params, f, indent=2)
    meta_str = json.dumps(norm_params)

    def fetch(cells):
        vols = []
        for r, c in cells:
            p = stack[:, r * args.patch:(r + 1) * args.patch,
                      c * args.patch:(c + 1) * args.patch]       # (Z, y, x)
            v = np.transpose(p, (1, 2, 0)).astype(np.float32)    # (Y, X, Z)
            v = np.clip((v - lo) / (hi - lo), 0, 1) * 2 - 1      # global window -> [-1, 1]
            v = eng.normalize(v)                                 # model nm (11g gamma)
            vols.append(torch.from_numpy(v)[None, None])
        return torch.cat(vols)

    n_scales, secs = encode_grid(eng, args, codec_dir, meta_str, rows, cols, fetch)
    if args.tsne:
        raw = stack.nbytes
        del stack                                   # free the RAM before sklearn
        finish_chunk(codec_dir, n_scales, secs, rows * cols, raw, True)
    else:
        finish_chunk(codec_dir, n_scales, secs, rows * cols, stack.nbytes, False)


if __name__ == '__main__':
    main()
