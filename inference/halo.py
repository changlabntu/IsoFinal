"""Latent-halo decode for the zarr grid (seam fix), ported from the THX10
seam-blending experiment (its scheme S2-4z4: stitched latents, h = hz = 4).

Why: decode_stack.py decodes every 384^3 cell on its own, so net_g never sees
the neighbouring cells and the tiles meet with a visible seam. Here each cell's
per-scale latents are rebuilt on the 3x3x3 block of neighbouring cells' stored
codes (stitched in LATENT space, per scale, coarse codes interpolated on the
stitched grid), cropped to the cell plus a halo of h latent px in Y/X and hz
in Z, decoded at (384+16h)^2 x (384+16hz) and cropped back to the 384^3 cell.
Interior voxels are identical to a plain decode (translation equivariance);
only the cell faces change, and they now continue into the neighbour.

Geometry (THX10 codec): 48x48 latent px per cell at the finest scale, 48 Z
planes per z-chunk, uprate 8 -> h = hz = 4 gives a 56^3 latent window and a
448^3 decode, cropped by 32 voxels per side. Cost ~1.49x a plain decode.

Missing neighbours (volume border, first/last z-chunk, a cell that was not
encoded) are replicate-padded in latent space, which on that side degenerates
to the single-cell behaviour. h = hz = 0 bypasses stitching entirely and is
bit-identical to engine.latents_from_indices.

Python 3.8 compatible (the zarr env).
"""

import os
import re
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CHUNK_RE = re.compile(r'^z(\d{4,})-(\d{4,})$')


# ---- Z-chunked full-res heads --------------------------------------------------

class ZChunkedHead(nn.Module):
    """Wrap a Conv3d(k3, p1) -> pointwise Sequential so it runs in exact Z slabs
    (1-voxel conv halo) when the input exceeds cuDNN's int32 element limit.
    Halo decodes (448^3+, 32 ch) hit that limit in net_g's conv7_k; SafeUpConv3d
    already guards up1 the same way. Below the limit it is the original module."""
    SAFE = 2 ** 30

    def __init__(self, seq):
        super().__init__()
        self.seq = seq
        flat = [m for m in seq.modules() if not isinstance(m, nn.Sequential)]
        assert isinstance(flat[0], nn.Conv3d) and flat[0].kernel_size == (3, 3, 3), \
            'ZChunkedHead expects a Conv3d(k3) followed by pointwise ops'
        self.conv, self.rest = flat[0], flat[1:]      # conv, then pointwise ops

    def forward(self, x):
        if x.numel() <= self.SAFE:
            return self.seq(x)
        zin = x.shape[4]
        plane = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
        step = max(1, self.SAFE // plane - 2)
        out = None
        for a in range(0, zin, step):
            b = min(a + step, zin)
            a2, b2 = a - (1 if a > 0 else 0), b + (1 if b < zin else 0)
            slab = F.pad(x[..., a2:b2], (1 if a == 0 else 0, 1 if b == zin else 0))
            y = F.conv3d(slab, self.conv.weight, self.conv.bias, stride=1,
                         padding=(1, 1, 0))
            del slab
            for m in self.rest:
                y = m(y)
            if out is None:                     # preallocate once, no final cat
                out = torch.empty(x.shape[:1] + (y.shape[1],) + x.shape[2:],
                                  dtype=y.dtype, device=y.device)
            out[..., a:b] = y
            del y
        return out


class NoOpHead(nn.Module):
    """Replaces net_g.conv7_g: its 'out1' head is never used by the decode path."""

    def forward(self, x):
        return None


def install_zchunked_heads(eng):
    """Make the engine's net_g decode 448^3+ windows: Z-chunk conv7_k, drop the
    unused conv7_g. In-memory only (never saved); numerically a no-op. Idempotent."""
    ng = eng.gan.net_g
    if not isinstance(ng.conv7_k, ZChunkedHead):
        ng.conv7_k = ZChunkedHead(ng.conv7_k)
        ng.conv7_g = NoOpHead()
    return eng


# ---- stored-index access -------------------------------------------------------

class CellIndexCache:
    """Stored per-scale indices of codec cells, LRU-cached (a halo decode reads
    each cell up to 27 times). get() -> list of (Z, hk, wk) int32 arrays, coarse
    -> fine, or None if the cell's npz does not exist."""

    def __init__(self, root, maxsize=512):
        self.root = root
        self.maxsize = maxsize
        self._d = OrderedDict()

    def path(self, chunk, r, c):
        return os.path.join(self.root, chunk, '%03d%03d.npz' % (r, c))

    def get(self, chunk, r, c):
        key = (chunk, r, c)
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        p = self.path(chunk, r, c)
        if r < 0 or c < 0 or not os.path.exists(p):
            val = None
        else:
            with np.load(p) as d:
                keys = sorted((k for k in d.files if k.startswith('scale_')),
                              key=lambda k: int(k.split('_')[1]))
                val = [d[k][0].astype(np.int32) for k in keys]     # stored (1, Z, hk, wk)
        self._d[key] = val
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)
        return val


def _z_range(name):
    m = _CHUNK_RE.match(name)
    if not m:
        raise ValueError("'%s' is not a z chunk dir name (zBBBB-EEEE)" % name)
    return int(m.group(1)), int(m.group(2))


def _z_adjacent(names, s, dz):
    """True if chunk s+dz is Z-contiguous with chunk s (no gap, no overlap)."""
    a, b = _z_range(names[s]), _z_range(names[s + dz])
    return (a[1] == b[0]) if dz > 0 else (b[1] == a[0])


# ---- stitched-latent construction ----------------------------------------------

def stitched_latents(eng, cache, names, s, r, c, h, hz):
    """Per-scale volume-form latents of cell (chunk s, row r, col c) with a halo:
    list over scales of (1, z_ch, L+2h, L+2h, Z+2hz), L = latent px per cell,
    Z = planes per chunk. names = sorted z-chunk dir names of the codec root.

    h = hz = 0 -> engine.latents_from_indices on the single cell (bit-identical
    legacy path). Otherwise the available neighbours (hz > 0: chunks s±1 if
    Z-contiguous and of equal depth; h > 0: rows/cols ±1) are stitched per scale,
    requiring every (dz, dr, dc) combination of the kept offsets to exist."""
    g = eng.gan
    dev = eng.device
    centre = cache.get(names[s], r, c)
    if centre is None:
        raise FileNotFoundError(cache.path(names[s], r, c))
    if h == 0 and hz == 0:
        idx = [torch.from_numpy(a[None].astype(np.int64)).to(dev) for a in centre]
        return eng.latents_from_indices(idx)

    Zc, L = centre[-1].shape[0], centre[-1].shape[-1]
    if centre[-1].shape[-2] != L:
        raise ValueError('non-square finest latent %s' % (centre[-1].shape,))
    dt = next(g.post_quant_convs.parameters()).dtype

    def usable(dz, dr, dc):
        if not 0 <= s + dz < len(names):
            return False
        if dz != 0 and not _z_adjacent(names, s, dz):
            return False
        a = cache.get(names[s + dz], r + dr, c + dc)
        return a is not None and a[-1].shape == centre[-1].shape

    dzs = [dz for dz in (-1, 0, 1) if (hz > 0 or dz == 0) and usable(dz, 0, 0)]
    drs = [dr for dr in (-1, 0, 1) if (h > 0 or dr == 0) and usable(0, dr, 0)]
    dcs = [dc for dc in (-1, 0, 1) if (h > 0 or dc == 0) and usable(0, 0, dc)]
    grid = {}
    for dz in dzs:
        for dr in drs:
            for dc in dcs:
                a = cache.get(names[s + dz], r + dr, c + dc)
                if a is None or a[-1].shape != centre[-1].shape:
                    raise FileNotFoundError('halo neighbour missing or mismatched: '
                                            + cache.path(names[s + dz], r + dr, c + dc))
                grid[(dz, dr, dc)] = a
    nz, nr, nc = len(dzs), len(drs), len(dcs)
    iz, ir, ic = dzs.index(0), drs.index(0), dcs.index(0)       # centre cell position
    Hs, Ws, Zs = L * nr, L * nc, Zc * nz

    out = []
    for k, p in enumerate(g.scale_factors):
        hk = L // p
        idx = np.zeros((Zs, hk * nr, hk * nc), np.int64)
        for (dz, dr, dc), arrs in grid.items():
            if arrs[k].shape != (Zc, hk, hk):
                raise ValueError('scale %d index shape %s, expected %s'
                                 % (k, arrs[k].shape, (Zc, hk, hk)))
            z0, y0, x0 = dzs.index(dz) * Zc, drs.index(dr) * hk, dcs.index(dc) * hk
            idx[z0:z0 + Zc, y0:y0 + hk, x0:x0 + hk] = arrs[k]
        t = torch.from_numpy(idx).to(dev)
        code = g.quantizers[k].get_codebook_entry(
            t.reshape(-1), shape=(Zs, hk * nr, hk * nc, g.embed_dim))   # (Zs, E, hk*nr, hk*nc)
        code = code.to(dt)
        if p > 1:   # coarse codes interpolated on the STITCHED grid (sees the neighbour)
            code = F.interpolate(code, size=(Hs, Ws), mode='bilinear', align_corners=False)
        lat = g.post_quant_convs[k](code)                                 # (Zs, z_ch, Hs, Ws)
        # crop the halo window around the centre cell; replicate-pad where short
        y0, x0, z0 = ir * L - h, ic * L - h, iz * Zc - hz
        y1, x1, z1 = y0 + L + 2 * h, x0 + L + 2 * h, z0 + Zc + 2 * hz
        pad = (max(0, -x0), max(0, x1 - Ws), max(0, -y0), max(0, y1 - Hs))
        lat = lat[max(z0, 0):min(z1, Zs), :, max(y0, 0):min(y1, Hs), max(x0, 0):min(x1, Ws)]
        if any(pad):
            lat = F.pad(lat, pad, mode='replicate')
        pz = (max(0, -z0), max(0, z1 - Zs))
        if any(pz):
            lat = torch.cat([lat[:1].expand(pz[0], -1, -1, -1), lat,
                             lat[-1:].expand(pz[1], -1, -1, -1)], 0)
        # .contiguous() is load-bearing: a strided view here makes conv3d
        # request >100 GB of workspace downstream
        out.append(g.slices_to_vol(lat.contiguous(), 1))                 # (1, z_ch, H, W, Z)
    return out


def batched_stitched_latents(eng, cache, names, s, cells, h, hz):
    """stitched_latents for several (r, c) cells of chunk s, concatenated along
    the batch dim per scale -> list over scales of (B, z_ch, L+2h, L+2h, Z+2hz)."""
    per_cell = [stitched_latents(eng, cache, names, s, r, c, h, hz) for r, c in cells]
    return [torch.cat([pc[k] for pc in per_cell], 0) for k in range(len(per_cell[0]))]


def crop_decoded(out, h, hz, up, patch):
    """(B, 1, patch+2*up*h, patch+2*up*h, Z'+2*up*hz) decode -> the (B, 1, patch,
    patch, Z') cell: drop `up` output voxels per latent halo px on every side."""
    oy = ox = up * h
    oz = up * hz
    return out[:, :, oy:oy + patch, ox:ox + patch, oz:out.shape[4] - oz]
