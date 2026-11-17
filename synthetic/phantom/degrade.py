"""Degrade an isotropic truth block the way THX10 was degraded.

    truth (counts above offset, isotropic)
      -> non-separable blur: FFT filter with the measured amplitude transfer
         sqrt(R(f_r, fz))                           (estimate/transfer.py)
      -> keep every 8th slice (point sampling, as the THX10 crop did)
      -> per-slice artefacts: intensity flicker, lateral jitter
      -> camera: offset + signal-dependent noise + fixed pattern
      -> in-plane resampling at a per-block sub-pixel shift (estimate/noise_shape.py)
      -> integer raw counts, and optionally the [-1, 1] training window.

`deg` everywhere is the dict stored in config/degradation.json.
"""
import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi

from synthetic import common
from synthetic.common import ZSTEP


def apply_transfers(truth, transfers, zpad=64, xypad=32):
    """Blur an isotropic (Z, Y, X) volume with each non-separable transfer in
    turn (one forward FFT for all of them); yields the blurred volumes.
    Reflect padding keeps the periodic FFT from wrapping structure around;
    pass zpad = xypad = 0 when the volume already carries a margin."""
    v = truth.astype(np.float32)
    if zpad or xypad:
        v = np.pad(v, ((zpad, zpad), (xypad, xypad), (xypad, xypad)), mode='reflect')
    F = sfft.rfftn(v, workers=common.FFT_WORKERS)
    shape = v.shape
    del v
    fz = np.abs(np.fft.fftfreq(shape[0])).astype(np.float32)[:, None, None]
    fy = np.fft.fftfreq(shape[1]).astype(np.float32)[None, :, None]
    fx = np.fft.rfftfreq(shape[2]).astype(np.float32)[None, None, :]
    fr = np.sqrt(fy ** 2 + fx ** 2)
    for t in transfers:
        # amplitude = exp(-0.5 (fz / f0)^p), in float32 to keep big blocks affordable
        amp = np.exp(np.float32(-0.5) * (fz / (np.float32(t['a']) + np.float32(t['b']) * fr)) ** np.float32(t['p']))
        out = sfft.irfftn(F * amp, s=shape, workers=common.FFT_WORKERS)
        del amp
        yield out[zpad:zpad + truth.shape[0], xypad:xypad + truth.shape[1], xypad:xypad + truth.shape[2]].astype(np.float32)


def apply_transfer(truth, transfer, zpad=64, xypad=32):
    return next(apply_transfers(truth, [transfer], zpad, xypad))


def slice_artefacts(vol, noise, rng):
    """Independent per-slice intensity gain and sub-pixel lateral shift."""
    out = vol
    fl = float(noise.get('flicker_std', 0.0))
    if fl > 0:
        out = out * (1.0 + rng.normal(scale=fl, size=(out.shape[0], 1, 1))).astype(np.float32)
    jit = float(noise.get('jitter_std_px', 0.0))
    if jit > 0:
        out = np.stack([ndi.shift(s, rng.normal(scale=jit, size=2), order=1, mode='nearest') for s in out])
    return out.astype(np.float32)


def camera(signal, noise, rng):
    """signal (counts above offset) -> raw counts with offset and white noise.

    Slice-independent noise has variance gain * signal + read_std^2. On top of
    it sits a pattern fixed to the pixel grid (the same in every slice), with
    variance var0 + slope * min(signal, sat).
    """
    gain = float(noise.get('gain', 1.0))
    read = float(noise.get('read_std', 0.0))
    S = np.clip(signal, 0, None)
    out = signal + float(noise.get('offset', 0.0)) + \
        np.sqrt(gain * S + read ** 2) * rng.standard_normal(signal.shape, dtype=np.float32)
    fp = noise.get('fixed_pattern')
    if fp:
        Vf = fp['var0'] + fp['slope'] * np.minimum(S, fp['sat'])
        out += np.sqrt(Vf) * rng.standard_normal((1,) + signal.shape[1:], dtype=np.float32)    # same in every slice
    return out.astype(np.float32)


def draw_resample(noise, rng):
    """Per-block in-plane interpolation fractions (wx, wy), drawn from the
    distribution measured on the real stacks (estimate/noise_shape.py)."""
    q = (noise.get('resample') or {}).get('w_quantiles')
    if not q:
        return 0.0, 0.0
    q = np.asarray(q, float)
    grid = np.linspace(0, 1, len(q))
    return float(np.interp(rng.random(), grid, q)), float(np.interp(rng.random(), grid, q))


def resample_xy(raw, wx, wy):
    """Linear interpolation at a fractional shift, as the real volume went
    through: (Z, Y + 1, X + 1) -> (Z, Y, X). Smooths signal and noise alike."""
    a = (1.0 - wx) * raw[:, :, :-1] + wx * raw[:, :, 1:]
    return ((1.0 - wy) * a[:, :-1] + wy * a[:, 1:]).astype(np.float32)


def acquire(signal, noise, rng, w=None, artefacts=True):
    """Low-resolution signal with one spare pixel in Y and X, (Z, Y + 1, X + 1)
    counts above offset -> integer raw counts (Z, Y, X): slice artefacts,
    camera noise, in-plane resampling, rounding."""
    if artefacts:
        signal = slice_artefacts(signal, noise, rng)
    wx, wy = draw_resample(noise, rng) if w is None else w
    return np.rint(resample_xy(camera(signal, noise, rng), wx, wy))


def to_window(raw, lo, hi):
    """Raw counts -> the [-1, 1] range of the training TIFFs."""
    return (np.clip((raw - lo) / (hi - lo), 0, 1) * 2 - 1).astype(np.float32)


def truth_to_window(truth, deg):
    """Noise-free truth in the same [-1, 1] units as the degraded input."""
    w = deg['window']
    return to_window(truth + deg['noise'].get('offset', 0.0), w['lo'], w['hi'])


def scale_transfer(transfer, width_scale):
    """Transfer whose axial kernel is `width_scale` times wider (f0 divided by it)."""
    return dict(a=transfer['a'] / width_scale, b=transfer['b'] / width_scale, p=transfer['p'])


def degrade(truth, deg, seed=0, with_noise=True, width_scale=1.0):
    """Convenience pipeline for a truth block without a rendered margin
    (reflect padding stands in for it). Returns raw counts, (Z/8, Y-1, X-1)
    with noise, (Z/8, Y, X) without."""
    rng = np.random.default_rng(seed)
    blurred = apply_transfer(truth, scale_transfer(deg['transfer'], width_scale))
    low = np.clip(blurred[::deg.get('zstep', ZSTEP)], 0, None)
    if with_noise:
        return acquire(low, deg['noise'], rng)
    return low + deg['noise'].get('offset', 0.0)
