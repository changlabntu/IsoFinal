"""Degrade an isotropic truth block the way THX10 was degraded.

    truth (counts above offset, isotropic)
      -> non-separable blur: FFT filter with the measured amplitude transfer
         sqrt(R(f_r, fz))                           (estimate/transfer.py)
      -> keep every 8th slice (point sampling, as the THX10 crop did)
      -> per-slice artefacts: intensity flicker, lateral jitter
      -> camera: offset + Poisson-Gaussian noise    (estimate/noise.py)
      -> raw counts, and optionally the [-1, 1] training window.

`deg` everywhere is the dict stored in config/degradation.json.
"""
import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi

from synthetic.common import ZSTEP
from synthetic.estimate.transfer import transfer_amplitude


def apply_transfer(truth, transfer, zpad=64, xypad=32):
    """Blur an isotropic (Z, Y, X) volume with the non-separable transfer.
    Reflect padding keeps the periodic FFT from wrapping structure around."""
    v = np.pad(truth.astype(np.float32), ((zpad, zpad), (xypad, xypad), (xypad, xypad)), mode='reflect')
    F = sfft.rfftn(v, workers=-1)
    fz = np.abs(np.fft.fftfreq(v.shape[0]))[:, None, None]
    fy = np.fft.fftfreq(v.shape[1])[None, :, None]
    fx = np.fft.rfftfreq(v.shape[2])[None, None, :]
    fr = np.sqrt(fy ** 2 + fx ** 2)
    F *= transfer_amplitude(fr, fz, transfer).astype(np.float32)
    out = sfft.irfftn(F, s=v.shape, workers=-1)
    return out[zpad:zpad + truth.shape[0], xypad:xypad + truth.shape[1], xypad:xypad + truth.shape[2]].astype(np.float32)


def sample_z(vol, step=ZSTEP, phase=None):
    """Every `step`-th slice starting at `phase` (default: the centre of the
    first step, so slice i of the result sits at truth slice phase + i*step)."""
    phase = step // 2 if phase is None else int(phase)
    return vol[phase::step], phase


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
    """signal (counts above offset) -> raw counts with offset and noise.

    var = gain * signal + read_var; `noise_corr_sigma` (px) gives the noise the
    slight lateral correlation measured in the real data.
    """
    gain = float(noise.get('gain', 1.0))
    read = float(noise.get('read_std', 0.0))
    n = rng.normal(size=signal.shape).astype(np.float32)
    cs = float(noise.get('noise_corr_sigma', 0.0))
    if cs > 0:
        n = ndi.gaussian_filter(n, (0, cs, cs))
        n /= n.std()
    fpn = float(noise.get('fixed_pattern_std', 0.0))
    temporal = np.sqrt(max(read ** 2 - fpn ** 2, 0.0))      # read_std includes the fixed pattern
    sd = np.sqrt(np.clip(gain * signal, 0, None) + temporal ** 2)
    out = signal + float(noise.get('offset', 0.0)) + sd * n
    if fpn > 0:                                             # same pattern in every slice
        out = out + fpn * rng.normal(size=(1,) + signal.shape[1:]).astype(np.float32)
    return out


def to_window(raw, lo, hi):
    """Raw counts -> the [-1, 1] range of the training TIFFs."""
    return (np.clip((raw - lo) / (hi - lo), 0, 1) * 2 - 1).astype(np.float32)


def truth_to_window(truth, deg):
    """Noise-free truth in the same [-1, 1] units as the degraded input."""
    w = deg['window']
    return to_window(truth + deg['noise'].get('offset', 0.0), w['lo'], w['hi'])


def degrade(truth, deg, seed=0, phase=None, with_noise=True):
    """Full pipeline. Returns dict(raw (Z/8, Y, X) counts, phase, blurred)."""
    rng = np.random.default_rng(seed)
    blurred = apply_transfer(truth, deg['transfer'])
    low, phase = sample_z(blurred, deg.get('zstep', ZSTEP), phase)
    if with_noise:
        low = slice_artefacts(low, deg['noise'], rng)
        raw = camera(low, deg['noise'], rng)
    else:
        raw = low + deg['noise'].get('offset', 0.0)
    return dict(raw=raw.astype(np.float32), phase=phase)
