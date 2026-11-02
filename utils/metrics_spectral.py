"""Spectral (FFT) metrics for detecting upsampling alias lattices.

Strided ConvTranspose upsampling imprints a periodic lattice (period-2/4,
diagonal-dominant) that generated structures anchor to, while the stride-2
patch discriminator aliases exactly those frequencies into invisibility —
so LPIPS/KID see the artifact only indirectly. These helpers measure it
directly. Diagnosis and probe recipe: doc/research_artifact_directions.md.
"""

import torch

# Probe frequencies in Nyquist units (1.0 = the sampling Nyquist, period 2 px).
# Keys become metric suffixes: val_lat_<key>. (fa, fb) are frequencies along the
# (H, W) axes of the probed planes; for lateral planes W is the synthesized Z.
LATTICE_PROBES = {
    'p2diag': (1.0, 1.0),   # period-2 diagonal — the checkerboard corner harmonic
    'p2a':    (1.0, 0.0),   # period-2 along the first (in-plane) axis
    'p2z':    (0.0, 1.0),   # period-2 along the second axis (Z for lateral planes)
    'p4diag': (0.5, 0.5),   # period-4 diagonal — the second upsampling stage
}


def mean_power_spectrum(planes):
    """Mean 2D power spectrum of a batch of planes.

    planes: real tensor (N, C, H, W). Returns (H, W), fftshifted so DC sits at
    (H//2, W//2). Per-plane mean subtraction plus a per-axis Hann window keep
    DC leakage out of the probed bins and handle H != W. Stays on the input
    device; the complex intermediate is only (N, C, H, W).
    """
    planes = planes - planes.mean(dim=(-2, -1), keepdim=True)
    wh = torch.hann_window(planes.shape[-2], periodic=False,
                           device=planes.device, dtype=planes.dtype)
    ww = torch.hann_window(planes.shape[-1], periodic=False,
                           device=planes.device, dtype=planes.dtype)
    spec = torch.fft.fft2(planes * (wh[:, None] * ww[None, :])).abs().pow(2)
    return torch.fft.fftshift(spec, dim=(-2, -1)).mean(dim=(0, 1))


# Radial bands in cycles/pixel (Nyquist = 0.5) for the spectral-retention metric
# (val_spec_*). 'mid' ~ the scale of real structures (filament widths, boundaries)
# — where vqclean's visible fidelity advantage was measured; 'hi' ~ fine grain,
# inflated by noise/dropout/lattice, read with care. On a 384^2 plane these are
# radii 32-96 and 96-192. 'lo' is omitted (always ~1, uninformative).
SPECTRAL_BANDS = {
    'mid': (1 / 12, 1 / 4),
    'hi':  (1 / 4, 1 / 2),
}


def band_power(spec, bands=SPECTRAL_BANDS):
    """Integrate an fftshifted (H, W) power spectrum over radial frequency bands.

    Bands are (lo, hi) in cycles/pixel; the radius grid is normalized per-axis so
    non-square planes probe the same physical frequencies. Returns
    {band_key: 0-dim tensor}.
    """
    H, W = spec.shape
    fy = (torch.arange(H, device=spec.device, dtype=spec.dtype) - H // 2) / H
    fx = (torch.arange(W, device=spec.device, dtype=spec.dtype) - W // 2) / W
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return {k: spec[(r >= lo) & (r < hi)].sum() for k, (lo, hi) in bands.items()}


def lattice_peak_ratios(spec, probes=LATTICE_PROBES,
                        peak_half=1, bg_half=10, exclude_half=2, eps=1e-12):
    """Peak-to-local-background ratio at each probe frequency. ~1 = clean.

    spec: (H, W) fftshifted power spectrum (e.g. from mean_power_spectrum).
    For each probe the target bin is torch.roll-ed to the center — after
    fftshift the Nyquist bin lives at index 0, so direct slicing would run off
    the array, while roll is periodic and exact. Peak = max of the center
    (2*peak_half+1)^2 window (Hann spreads a tone over ~3 bins); background =
    median of the (2*bg_half+1)^2 patch with the center (2*exclude_half+1)^2
    masked out. Returns {probe_key: 0-dim tensor}; empty dict if the spectrum
    is too small to probe.
    """
    H, W = spec.shape
    bg_half = min(bg_half, (min(H, W) - 1) // 2)
    if bg_half <= exclude_half:
        return {}
    cy, cx = H // 2, W // 2
    out = {}
    for key, (fa, fb) in probes.items():
        iy = cy + int(round(fa * H / 2))
        ix = cx + int(round(fb * W / 2))
        rolled = torch.roll(spec, shifts=(cy - iy, cx - ix), dims=(0, 1))
        patch = rolled[cy - bg_half: cy + bg_half + 1,
                       cx - bg_half: cx + bg_half + 1]
        k = bg_half
        peak = patch[k - peak_half: k + peak_half + 1,
                     k - peak_half: k + peak_half + 1].max()
        mask = torch.ones_like(patch, dtype=torch.bool)
        mask[k - exclude_half: k + exclude_half + 1,
             k - exclude_half: k + exclude_half + 1] = False
        bg = patch[mask].median()
        out[key] = peak / bg.clamp_min(eps)
    return out
