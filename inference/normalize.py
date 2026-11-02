"""Intensity normalization — the same math training applies in
dataloader.data_multi.PairedImageDataset._normalize_image, as pure functions.

All functions take numpy arrays or torch tensors (only .clip/.min/.max/**
are used) and return the same type. No file I/O here.
"""


def normalize(vol, nm, gamma=0.25, gamma_lo=-1.0, norm_stats=None, key=None):
    """Apply the run's --nm normalization to a raw volume.

    '00'  untouched
    '01'  min-max to [0, 1]
    '11'  min-max to [-1, 1]
    '11p' percentile clip to [-1, 1] using norm_stats[key] = (lo, hi)
    '11g' noise floor + compressive gamma; input must already be in [-1, 1]:
          clip((v - gamma_lo) / (1 - gamma_lo), 0, 1) ** gamma * 2 - 1
    """
    if nm == '01':
        vol = (vol - vol.min()) / (vol.max() - vol.min())
    elif nm == '11':
        vol = (vol - vol.min()) / (vol.max() - vol.min())
        vol = vol * 2 - 1
    elif nm == '11p':
        if norm_stats is None or key not in norm_stats:
            raise ValueError(f"nm='11p' needs norm_stats with key '{key}'")
        lo, hi = norm_stats[key]
        vol = ((vol - lo) / (hi - lo + 1e-8)).clip(0, 1) * 2 - 1
    elif nm == '11g':
        # '11g' expects input already in [-1, 1]; raw counts saturate the clip
        # to a constant +1 SILENTLY, so guard rather than return garbage (a
        # forgotten window -> [-1, 1] step is the usual cause).
        vlo, vhi = float(vol.min()), float(vol.max())
        if vlo < -1.5 or vhi > 1.5:
            raise ValueError(
                f"nm='11g' input out of range [{vlo:.3g}, {vhi:.3g}]; it must "
                "already be in [-1, 1]. Map raw values through the intensity "
                "window first (e.g. clip((v - lo) / (hi - lo), 0, 1) * 2 - 1).")
        vol = ((vol - gamma_lo) / (1 - gamma_lo)).clip(0, 1)
        vol = vol ** gamma * 2 - 1
    elif nm != '00':
        raise ValueError(f"Unknown nm '{nm}'")
    return vol


def invert_gamma(vol, gamma, gamma_lo):
    """Map an nm='11g' gamma-space volume back to the pre-gamma [-1, 1] scale.

    Exact inverse of normalize(..., '11g') above the --gamma_lo noise floor
    (the forward clip flattened the floor band, which returns as constant
    gamma_lo). Model outputs slightly outside [-1, 1] are clipped.
    """
    u = ((vol + 1) / 2).clip(0, 1) ** (1.0 / gamma)
    return u * (1 - gamma_lo) + gamma_lo
