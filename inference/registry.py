"""Model registry — named, ready-to-load model specs for inference.

Each entry pins everything needed to reproduce a model's inference behavior:
which checkpoint dir, which models/*.py implements the staged generation_test
for those weights, which epoch, and the normalization the run was trained with.

Fields (only `checkpoint` is required):
    checkpoint  path accepted by inference.load.resolve_checkpoint_dir
    model_file  repo-relative models/*.py overriding the run's source snapshot
                (omit to fall back to Engine's auto-map / the snapshot)
    epoch       checkpoint epoch (omit/None = latest complete one)
    nm          normalization mode ('00', '01', '11', '11p', '11g');
                omit to use the run's config.json
    gamma, gamma_lo
                nm='11g' parameters; omit to use the run's config.json

Usage:
    from inference import Engine
    eng = Engine.from_registry('skipU_qd')
    x = eng.normalize(vol)                    # vol: (Y, X, Z) in [-1, 1]
"""

import os

# Checkpoints live inside this repo (checkpoints/<name>/). The path is made
# absolute here because resolve_checkpoint_dir() calls abspath() BEFORE
# inference.load chdirs to the repo root, so a bare relative path would resolve
# against the caller's cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    # skipU_qd: the canonical THX10 codec (decision 2026-09-12). MScleanQD =
    # MSclean (multi-scale residual VQ, FPN netG ed023emsfpnu, patch_16 D) +
    # quantizer dropout (p_prefix 0.5, prefix L1+LPIPS vs pooled target,
    # adv_prefix 0) + VQ collapse fixes (--vq_normalize --vq_restart), trained
    # on thx10/roiD (192/24 crops, uprate 8, nm 11g gamma 0.7 / gamma_lo -0.8).
    # Epoch 500 = final checkpoint; the thx10codec_qd store and all reported
    # numbers were produced at this epoch. Weights: checkpoints/skipU_qd/
    # (see PROVENANCE.md there; original run dir recorded in its config.json).
    'skipU_qd': dict(
        checkpoint=os.path.join(_ROOT, 'checkpoints', 'skipU_qd'),
        model_file='models/MScleanQD.py',
        epoch=500,
        nm='11g', gamma=0.7, gamma_lo=-0.8,
    ),
}


def available():
    """Registered model names."""
    return sorted(MODELS)


def get(name):
    """The spec dict for `name` (a copy — mutating it does not edit the registry)."""
    if name not in MODELS:
        raise KeyError("Unknown model '%s'. Registered: %s" % (name, ', '.join(available())))
    return dict(MODELS[name])


def register(name, **spec):
    """Add or replace a registry entry at runtime (e.g. from a notebook)."""
    if 'checkpoint' not in spec:
        raise ValueError("spec needs at least 'checkpoint'")
    MODELS[name] = spec
