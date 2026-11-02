"""Model registry — named, ready-to-load model specs for inference.

Each entry pins everything needed to reproduce a model's inference behavior:
which checkpoint (a timestamped run dir, its checkpoints/ parent, or the
experiment root), which models/*.py implements the staged generation_test for
those weights, which epoch, and the normalization the run was trained with.

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
    eng = Engine.from_registry('skipU')
    x = eng.normalize(vol)                    # vol: (Y, X, Z) in [-1, 1]
"""

MODELS = {
    # skipU: vqcleanM0aMSskipU recipe (skipE class + resize-conv netG
    # ed023emsfpnu), thx10 roiD192gfC/max5skip4. The newer of the two complete
    # runs in this experiment (both epochs 0-1100 at epoch_save=100); pinned to
    # the timestamped dir to disambiguate. Weights copied to this box under
    # /home/cheese/workspace/logs (2026-07-18).
    'skipU': dict(
        checkpoint='/home/cheese/workspace/logs/skipU/roiD192gfC/max5skip4/checkpoints/20260716_055927',
        model_file='models/MSclean.py',
        epoch=300,                  # 0-1100 available at epoch_save=100
        nm='11g', gamma=0.7, gamma_lo=-0.8,
    ),
    # sa635: MSclean run on the Chulab SA635 patch set (direction SA635/;
    # cropz 48 / cropsize 192 -> uprate 4, Z 64 -> 256). nm/gamma/gamma_lo
    # omitted so the Engine uses the run's own config.json (11g, gamma 0.8,
    # gamma_lo -0.85) — the single source of truth.
    'sa635': dict(
        checkpoint='/home/cheese/workspace/logs/sa635/MSclean/b4/checkpoints/20260720_150240',
        model_file='models/MSclean.py',
        epoch=100,
    ),
    # sa635_g03: retrain of the above with a lower training gamma to preserve
    # dim fine detail (config gamma 0.3, gamma_lo -0.9 vs the 0.8/-0.85 of
    # 'sa635'). nm/gamma/gamma_lo omitted -> Engine uses the run's config.json.
    'sa635_g03': dict(
        checkpoint='/home/cheese/workspace/logs/sa635/MSclean/b4/checkpoints/20260722_143233',
        model_file='models/MSclean.py',
        epoch=300,
    ),
    # vqclean: single-VQ baseline (ConvTranspose netG ed023e), trained as the
    # direct comparator to 'skipU' — same thx10 roiD192gfC recipe (nm 11g,
    # gamma 0.7/-0.8, 192/24, lamb 5, lr 5e-4, b200) minus the MS flags.
    # Weights copied to this box 2026-08-04; epochs 0-500 at epoch_save=100.
    'vqclean': dict(
        checkpoint='/home/cheese/workspace/logs/vqclean/roiD192gfC/max5skip4/checkpoints/20260804_163952',
        model_file='models/vqclean.py',
        epoch=300,                  # match the skipU comparison epoch
        nm='11g', gamma=0.7, gamma_lo=-0.8,
    ),
    # qdcut: MScleanQDCUT (MScleanQD2 + PatchNCE, q=generated so the NCE
    # gradient reaches net_g) on symmetricbead/CamA — winner of the 2026-08-10/11
    # lamb x lbNCE sweep (mlflow exp 30 run f12afb81: lamb 5, lbNCE 5, l1how dsp,
    # plain VQ + vq_restart, netG ed023emsfpnu). Picked on val_kid/val_lpips_pred/l1
    # at the epoch_save=100 checkpoints; epoch 500 was the composite best (its
    # ep400 is the fidelity-leaning fallback). Sibling 024953/024954 dirs are
    # empty DDP rank stubs; failed sweep runs moved to ../_trash_failed_runs.
    # nm/gamma omitted -> Engine uses the run's config.json (11g, 1.0, -1.0).
    'qdcut': dict(
        checkpoint='/home/cheese/workspace/logs/symmetricbead/scratch/QDCUTsmoke/checkpoints/20260811_024946',
        model_file='models/MScleanQDCUT.py',
        epoch=500,
    ),
    # skipU_qd: MScleanQD (MSclean + quantizer dropout: p_prefix 0.5, prefix
    # L1+LPIPS vs pooled target, adv_prefix 0) + the VQ collapse fixes
    # (--vq_normalize --vq_restart), same thx10 roiD192gfC recipe as 'skipU' —
    # trained so truncated-prefix decodes (E3) stop being out-of-distribution.
    # Pinned to the one timestamped dir with weights (siblings are aborted
    # stubs). Run stopped at epoch 500 (final checkpoint; epochs 0-500 at
    # epoch_save=100). CANONICAL THX10 codec (decision 2026-09-12): the
    # thx10codec_qd store, experiment_compression_QD and isotropic_evaluation_QD
    # were all produced at epoch 500, so that is the pin; pass epoch=300 to
    # match the skipU/vqclean comparison point.
    'skipU_qd': dict(
        checkpoint='/home/cheese/workspace/logs/THX10SDM20xw/thx10/MScleanQD/roiD192gfC/max5skip4/checkpoints/20260805_153500',
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
        raise KeyError(f"Unknown model '{name}'. Registered: {', '.join(available())}")
    return dict(MODELS[name])


def register(name, **spec):
    """Add or replace a registry entry at runtime (e.g. from a notebook)."""
    if 'checkpoint' not in spec:
        raise ValueError("spec needs at least 'checkpoint'")
    MODELS[name] = spec
