# checkpoints/skipU_qd — provenance

Epoch-500 weights of the canonical THX10 model (`--models MScleanQD`), copied on
2026-11-02 from

    /home/cheese/workspace/logs/THX10SDM20xw/thx10/MScleanQD/roiD192gfC/max5skip4/checkpoints/20260805_153500/

Only epoch 500 was copied (the run saved 0, 100, ..., 500 at `epoch_save=100`).
The seven `*_model_epoch_500.pth` files are EMA weights of the generator-side
modules (`encoder`, `decoder`, `quantizers`, `quant_convs`, `post_quant_convs`,
`inject_convs`, `net_g`); the discriminators were not saved (`save_d=false`).
They are git-ignored (`*.pth`, 61.7 MB) — re-copy from the path above, or
retrain with `run.sh` section A, to restore them.

Committed alongside: `config.json` (the full effective hyperparameters and the
exact launch command under `bash`), `aisr.yaml` (the config snapshot) and
`MScleanQD.py` (the model source snapshot; identical to `models/MScleanQD.py`
apart from a trailing newline).

The run trained on b200 (4 GPUs, `--env b200 --dataset thx10/`); the
`THX10SDM20xw` element of the logs path is from a later copy, not the run.
