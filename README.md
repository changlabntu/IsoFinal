# IsoFinal

The minimal, self-contained version of IsoScopeXX that reproduces exactly two things:

- **A. Training** the canonical THX10 model `skipU_qd` (`--models MScleanQD`, epoch 500):
  self-supervised isotropic super-resolution + VQ compression of an anisotropic 3D volume.
- **B. Inference** on the whole THX10 volume: OME-Zarr → codec store (`encode_stack.py`) →
  isotropic OME-Zarr (`decode_stack.py`), with the latent-halo seam fix built in.

Nothing else from the parent repo (other model variants, registration, segmentation,
clustering, edge maps, TIFF-patch tooling) is included. The two commands live in `run.sh`.

## Layout

| path | role |
|---|---|
| `train.py`, `cfg/aisr.yaml`, `cfg/env.json` | training entry point, hyperparameter defaults, per-machine `DATASET`/`LOGS` roots |
| `models/base.py` → `MSclean.py` → `MScleanQD.py`, `models/quantize.py` | the Lightning model: base scaffold, multi-scale residual VQ recipe, quantizer dropout; the VQ with normalize + dead-code restart |
| `networks/registry.py`, `networks/EncoderDecoder/ed023eMSfpn.py`, `networks/cyclegan/models.py` | generator `ed023emsfpnu` (3D FPN U-Net), discriminator `patch_16` |
| `ldm/vqgan.yaml`, `ldm/modules/diffusionmodules/modelcut.py` | 2D VQ encoder/decoder architecture (vendored from latent-diffusion) |
| `taming/modules/autoencoder/lpips/vgg.pth` | LPIPS linear weights (the `taming` Python code comes from the pip package; do **not** add an `__init__.py` here) |
| `dataloader/data_multi.py` | 3D TIFF dataset with the `11g` normalization and albumentations crops |
| `inference/` | `engine.py` / `load.py` (checkpoint → `Engine`), `registry.py` (`skipU_qd`), `zarr_io.py`, `encode_stack.py`, `decode_stack.py`, `halo.py` |
| `checkpoints/skipU_qd/` | epoch-500 weights (git-ignored) + `config.json`, `aisr.yaml`, model snapshot, `PROVENANCE.md` |
| `tools/check_closure.py` | asserts every local import resolves inside this repo |

Tensor layout everywhere is `(B, C, Y, X, Z)`; TIFF pages are read as `(Z, Y, X)` and the zarr
stores are native `(z, x, y)`.

## Environment

One conda env, `isofinal` (python 3.10, torch 2.5.1+cu121, pytorch-lightning 1.9.5,
tensorstore), built as described at the top of `requirements.txt`. Pick the torch CUDA build
your driver supports: this box runs driver 535 (CUDA 12.2), so a cu130 torch reports
`cuda.is_available() == False` and training aborts with "ProcessGroupNCCL is only supported
with GPUs". All `inference/` code is also Python-3.8 compatible (it was validated in a
py3.8 / torch 2.4.1 env as well).

Two pretrained files are fetched by torch/torchmetrics on first use and are already cached
on this machine in `~/.cache/torch/hub/checkpoints/`: `vgg16-397923af.pth` (LPIPS backbone)
and `weights-inception-2015-12-05-6726825d.pth` (KID validation metric).

Always launch from the repo root: `ldm/vqgan.yaml`, `cfg/*`, the LPIPS weights and `out/`
are resolved relative to the working directory (the inference scripts `chdir` there themselves).

## A. Training

Data: `/home/cheese/workspace/Data/thx10_training_data.zip` → `unzip -n ... -d $DATASET/thx10`
gives `train/roiD/` (1000 TIFFs) and `val/roiD/` (73), each (48, 384, 384) float32.
`cfg/env.json["cheese"]` sets `DATASET=/home/cheese/workspace/Data/`, `LOGS=/home/cheese/workspace/logs/`.

```
CUDA_VISIBLE_DEVICES=0,1 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr \
  --prj thx10/MScleanQD/roiD192gfC/max5skip4 --env cheese --dataset thx10/ --direction roiD/ \
  --nm 11g --gamma 0.7 --gamma_lo -0.8 --cropsize 192 --cropz 24 --dsp 1 --lamb 5 \
  --models MScleanQD --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 \
  --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri isofinal \
  -b 3 --vq_normalize --vq_restart --n_embed_scales 128,128,256,512
```

This is the original command (`checkpoints/skipU_qd/config.json["bash"]`) with two changes:
`--env cheese` instead of `b200`, and a local MLflow name instead of the remote server
(`--tracking_uri isofinal` → `$LOGS/isofinal.db` + `$LOGS/isofinal/`). Effective batch is
`2 × b × GPUs` (`accumulate_grad_batches=2` is hardcoded); the original used 4 GPUs. There is
no seeding, so a rerun reproduces the recipe, not the run bit-for-bit.

**GPU memory.** The 192 × 192 × 24 recipe needs more than 24 GB per GPU: on this box's
RTX 3090s the first training step allocates 22.5 GB at `-b 1` and fails. Train it on a
large-memory GPU (the original: b200, ~180 GB, `-b 3`). Validation needs even more
(it decodes the full 384 × 384 × 48 val volumes to 384³); on GPUs where only validation
fails, `--skip_val` disables it (monitoring metrics only, training losses and checkpoints
unchanged). The pipeline itself was verified end to end on a 3090 with a reduced crop,
`--cropsize 128 --cropz 16 --skip_val -b 1` (same uprate 8): one epoch of 1000 steps,
checkpoint with all seven module files, MLflow metrics, no weight downloads.

Checkpoints land in `$LOGS/thx10/thx10/MScleanQD/roiD192gfC/max5skip4/checkpoints/<timestamp>/`
every `epoch_save=100` epochs; the canonical model is epoch 500. To use a new run for
inference, copy its epoch-N files into `checkpoints/<name>/` and add an entry to
`inference/registry.py`.

## B. Inference (zarr → codec → isotropic zarr)

Input: `/media/cheese/Ghc_data3/THX10/thx10cropzarr` (OME-NGFF 0.5, zarr v3, axes z,x,y,
level 0 = 816 × 14208 × 8448 uint16, chunks 48 × 384 × 384).

**B1 encode** (fp32 — never `--half` here, fp16 flips 8–23 % of the codebook indices):
```
CUDA_VISIBLE_DEVICES=0 python inference/encode_stack.py --model skipU_qd --exp thx10codec_qd \
  --zarr /media/cheese/Ghc_data3/THX10/thx10cropzarr --out_base /media/cheese/Ghc_data3/THX10 \
  --patch 384 --batch 2 --window 113 588
```
Writes `<out_base>/thx10codec_qd/codec/zBBBB-EEEE/{rrr}{ccc}.npz` (one 384 × 384 × 48 cell each,
four per-scale index arrays) plus a `norm_params.json` per z-chunk recording the intensity
window, model/epoch and `uprate`. The canonical store already exists at that path.

**B2 decode** (whole volume, latent halo on, one MC draw per cell):
```
CUDA_VISIBLE_DEVICES=0 python inference/decode_stack.py \
  --codec /media/cheese/Ghc_data3/THX10/thx10codec_qd/codec \
  --zarr /media/cheese/Ghc_data3/THX10/thx10iso_qd_halo4.zarr \
  --what decode --half --batch 1 --halo 4 --zhalo 4
```
Output: one native `(z, x, y)` uint16 OME-Zarr of 6528 × 14208 × 8448 (Z × 8), chunks 384³,
restored to source intensity units, plus `decode_params.json`. ROI runs: `--chunk z0384-0432
--cells 9 14 16 22 --crop` (a small, napari-friendly store); `--orig` writes the trilinear
Z-upsampled source alongside.

Options that matter:
- `--halo H --zhalo HZ` (default 4/4): each cell's per-scale latents are stitched from the
  3 × 3 × 3 neighbouring cells' stored codes, decoded at (384 + 16H)² × (384 + 16HZ) and cropped
  back to the cell, so tile faces continue into the neighbour (`inference/halo.py`). Interior
  voxels are unchanged. `--halo 0 --zhalo 0` is the legacy cell-by-cell decode. Cost ≈ 1.49×.
  A 448³ fp16 decode needs ~14 GB: `--half --batch 1` on a 24 GB card.
- `--deterministic`: running-stat BatchNorm and dropout off. The default runs the generator
  components in `.train()` (batch-stat BN + `Dropout(0.5)` live in two decoder blocks), one
  MC draw per cell. Note that independent draws per cell can leave a faint stochastic texture
  step at the faces even with the halo; `--deterministic` is the exactly seam-free setting.
- Encoding is deterministic in both modes (the encoder has no dropout and the quantizer's
  restart logic is gated on gradients).

## Verification recipes

1. Imports and closure (both envs): `python tools/check_closure.py` and
   `python -c "import models.MScleanQD, networks.registry, inference"`.
2. Engine parity: encode one cell with `Engine.from_registry('skipU_qd')` in fp32 and compare
   the indices with the stored npz (expect 100 % equal); decode with `--halo 0 --zhalo 0
   --deterministic` and compare against the parent repo's output (expect bit-identical).
3. Seam test: decode a 2 × 2 × 2 block with and without the halo (`--crop`), check that the
   central 320³ of each cell is unchanged and that the face discontinuity drops to the level
   of neighbouring interior planes.
4. Training smoke test on a 24 GB GPU: the command in A with `--prj scratch/smoke --cropsize 128
   --cropz 16 -b 1 --skip_val --n_epochs 1 --epoch_save 1` (the full recipe needs a larger GPU).
