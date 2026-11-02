#!/usr/bin/env bash
# IsoFinal — the two canonical THX10 commands. Run from this directory.
# Env: conda activate isofinal (python 3.10, torch 2.5.1+cu121, tensorstore) for every step

# 0. data (once, ~30 GB unpacked) -> $DATASET/thx10/{train,val}/roiD/
unzip -n /home/cheese/workspace/Data/thx10_training_data.zip -d /home/cheese/workspace/Data/thx10

# A. training. Original run: 4 GPUs on b200 (--env b200, remote mlflow). NEEDS > 24 GB per GPU: on this
#    box's 3090s the first step allocates 22.5 GB at -b 1 and OOMs, so run this on a large-memory GPU.
#    Validation needs even more (full 384^3 val decode) -> --skip_val disables it (monitoring only).
#    MLflow -> $LOGS/isofinal.db (view: mlflow server --backend-store-uri sqlite:////home/cheese/workspace/logs/isofinal.db --artifacts-destination /home/cheese/workspace/logs/isofinal)
#    24 GB smoke test of the pipeline (not the recipe): add --prj scratch/smoke --cropsize 128 --cropz 16 -b 1 --skip_val --n_epochs 1 --epoch_save 1
CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/MScleanQD/roiD192gfC/max5skip4 --env cheese --dataset thx10/ --direction roiD/ --nm 11g --gamma 0.7 --gamma_lo -0.8 --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models MScleanQD --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri isofinal -b 3 --vq_normalize --vq_restart --n_embed_scales 128,128,256,512

# B1. encode the THX10 OME-Zarr to the codec store. fp32 on purpose: --half flips 8-23 % of indices.
#     The canonical store /media/cheese/Ghc_data3/THX10/thx10codec_qd/codec already exists; rerun only to regenerate it.
CUDA_VISIBLE_DEVICES=0 python inference/encode_stack.py --model skipU_qd --exp thx10codec_qd --zarr /media/cheese/Ghc_data3/THX10/thx10cropzarr --out_base /media/cheese/Ghc_data3/THX10 --patch 384 --batch 2 --window 113 588

# B2. decode the whole volume to an isotropic OME-Zarr with the latent halo (~12 h on one 3090)
CUDA_VISIBLE_DEVICES=0 python inference/decode_stack.py --codec /media/cheese/Ghc_data3/THX10/thx10codec_qd/codec --zarr /media/cheese/Ghc_data3/THX10/thx10iso_qd_halo4.zarr --what decode --half --batch 1 --halo 4 --zhalo 4
