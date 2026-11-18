#!/bin/bash
# One proxy training run on synthF with the unchanged train.py.
#
#   synthetic/sweep/proxy.sh NAME GPU EPOCHS [train.py arguments that override the baseline]
#   e.g.  synthetic/sweep/proxy.sh MScleanQD/base_r1 0 100
#         synthetic/sweep/proxy.sh MScleanQD/mean_r1 1 60 --l1how mean
#
# The recipe is the README baseline (the skipU_qd command) with the crop reduced to
# 128 x 128 x 16 and batch 1 so that it fits a 24 GB GPU. --n_epochs is left at its
# default on purpose: the cosine schedule spans n_epochs, and the default keeps the
# learning rate flat as in the original run. The run is stopped here once the
# checkpoint of epoch EPOCHS exists (checkpoints are written every 10 epochs).
#
# Run inside the isofinal env. Checkpoints: $LOGS/synthF/synthF/NAME/checkpoints/<time>/
set -u
NAME=$1; GPU=$2; EPOCHS=$3; shift 3
cd "$(dirname "$0")/../.."
LOGS=${LOGS:-/home/cheese/workspace/logs/}
RUN=$LOGS/synthF/synthF/$NAME
LOG=synthetic/out/logs/train_${NAME//\//_}.log
mkdir -p synthetic/out/logs
CUDA_VISIBLE_DEVICES=$GPU NO_ALBUMENTATIONS_UPDATE=1 python -u train.py --yaml aisr \
  --prj synthF/$NAME --env cheese --dataset synthF/ --direction deg/ \
  --nm 11g --gamma 0.7 --gamma_lo -0.8 --cropsize 128 --cropz 16 --dsp 1 --lamb 5 \
  --models MScleanQD --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 \
  --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri synth \
  -b 1 --vq_normalize --vq_restart --n_embed_scales 128,128,256,512 --skip_val --epoch_save 10 \
  "$@" > "$LOG" 2>&1 &
PID=$!
until ls "$RUN"/checkpoints/*/net_g_model_epoch_"$EPOCHS".pth > /dev/null 2>&1 || ! kill -0 $PID 2> /dev/null; do
  sleep 60
done
sleep 60                      # let the other component files of that epoch be written
pkill -P $PID 2> /dev/null
kill $PID 2> /dev/null
echo "stopped $NAME at epoch $EPOCHS" >> "$LOG"
