# Rebuilding the `synthF` dataset on another machine

The dataset is not in git (16 GB). It is generated from two committed files,
`synthetic/config/phantom.yaml` (the scenes) and `synthetic/config/degradation.json` (blur,
noise, window), and fixed seeds, so it can be rebuilt anywhere.

## Needs

- This repo on branch `synthetic`, and the `isofinal` env (numpy, scipy, tifffile, pyyaml).
- No GPU, no THX10 data. (Re-estimating `degradation.json` would need the THX10 zarr; a
  rebuild does not.)
- 8 CPU cores, about 5 GB RAM per worker (4.6 GB measured), 16 GB disk.

## Build

From the repo root:

```
export ENV_NAME=<entry>              # the machine's entry of cfg/env.json (or: export DATASET=/path/to/Data/)
python -m synthetic.phantom.build_dataset --name synthF --n_train 400 --n_val 24 --workers 8 \
    --variants F,A,F,B,F,F,A,F,C,F
```

About 40 minutes with 8 workers (each uses 4 FFT threads). Without `ENV_NAME` or `DATASET`
the script falls back to the `cheese` entry of `cfg/env.json`. Below, `$DATASET` stands for
the `DATASET` folder of the entry.

Result:

```
$DATASET/synthF/
  train/deg/*.tif        400 stacks (48, 384, 384) float32 in [-1, 1], blur width 0.96-1.19
  val/deg/*.tif          24 held-out blocks, blur width 1.07
  val_narrow/deg/*.tif   the same blocks at 0.96
  val_wide/deg/*.tif     the same blocks at 1.19
  gt/val/*.tif, *.npz    their isotropic truth (384^3 float16) and fibre geometry
  manifest.json          seed, variant, width and scene parameters of every block
```

`--variants` sets the mix: block i gets entry `i % 10`, i.e. 60 % F (fibre-dominated),
20 % A, 10 % B, 10 % C. Training blocks use seeds 100000 + i, held-out blocks 900000 + i.

## Check

1. File counts:

   ```
   for d in train/deg val/deg val_narrow/deg val_wide/deg gt/val; do echo $d $(ls $DATASET/synthF/$d | wc -l); done
   # 400 24 24 24 48
   ```

2. Checksums. On the machine where the dataset was first built, a rebuild is bit-identical.
   `md5sum FILE | cut -c1-12` there gives:

   | file | md5 (first 12) |
   |---|---|
   | `train/deg/s00000.tif` | `45075d9f2d88` |
   | `train/deg/s00399.tif` | `f27e0785f8d5` |
   | `val/deg/v00000.tif` | `16a78c1eb5a1` |
   | `val/deg/v00001.tif` | `cc175c02a7b9` |
   | `gt/val/v00000.tif` | `746c972ca25b` |
   | `gt/val/v00001.tif` | `af31b5223fe1` |

3. If the checksums differ (another CPU or FFT library can round differently, which may move
   a voxel by one count), check the content instead. Scoring cubic interpolation needs no GPU
   and takes about 8 minutes:

   ```
   python -m synthetic.evaluate run --method cubic
   ```

   Expected (psnr / corr / z_res / false / recall):

   | set | psnr | corr | z_res | false | recall |
   |---|---|---|---|---|---|
   | `val` | 21.05 | 0.703 | 17.2 | 0.077 | 0.928 |
   | `val_narrow` | 21.35 | 0.727 | 15.5 | 0.065 | 0.940 |
   | `val_wide` | 20.77 | 0.679 | 18.9 | 0.089 | 0.915 |

## Use

With the entry for the machine in `cfg/env.json` (its `DATASET` and `LOGS` folders) and
`ENV_NAME` exported:

- **Train:** the unchanged `train.py` with `--dataset synthF/ --direction deg/ --env ENTRY`.
  The proxy recipe: `bash synthetic/sweep/proxy.sh MScleanQD/NAME GPU EPOCHS`; arguments
  after `EPOCHS` override the recipe (e.g. `--cropsize 192 --cropz 24`).
- **Score:** `python -m synthetic.evaluate run --method MScleanQD/NAME --epoch N`; the run is
  looked up under `$LOGS/synthF/synthF/`.

`README.md` has the baseline numbers (Stage 3, "Proxy baseline") and the commands for the
full-crop run ("Full recipe").

See `README.md` (Stage 3) for the meaning of the scores.
