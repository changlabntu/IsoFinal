# synthetic/ — ground-truth validation with a synthetic volume

THX10 has no isotropic ground truth. This folder builds one synthetically:

1. **Estimate** the THX10 degradation (blur along Z, Z sampling, noise) from the anisotropic data.
2. **Generate** isotropic phantom blocks with known truth, and **degrade** them the same way.
3. **Train** the unchanged IsoFinal model on the degraded blocks and **score** its output against truth.
4. **Sweep** hyperparameters / designs on the phantom, then train THX10 with the winner.

Nothing outside `synthetic/` is modified. Run everything from the repo root in the `isofinal`
env, as modules (`python -m synthetic....`). Figures and caches go to `synthetic/out/`
(git-ignored); the committed results are `config/degradation.json` and `config/phantom.yaml`.

Conventions: arrays are `(Z, Y, X)`; lengths are in XY pixels; one Z slice = 8 XY pixels;
frequencies are in cycles per XY pixel (Z Nyquist = 1/16); intensities are raw camera counts
unless a name says "window" (`[-1, 1]`, the training TIFF units).

## Status

| Stage | State |
|---|---|
| 1. Degradation estimate | done |
| 2. Phantom, blur-width calibration, dataset, realism | done, **awaiting review** (gate) |
| 3. Training / evaluation harness | not started |
| 4. Sweep | not started |
| 5. Confirm on the A6000 machine, train THX10 | not started |

## Stage 1 — degradation estimate

```
python -m synthetic.estimate.blocks --per_band 6   # pick fibre-rich blocks (192 x 768 x 768)
python -m synthetic.estimate.transfer              # main model: non-separable transfer
python -m synthetic.estimate.spectra               # 1D shell-averaged summary
python -m synthetic.estimate.lsf                   # independent check on single fibres
python -m synthetic.estimate.noise                 # offset, flicker, jitter
python -m synthetic.estimate.noise_shape           # gain, read noise, fixed pattern, in-plane resampling
python -m synthetic.estimate.window                # intensity window of the training TIFFs
python -m synthetic.estimate.calibrate_width       # bias-corrected blur width (stage 2c, needs the phantom)
python -m synthetic.estimate.report                # cross-check + config/degradation.json
```

| file | what it measures |
|---|---|
| `estimate/transfer.py` | `R(f_r, fz)`: power transfer from "isotropic, as sharp as XY" to the data, from the 3D power spectrum. Assumes only that the tissue is statistically isotropic. |
| `estimate/spectra.py` | the same, averaged over shells into one 1D kernel (for orientation; not used by the pipeline) |
| `estimate/lsf.py` | lateral and axial profiles of individual in-plane fibres; no isotropy assumption |
| `estimate/noise.py` | offset, per-slice flicker and jitter (its gain is superseded, see below) |
| `estimate/noise_shape.py` | camera gain and read noise, fixed pattern, and the in-plane resampling of the volume, from the part of the spectrum that holds only noise |
| `estimate/window.py` | locates a training TIFF in the zarr and fits its raw-count window |
| `estimate/calibrate_width.py` | runs both blur estimators on phantom blocks degraded at known widths |
| `estimate/report.py` | per-fibre comparison of the two methods, writes `config/degradation.json` |

### Blur (18 blocks, three depth bands)

- **The blur is not a 1D kernel along Z.** Fine lateral detail is sectioned better than coarse
  detail: `R = exp(-(|fz| / f0)^p)` with `f0 = 0.0060 + 0.043 f_r`, `p = 1.39`. The equivalent
  axial kernel is about 41 px FWHM for coarse structure (`f_r` = 0.01), 36 px at 0.03 and
  25 px at 0.10, i.e. 3 to 5 slices. The kernel has heavier tails than a Gaussian (`p` < 2).
- It is stable: per-block `a` 0.0050-0.0073, `b` 0.034-0.050, and the three depth bands agree
  within 5 %.
- The Z stack is oversampled relative to this blur (kernel 3-5 slices wide), so the task the
  model faces is mostly axial *deblurring*, not only interpolation.
- Not verifiable from this data: the transfer above the Z Nyquist (extrapolated by the model).

### Cross-check of the two blur estimates

For 350 fibres the transfer model predicts the axial width each should have from its lateral
width. Measured / predicted: median 1.24, quartiles 1.14-1.43; pooled profile 49 px measured
against 40 px predicted. The disagreement is resolved by simulation in stage 2c.

### Noise

- Offset 101 counts. Flicker (0.4 %) and jitter (0.04 px) between slices are negligible.
- **The volume was resampled in-plane.** Noise is smoothed along X and Y by an amount that
  differs from stack to stack and clusters by tile: the signature of linear interpolation at a
  sub-pixel shift (fraction `w` per axis, measured 0.03-0.47, median 0.22).
- Before that resampling the slice-independent noise is white, with variance
  `1.40^2 + 1.39 x signal`. A pattern fixed to the pixel grid (identical in every slice) adds
  `0.74 + 0.145 x min(signal, 44)`.
- An earlier estimate (gain 0.83, isotropic noise correlation) assumed one correlation model
  for all stacks and is superseded. The estimator in `noise_shape.py` reads the noise from Z
  frequencies that the blur leaves empty of structure, and returns the same numbers on
  synthetic stacks as on real ones (gain 1.38 against 1.39).

### Intensity window

The training TIFFs are `(raw - 99) / (673 - 99) * 2 - 1`, recovered exactly (fit residual
~1e-8) by locating two TIFFs in the zarr. Inference (`encode_stack.py --window 113 588`) uses
a different window, a 1.21x contrast stretch relative to training; left as is for now.

## Stage 2 — phantom, blur width, dataset, realism

```
python -m synthetic.realism stats --real                          # statistics of 200 real training stacks
python -m synthetic.phantom.build_dataset --preview 36 --tag r    # calibration set (out/phantom/preview_r)
python -m synthetic.realism compare --synth DIR --tag r           # table, figure, montage
python -m synthetic.estimate.noise_shape --synth DIR              # noise estimator on synthetic stacks
python -m synthetic.estimate.calibrate_width --n_blocks 6         # 2c, then rerun estimate.report
python -m synthetic.phantom.build_dataset --name synthA --n_train 400 --n_val 24
python -m synthetic.realism classify --synth $DATASET/synthA/train/deg
python -m synthetic.realism headroom --root $DATASET/synthA
python -m synthetic.modelcheck codebook --synth $DATASET/synthA/train/deg
python -m synthetic.modelcheck preview  --root $DATASET/synthA
```

| file | role |
|---|---|
| `config/phantom.yaml` | phantom parameters and their per-block priors; three variants (A mixed, B fibre-dominated, C cell-dominated) |
| `phantom/generate.py` | `make_block(size, seed, params)`: Voronoi cells with dark interiors and patchy walls, tortuous branching fibres with varicosities, bright bodies, fine grain; isotropic, noise-free |
| `phantom/degrade.py` | transfer filter, every 8th slice, camera noise, fixed pattern, in-plane resampling, window |
| `phantom/build_dataset.py` | writes the dataset tree (below) |
| `realism.py` | the same statistics on real and synthetic stacks; comparison table, figure, montage; headroom |
| `classify.py` | real-vs-synthetic CNN on XY and XZ patches |
| `modelcheck.py` | the existing `skipU_qd` checkpoint on synthetic stacks: codebook usage, input / output / truth preview |

### Dataset layout

```
$DATASET/synthA/
  train/deg/*.tif        (48, 384, 384) float32 in [-1, 1]; blur width drawn per block from the range
  val/deg/*.tif          held-out blocks, nominal width
  val_narrow/deg/*.tif   the same blocks at the narrow end of the range
  val_wide/deg/*.tif     the same blocks at the wide end
  gt/val/*.tif           (384, 384, 384) float16, counts above offset; truth plane 8 i = slice i
  gt/val/*.npz           fibre centrelines, radii, brightness, bodies
  manifest.json          seed, variant, width and scene parameters of every block
```

Truth lives outside `train/` and `val/`, so the training loader cannot reach it. A block is
rendered with a margin (96 px in Z, 48 px in XY) that is blurred with it and then cropped, so
slices near the faces receive light from structure outside the block, as real data does.

### 2c — blur width, settled by simulation

Both estimators were run on 6 phantom blocks (192 x 768 x 768) degraded at five known width
scales (the fitted transfer with `f0` divided by the scale). Both respond linearly:

| true scale | spectral fit: kernel width / real fit | fibres: measured / predicted (real: 1.24) |
|---|---|---|
| 0.85 | 0.87 | 0.93 |
| 1.00 | 1.03 | 1.07 |
| 1.15 | 1.18 | 1.22 |
| 1.30 | 1.34 | 1.36 |
| 1.50 | 1.54 | 1.54 |

- At a known width the spectral estimator reads 3 % wide and the fibre estimator 7 % wide, so
  estimator bias explains only a small part of the 24 % seen on real data.
- The scale that reproduces the real reading is **0.97 +- 0.01 for the spectral method and
  1.17 +- 0.02 for the fibre method**. The two still disagree by about 20 %. Either real
  fibres are widened by something the phantom lacks (bundles, overlap), or real tissue is not
  statistically isotropic, which the spectral method assumes. This data cannot say which.
- The dataset therefore spans both: **nominal 1.07, range 0.96-1.19** (`width` in
  `config/degradation.json`). Training blocks draw a width from the range; held-out blocks
  exist at 0.96, 1.07 and 1.19.
- The calibration was run before the last narrowing of the phantom's per-block priors; it uses
  its own dense-scene settings, so it was not repeated.

### Realism of `synthA` (400 training stacks; compared with 200 real training stacks)

| check | real | synthetic |
|---|---|---|
| per-stack mean, median over stacks (counts) | 146 | 140 |
| per-stack 99th percentile, median | 342 | 289 |
| fraction of voxels above 250 counts, median | 3.2 % | 1.8 % |
| two-sample KS distance of the per-stack statistics | | 0.09-0.27 |
| XY power spectrum, synthetic / real, 0.008-0.4 cycles/px | | 0.71-1.04 |
| slice-to-slice correlation, lag 1 / 4 / 8 | 0.941 / 0.596 / 0.265 | 0.943 / 0.599 / 0.242 |
| transfer fitted on 48-slice stacks (`a`, `b`, `p`) | 0.0067, 0.046, 1.50 | 0.0065, 0.040, 1.56 |
| noise estimator (read std, gain) | 1.40, 1.391 | 1.37, 1.386 |
| codebook usage of `skipU_qd`, JS divergence per scale | 0.009-0.020 (real vs real) | 0.013-0.028 (real vs synthetic) |

Classifier (120 stacks per source, tested on stacks it never saw; real-vs-real control
0.45-0.48):

| view | AUC | AUC on the model's input (below 156 counts flattened) |
|---|---|---|
| XY, full resolution | 0.72 | 0.87 |
| XY, 2x2 averaged | 0.86 | 0.92 |
| XY, 4x4 averaged | 0.91 | 0.96 |
| XZ | 0.80 | 0.94 |

**The synthetic stacks match the real ones in their first- and second-order statistics, but a
CNN can tell them apart**, more easily at coarse scale. What it picks up is morphology: the
shapes of procedural cells, fibres and bodies are not those of real tissue. This is above the
plan's "investigate above 0.65" line and is a limit of a procedural phantom. Its consequence is
that a ranking of designs on `synthA` must be confirmed on real data (the stage 5 transfer
check) before it is trusted.

### 2d — headroom (24 held-out blocks)

The blur spreads a thin structure over many planes, so the isotropic truth is brighter than
the data it produces:

- 99.9th percentile above offset: truth 2.2x the data (998 against 522 counts); the brightest
  truth voxel is 4819 counts, 8x the window ceiling (673).
- 0.36 % of truth voxels lie above the ceiling; among voxels above the model's input floor
  (156 counts) it is 2.7 %, carrying 6.8 % of the above-floor signal.

So a model whose output is confined to the data's window cannot reproduce the brightest
structures of the truth. Stage 3 scores therefore have to state how truth is clipped.

### Existing model on synthetic input

`out/realism/model_preview_synthA.png` shows input, `skipU_qd` output and truth for held-out
synthetic crops, and input and output for real crops. On both, the output is sharper in XZ
than the input but much dimmer and sparser; against the truth it misses most dim structure.
The baseline input mapping (`--gamma_lo -0.8`) discards everything below 156 counts, which is
about three quarters of the voxels in a typical stack.

### Verification

- `python tools/check_closure.py` passes; `import synthetic` works in the `isofinal` env.
- A degraded stack has the shape, dtype, range and 1/287 quantisation of a real training TIFF
  and loads through `dataloader/data_multi.py` (train and val splits) exactly as `thx10` does.
- Both blur estimators recover a known width on the phantom within 3 % and 7 % (table above).
- The noise estimator returns the camera parameters put into the phantom (gain 1.386 for
  1.391, read noise 1.37 for 1.40) and the same resampling distribution.

## Decisions at the stage 2 gate

1. Is the phantom close enough to proceed, given that a classifier separates it from real data
   by morphology?
2. Blur width: accept nominal 1.07 with range 0.96-1.19, or first measure it on a sub-block of
   the dense Z stack?
3. How should truth be scored given the headroom: clipped to the data window, or with a wider
   output range treated as a design option?
