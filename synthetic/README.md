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
- The fit only sees frequencies where structure is above the noise. At the highest axial
  frequencies the real data keeps slightly more structure than this model predicts (stage 2,
  "Noise band").

### Cross-check of the two blur estimates

For 350 fibres the transfer model predicts the axial width each should have from its lateral
width. Measured / predicted: median 1.24, quartiles 1.14-1.43; pooled profile 49 px measured
against 40 px predicted. The disagreement is resolved by simulation in stage 2c.

### Noise

- Offset 101 counts. Flicker (0.4 %) and jitter (0.04 px) between slices are negligible.
- **The volume was resampled in-plane.** Noise is smoothed along X and Y: the signature of
  linear interpolation at a sub-pixel shift (fraction `w` per axis, measured 0.03-0.47, median
  0.22). The pair (`wx`, `wy`) is one constant per XY tile, identical at every depth, and the
  two are uncorrelated across tiles. `degradation.json` stores the 200 measured pairs and the
  phantom draws one pair per block.
- Before that resampling the slice-independent noise is white, with variance
  `1.40^2 + 1.39 x signal`. A pattern fixed to the pixel grid (identical in every slice) adds
  `0.74 + 0.145 x min(signal, 44)`.
- An earlier estimate (gain 0.83, isotropic noise correlation) assumed one correlation model
  for all stacks and is superseded. The estimator in `noise_shape.py` reads the noise from Z
  frequencies that the blur leaves empty of structure, and returns the same numbers on
  synthetic stacks as on real ones (gain 1.39 against 1.39, read noise 1.30 against 1.40).

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
python -m synthetic.phantom.build_dataset --name synthF --variants F,A,F,B,F,F,A,F,C,F
python -m synthetic.realism classify --synth $DATASET/synthF/train/deg [--only F] [--model_domain]
python -m synthetic.realism noiseband [--synth $DATASET/synthF/train/deg]   # noise band by signal level
python -m synthetic.realism headroom --root $DATASET/synthF
python -m synthetic.modelcheck codebook --synth $DATASET/synthF/train/deg
python -m synthetic.modelcheck preview  --root $DATASET/synthF
python -m synthetic.phantom.export --size 512 768 768 --seed 2 --variant F --blurred   # TIFFs to look at
```

| file | role |
|---|---|
| `config/phantom.yaml` | phantom parameters and their per-block priors; four variants (below) |
| `phantom/generate.py` | `make_block(size, seed, params)`: Voronoi cells with dark interiors and patchy walls, branching fibres with varicosities, bright bodies, fine grain; isotropic, noise-free |
| `phantom/degrade.py` | transfer filter, every 8th slice, camera noise, fixed pattern, in-plane resampling, window |
| `phantom/build_dataset.py` | writes the dataset tree (below) |
| `phantom/export.py` | one phantom volume as TIFFs (truth, blurred, degraded) plus a preview image, in `out/inspect/` |
| `realism.py` | the same statistics on real and synthetic stacks; comparison table, figure, montage; headroom |
| `classify.py` | real-vs-synthetic CNN on XY and XZ patches and on the noise band |
| `modelcheck.py` | the existing `skipU_qd` checkpoint on synthetic stacks: codebook usage, input / output / truth preview |

### Phantom variants

| variant | content |
|---|---|
| F | **fibre-dominated**: long smooth fibres (300-3000 px, 15 % thick processes) with varicosities, brightness varying by region, few bodies, faint cell outlines. Fibres that leave the block re-enter at the opposite face, so the density is uniform and fibres are not cut short. |
| A | mixed: cell outlines, short tortuous fibres (60-700 px), bright bodies |
| B | as A with more fibres and weaker cell outlines |
| C | cell-dominated, few fibres |

Each variant was calibrated on its own against the real per-stack statistics. `synthF` is
60 % F, 20 % A, 10 % B, 10 % C. An earlier dataset, `synthA` (A, B, C in equal parts), predates
variant F and the paired resampling fractions; it is superseded.

### Dataset layout

```
$DATASET/synthF/
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
- The calibration was run on variant A scenes with its own dense-scene settings, before the
  last narrowing of the per-block priors and before variant F existed. It was not repeated.

### Realism of `synthF` (400 training stacks; compared with 400 real training stacks)

| check | real | synthetic |
|---|---|---|
| per-stack mean, median over stacks (counts) | 147 | 144 |
| per-stack 99th percentile, median | 342 | 312 |
| per-stack 99.9th percentile, median | 506 | 573 |
| fraction of voxels above 250 counts, median | 3.1 % | 2.2 % |
| two-sample KS distance of the per-stack statistics | | 0.05-0.26 |
| XY power spectrum, synthetic / real, 0.008-0.4 cycles/px | | 0.94-1.30 |
| slice-to-slice correlation, lag 1 / 4 / 8 | 0.941 / 0.598 / 0.263 | 0.945 / 0.603 / 0.248 |
| transfer fitted on 48-slice stacks (`a`, `b`, `p`) | 0.0066, 0.046, 1.50 | 0.0065, 0.041, 1.54 |
| noise estimator (read std, gain) | 1.40, 1.391 | 1.30, 1.389 |
| fixed pattern as read by the estimator | 0.74 + 0.145 min(S, 44) | 0.99 + 0.103 min(S, 47) |
| codebook usage of `skipU_qd`, JS divergence per scale | 0.009-0.020 (real vs real) | 0.012-0.024 (real vs synthetic) |

The F blocks alone give the same picture (KS 0.05-0.28, spectrum ratio 0.94-1.36). Known
residuals: the synthetic stacks have somewhat brighter extremes (99.9th percentile, and
0.05 % against 0.00 % of voxels at the window ceiling in the median stack) and about 30 % more
power at 0.008-0.02 cycles/px.

Classifier (120 stacks per source, 70 / 30 split by stack, tested on stacks it never saw):

| view | real vs real (control) | `synthF` | F blocks only | `synthF`, model's input | F only, model's input |
|---|---|---|---|---|---|
| XY, full resolution | 0.56 | 0.97 | 0.98 | 0.90 | 0.92 |
| XY, 2x2 averaged | 0.53 | 0.99 | 0.99 | 0.95 | 0.97 |
| XY, 4x4 averaged | 0.47 | 0.99 | 0.99 | 0.98 | 0.98 |
| XZ | 0.55 | 0.98 | 0.97 | 0.95 | 0.94 |
| noise band only (see below) | 0.51 | 0.88 | 0.90 | | |

"Model's input" applies the baseline input mapping first (below 156 counts flattened).

**The synthetic stacks match the real ones in their first- and second-order statistics, but a
CNN tells them apart almost every time.** Variant F is no closer to real than A, B or C by
this test (`synthA` scores 0.97-0.99 and 0.89-0.98), and calibrating statistics did not move
it. No single patch statistic explains the separation (about 30 were tried: brightness
percentiles, moments, gradients, texture strength at several scales; each alone gives AUC
0.38-0.60), so the main cue is the shape of the structures: procedural fibres, cells and
bodies are not those of real tissue. This is far above the plan's "investigate above 0.65"
line and is a limit of a procedural phantom. A ranking of designs on `synthF` therefore has to
be confirmed on real data (the stage 5 transfer check) before it is trusted.

An earlier version of this table reported 0.72-0.91. Those numbers were wrong: the real stacks
were split into training and test sets by file name, which is position in the volume, and the
test end is dimmer. The split is now random.

**Noise band.** High-passing a stack along Z (above 0.3 cycles/slice) leaves the noise and
almost no structure, so this view tests the degradation model rather than the morphology. It
separates as well. On 400 stacks per source (`realism noiseband`): 0.99 in empty 64 x 64
tiles (below 15 counts above offset), 0.92-0.96 in tiles with signal, 0.85 after dividing out
the local noise level; real-vs-real controls 0.47-0.53. Three differences were measured in
that band and none is modelled. The first two are a few percent of the band's noise power or
less; the third is a matter of which pairs go with which content:

- Real stacks keep 3-4x more correlated content at 2-4 px (lag-2 correlation 0.02-0.06
  against 0.00-0.02, growing with signal, equal in X and Y). Most likely this is structure
  that the real blur passes at the highest axial frequencies and the fitted transfer does
  not: about 1 count rms on signals of 80-400 counts.
- Real camera noise has a row-wise component: in empty regions the correlation along X stays
  near 0.01 out to at least 24 px. The phantom has none.
- In the real volume the resampling fractions depend on position, and so does content: empty
  regions lie mostly in tiles with a large X and a small Y fraction (lag-1 correlation
  0.48 / 0.13, against 0.28 / 0.26 in the phantom, which draws a measured pair at random).

So the degradation is reproduced in everything the estimators measure (transfer, noise
variance, resampling), but not exactly.

### 2d — headroom (24 held-out blocks)

The blur spreads a thin structure over many planes, so the isotropic truth is brighter than
the data it produces:

- 99.9th percentile above offset: truth 2.2x the data (1022 against 527 counts); the brightest
  voxel of a typical truth block is 6277 counts, 11x the window ceiling (673) above offset.
- 0.37 % of truth voxels lie above the ceiling; among voxels above the model's input floor
  (156 counts) it is 2.7 %, carrying 9.7 % of the above-floor signal.

So a model whose output is confined to the data's window cannot reproduce the brightest
structures of the truth. Stage 3 scores therefore have to state how truth is clipped.

### Existing model on synthetic input

`out/realism/model_preview_synthF.png` shows input, `skipU_qd` output and truth for held-out
synthetic crops, and input and output for real crops. On both, the output is sharper in XZ
than the input but much dimmer and sparser; against the truth it misses most dim structure.
The baseline input mapping (`--gamma_lo -0.8`) discards everything below 156 counts, which is
about three quarters of the voxels in a typical stack.

### Verification

- `python tools/check_closure.py` passes; `import synthetic` works in the `isofinal` env.
- A `synthF` stack has the shape, dtype, range and 1/287 quantisation of a real training TIFF.
  The loader test (`dataloader/data_multi.py`, train and val splits) was run on `synthA`,
  written by the same code.
- Both blur estimators recover a known width on the phantom within 3 % and 7 % (table above).
- The noise estimator returns the camera parameters put into the phantom (gain 1.389 for
  1.391, read noise 1.30 for 1.40) and the same resampling fractions (median 0.24 / 0.22
  against 0.22 / 0.23).
- The generator changes for variant F leave A and B bit-identical to the truth stored in
  `synthA`, and a block renders identically in separate processes.

## Decisions at the stage 2 gate

1. Is the phantom good enough to proceed, given that a classifier separates it from real data
   almost every time? The alternatives are more work on morphology, or truth taken from real
   images (a near-isotropic dataset, or THX10 itself reduced in XY).
2. Blur width: accept nominal 1.07 with range 0.96-1.19, or first measure it on a sub-block of
   the dense Z stack?
3. How should truth be scored given the headroom: clipped to the data window, or with a wider
   output range treated as a design option?
4. Noise band: leave the three small residuals unmodelled, or add row noise and
   position-linked resampling to the phantom and refit the transfer tail?
