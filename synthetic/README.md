# synthetic/ — ground-truth validation with a synthetic volume

THX10 has no isotropic ground truth. This folder builds one synthetically:

1. **Estimate** the THX10 degradation (blur along Z, Z sampling, noise) from the anisotropic data.
2. **Generate** isotropic phantom blocks with known truth, and **degrade** them the same way.
3. **Train** the unchanged IsoFinal model on the degraded blocks and **score** its output against truth.
4. **Sweep** hyperparameters / designs on the phantom, then train THX10 with the winner.

Nothing outside `synthetic/` is modified. Run everything from the repo root in the `isofinal`
env, as modules (`python -m synthetic....`). Figures and caches go to `synthetic/out/`
(git-ignored); the only committed result is `config/degradation.json`.

Conventions: arrays are `(Z, Y, X)`; lengths are in XY pixels; one Z slice = 8 XY pixels;
frequencies are in cycles per XY pixel (Z Nyquist = 1/16).

## Status

| Stage | State |
|---|---|
| 1. Degradation estimate | done, **awaiting review** (see the cross-check below) |
| 2. Phantom + realism gate | generator and degradation pipeline written, not yet calibrated |
| 3. Training / evaluation harness | not started |
| 4. Sweep | not started |
| 5. Confirm on the A6000 machine, train THX10 | not started |

## Stage 1 — degradation estimate

```
python -m synthetic.estimate.blocks --per_band 6   # pick fibre-rich blocks (192 x 768 x 768)
python -m synthetic.estimate.transfer              # main model: non-separable transfer
python -m synthetic.estimate.spectra               # 1D shell-averaged summary
python -m synthetic.estimate.lsf                   # independent check on single fibres
python -m synthetic.estimate.noise                 # camera noise, flicker, jitter
python -m synthetic.estimate.window                # intensity window of the training TIFFs
python -m synthetic.estimate.report                # cross-check + config/degradation.json
python -m synthetic.estimate.selftest              # estimators on a phantom with a known blur
```

| file | what it measures |
|---|---|
| `estimate/transfer.py` | `R(f_r, fz)`: power transfer from "isotropic, as sharp as XY" to the data, from the 3D power spectrum. Assumes only that the tissue is statistically isotropic. |
| `estimate/spectra.py` | the same, averaged over shells into one 1D kernel (for orientation; not used by the pipeline) |
| `estimate/lsf.py` | lateral and axial profiles of individual in-plane fibres; no isotropy assumption |
| `estimate/noise.py` | offset, read noise, gain, noise correlation, per-slice flicker and jitter |
| `estimate/window.py` | locates a training TIFF in the zarr and fits its raw-count window |
| `estimate/report.py` | per-fibre comparison of the two methods, writes `config/degradation.json` |

### Result (18 blocks, three depth bands)

- **The blur is not a 1D kernel along Z.** Fine lateral detail is sectioned better than coarse
  detail: `R = exp(-(|fz| / f0)^p)` with `f0 = 0.0060 + 0.043 f_r`, `p = 1.39`. The equivalent
  axial kernel is about 41 px FWHM for coarse structure (`f_r` = 0.01), 36 px at 0.03 and
  25 px at 0.10, i.e. 3 to 5 slices. The kernel has heavier tails than a Gaussian (`p` < 2).
- It is stable: per-block `a` 0.0050-0.0073, `b` 0.034-0.050, and the three depth bands agree
  within 5 %.
- Noise: offset 101 counts, variance = 0.83 x signal + 1.3^2, slight lateral correlation
  (lag-1 0.25). Flicker (0.4 %) and jitter (0.04 px) are negligible.
- The Z stack is oversampled relative to this blur (kernel 3-5 slices wide), so the task the
  model faces is mostly axial *deblurring*, not only interpolation.

### Cross-check (the stage 1 gate)

For 350 fibres the transfer model predicts the axial width each should have from its lateral
width. Measured / predicted: median 1.24, mode 1.16, quartiles 1.14-1.43. Pooled profile:
49 px measured against 40 px predicted. The plan's target was agreement within about 15 %.

Known biases, both pushing the ratio above 1:
- the fibre measurement is widened by bundles, overlapping fibres and steep fibres;
- the self-test (known blur put in, estimators run) shows the transfer estimator returns `f0`
  about 8 % too high, i.e. a kernel about 8 % too narrow.

Not yet verifiable from this data: the transfer above the Z Nyquist (extrapolated by the
model; matters for aliasing of fine lateral detail). The dense original Z stack would settle
both this and the cross-check.

## Stage 2 — phantom (in progress)

| file | role |
|---|---|
| `phantom/generate.py` | `make_block(size, seed)`: fibres with varicosities, cell bodies, puncta, textured background; isotropic, noise-free |
| `phantom/degrade.py` | `degrade(truth, deg)`: transfer filter, every 8th slice, slice artefacts, camera noise, window |

Still to do: calibrate the phantom parameters against real XY statistics, the realism tests,
and the dataset writer.
