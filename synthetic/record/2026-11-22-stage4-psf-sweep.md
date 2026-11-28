# Stage 4 — PSF-loss proxy sweep, full-crop run, batch isolation

Machine: brcb (8× RTX A6000 48 GB), env `isofinal`, dataset `synthF` rebuilt and
verified bit-identical per `REBUILD.md`. Sweep run 2026-11-22, follow-ups 2026-11-23.
Raw per-run scores: `synthetic/out/eval/*.json` (git-ignored, local); training
curves in `$LOGS/synth.db` (MLflow).

## Summary (as of 2026-11-24)

Metric: z_res = 0.5/fz50 (px, lower is better), from spectral correlation
against the aligned ground truth. Reference points on `val`: ~1 px metric
ceiling, 8 px Z-sampling Nyquist HALF-period (z_res is a half-period, so
8 — not the 16-px full period — is the sampling-limit threshold in these
units), ~17.2 px interpolation floor.

1. **The PSF Z-consistency loss wins.** On `val` at 100 epochs, the floored
   PSF variants (psf, bg085, wide) reach z_res 13.4–14.1 vs 14.9 for the
   MScleanQD baseline and ~17.2–17.3 for interpolation — clearly past the
   interpolation floor, i.e. real deconvolution within the sampled band
   (NOT past the sampling Nyquist: that threshold is 8 in half-period
   units; see the 2026-11-28 erratum). The corr margin over the baseline
   (~0.06) is ~10× the repeat spread; the z_res margin over the baseline is
   0.8–1.5 px, modest but consistent, and over interpolation 3–4 px. The
   same ordering holds on `val_narrow` and `val_wide`. (nofloor collapses to
   the interpolation floor and is rejected; bg075 reaches 14.6.)
2. **`--psf_bg -0.85` is the chosen variant.** Resolution is indistinguishable
   from plain psf, but output is cleaner (false 0.026–0.029 vs 0.038 at
   similar recall: 0.984–0.986 vs 0.989–0.990). Replicated over three seeds:
   corr 0.797–0.805, z_res 13.4–14.1. psf_bg acts as a precision/recall
   dial; −0.75 and −0.95 are both worse. A 1.3× PSF-width error costs only
   ~0.1–0.2 px vs plain psf (0.7 px vs the best bg085 seed).
3. **Crop × batch 2×2** (z_res): proxy crop 128×128×16 gives 13.4 at
   effective batch 2 and 13.6 at batch 12; full crop 192×192×24 at batch 2
   plateaus at 14.1–14.7 (e30–e90) then drifts to 15.2 at e100, and at
   batch 12 it regresses outright to 16.9→17.3. Only the proxy crop is
   stable in every cell; the full crop decays — mildly at batch 2,
   catastrophically at batch 12 — and the two batch-12 runs had matched
   (~33/epoch) update counts, so it is not an update-budget artifact.
   Mechanism unknown. **Stage-5 recipe: proxy crop 128×128×16.**
4. **Caveat: multi-GPU training-log metrics are unreliable** in this codebase
   (per-rank-random prefix logging × PL `sync_dist` scrambles values across
   keys). Diagnose multi-GPU runs only by scoring checkpoints. The taming
   slice discriminator never engages at this scale (`disc_start` 250001
   steps); the six-way axial discriminators are always active.
5. **Longer training does not help.** `bg085_long_r1` (winner recipe, proxy
   crop, 1000 epochs, 6 GPUs = effective batch 12) peaks at e200
   (14.1 / 0.795) — no better than the 100-epoch batch-2 runs — then
   regresses to ~16.5–17 from e300 onward (checkpoint-scored, so real, unlike
   the earlier log-based false alarm). All runs train at effectively
   constant lr 5e-4 (`n_epochs` left at 10001 by design, proxy.sh:9-10, so
   the cosine never moves — a corrected earlier misreading); at identical
   lr, decay onset is monotone in discriminator data-throughput per update
   (full+b12 by ~e60, proxy+b12 ~e250, full+b2 mild at e100, proxy+b2
   stable ≥e100). lr annealing has never been tested. Best models remain
   the 100-epoch proxy-crop batch-2 runs.

## The six proxy runs (crop 128×128×16, effective batch 2, lr 5e-4, 100 epochs)

The run list of `README.md` "Runs for the A6000 machine", at 100 epochs instead
of 50, GPUs 2–7. `P = --models MScleanQDPSF --l1how psf --skipl1 1`.

Epoch 100, set `val` (spread over blocks: ±0.010 corr, ±1.6 z_res):

| run | flags | psnr | corr | z_res | false | recall |
|---|---|---|---|---|---|---|
| **bg085_r1** | `P --psf_bg -0.85` | 22.49 | 0.797 | **13.4** | 0.029 | 0.984 |
| psf_r2 | `P` | 22.37 | 0.803 | 13.9 | 0.038 | 0.989 |
| psf_r1 | `P` | 22.28 | 0.797 | 14.0 | 0.038 | 0.990 |
| wide_r1 | `P --psf_width 1.39` | 22.30 | 0.799 | 14.1 | 0.040 | 0.989 |
| base_r3 | baseline MScleanQD | 20.29 | 0.742 | 14.9 | 0.012 | 0.891 |
| nofloor_r1 | `P --gamma_lo -1 --psf_bg -1` | 21.62 | 0.747 | 17.3 | 0.036 | 0.966 |
| cubic interpolation | — | 21.05 | 0.703 | 17.2 | 0.077 | 0.928 |
| linear interpolation | — | 21.00 | 0.699 | 17.3 | 0.079 | 0.923 |

Epoch 100 on the other blur widths (corr / z_res):

| run | val_narrow | val_wide |
|---|---|---|
| bg085_r1 | 0.817 / 12.2 | 0.773 / 15.7 |
| psf_r2 | 0.822 / 12.6 | 0.780 / 15.7 |
| psf_r1 | 0.817 / 12.6 | 0.772 / 16.0 |
| wide_r1 | 0.815 / 12.9 | 0.778 / 15.8 |
| base_r3 | 0.753 / 13.7 | 0.727 / 16.1 |
| nofloor_r1 | 0.766 / 15.8 | 0.724 / 18.9 |
| cubic | 0.727 / 15.5 | 0.679 / 18.9 |

Reference points for z_res (px): 1 = metric ceiling (perfect truth recovery),
8 = Z sampling Nyquist half-period (8× slice spacing; z_res is a half-period,
so 8 is the sampling limit in these units), ~17.2 = interpolation floor on `val`.

### Conclusions

- **The PSF loss wins**: every floored PSF run (all but nofloor) beats the
  baseline and both interpolations on corr and z_res at all three widths.
  The corr margins are ~10× the repeat spread (psf_r1 vs psf_r2 ≈ 0.006
  corr, 0.1 z_res); the z_res margin is 0.8–1.5 px over the baseline and
  3–4 px over interpolation.
- All PSF runs resolve **well past the interpolation floor** — genuine
  deconvolution, not interpolation, by the hallucination-proof spectral
  metric. (Erratum 2026-11-28: an earlier version said "beyond the 16-px
  sampling Nyquist" — a unit error; 13.4 half-period = f50 0.037 cyc/px,
  still below the 0.0625 Nyquist frequency. Crossing requires z_res < 8.)
- **bg085 (`--psf_bg -0.85`) is the best variant** (13.4 on val, 12.2 narrow).
- **nofloor is rejected**: removing the input floor collapses z_res to the
  interpolation floor at every width.
- **Robust to PSF-width error**: wide_r1 trains with a 1.3× too-wide PSF and
  loses only ~0.1–0.2 px z_res vs plain psf — the loss does not need the
  width to be exact.
- The baseline's near-zero false (0.012) comes with low recall (0.891) and a
  14-plane z offset; the PSF runs trade a little false rate for finding nearly
  everything at the right place.

## Replication + psf_bg refinement (2026-11-24, 5 runs, 100 epochs, proxy crop)

Epoch 100, `val`:

| run | psf_bg | corr | z_res | false | recall | bright |
|---|---|---|---|---|---|---|
| bg085_r2 | -0.85 | 0.799 | 14.1 | 0.027 | 0.985 | 0.088 |
| bg085_r3 | -0.85 | 0.805 | 14.1 | 0.026 | 0.986 | 0.089 |
| bg075_r1 | -0.75 | 0.772 | 14.6 | 0.020 | 0.939 | 0.054 |
| bg095_r1 | -0.95 | 0.795 | 14.1 | 0.056 | 0.992 | 0.146 |
| base_r4 | baseline | 0.736 | 14.9 | 0.009 | 0.881 | 0.029 |

- bg085 replicates (corr 0.797-0.805, z_res 13.4-14.1 over three seeds); the
  original 13.4 was the lucky end — typical is ~14. On corr/z_res bg085 and
  plain psf are indistinguishable; bg085's replicated advantage is **cleaner
  output** (false 0.026-0.029 vs 0.038, ~25 % less bright volume at the same
  recall).
- psf_bg is a precision/recall dial: -0.75 paints too little (recall 0.939,
  corr 0.772), -0.95 too much (false 0.056). Resolution barely moves.
  **-0.85 stands.**
- Baseline spread is tiny (0.736/0.742): the PSF corr margin (+0.06) is ~10×
  the run noise; the z_res margin is 0.8-1.5 px, modest but consistent; the
  recall gap (0.98 vs 0.88) is large and stable.
- All five runs stayed GAN-dormant and PSF-consistent (disc = 0, l1p ≈ 0.06).

## Full-crop run of the winner: regression

`bg085_full_r1`: same recipe at crop 192×192×24, 6 GPUs × b1 × accum 2 =
effective batch 12, lr unchanged 5e-4, ~33 updates/epoch, target 1000 epochs
(needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; 38.6 GB/GPU).
Stopped at epoch ~396 after the metric curve showed it had plateaued-regressed.
Set `val`:

| epoch | psnr | corr | z_res |
|---|---|---|---|
| 10 | 21.20 | 0.715 | 18.5 |
| 30 | 21.80 | 0.758 | **16.0** |
| 60 | 21.83 | 0.760 | 16.2 |
| 100 | 21.89 | **0.763** | 16.9 |
| 150 | 21.80 | 0.757 | 17.1 |
| 200 | 21.69 | 0.750 | 17.7 |
| 250–360 | ~21.7 | ~0.754 | ~17.3 |

It peaks early (z_res 16.0 at e30 — barely better than interpolation), then
decays to the interpolation floor while corr holds ~0.75: the model keeps coarse
structure but stops resolving. Worse than the proxy even at matched update
counts (12k updates ≈ proxy e60: 0.797 / 13.2). Checkpoints 0–390 remain at
`$LOGS/synthF/synthF/MScleanQDPSF/bg085_full_r1/`.

Two candidate causes: (a) effective batch 12 vs 2 at the same lr, (b) a genuine
crop-size × loss interaction.

**Do not read multi-GPU training-log metrics** (caveat found 2026-11-24): on
6-rank DDP runs the progress-bar/epoch metrics are scrambled across keys.
MScleanQD logs `l1p`/`lpipsp` only on random prefix batches (`--p_prefix`
0.5), each rank draws independently, so ranks log different key sets per
batch, and PL 1.9's `sync_dist` epoch reduction then crosses values between
keys. Verified by instrumentation: in a 6-GPU run the real taming `disc` loss
is exactly 0 on every call (gated behind `disc_start` 250001 steps, never
reached at this scale — by design the slice-realism GAN never engages) while
the bar displayed disc=0.24; relaunching with `--p_prefix 0` makes the display
clean. An earlier reading of full_r1's log ("PSF term collapsed, GAN engaged")
was this artifact — full_r1's directly-logged `l1` (0.034) matched healthy
runs; nothing is wrong with its loss. For multi-GPU runs trust only
checkpoint-based evaluation; 1-GPU logs are unaffected.

## Batch-isolation tests: the failure needs crop AND batch together

`bg085_b12_r1`: proxy crop, but 2 GPUs × b3 × accum 2 = effective batch 12,
lr 5e-4, 100 epochs. Run 2026-11-23, GPUs 2–3 (GPUs 0,1 hung at NCCL
rendezvous with this command; the identical launch on 2,3 trained immediately).
Read-out agreed beforehand: z_res ≈ 17 ⇒ batch/lr is the culprit; z_res ≈ 13.4
⇒ the crop is the culprit. Set `val`:

| epoch | psnr | corr | z_res |
|---|---|---|---|
| 10 | 21.27 | 0.721 | 17.9 |
| 30 | 21.89 | 0.764 | 16.6 |
| 50 | 22.17 | 0.780 | 15.3 |
| 70 | 22.28 | 0.787 | 14.4 |
| 90 | 22.39 | 0.792 | **13.5** |
| 100 | 22.41 | 0.793 | 13.6 |

**At effective batch 12 the proxy crop still converges to the proxy result**
(13.6 / 0.793 vs bg085_r1's 13.4 / 0.797) — the large batch only slows the
early trajectory (16.6 at e30 vs 13.9). So batch alone is not the problem.

`bg085_fullb2_r1`: the remaining cell of the 2×2 — crop 192×192×24 at
effective batch 2 (1 GPU, 100 epochs, lr 5e-4). Run 2026-11-23/24, GPU 2.
Set `val`:

| epoch | psnr | corr | z_res |
|---|---|---|---|
| 10 | 21.82 | 0.759 | 16.3 |
| 20 | 22.02 | 0.772 | 15.2 |
| 30 | 22.13 | 0.778 | **14.1** |
| 40 | 22.24 | 0.785 | 14.4 |
| 50 | 22.21 | 0.783 | 14.7 |
| 60 | 22.24 | 0.785 | 14.3 |
| 70 | 22.24 | 0.785 | 14.6 |
| 80 | 22.26 | 0.786 | 14.3 |
| 90 | 22.19 | 0.783 | 14.5 |
| 100 | 22.14 | 0.779 | 15.2 |

At e100 the other widths read 13.4 narrow / 16.7 wide — consistently ~1 px
behind the proxy runs (12.2–12.9 / 15.7–16.0).

**The full crop at effective batch 2 mostly works**: it plateaus at
14.1–14.7 / 0.785 over epochs 30–90 — at the edge of the proxy run's
replicated range (13.4–14.1 / 0.797–0.805), a shade worse, and nowhere near
full_r1's 16–17 floor — then drifts to 15.2 / 0.779 at e100, a mild echo of
the decay that destroyed full_r1 at batch 12.
Completed 2×2 (z_res, best → e100):

| | eff. batch 2 | eff. batch 12 |
|---|---|---|
| proxy crop 128×128×16 | 13.4 (bg085_r1) | 13.5→13.6 (b12_r1) |
| full crop 192×192×24 | 14.1→15.2 (fullb2_r1) | 16.0→17.3 (full_r1) ✗ |

So the full crop carries a late-decay tendency of its own (mild at batch 2),
and batch 12 amplifies it into a collapse; the batch alone is harmless on the
proxy crop. b12_r1 and full_r1 have the *same* ~33 updates/epoch, so
update-count doesn't explain the batch-12 difference; the amplification is
genuinely crop-size × batch-size (speculation: averaging 12 large diverse
crops per step smooths the gradient more than 12 small redundant ones, which
may upset the adversarial/PSF balance). Practical conclusion for stage 5:
**use the proxy crop (128×128×16)** — it is better and stable at every batch
size, 3× faster per epoch, and robust to the multi-GPU batch inflation that
DDP brings.

## Long run of the winner recipe: peaks at e200, then regresses

`bg085_long_r1`: the pure winner recipe (proxy crop, `P --psf_bg -0.85`),
1000 epochs on 6 GPUs (effective batch 12, ~33 updates/epoch; lr effectively
constant 5e-4 — the cosine spans the default `n_epochs` 10001, as in every
run here). Relaunched fresh 2026-11-24 08:03 after the first attempt was
stopped on the (false) log-metric alarm. Set `val`:

| epoch | psnr | corr | z_res |
|---|---|---|---|
| 50 | 22.24 | 0.784 | 14.7 |
| 100 | 22.42 | 0.793 | 14.4 |
| 150 | 22.45 | 0.795 | 14.6 |
| **200** | 22.45 | **0.795** | **14.1** |
| 250 | 22.34 | 0.789 | 15.0 |
| 300 | 22.21 | 0.782 | 16.4 |
| 400–700 | ~22.1 | ~0.777 | 16.5–16.9 |
| 800–850 | ~22.0 | ~0.772 | 16.8–17.3 |

- The peak (e200, 14.1 / 0.795) never beats the 100-epoch batch-2 runs
  (13.4–14.1 / 0.797–0.805); decay starts between e200 and e250 and settles
  near the interpolation floor while recall/false stay flat — the same
  "keeps coarse structure, stops resolving" signature as full_r1.
- Combined picture across runs: every effective-batch-12 run eventually
  regresses (full crop by ~e100, proxy crop by ~e250); batch-2 runs were
  stable through their 100-epoch windows (full crop showed a mild e100
  drift). All at identical, effectively constant lr 5e-4 (the cosine spans
  `n_epochs` 10001 by design — an earlier "stable runs annealed to ~0"
  reading of this record was wrong). Decay onset is monotone in the
  discriminators' data-throughput per update, consistent with D
  saturation/overfit killing the adversarial gradient — the only loss term
  that constrains the band beyond the PSF data support. lr annealing
  (`--n_epochs <run length>`) is untested and free to try.
- Practical: stage 5 should train ~100 epochs, proxy crop, small effective
  batch; pick checkpoints by scored evaluation, not by training length.

`synthetic/visualize.py` (added 2026-11-23) renders XZ side views of
raw (slices repeated ×8) / cubic / model, each in grayscale and as a
green/magenta overlay on the aligned truth (truth = green, method = magenta,
agreement = white; per-volume 99.5-percentile normalization so the dimmer,
blur-diluted methods are comparable to the truth channel):

```
ENV_NAME=brcb CUDA_VISIBLE_DEVICES=0 python -m synthetic.visualize \
    --method MScleanQDPSF/bg085_r1 --epoch 100 --set val --block 0 --crop 0
```

![side views, val block 0](fig/side_val_b0_c0_MScleanQDPSF_bg085_r1_e100.png)

The picture matches the numbers: raw/cubic show each truth blob as a magenta
column stretched 2–3× along Z (z_res ~17); bg085's cores go white with short
magenta fringes (13.4 recovered vs ~2 lateral) and it traces the dim filament
network cubic misses entirely (recall 0.984 vs 0.928); almost no isolated
magenta (false 0.029); truth structure the blur pushed below the floor stays
green in every method — the information no model can recover. The model's
systematic ~+3-plane placement offset is estimated and removed before scoring
(cubic: −0.4).

Long-run decay, visually (same overlay method, epochs 100–400 and 500–800):
through e200 the blob cores are compact and white (agreement) with the dim
filament network traced; from e300 the output smooths out — axial magenta
fringes lengthen back toward the interpolation look and dim truth detail
drifts green (present in truth, fading from the output) — while the bright
coarse blobs persist, which is why corr barely moves (0.795→0.77) as z_res
decays 14.1→17.

![long run e100–400](fig/side_val_b0_c0_long_e100-400.png)
![long run e500–800](fig/side_val_b0_c0_long_e500-800.png)

Two more views of the same script:

- `--spectrum tag1,tag2,...` plots the axial band-correlation curves behind
  z_res from the eval JSONs (no GPU). On `val`, all methods hold ≥0.8
  correlation below ~0.016 cycles/px; the PSF win lives at 16–8 px
  half-period, where bg085 keeps 0.2–0.6 correlation while linear falls
  from ~0.4 at 16 px to ~0 below ~13 px; below ~5 px every method is zero
  (the blur attenuates those bands below the noise — gone). The regressed
  full-crop run sits between the two (e.g. 0.22 at 10.7 px half-period vs
  linear 0.06 and bg085 0.35), but its 0.5-crossing lands at the
  interpolation floor.

  ![axial spectra, val](fig/spectrum_val.png)

- `--fplane` maps correlation over the 2D (lateral × axial) frequency plane,
  over a block's four crops. Raw/cubic's trusted region is a pancake (lateral
  to 0.25 at fz < 0.03): their errors are purely axial. bg085 doubles the
  axial bandwidth (0.5 contour to fz ≈ 0.053) but retreats laterally beyond
  ~0.10: the GAN re-synthesizes the volume, trading a little fine lateral
  fidelity for the axial gain, while interpolation passes the original XY
  planes through untouched. High-frequency artifacts are therefore
  directional in opposite ways: axial for interpolation, lateral for the model.

  ![frequency plane, val block 0](fig/fplane_val_b0_MScleanQDPSF_bg085_r1_e100.png)

## Update 2026-11-25: audit slate result

The six-run audit slate (`2026-11-25-stage4-audit-plan.md`) found a new
best: `--adv_ms 0` (all adversarial weight on the full-resolution D) gives
**z_res 12.7** on val at e100 (advms0_r1), 0.7 px below this record's 13.4
best, with better false rate (0.025) and recall (0.983), and still
improving at 100 epochs. Pending replication seeds + a 300-epoch annealed
advms0+gnorm run. Residual output, BlurPool D, and lamb 2 all failed to
beat baseline; details and per-run tables in the audit-plan note.

Update 2026-11-26: the graduation runs improved this further — **advms0 +
GroupNorm/no-dropout + lr annealing reaches z_res 12.4 (e140, val)**, with
mild late decay only (13.5 at e290). See the audit-plan note's final section.

**Update 2026-11-27:** new campaign best z_res 11.1 (r1ttur_r1 e260,
R1 gradient penalty + TTUR on top of advms0+gnorm+annealing); see
2026-11-26-stage4-next-models.md.

**Update 2026-11-27 (2):** campaign best now z_res 10.8 (combo_r1 e200:
inject_x + R1 + TTUR on the advms0+gnorm+annealing base), flat e200–e290;
see 2026-11-26-stage4-next-models.md.

**Update 2026-11-28:** campaign best now z_res 10.5 (mpd_r1 e140:
multi-plane 2.5D patch D `--mp_d 5` on the combo recipe), flat e140–e290;
see 2026-11-26-stage4-next-models.md.

**Update 2026-11-28 (2) — framing:** the compression/latent-only claim is
kept: the claimed model decodes from the quantized latent alone.
Claim-intact best: z_res 11.1 (r1ttur_r1 e260). The inject_x lineage
(10.8 combo, 10.5 mpd) is re-labeled a bottleneck-cost ablation, not the
claimed model. See 2026-11-28-stage4-status.md.
