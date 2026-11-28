# Stage 4 summary — synthetic validation of MScleanQDPSF (as of 2026-11-26)

One-page state of the experiment. Details and per-run tables:
`2026-11-22-stage4-psf-sweep.md` (PSF sweep, crop×batch, long run) and
`2026-11-25-stage4-audit-plan.md` (code audit, slate, graduation runs).

## Problem and metric

Self-supervised 8× axial super-resolution on synthetic THX10-like volumes
(proxy crop 128×128×16, dataset synthF, 400 blocks). Primary metric:
**z_res = 0.5/fz50 in px (lower is better)** against aligned truth on the
`val` set, scored from checkpoints (training logs are unreliable under DDP).
Secondary: corr, false rate (hallucination), recall.

Reference points (val): truth ceiling ~1 · Z-sampling Nyquist 16 (the full
Nyquist period at 8-px slice spacing; as a half-period in z_res units the
sampling limit is 8) · interpolation floor 17.2 (cubic) / 17.3 (linear) ·
lateral resolution ~2 px (from the fplane analysis, not a scored metric).
corr saturates near 0.80 early and is not a progress metric after that.

## Chronology of results

1. **PSF-loss sweep (2026-11-22).** A Z-consistency loss that replays the
   acquisition (`--l1how psf --skipl1 1 --psf_bg -0.85`) wins; three seeds
   reach z_res 13.4 / 14.1 / 14.1 at 100 epochs — the first results clearly
   past the interpolation floor (not past Z-Nyquist: in half-period units
   that threshold is 8; see the 2026-11-28 erratum). This became the
   "baseline winner" (13.4 best, band 13.4–14.1). *(Erratum 2026-11-28,
   independent re-audit: on the stored eval JSONs psf_r2 (−0.9) beat
   bg085_r1 on both z_res (13.026 vs 13.193) and corr (0.8055 vs
   0.7959); bg085 was carried for its lower false rate (0.027 vs 0.039)
   and stability, not for z_res.)*
2. **Crop × batch 2×2.** best→e100 z_res: proxy crop b2 13.2→13.4
   (stable); proxy b12 13.5→13.6; full crop 192×192×24 b2 14.1→15.2 (mild
   decay); full b12 16.0→16.9, reaching 17.3 by e300 (collapse). Proxy
   crop is both best and most stable; large batch amplifies a decay
   inherent to the full crop.
3. **1000-epoch run** (proxy crop, 6 GPU, eff. batch 12; ran to completion
   at e1000, scored through e850): peak 14.1 / corr 0.795 at e200, then
   genuine regression — 16.4 at e300, 16.5–17.3 over e400–e850. Longer
   training did not help. SWA-style averaging of
   e150–250 checkpoints scored 15.0 — negative (directional drift, not
   noise). Diagnosis: discriminator saturation/overfit; decay onset is
   monotone in D data-throughput per update. *(Erratum 2026-11-28:
   measured log trajectories later showed NO saturation in the pre-R1
   main game — axx flat ~0.89 over 300 epochs; the diagnosis was
   overstated. See the round-2 erratum section.)*
4. **Three-agent code audit (2026-11-25).** Convergent diagnosis: the PSF
   data term is numerically blind in the target band (axial PSF transfer
   ≈ 1–2e-3 at 13.4 px half-period, depending on the trained psf_width
   1.07 vs 1.0) *(erratum 2026-11-28: that transfer figure is the fr=0
   worst case of the non-separable blur; erratum to the erratum, same
   date, independent re-audit: the "11–13.5" support edge was itself
   refuted — corrected per-bin SNR=1 edge z_res ≈ 14.0–18.5, Wiener
   bound ≈ 14.7–15.8 — see the correction under the round-2 erratum
   section)*, so the
   band beyond the PSF data support is carried
   only by the adversarial loss — which was misallocated (half its weight
   on coarse heads via `--adv_ms 0.5`, never swept), starved of input
   information (no skip/residual from the measured volume), and unstable
   (vanilla BCE, no regularization, train/eval mismatches from
   BatchNorm-at-batch-1 + EMA + Dropout). All runs had trained at constant
   lr 5e-4 (cosine spans the 10001-epoch default); annealing was untested.
5. **Six-run slate** (100 epochs each; best val z_res):
   `--adv_ms 0` **12.7** ✓ · `--adv_ms 0 --lamb 2` 13.3 · GroupNorm+no-
   dropout 13.6 · `--lamb 2` 14.2 · anti-aliased D (patchblur) 14.7 ✗ ·
   output-level residual 16.5 ✗ (pinned at interpolation floor, false rate
   ×5). The win is entirely in `--adv_ms 0`; touching `--lamb` is neutral
   to harmful; both "design" changes failed.
6. **Graduation runs (2 GPU each).** advms0 replication seed: 13.7 (so
   plain advms0 at e100 averages ≈13.2 over its three draws:
   12.7/13.7/13.2 — worth ~0.5 px alone vs the 13.4–14.1 baseline band,
   not the full 0.7; the gnorm run adds a fourth consistent e100 point at
   12.8). 300-epoch annealed advms0:
   13.2, decays mildly to 14.8 by e290. **300-epoch advms0 + GroupNorm/no-
   dropout + annealing: 12.4 at e140** — new best, with the lowest false
   rates in the PSF-loss family (0.019 at e50–60, 0.022 at the e120–140
   peak; the older pre-PSF MScleanQD base runs scored lower still, ~0.010),
   improvement continues past e100, and late decay is mild (13.5 at e290)
   instead of the old collapse.

## Current best (framing: latent-only decode — see the 2026-11-28
## framing decision below; two tracks are kept)

**Claimed model (decodes from quantized latent alone):
z_res 11.1 / corr 0.841 / false 0.012 — r1ttur_r1, epoch 260** (val).
Recipe = proxy.sh base + `--l1how psf --skipl1 1 --psf_bg -0.85 --save_d
--adv_ms 0 --n_epochs 300 --yaml aisr_gn --r1 1 --lr_g_factor 0.5`
(GroupNorm generator, dropout off, R1 γ=1 on real XY slices, TTUR;
`--n_epochs 300` ends at epoch 299, last checkpoint e290). Band
11.1–11.5 across two runs *(erratum 2026-11-28, independent re-audit:
both runs have seed=None — an unseeded re-run, not a seeded
replication; the 0.36 gap ≈ 1.3 combined SE)*. Claim-intact ladder: interpolation 17.2 → first
PSF winner 13.4 → advms0 ~13.2 → +gnorm+annealing 12.4 → **+R1+TTUR
11.1**. All rungs are losses/training dynamics; the decode path is
codes-only throughout.

**Conditioned upper bound (bottleneck-cost ablation, NOT the claimed
model): z_res 10.5 / corr 0.852 / false 0.013 — mpd_r1, epoch 140**
(adds `--inject_x` raw-measurement conditioning of the decoder +
`--mp_d 5`). Prices the quantized-latent-only constraint at ~0.3–0.7 px
at matched epochs. Earlier sections of this doc that call the inject_x
lineage "best" predate the framing decision and should be read as this
ablation track.

Truth ceiling ~1, lateral ~2: the gap to "drastic" improvement remains
large on both tracks.

## What moved the needle, and what didn't

Worked: PSF replay loss · proxy crop, small batch · all adversarial weight
on the full-res D (`--adv_ms 0`) · GroupNorm + dropout-off generator ·
lr annealing (softens late decay) · checkpoint selection by scored eval.
Didn't: longer unannealed training · big batch · full crop · SWA averaging
· lowering `--lamb` · anti-aliased (BlurPool) discriminator · output-level
residual head · corr as a progress signal.

## Next directions (ranked candidates for a drastic z_res move)

Still-untested audit items, roughly by expected leverage:

1. **Inject the measured volume into the decoder properly** (oriX/skip at
   the up-stages, not an output-level anchor — the minimal residual failed
   because the anchor satisfied the data term and freed the adversarial
   gradient to hallucinate).
2. **D regularization for sustained pressure**: R1 penalty on real XY
   slices, TTUR (`--lr_g_factor`), instance noise/flips — aimed at pushing
   the improvement window past e140 instead of merely softening decay.
3. **Rebalance the remaining dead weight**: `p_prefix 0` (z-blind prefix
   terms: weighted magnitude ≈0.56 on half the steps, ~19 % of the
   generator objective — more than the whole PSF term), `lamb_coarse 0.25`,
   fix the qloss double-count, re-weight `adv_loss_six_way` away from the
   near-real XY planes.
4. **More synthetic data** (nearly free) to delay D memorization in long
   runs.
5. Larger/stronger full-res D (capacity in the 8–16 px band), hinge or
   non-saturating loss instead of vanilla BCE.

## Update 2026-11-27 (post-audit)

The proposal slates ran (2026-11-26-stage4-next-models.md). Slate 1: R1
γ=1 on real XY slices + TTUR (`--lr_g_factor 0.5`) on top of P* reached
11.1 with no decay over 300 epochs — D regularization delivered both gain
and stability; `--inject_x` alone gave 12.4 at e100 then decayed; the
loss-cleanup bundle was negative. Slate 2: the combination
**inject_x + R1 + TTUR (combo_r1) = z_res 10.8 / corr 0.848 / false
0.011 at e200, flat at 10.8 through e290** — injection adds ~0.3–1.0 px
at matched epochs once R1 keeps D healthy (one combo seed so far; the
combo_r2 replication is running); a replication seed put
R1+TTUR at 11.5 (seed band 11.1–11.5); γ=2 was slightly worse than γ=1.
The +3.4 px z-offset was shown to be rigid
(crop scatter ~0.3 px → ~1–2 % spectral attenuation): compensated in
scoring, not a z_res limiter.

Slate 3 (2026-11-28): **multi-plane 2.5D patch discriminator (`--mp_d 5`)
on top of the combo recipe = z_res 10.5 / corr 0.852 / false 0.013 at
e140, held flat to e290** — new campaign best and the first architecture
change to beat the combo recipe (gain 0.3 px, below the 0.5 px
graduation bar). combo_r2 reproduced the combo recipe at 10.6–10.8
(second seed), settling reproducibility. hinge loss converged fastest
early and touched 10.5 late, but false climbed monotonically to 0.020 —
kept only as a combination candidate. Ladder now 17.2 → 13.4 → 12.4 →
11.1 → 10.8 → **10.5** (carry-forward: MScleanQDPSFX/mpd_r1 e140).
*(2026-11-28: the mp_d attribution was later invalidated — see the
"Superseded" note under the framing decision below; the score stands.)*

## Audit note

Independently audited 2026-11-26 against primary sources (raw eval outputs,
persisted eval JSONs in synthetic/out/eval/, run config.json files, MLflow
synth.db, checkpoint directories, and the cited code). All headline numbers
(slate bests, graduation results, 12.4/0.805/0.022 at advms0gn_e300 e140,
recipe flags, interpolation floors, long-run peak and SWA probe) verified to
printed precision; the mechanism claims (constant lr, adv_ms origin, unused
Xup, qloss double-count, EMA/BN/dropout mismatches, six-way weighting,
BlurPool/InstanceNorm D) verified in code. Corrections applied after audit:
the 1000-epoch run completed (scored through e850; it was not stopped at
~e880); full-crop b12 ends e100 at 16.9 (17.3 is the e300 value); the
z-blind prefix terms are ~19 % of the generator objective (0.56 is their
absolute magnitude, not a share); the ≈13.1 four-draw advms0 mean mixed in
the gnorm run — the plain-advms0 three-draw mean is 13.2. Not verifiable
from primary sources: per-run GPU counts (configs don't record them), the
SWA e150–250 membership, and the ~2 px lateral number (figure-based).

## Framing decision (2026-11-28): compression / latent-only claim kept

Project constraint adopted: the published model decodes from the
quantized latent alone — the measured volume conditions training
(losses, discriminators) but never the generation path. Under this
framing the campaign reads:

**Claim-intact ladder (codes-only decode):**
17.2 interpolation → 13.4 PSF replay loss → ~13.2 adv_ms 0 →
12.4 +GroupNorm/no-dropout/annealing → **11.1 +R1+TTUR**
(r1ttur_r1 e260; seed band 11.1–11.5). This is the current
claim-intact best. All of these levers are losses/training dynamics —
fully compatible with the compression statement.

**Injection runs re-labeled as bottleneck-cost ablations:** the
inject_x lineage (combo 10.6–10.8, mpd 10.5, slate 4 pending) decodes
with raw-measurement conditioning and is NOT the claimed model; it
quantifies what the quantized-latent-only path currently costs:
~0.3–0.7 px at matched epochs. Useful as the paper's ablation defending
(or pricing) the bottleneck.

**Gap to close:** the newest D-side wins (multi-plane 2.5D D, hinge)
were only ever measured on top of inject_x, but are themselves
claim-compatible (they never touch the decode path). Porting them back
to the latent-only recipe is the top-priority run:
`base + R1 + TTUR + mp_d 5 (− inject_x)` — expected ~10.8 if the mpd
gain transfers.

*Superseded 2026-11-28 (external audit, point 2): the mp_d port was
withdrawn before launch. The mp_d real construction (shifted copies of
one slice) is degenerate — real channel k is exactly channel 0
translated k px, a watermark the D detects trivially; the mp
adversarial game collapsed by ~e15 of mpd_r1 (axmp → 24, dmp → 4e-8)
while the main slice game sat at equilibrium (axx 0.71) in the same
run. The 10.5 score stands, but the "multi-plane D helps" attribution
does not (the 10.6→10.5 step is within the combo seed band; mpd_r2 is
the direct test). Full analysis: next-models.md, "mp_d construction
audit".*

**Allowed generator directions under this framing:** decode-tail
capacity (`--refine_hr` WITHOUT inject_x_hr), axial sub-pixel head
(`--zshuffle`), band-split losses (`--band_split`), decoder noise
(`--g_noise`), axial FFC — all latent-only-safe. Excluded for now:
`--inject_x_hr`, `--inject_h`, `--skip_hs` (they condition the decode
path on measurement or un-quantized features).

## Erratum (2026-11-28, from external audit): Nyquist units

z_res = 0.5/fz50 is a HALF-period. At 8-px slice spacing the sampling
Nyquist frequency is 1/16 cyc/px, i.e. a 16-px full period = an 8-px
half-period. The sampling-limit threshold in z_res units is therefore
**8, not 16**. Early records compared z_res scores against 16 (full
period) — a unit mix-up — and claimed "past Z-Nyquist" at 13.4. In
fact every score to date (13.4 → 11.1 claim-intact, 10.5 conditioned)
has f50 = 0.037–0.048 cyc/px, BELOW the 0.0625 Nyquist frequency: the
gains are genuine deconvolution past the interpolation floor and past
the PSF data-support edge (~13–14), but still within the sampled band.
Crossing the sampling limit requires z_res < 8 — which is and remains
the campaign milestone. Affected phrasing corrected throughout the
records ("beyond-Nyquist band" → "band beyond the PSF data support";
reference ladders now state 8 as the threshold); the underlying scores,
rankings and conclusions are unaffected (the between-run comparisons
never depended on the threshold).

## Erratum (2026-11-28, external audit round 2): PSF cutoff, decoder, D dynamics

Three corrections from a second external audit, all verified against
code, data, and logs (full analysis: next-models.md, "External audit
round 2"):

1. **PSF information cutoff**: the degradation blur is non-separable
   (axial cutoff widens with lateral frequency), so "axial transfer
   ~1e-3" only held at fr=0. The frequency-resolved SNR was then
   measured directly (truth vs degraded spectra, val block 1): the
   SNR=1 support edge sits at fz ≈ 0.037–0.045 (z_res ≈ 11–13.5),
   nearly flat across lateral frequency. The 13.4→11.1 gains ran
   through a partial-information transition band (SNR ~0.3); below
   z_res ≈ 11 the reconstruction is genuinely prior-driven. Scores and
   rankings unaffected; the "purely prior-driven beyond 13.4" phrasing
   was too strong.

   > **Erratum (2026-11-28, independent re-audit — two-agent
   > validation): this item's SNR measurement is itself refuted.**
   > Three method errors: the corner noise floor underestimates the
   > noise (the noise is COLORED by the per-block in-plane resample —
   > 3–5× power drop toward the corner, so the corner is the spectrum
   > minimum, not conservative); the white-camera "8.7/bin" prediction
   > is ~10× wrong; and the UNSCALED transfer was used though val
   > blocks use width 1.07 (signal overestimated 2–3.5× near the
   > edge). Corrected by bit-exact degradation replay: the SNR=1
   > contour is NOT flat in fr (falls ~2× across fr); the per-bin
   > SNR=1 edge is z_res ≈ **14.0–18.5**; the Wiener
   > (best-linear-reweighting) bound is z_res ≈ **14.7–15.8**; pooled
   > raw-data coherence crosses 0.5 at z_res ≈ **17.0–17.2** — exactly
   > the cubic score. Everything below z_res ≈ 15–16, including 12.4
   > and 11.1, is prior-driven reconstruction (legitimate — priors
   > couple frequencies — but to be framed so). Also newly found: the
   > output head has NO tanh (configs set final='none'; out0 is an
   > unbounded conv — eval clips, but the D sees unbounded fakes vs
   > [-1,1] reals). Full record: 2026-11-28-trackA-validation.md.
2. **Decoder**: netG=ed023emsfpnu instantiates nearest-upsample +
   Conv3d(k3) stages, not ConvTranspose3d; the full-res tail is two
   convs, not one. Thin-tail motivation for --refine_hr stands with
   corrected arithmetic.
3. **D dynamics**: the G adversarial loss is the non-saturating BCE
   form; the 24:1 fake:real figure is a sample-diversity asymmetry,
   not a loss weighting; and measured trajectories show the pre-R1
   main game never saturated (axx ~0.89 flat). R1+TTUR's 12.4→11.1
   gain is real but its mechanism attribution (and the R1-vs-TTUR
   split) is open — r1only/tturonly ablations queued.
