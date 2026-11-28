# Track A validation record (2026-11-28) — three-agent independent audit

## Purpose and method

Independent validation of the stage-4 Track A results (claimed model,
codes-only decode, no `--inject_x`) by three agents:

- **Auditor 1 (results & provenance)**: re-derived every scoreboard
  number from the stored eval JSONs, re-ran representative checkpoint
  evaluations on GPU (bit-exact to 1e-6), verified claim-intact flags in
  every cited run's config and in the model code, and checked for
  Track B contamination.
- **Auditor 2 (methods & analysis)**: validated the analysis layer with
  a bit-exact replay of the degradation pipeline (reproduced the stored
  degraded TIFFs with max|sim−deg| = 0.0), then re-measured the SNR /
  information-support question from primary data, re-derived the decoder
  and adversarial-loss facts from code, and recomputed D-health
  trajectories over all 300 epochs.
- **Finalizer (this document)**: reconciled the two reports into what is
  established, corrected, or open, and applied errata to the existing
  record docs.

Where Auditor 2's re-measurement conflicts with earlier team analysis,
Auditor 2 wins: the degradation replay is bit-exact, so its noise and
transfer terms are grounded where the earlier analysis estimated them.

## Certified Track A scoreboard (val; z_res = 0.5/fz50, half-period px, lower better)

All numbers confirmed against eval JSONs; r1ttur_r1 e260, advms0gn
e140 and cubic re-evaluated on GPU, bit-exact to 1e-6.

| run | z_res | corr | false | best ep | description | validity notes |
|---|---|---|---|---|---|---|
| cubic | 17.199 | 0.7029 | — | — | interpolation baseline | also equals the pooled coherence-0.5 limit of the raw data (17.0–17.2) — independent cross-validation |
| base_r3 | 13.805 | — | — | e70 | pre-PSF baseline | best-of-two (base_r4 14.46) |
| psf_r2 | 13.026 | 0.8055 | 0.039 | e40 | PSF loss, psf_bg −0.9 | beat bg085_r1 on z_res AND corr |
| bg085_r1 | 13.193 | 0.7959 | 0.027 | e50 | PSF loss, psf_bg −0.85 (carried recipe) | "sweep winner" label corrected: chosen on lower false rate / stability, not z_res (family means 13.55 bg085 vs 13.44 psf) |
| advms0 (3 draws) | 12.672 / 13.238 / 13.652 | — | — | — | `--adv_ms 0` | mean 13.187 → "~13.2" is fair |
| advms0gn | 12.437 | — | — | e140 | + GroupNorm / no dropout | single draw, inside the advms0 seed spread (12.67–13.65) — step not clearly outside seed noise; "+annealing" is the shared e300 schedule, not unique to the gn change |
| **r1ttur_r1** | **11.090** | 0.8409 | 0.0125 | e260 | + R1 γ=1 + TTUR; **carry-forward checkpoint** | plateau e140–e290 mean ~11.32; 11.09 is min-of-~18-scored-epochs, ~0.2 optimism (same convention for all runs — rankings unaffected). false rises monotonically after e100 (0.0104→0.0129); "0.012" is a favorable round of 0.0125 |
| r1ttur_r2 | 11.454 | — | — | e280 | unseeded re-run (seed=None in both) | 0.36 gap ≈ 1.3 combined SE — "seed band 11.1–11.5" supported but not a seeded replication |
| r1g2_r1 | 11.669 | — | — | e290 | R1 γ=2 | γ=2-worse directionally supported, statistically weak (n=1, ~1 SE, still improving at e290) |

Milestone: z_res < 8 (sampling half-period). Unchanged, and nothing
approaches it.

## Established (both auditors, primary sources)

1. **Scores are real and reproducible.** Every headline number matches
   the stored eval JSONs; GPU re-evaluation of r1ttur_r1 e260, advms0gn
   e140 and cubic is bit-exact to 1e-6 (even with the working-tree
   MScleanQDPSFX.py modified).
2. **Claim-intact verified.** All cited Track A runs have
   `inject_x=False` and no `mp_d` in their configs; recipe flags match
   the records exactly (l1how=psf, skipl1=1, psf_bg=−0.85, adv_ms=0,
   r1=1.0 [2.0 for r1g2], lr_g_factor=0.5, n_epochs=300); the model
   code is structurally codes-only without inject_x.
3. **No Track B contamination.** All 8 inject_x runs checked; none is
   cited in Track A. The near-collision injx_r1 12.443 vs advms0gn
   12.437 was checked: the table's 12.44 is genuinely advms0gn.
4. **Nyquist units.** Threshold 8 in half-period units confirmed; f50
   of all scores < 0.0625 cyc/px; best run at 72% of Nyquist. The
   sampling limit is NOT the binding constraint — blur/noise support
   (~15–17) is.
5. **Adversarial loss form.** Non-saturating G loss confirmed
   (BCEWithLogits target-1 on fakes = softplus(−logit)); all terms
   mean-reduced, 1:1 fake:real loss weighting (equilibrium dxx_x =
   1.386). `adv_loss_six_way` averages 3 orientations — "six_way" is a
   misnomer.
6. **Decoder facts.** Nearest-upsample + Conv3d(k3) (not
   ConvTranspose); exactly TWO full-res 3×3×3 convs after the final
   upsample op (up1's conv + conv7_k) with GroupNorm+ReLU between.
7. **The R1+TTUR gain is real.** The 12.44 → 11.09 step (~1.3 px,
   SE ~0.2) is solid; the game moved to fake-side equilibrium
   (r1ttur axx flat at 0.70 ≈ ln 2 — D at chance on fakes) vs the
   pre-R1 mild persistent D advantage (advms0gn axx ~0.88–0.90, dxx_x
   0.89→0.80–0.85, min 0.755, no saturation; fake logit ≈ −0.36, real
   ≈ +1.0). No epoch-level D-win episodes in either run.
8. **mp_d never touched Track A** (confirmed independently by both).

## Corrected (errata applied to the record docs, 2026-11-28)

1. **SNR support edge — the central correction.** The 2026-11-28
   team measurement ("SNR=1 edge at z_res ≈ 11–13.5, nearly flat in
   fr") is REFUTED by bit-exact degradation replay. Three method
   errors: (a) the noise is COLORED by the per-block in-plane resample
   — 3–5× power drop toward the spectral corner, so the corner floor
   is the spectrum MINIMUM, not conservative (v00001 corner 47 vs
   in-band 56–86 counts²; v00000 corner 10 vs in-band 33–50); (b) the
   white-camera prediction "8.7/bin" is ~10× wrong; (c) the UNSCALED
   transfer was used although val blocks use width 1.07 — overestimates
   signal 2–3.5× near the edge. Additionally, ceiling-clip harmonics
   (~0.04–0.07% clipped voxels) inflate apparent high-fz "signal" (85%
   of the naive residual variance there). Corrected numbers:
   - The SNR=1 contour is NOT flat in fr: the edge falls ~2× across fr
     (fz ≈ 0.033–0.036 at fr < 0.05; 0.025–0.030 at fr 0.10–0.15;
     0.016–0.024 at 0.15–0.25; fr > 0.25 never reaches SNR 1).
   - True per-bin SNR=1 edge: z_res ≈ **14.0–18.5** (best bin 14.0).
   - Best possible per-frequency linear reweighting (empirical Wiener
     bound): z_res ≈ **14.7–15.8**; at fz = 0.045 coherence ≈ 0.09.
   - Pooled coherence of raw data with truth (the metric's own
     weighting) crosses 0.5 at fz ≈ 0.029 → z_res **17.0–17.2** —
     exactly the cubic score, an independent cross-validation of the
     interpolation floor.
   - Consequence: **everything below z_res ≈ 15–16 is prior-driven** —
     the pre-R1 12.44 AND the r1ttur 11.09 both measure prior-driven
     reconstruction, not transferred axial information. This is
     legitimate (priors couple frequencies) but must be framed so. The
     earlier framing "13.4→11.1 ran through a partial-information
     transition band (SNR ~0.3)" overstated the data support.
2. **No tanh on the output head.** All cited configs set
   `final='none'` → Identity: the out0 head is a bare unbounded conv.
   Eval clips, so scores are unaffected, but the D sees unbounded
   fakes vs [-1,1]-bounded reals — an undiscussed free discriminator
   cue. Record phrases "the final tanh", "tanh removed", "final
   activation stays tanh" are wrong and have been annotated.
3. **"Seed replication" → unseeded re-run.** r1ttur_r1 and r1ttur_r2
   both ran with seed=None; the band 11.1–11.5 reflects two unseeded
   draws (gap 0.36 ≈ 1.3 combined SE), not a controlled seed study.
4. **bg085 "PSF sweep winner" label.** psf_r2 (−0.9) beat bg085_r1 on
   both z_res (13.026 vs 13.193) and corr (0.8055 vs 0.7959); family
   means 13.44 (psf) vs 13.55 (bg085). bg085 was carried on lower
   false rate (0.027 vs 0.039) / stability; the records must say so.
5. **Min-selection optimism ~0.2 px.** Headline numbers are
   min-of-many-scored-epochs (r1ttur_r1 plateau mean ~11.32 vs
   headline 11.09). Convention is uniform across runs, so rankings
   stand, but absolute claims carry ~0.2–0.5 px optimism.
6. **GroupNorm step not established.** 13.2 → 12.44 rests on a single
   draw inside the advms0 seed spread (12.67–13.65); the
   "+annealing" attribution in that row is the shared e300 schedule.
7. **"six_way" is a misnomer** (averages 3 orientations) and
   **"equilibrium" holds only on the fake side** for r1ttur: real-side
   BCE ≈ 0.24 (real logit ≈ +1.2) — D still confidently separates
   reals.
8. **DDP trust caveat on D-trajectory evidence.** The per-epoch
   trajectories come from the 2-GPU DDP logging path that the team's
   own memory flags as scrambled for some metrics; epoch means can
   also hide sub-epoch episodes. The D-health conclusions above carry
   that caveat.
9. Minor: base_r3 13.80 is best-of-two (base_r4 14.46); r1ttur "false
   0.012" is a favorable round of 0.0125.

## Open

1. **R1 vs TTUR attribution.** No isolating run exists (all 15
   X-configs scanned); r1g2-slightly-worse is weak evidence R1
   strength matters. r1only/tturonly ablations remain queued.
2. **Sub-epoch D dynamics.** Epoch-mean logs cannot exclude brief
   D-win episodes; would need step-level logging on a trusted path.
3. **Multi-block SNR.** The colored, block-dependent noise makes any
   single-block constant-floor analysis structurally inadequate; the
   corrected edge numbers come from blocks v00000/v00001 and should be
   confirmed across the val set.
4. **Unbounded head as D cue.** Whether giving the fakes a bounded
   head (or the D clipped inputs) changes the game is untested.

## Verdict

The Track A scoreboard and the claim-intact property are **certified**:
cubic 17.20 → ~13.2 → 12.44 → **11.09** (r1ttur_r1 e260, corr 0.8409,
false 0.0125), codes-only throughout, bit-exact reproducible, no
Track B contamination. The R1+TTUR step is the only inter-rung step
clearly outside noise. The *interpretation* changes: by the corrected
information analysis, the entire leaderboard below z_res ≈ 15–16 —
including both 12.44 and 11.09 — measures prior-driven reconstruction
rather than transferred axial information, and must be framed as such.
Milestone z_res < 8 unchanged; nothing approaches it.

## Addendum (same day): unbounded-head cue measured

Open item 4 partially settled: the raw head output of r1ttur_r1 e260
on a val crop (pre-denormalize, the training-domain tensor the D sees)
has ZERO voxels outside [-1, 1] — no clamp exists on that path, the
generator learned the range from its reconstruction losses (background
mass sits at the floor). Together with axx = ln 2 (D at chance on
fakes), the theoretical out-of-range D cue is not binding in practice.
Adding tanh is NOT recommended: it would do nothing at eval and would
saturate exactly where the data lives (boundary-pinned background and
ceiling-clipped peaks), starving gradients. If ever closed for rigor,
clamp the D inputs (loss-side, claim-intact), not the generator head.

## Proposed next round (slate 6) — approved direction, pending launch

Rationale. r1ttur_r2 already replicates the RECIPE EFFECT (two
unseeded draws 11.09 / 11.45 vs pre-R1 12.44; step ~1.3 px vs SE
~0.2) — so a third confirmation draw would only tighten the 11.1–11.5
band, not add knowledge. The slate instead completes the full 2×2 of
the validation's biggest open item (R1 vs TTUR), using the existing
r1ttur runs as the both-on cell and advms0gn as the both-off cell,
plus the one surviving forward-looking architectural lever.

All three claim-intact (codes-only), Track-A base recipe
(`--l1how psf --skipl1 1 --psf_bg -0.85 --save_d --adv_ms 0
--n_epochs 300 --yaml aisr_gn`), GPU pairs 2,3 / 4,5 / 6,7,
staggered launch, e50 gate (kill z_res > 15 or false > 0.05):

| run | flags added | question answered | epochs | est. time |
|---|---|---|---|---|
| r1only_r1 | `--r1 1 --lr_g_factor 1.0` | is the 12.4→11.1 gain the gradient penalty? | 300 | ~11 h |
| tturonly_r1 | `--r1 0 --lr_g_factor 0.5` | ...or the halved generator/AE lr? | 300 | ~11 h |
| zctx_r1 | `--zctx 3` | can Z-aware quantization recover the sub-slice axial cue? (implemented + smoke-tested 2026-11-28; 4-channel latent caps block capacity — this run prices it) | 300 | ~11 h |

Readout: r1only ≈ 11.1 ⇒ R1 is the driver; r1only ≈ 12.4 and
tturonly ≈ 11.1 ⇒ the lr change was; both intermediate ⇒ synergistic.
zctx_r1 compares against the r1ttur band (11.1–11.5). Deferred:
r1ttur_r3 (n=3 band polish), genband/genhr (after the decomposition),
D-input clamp (not binding), anything mp_d (invalidated).
