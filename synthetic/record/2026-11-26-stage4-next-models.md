# Stage 4 — proposed next models (toward a drastic z_res move)

> **Framing note (added 2026-11-28, applies to this whole log):** the
> project keeps the compression / latent-only claim — the published
> model decodes from the quantized multi-scale latent alone; the
> measured volume may shape training losses but not the decode path.
> Runs in this log therefore split into two tracks:
> **claim-intact** (codes-only decode: base, advms0gn, r1ttur_r1/r2,
> r1g2_r1, clean_r1 — best 11.1, r1ttur_r1 e260) and
> **conditioned ablations** (any run with `--inject_x`: injx_r1, the
> combo family, and ALL of slates 3–4 — best 10.5, mpd_r1 e140). The
> conditioned track is not the claimed model; it prices the bottleneck
> (~0.3–0.7 px at matched epochs). "Campaign best" below, where it
> refers to the inject_x lineage, predates this decision. D-side
> changes (adv_ms 0, R1, TTUR, hinge, mp_d) are claim-compatible but
> mp_d/hinge are so far only measured on the conditioned track; the
> port `R1+TTUR+mp_d 5 − inject_x` is the top-priority claim-intact
> run. Current status doc: 2026-11-28-stage4-status.md.

2026-11-26. Current best: advms0gn_e300 e140, z_res 12.4 (val). Goal: push
toward single digits (≤10; ~8 = unambiguously past the 8-px sampling
half-period, pure-prior territory). Everything below ~13 is carried by the
adversarial prior, so the candidates either feed the generator more
evidence or sustain/sharpen the adversarial pressure. Base recipe for all:
the current winner `P* = --l1how psf --skipl1 1 --psf_bg -0.85 --save_d
--adv_ms 0 --n_epochs 300 --yaml aisr_gn` (GroupNorm, no dropout,
annealed). One change per run.

## Proposal 1 — decoder-stage input injection (`--inject_x`) [highest leverage]

**Mechanism.** The generator's only input today is the per-slice VQ
16×16×4 latent; quantization destroys the small intensity gradations
(partial-volume/PSF ratios across neighboring slices) that carry
sub-Nyquist axial position, and fine lateral detail is re-hallucinated
from 16×16 codes. Hence the +3-plane z-offset and lateral retreat. Fix:
concatenate a trilinearly resized copy of the measured volume (oriX) into
the decoder at each up-stage, each followed by a 1–3³ merge conv, keeping
the VQ path untouched as the structural prior. U-Net-style skip from the
*input* only — no encoder surgery, slots in next to the existing
inject_convs pattern in ed023eMSfpn.

**Why not skip_r1 again.** skip_r1 anchored the *output* (after all convs,
tanh removed): the data term was satisfied instantly by the anchor and the
adversarial gradient bought hallucinated texture (false 0.063→0.140).
Feature-level injection arrives *before* several conv stages and the final
tanh, so all losses still shape how the evidence is used.

**Implementation.** New flag `--inject_x` in MSclean.py; in
`_netg_decode`/ed023eMSfpn decode trunk, at each up-stage s: `h = conv_s(
cat(h, interpolate(oriX, size=h.shape[2:])))`. New conv weights only; the
rest of the checkpoint structure unchanged. Final activation stays tanh.

**Read-out.** Expect z_off → ~0 and lateral spectrum tracking
interpolation out to high f (both scored/visible in fplane); z_res should
drop if the freed capacity goes axial. Failure mode: interpolation
collapse (z_res →17, like skip_r1) or false-rate climb — both visible by
e50, so a bad run can be killed early.

## Proposal 2 — sustained adversarial pressure: R1 + TTUR (`--r1 γ`, `--lr_g_factor 0.5`)

**Mechanism.** Improvement stalls at ~e140 even annealed; diagnosis is D
saturation/overfit (vanilla BCE, no regularization, ~24:1 fake:real slice
imbalance). R1 gradient penalty on real XY slices (~15 lines in
backward_d, applied to net_d only) is the standard fix for D overfit on
small real sets; TTUR via the existing `--lr_g_factor` (G at lr/2, D at
lr) keeps D ahead without saturating. Target: keep the e100–e140 slope
going to e250+ instead of plateauing.

**Implementation.** `--r1` flag (every-16-step lazy R1, γ≈1) in
backward_d. TTUR: verified in code (MSclean.py:720-723) that lr_d = lr and
lr_g = lr_g_factor × lr, so `--lr_g_factor 0.5` gives G 2.5e-4 / D 5e-4
with no other change (note lr_g also covers the encoder/decoder/VQ stack —
the whole generator side, which is what we want slower). Flips/instance
noise on D inputs held in reserve.

## Proposal 3 — loss cleanup bundle (`--p_prefix 0 --lamb_coarse 0.25`)

**Mechanism.** Remaining dead/harmful weight: z-blind prefix terms
(~19 % of the generator objective on half the steps), coarse PSF term l1c
outweighing full-res l1 at equal weights, qloss double-count, 1/3 of
adversarial weight on near-real XY planes. Cheapest run of the three; also
fixes the DDP log scrambling (p_prefix off). qloss and six-way re-weight
are code edits — hold for a second bundle so this run stays flag-only.

## Proposal 4 — more synthetic data (800+ blocks) [enabler, not a model]

Nearly free to generate; delays D memorization in long runs. Do before any
500+-epoch training; not expected to move 100–300-epoch results.

## Proposal 5 (reserve) — stronger full-res D / hinge loss

Wider patch D or an extra D at native resolution with hinge loss instead
of vanilla BCE. Only if 1–3 stall: more D capacity without regularization
historically accelerates the decay.

## Proposed slate (2 GPU per run, pairs from 2+3/4+5/6+7; 0+1 NCCL-unsafe as a pair)

| run | change vs P* | epochs | expected time |
|---|---|---|---|
| injx_r1 | `--inject_x` (proposal 1) | 300 | ~10.5 h |
| r1ttur_r1 | R1 + TTUR (proposal 2) | 300 | ~10.5 h |
| clean_r1 | `--p_prefix 0 --lamb_coarse 0.25` (proposal 3) | 300 | ~10.5 h |

All three are independent single-change runs off P*, so winners compose;
any ≥0.5 px gain (vs 12.4, spread caveats as before) graduates to a
combined run + replication seed. Kill criteria at e50: z_res > 15 or
false > 0.05 (injx interpolation-collapse / hallucination check).

## e50 gate (scored 2026-11-26, val) — all pass, no kills

| run | e10 z_res/false | e30 | e50 | note |
|---|---|---|---|---|
| injx_r1 | 31.9 / 0.062 | 14.8 / 0.031 | **12.9 / 0.025** | best e50 ever seen (advms0gn hit 12.4 only at e140); z_off 3.60 — NOT reduced vs peers (~3.4–3.5), so injection helps z_res without fixing plane offset yet |
| r1ttur_r1 | 19.1 / 0.055 | 15.1 / 0.016 | **13.1 / 0.013** | lowest false rate in the PSF family |
| clean_r1 | 19.4 / 0.040 | 15.0 / 0.025 | **14.4 / 0.025** | slowest; on the baseline trajectory |

## e60–e100 (scored 2026-11-26, val, z_res by epoch)

| run | e60 | e70 | e80 | e90 | e100 | read |
|---|---|---|---|---|---|---|
| r1ttur_r1 | 12.7 | 12.3 | 12.0 | 11.8 | **11.8** | first sub-12 of the campaign (crossing at e80); corr 0.835 best-ever, false 0.010 near-best (MScleanQD base_r4 0.0088 remains lower); already ≥0.5 px past 12.4 → graduates on completion |
| injx_r1 | 13.1 | 12.8 | 12.7 | 12.6 | **12.4** | ties old best at e100 (advms0gn needed e140); z_off ~3.9 unchanged — helps z_res, not plane offset |
| clean_r1 | 14.5 | 14.9 | 15.1 | 15.4 | **15.2** | decays after e50; proposal 3 negative. Left running on idle GPUs for the annealing tail only |

R1+TTUR is doing exactly what proposal 2 predicted: false rate at the
pre-PSF base level (0.010–0.012) while z_res still falls — the first run
where hallucination and axial resolution improve together. Obvious
combination candidate: r1ttur + injx (independent changes, both ≥ old
best at e100).

## e120–e200 (scored 2026-11-27, val, z_res by epoch)

| run | e120 | e140 | e160 | e180 | e200 | read |
|---|---|---|---|---|---|---|
| r1ttur_r1 | 11.6 | **11.3** | 11.4 | 11.5 | **11.3** | new campaign best; flat e140–e200 (corr 0.840, false 0.011–0.012) — late decay eliminated, not just softened. Graduates (−1.1 px vs 12.4) |
| injx_r1 | 12.4 | 13.1 | 13.4 | 14.1 | 14.3 | peaked e100–120 at 12.4, then the familiar decay — injection adds evidence but doesn't protect D |
| clean_r1 | 15.7 | 15.5 | 16.0 | 15.5 | 16.4 | confirmed negative |

Interpretation: the two mechanisms are complementary — injx improves the
early evidence path (ties old best at e100) but still suffers D
saturation; r1ttur keeps D healthy and gets both the gain and the
stability.

## Slate 2 (launched 2026-11-27, after slate-1 completion; all MScleanQDPSFX, base P*)

| run | change vs P* | GPUs | epochs | expected time |
|---|---|---|---|---|
| combo_r1 | `--inject_x --r1 1 --lr_g_factor 0.5` (injx + r1ttur) | 2,3 | 300 | ~10.5 h |
| r1ttur_r2 | `--r1 1 --lr_g_factor 0.5` (replication seed) | 4,5 | 300 | ~10.5 h |
| r1g2_r1 | `--r1 2 --lr_g_factor 0.5` (stronger R1) | 6,7 | 300 | ~10.5 h |

Same e50 kill gate (z_res > 15 or false > 0.05). Reference to beat:
r1ttur_r1 11.3 (e140/e200).

## Slate 1 final (e220–e290 scored 2026-11-27, val)

| run | e220 | e240 | e260 | e280 | e290 | best overall |
|---|---|---|---|---|---|---|
| r1ttur_r1 | 11.5 | 11.2 | **11.1** | 11.3 | 11.4 | **11.1 at e260** (corr 0.841, false 0.012) — new campaign best; flat 11.1–11.5 from e140 to e290, no collapse in 300 epochs |
| injx_r1 | 14.7 | 14.7 | 14.9 | 14.8 | 15.1 | 12.4 at e100–120, then decay; mechanism retested with R1 in combo_r1 |
| clean_r1 | 16.6 | 17.1 | 17.3 | 16.8 | 16.8 | 14.4 at e50; confirmed negative, proposal 3 closed |

**Checkpoint to carry forward: MScleanQDPSFX/r1ttur_r1 e260** — z_res
11.1 / corr 0.841 / false 0.012. Progress ladder: interpolation 17.2 →
PSF 13.4 → advms0+gn 12.4 → **R1+TTUR 11.1** (~40 % of floor→ceiling).
Annealing tail still improving at e240–260 suggests a longer run
(600 ep, with the 800-block dataset per proposal 4) once slate 2 reads
out.

## Slate 2 — e50 gate (scored 2026-11-27, val) — all pass

| run | e10 z_res/false | e30 | e50 | note |
|---|---|---|---|---|
| combo_r1 | 14.7 / 0.020 | 11.9 / 0.011 | **11.8 / 0.010** | injx+R1 synergy confirmed early: slate-1 winner needed ~e100 for 11.8; injx's early gain survives under R1 |
| r1g2_r1 | 17.7 / 0.046 | 14.3 / 0.020 | **12.7 / 0.015** | γ=2 converges faster than γ=1 did (13.1 at e50) |
| r1ttur_r2 | 17.5 / 0.041 | 14.1 / 0.016 | **13.3 / 0.013** | replicates r1ttur_r1's trajectory (13.1 at e50) |

## Slate 2 — e60–e100 (scored 2026-11-27, val, z_res by epoch)

| run | e60 | e70 | e80 | e90 | e100 | read |
|---|---|---|---|---|---|---|
| combo_r1 | 11.6 | 11.5 | 11.3 | 11.3 | **11.2** | within 0.1 of the all-time best (11.1 @ e260) at e100, monotone falling; corr 0.844 best-ever (recall 0.988 is high but below bg095_r1's 0.992–0.993, which came with far higher false rates) |
| r1g2_r1 | 12.3 | 12.1 | 12.3 | 12.0 | **12.1** | early γ=2 advantage fades; slightly behind γ=1 at e100 (11.8) |
| r1ttur_r2 | 13.0 | 12.8 | 12.6 | 12.5 | **12.2** | ~0.4 px behind seed r1 at e100, same shape — recipe replicates |

## Slate 2 — e120–e200 (scored 2026-11-27, val, z_res by epoch)

| run | e120 | e140 | e160 | e180 | e200 | read |
|---|---|---|---|---|---|---|
| combo_r1 | 11.1 | 11.0 | 10.9 | 11.0 | **10.8** | **first sub-11 run (crossing at e140–160); new campaign best** (corr 0.848 best-ever; false 0.010–0.011 and recall 0.991 near-best); r1ttur-stability + injection gain compose |
| r1ttur_r2 | 12.0 | 11.9 | 11.8 | 11.7 | 11.8 | plateau 11.7–12.0 — replicates sibling shape ~0.5 px higher; R1+TTUR seed spread ≈0.5 px |
| r1g2_r1 | 12.1 | 12.1 | 12.3 | 12.0 | 12.0 | flat 12.0–12.3 — γ=2 slightly worse than γ=1; keep γ=1 |

Ladder: 17.2 → 13.4 → 12.4 → 11.1 → **10.8** (combo_r1 e200).

## Slate 3 (launched 2026-11-27 after slate-2 completion; all MScleanQDPSFX)

Base = combo recipe C* = P* + `--inject_x --r1 1 --lr_g_factor 0.5`.
Per user: 1 confirmation + 2 innovations, 300 epochs each (no long run yet).

| run | change vs C* | GPUs | epochs | expected time |
|---|---|---|---|---|
| combo_r2 | none (replication seed of combo_r1) | 2,3 | 300 | ~10.5 h |
| mpd_r1 | `--mp_d 5`: extra multi-plane (2.5D) patch D on 5-adjacent-plane stacks (fake: Y/X/Z unfolds of XupX; real: 1-px-shifted XY-slice stacks, SliceGAN-style); R1 covers it | 4,5 | 300 | ~10.5 h |
| hinge_r1 | `--d_hinge`: hinge margins for D, non-saturating −logits for G, all add_loss_adv discriminators | 6,7 | 300 | ~10.5 h |

New flags implemented in models/MScleanQDPSFX.py (off by default; smoke-
tested 1 epoch with all features on: net_d_mp checkpoints and evaluator
loads 13 components; hinge signatures correct at init; ~10 % step cost).
Pitfall found: LitEma.forward crashes on any requires_grad param missing
from its map — every D is EMA-tracked, so net_d_mp requires LitEma
re-registration (the net_d_prefix "created after LitEma = untracked"
comment in MScleanQD is wrong; it would crash if adv_prefix were used
with EMA). Reference to beat: combo_r1 10.8 (e200, tail pending).

## Slate 2 final (e220–e290 scored 2026-11-27, val)

| run | e220 | e240 | e260 | e280 | e290 | verdict |
|---|---|---|---|---|---|---|
| combo_r1 | 10.8 | 10.8 | 10.8 | 10.8 | 10.8 | **winner — flat at 10.8 for 90+ epochs, zero decay; psnr/corr still rising (23.58/0.854 at e290, best ever).** Carry e200 (10.8 / corr 0.848 / false 0.011 — same z_res, lowest false) |
| r1ttur_r2 | 11.7 | 11.6 | 11.7 | 11.5 | 11.5 | replication ✓: R1+TTUR spans 11.1–11.5 across seeds |
| r1g2_r1 | 11.8 | 11.8 | 11.9 | 11.7 | 11.7 | γ=2 consistently behind γ=1 — closed |

inject_x on top of R1+TTUR is worth ~0.3–1.0 px at matched epochs
(0.33–0.71 vs the better R1 seed, 0.67–0.98 vs the worse one); one combo
seed so far — combo_r2 (slate 3) is the reproducibility test.
**Checkpoint to carry forward: MScleanQDPSFX/combo_r1 e200** (z_res 10.8 /
corr 0.848 / false 0.011). Ladder: 17.2 → 13.4 → 12.4 → 11.1 → **10.8**.

## Audit note (2026-11-27)

Independently audited against primary sources (eval JSONs in
synthetic/out/eval/, run config.json files, checkpoint inventories, the
diagnostic script+output, and the cited code): all z_res table cells,
configs, the ladder, graduation decisions, the z-offset numbers/method,
the LitEma analysis and the MScleanQDPSFX code claims verified; slate-3
GPU pairs verified live from /proc. Corrections applied: recall/false
"best-ever" superlatives removed (bg095_r1 recall 0.992–0.993 and
MScleanQD base_r4 false 0.0088 predate them; corr claims stand), "ties
all-time best at e100" softened to "within 0.1" (11.16 vs 11.09), r1ttur
e200 false prints 0.012, sub-12/sub-11 crossings noted at e80/e140–160,
and the inject_x gain restated as ~0.3–1.0 px pending the combo_r2
replication. Not verifiable from durable records: slate-1/2 GPU pairs
(session logs only), smoke-run step-cost and loss-signature observations.

## z-offset scatter diagnostic (2026-11-27) — shift is rigid, not a z_res tax

Per-crop and per-tile (2×2 lateral) lag-curve peaks over val
(scratchpad zoff_scatter.py, same estimator as evaluate.z_offset):
r1ttur_r1 e260 global +3.35 px, crop std 0.31 (range +2.5..+4.3),
within-crop tile std 0.59; combo_r1 e200 global +3.38, crop std 0.28,
tile std 0.52 (slightly tighter — injection stabilizes placement a bit).
Gaussian-jitter attenuation at fz50 ≈ 0.046 cyc/px: exp(−2π²f²σ²) ≈
1–2 % — negligible, likewise at the 8-px target band. The +3.4 px bias
is compensated by the evaluator's global alignment and is a one-line
constant correction at deployment. Conclusion: no z-consistency loss
needed; the offset is a symptom of lost sub-slice metrology, not an
active limiter. (cubic is not a valid estimator-noise floor: its blurred
output makes the lag peak itself noisy — crop std 2.87.)

## Slate 3 e50 gate (scored 2026-11-27, val) — all pass

Kill criteria at e50: z_res > 15 or false > 0.05. z_res (false):

| run | e10 | e30 | e50 | gate |
|---|---|---|---|---|
| combo_r2 | 15.5 (0.030) | 11.9 (0.010) | 11.4 (0.008) | pass |
| mpd_r1 | 19.9 (0.058) | 13.5 (0.071) | 11.9 (0.011) | pass |
| hinge_r1 | 14.9 (0.027) | 11.7 (0.013) | 11.2 (0.011) | pass |

For scale, combo_r1 (the 10.8 winner) scored 14.7 / 11.9 / 11.8 at the
same epochs. combo_r2 tracks it through e30 and is 0.4 ahead at e50;
hinge_r1 at 11.2 is the fastest e50 in the campaign (0.6 ahead of
combo_r1's pace) with false already at 0.011. mpd_r1 starts rough (the
extra D destabilizes early training: false 0.071 at e30) but converges
hard, 13.5 → 11.9 over e30–e50 with false collapsing to 0.011. Early
pace has not always predicted the e200 ordering (combo_r1's lead opened
late), so no conclusions yet; next read at e100.

## Slate 3 e60–e100 (scored 2026-11-27, val) — both innovations ahead of the winner's pace

z_res (combo_r1 same-epoch reference: 11.6 / 11.5 / 11.3 / 11.3 / 11.2):

| run | e60 | e70 | e80 | e90 | e100 | false @e100 |
|---|---|---|---|---|---|---|
| combo_r2 | 11.2 | 11.2 | 11.1 | 11.1 | 11.1 | 0.008 |
| mpd_r1 | 11.6 | 11.3 | 11.1 | 11.0 | 10.9 | 0.012 |
| hinge_r1 | 11.2 | 11.0 | 10.9 | **10.7** | 10.8 | 0.010 |

hinge_r1 reaches 10.7 at e90 — at printed precision a tick below the
all-time best (combo_r1's 10.8, which took e200), at under half the
epochs; one checkpoint, within eval noise of 10.8, so read it as
"matches the best so far, ~2× faster", not a new record yet. mpd_r1 is
also ahead of combo_r1's pace (10.9 at e100) and still descending
steeply; its early instability is fully resolved. combo_r2 tracks
combo_r1 almost exactly (11.1 vs 11.16 at e100) with the lowest false
of the slate (0.008) — the replication is so far confirming both the
recipe and the seed band. Next read e120–e200.

## Slate 3 e120–e200 (scored 2026-11-28, val) — mpd_r1 10.5, new campaign best

z_res (combo_r1 same-epoch reference: 11.1 / 11.0 / 10.9 / 11.0 / 10.9):

| run | e120 | e140 | e160 | e180 | e200 | false @e200 |
|---|---|---|---|---|---|---|
| combo_r2 | 10.9 | 10.9 | 11.1 | 10.9 | 10.8 | 0.011 |
| mpd_r1 | 10.8 | **10.5** | 10.5 | 10.5 | 10.5 | 0.014 |
| hinge_r1 | 11.0 | 11.1 | 10.9 | 10.7 | 10.6 | 0.016 |

- **mpd_r1: 10.5 from e140, flat for 60 epochs** — 0.3 px below the
  previous best (combo_r1 10.8/10.9 in this window). The multi-plane
  2.5D D is the first architecture change to beat the combo recipe.
  Not yet at the ≥0.5 graduation bar (needs ≤10.3); tail pending.
- **hinge_r1: 10.6 at e200 and still descending** (11.0 → 10.6 over
  e120–e200), but false is creeping monotonically (0.010 → 0.016, the
  highest of the slate) — watch the tail for a quality/hallucination
  trade.
- **combo_r2: 10.8 at e200 = combo_r1's e200 exactly** — the combo
  recipe and its 10.8 level are reproduced across seeds. inject_x gain
  confirmed on a second seed.

Ladder (pending tails): 17.2 → 13.4 → 12.4 → 11.1 → 10.8 → **10.5**.
Next read e220–e290 (last checkpoints; graduation decision there).

## Slate 3 final (e220–e290 scored 2026-11-28, val)

| run | e220 | e240 | e260 | e280 | e290 | verdict |
|---|---|---|---|---|---|---|
| combo_r2 | 10.7 | 10.7 | 10.6 | 10.7 | 10.7 | replication ✓ — tail edges below combo_r1's 10.8; combo seed band 10.6–10.8. false 0.012→0.015 over the tail |
| mpd_r1 | 10.5 | 10.5 | 10.5 | 10.6 | 10.6 | **campaign best — 10.5 from e140, held ~150 epochs, no decay.** Carry e140 (10.5 / corr 0.852 / false 0.013). false creeps to 0.017 by e290 |
| hinge_r1 | 10.6 | 10.6 | 10.5 | 10.5 | 10.6 | touches 10.5 late, but false climbs monotonically 0.010→0.020 (highest of the slate, ~1.5× the others) — resolution partly bought with hallucination |

All three finished cleanly at e290 (training ~16:06 → ~03:30, ~11.5 h
with eval contention). Outcomes:

- **mpd_r1: best run, below the ≥0.5 px graduation bar.**
  Gain vs combo 10.8 = 0.3 px (10.5), so no automatic graduation; the
  standing next step is a replication seed + a combination trial.
- **hinge**: works (faster early convergence, matches 10.5 eventually)
  but its false-rate trend is the wrong direction; keep it only as a
  candidate *in combination* with mp_d, where the e50–e100 acceleration
  may compound, and watch false closely.
- **combo_r2** closes the reproducibility question: combo recipe =
  10.6–10.8 across two seeds.

Ladder: 17.2 → 13.4 → 12.4 → 11.1 → 10.8 → **10.5** (mpd_r1 e140).
**Checkpoint to carry forward: MScleanQDPSFX/mpd_r1 e140** (z_res 10.5 /
corr 0.852 / false 0.013).

## Slate 4 (launched 2026-11-28) — probe the multi-plane D

Base = combo recipe C*; mpd_r1's `--mp_d 5` proved the multi-plane D is
the active mechanism, so this slate replicates it, combines it with
hinge, and probes its strength. No new code (all flags existed).
Reference to beat: mpd_r1 10.5 (e140). e50 kill gate armed on GPU 0
(z_res > 15 or false > 0.05); extra watch: kill mpdhinge_r1 if false
exceeds ~0.025 at any scored epoch (hinge's false creep).

| run | change vs C* | GPUs | epochs | expected time |
|---|---|---|---|---|
| mpd_r2 | `--mp_d 5` (replication seed of mpd_r1) | 2,3 | 300 | ~10.5 h |
| mpdhinge_r1 | `--mp_d 5 --d_hinge` (hinge's early acceleration on the mpd winner) | 4,5 | 300 | ~10.5 h |
| mpd9_r1 | `--mp_d 9` (9-plane stacks — more axial context in the D) | 6,7 | 300 | ~10.5 h |

## Generator-architecture research (2026-11-28, three independent agents)

Agents: (1) literature survey of isotropic-recon generators, (2) code
audit of our generator path, (3) SR-mechanism assessment for the
beyond-Nyquist regime. Full reports in session; convergent findings:

**Code-audit facts (checked against files):**
- The 8× axial upsample is implicit: 2D encoder folds Z into batch
  (MSclean.py:271-281), per-slice VQ latents form a 16³ grid, and net_g
  decodes with three ConvTranspose3d(k4,s2) stages. **After the final
  ×2 upsample there is exactly ONE 3×3×3 conv (32ch, conv7_k) before
  tanh** (ed023eMSfpn.py:306-307) — the whole beyond-Nyquist band must
  be synthesized by that single conv. Classic SR anti-pattern.
- Everything the decoder sees is **quantized per-slice 2D** latents
  (~2.7 kbit/slice): inter-slice partial-volume gradations — the main
  sub-Nyquist axial cue — are destroyed before any Z mixing happens.
  Only bypass is inject_x, at (Y/4,X/4) and (Y/2,X/2) — never at full
  lateral resolution. inject_x helping confirms the bottleneck binds.
- U-Net skips: encoder returns hs but MSclean.py:303 discards them.
- Literature: no published isotropic-recon method uses a VQ bottleneck;
  tokenizer papers (SeQ-GAN'24, HiTokSR'26) show VQ is a HF ceiling;
  skip-rich U-Nets (CARE/Deep-Z/IsoNet) and learned sub-pixel Z-
  upsampling (Heinrich'17, ESRGAN, FGA'25) are the recurring winners.
- Mechanisms: the target band is conditional generation (data term is
  blind there) → fix spectral bias (axial 1D FFC / frequency losses),
  give the generator stochasticity (per-layer noise), and keep adv/data
  gradients band-separated (PSF loss on axial low band, adversarial on
  high band — repairs the failed --residual, which anchored on the
  interpolated volume instead of band-splitting the generator's own
  output).

**Proposed generator slate (slate 5) — REVISED 2026-11-28 for the
latent-only framing (user: bottleneck + compression claim must hold in
every run).** All three runs are claim-intact: decode path = quantized
codes only; no run carries inject_x or any measurement/un-quantized
conditioning. Base = Track-A recipe (P* + `--r1 1 --lr_g_factor 0.5`,
NO inject_x). All opt-in flags in MScleanQDPSFX.py, zero-init.

| run | change | mechanism | claim check |
|---|---|---|---|
| latent_mpd_r1 | `--mp_d 5` (+ `--d_hinge` only if slate-4 e120–200 revives it) | port the D-side wins to codes-only decode; expected ~10.8 if the mpd gain transfers | D/loss-side only ✓ |
| genhr_r1 | `--refine_hr 2 --zshuffle` (NO inject_x_hr) | decode-tail capacity: 2 residual 3×3×3 blocks after the last upsample (today: ONE conv) + learned axial sub-pixel head; inputs are code-derived features only | decode capacity, same inputs ✓ |
| genband_r1 | `--band_split --g_noise` | PSF loss on axial low band / adversarial on high band (loss-side filter only); StyleGAN-style decoder noise, **zero at eval** (training-only stochasticity; noise carries no sample information — codes remain the sole information source; inference stays deterministic) | losses + information-free noise ✓ |

Dropped from the original proposal under the framing: `--inject_x_hr`,
`--inject_h` (conditioned decode — Track-B ablations only, not run for
now). Reserve (claim-intact): axial 1D FFC, Z-aware encoder/quantizer
context (codes carry inter-slice gradations), per-scale code-masking
ablation, ngf increase (breaks checkpoint compat). Reserve (Track B):
`--skip_hs`, Z-attention. Each new module requires the LitEma
re-register + configure_optimizers rebuild pattern
(MScleanQDPSFX.py:79-83).

## Slate 4 e50 gate (scored 2026-11-28, val) — all pass; mpdhinge 10.8 at e50

z_res (false):

| run | e10 | e30 | e50 | gate |
|---|---|---|---|---|
| mpd_r2 | 21.0 (0.084) | 13.5 (0.080) | 11.8 (0.008) | pass |
| mpdhinge_r1 | 17.0 (0.035) | 12.0 (0.010) | **10.8** (0.011) | pass |
| mpd9_r1 | 20.1 (0.071) | 15.1 (0.023) | 12.3 (0.016) | pass |

mpdhinge_r1 at e50 already matches the pre-slate-3 all-time best (10.8,
which took combo_r1 200 epochs) — fastest e50 of the campaign by 0.4 px
(prev: hinge_r1 11.2); its false watch (kill >0.025) reads 0.011, clean.
mpd_r2 tracks mpd_r1's gate almost exactly (19.9/13.5/11.9) incl. the
characteristic unstable start — good replication signal. mpd9_r1 is
0.4–0.5 behind the 5-plane runs at e50 with a rougher start: wider
stacks look worse so far, pending the recovery pattern. Next read e100.

## Slate 4 e60–e100 (scored 2026-11-28, val)

z_res (mpd_r1 same-epoch reference: 11.6 / 11.3 / 11.1 / 11.0 / 10.9):

| run | e60 | e70 | e80 | e90 | e100 | false @e100 |
|---|---|---|---|---|---|---|
| mpd_r2 | 11.4 | 11.3 | 10.9 | 10.8 | **10.6** | 0.011 |
| mpdhinge_r1 | 10.7 | 10.9 | 10.9 | 10.8 | 11.1 | 0.013 |
| mpd9_r1 | 11.8 | 11.5 | 11.1 | 10.9 | 10.8 | 0.012 |

mpdhinge's e50 head start (10.8) stalled: flat-to-worse through e100,
now behind both plain-mpd runs — same signature as solo hinge in slate
3 (acceleration, not a better endpoint); false stable 0.013–0.015, no
kill. mpd_r2 is ahead of mpd_r1's pace (10.6 at e100 vs 10.9) — strong
replication. mpd9 recovered its slow start and tracks the 5-plane pace
(wash so far). Next read e120–e200 (mpd_r1 bottomed at 10.5 from e140).
