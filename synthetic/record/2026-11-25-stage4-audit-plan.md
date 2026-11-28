# Stage 4 — code audit of loss/architecture/training dynamics, and the next sweep

2026-11-25. Context: after the crop×batch 2×2 and the 1000-epoch run
(`2026-11-22-stage4-psf-sweep.md`), the best model sits at z_res 13.4–14.1
(interpolation floor 17.2, Z Nyquist 16, metric ceiling ~1, lateral ~2) and
longer training regresses after a peak. Three independent code audits
(loss weights / architecture / training dynamics) were run to find the
highest-leverage changes before spending more GPU time. This note records
their findings and the agreed experiment slate.

## Corrections to the previous record established by the audit

- **All runs so far trained at effectively constant lr 5e-4.** The cosine
  scheduler spans `n_epochs` (networks/networks.py:81) and every launch
  leaves `n_epochs` at its 10001 default on purpose (proxy.sh:9-10;
  confirmed in every run's saved config.json and MLflow `lr_g`). The
  earlier "stable runs ended with lr annealed to ~0" reading was wrong;
  **lr annealing has never been tested.**
- The patch discriminators use InstanceNorm (hard-coded,
  networks/cyclegan/models.py:139), not BatchNorm; only the generator uses
  BatchNorm (`norm: batch`, cfg/aisr.yaml:9).

## Convergent diagnosis

The PSF Z-consistency loss is blind above the blur cutoff (axial transfer
≈ 2e-3 at 13.4 px half-period, 3e-6 at 8 px), so everything beyond the
current frontier is carried **only by the adversarial term**. That term is:

1. **Misallocated** (loss audit). Adversarial terms are ~69 % of the
   generator objective (1-GPU log, bg085_r1 e100: axx 0.890 + axx128/64
   0.868 weighted, vs PSF l1+l1c 0.39). Half the adversarial budget sits on
   the 1/2- and 1/4-res discriminators (`--adv_ms 0.5`, inherited from a
   code comment, MSclean.py:21, never swept), which cannot represent the
   16–8 px band where z_res lives. The coarse PSF term (l1c, 0.247)
   accidentally outweighs the full-res one (l1, 0.147) at equal weights.
   Also: `adv_loss_six_way` (MSclean.py:344-356) spends 1/3 of the
   adversarial weight on the near-real XY planes; qloss is double-counted
   (MSclean.py:629-632 + vqperceptual.py:107); `lamb` is 10 in aisr.yaml
   but 5 in proxy.sh; prefix terms (l1p+lpipsp ≈ 0.56 on half the steps,
   z-blind) exceed the whole PSF term.
2. **Starved of evidence** (architecture audit). The measured voxels never
   reach the 3D generator: the only input path is the per-slice quantized
   16×16×4 latent (encoder MSclean.py:278-328); `self.Xup` (trilinear
   input) is computed (MSclean.py:449) but unused by the decode path — no
   skip, no residual. This matches the fplane signature (lateral retreat
   beyond ~0.10 c/px where interpolation is perfect; +3-plane Z offset):
   measured information is destroyed by the bottleneck and re-hallucinated.
   Additional findings: Dropout(0.5) is active in the decode trunk
   (`mc: true` in cfg/aisr.yaml:10 → ed023eMSfpn.py:204-241) at the exact
   stages synthesizing axial detail; generator BatchNorm at per-GPU batch 1
   plus EMA (LitEma tracks only parameters, ldm/modules/ema.py:16-21, so
   checkpoints pair EMA weights with non-EMA BN buffers) plus dropout =
   triple train/eval mismatch; the patch D's stride-2 k4 convs alias the
   8–16 px band (the repo's own BlurPool comment, models.py:95-104) and an
   anti-aliased `patchblur_16` exists unused (registry.py:101-108).
3. **Unstable by construction** (dynamics audit). Vanilla saturating BCE
   (`gan_mode: vanilla`), no R1/spectral norm/instance noise/D
   augmentation anywhere, ~24:1 fake:real slice imbalance per step (D sees
   3 orientations × 128+ slices fake vs 16-24 real XY slices from the same
   crop). At identical constant lr, decay onset is monotone in D
   data-throughput per update: full+b12 ~e60 < proxy+b12 ~e250 < full+b2
   (mild, e100) < proxy+b2 (stable ≥e100). Diagnosis: D saturation/overfit
   kills the adversarial gradient; the other losses hold everything they
   can see ("keeps coarse structure, stops resolving"). Generator EMA is
   already on (decay 0.9999, ~10-20-epoch horizon — cannot mask a
   50-epoch decay). `--lr_g_factor` (TTUR) exists but defaults to 1.
   `--save_d` exists but was never passed: no past D can be probed.
   Data (400 blocks, rotation+crop aug, no flips) is not the current
   binding constraint, but more synthetic blocks are nearly free and would
   delay D memorization for long runs.

## Ranked conclusions

- **Most impactful hyperparameter: `--adv_ms` → 0** (redirects the whole
  adversarial budget to the only D that sees the target band), then the
  adv:PSF ratio (`--lamb` 5 → 2; prefer lowering lamb over raising adv,
  since `--adv` also scales the D loss, MSclean.py:651).
- **Most impactful design change: residual over interpolation** — output =
  Xup + learned residual (minimal: final activation 'none' +
  `XupX = Xup + out0` in generation(); better: also inject oriX at the up1
  stage). Only change that adds information; makes the measured band exact
  by construction; converts the decay mode into "degrade toward
  interpolation" instead of below it.
- **Most impactful for sustained training: D stabilization** — TTUR
  (`--lr_g_factor 2` with halved base lr), R1 penalty on real XY slices
  (~15 lines in backward_d), `patchblur_16`, flips/instance noise on D
  inputs, lr annealing (`--n_epochs <run length>`).
- Secondary cleanups: `--norm group` + `mc` off (removes the triple
  train/eval mismatch), `lamb_coarse 0.25`, `p_prefix 0` (z-blind terms;
  also fixes DDP log scrambling), fix qloss double-count, never enable
  `lbm_ms_ssim` (its target is interpolation — anti-deconvolution).

## Experiment slate (planned; proxy crop, 100 epochs, 1 GPU × b1 each,
all with `--save_d`, judged by checkpoint-scored z_res/false/recall)

Base recipe `P` = `--models MScleanQDPSF --l1how psf --skipl1 1 --psf_bg -0.85`.

| run | change vs winner | tests |
|---|---|---|
| advms0_r1 | `--adv_ms 0` | adversarial reallocation |
| lamb2_r1 | `--lamb 2` | adv:PSF ratio |
| advms0lamb2_r1 | `--adv_ms 0 --lamb 2` | combination |
| blurD_r1 | `--netD patchblur_16` | D anti-aliasing in the win band |
| gnorm_r1 | `--norm group` + mc off | train/eval mismatch, batch-robustness |
| skip_r1 | residual-over-Xup code change | information path (biggest swing) |
| anneal_r1 | winner + `--n_epochs 100` | first-ever lr-annealed run (control+) |
| (CPU probe) | average long_r1 e150–250 net_g, score | free peak-exploitation check |

Launched 2026-11-25: runs 1–6 on GPUs 0–5 (anneal_r1 control skipped — three
winner seeds already serve as baseline; GPUs 6–7 spare). The averaging probe
is done and **negative**: the e150–250 weight average (all components) scores
15.0 / 0.794 vs the best single checkpoint's 14.1 / 0.795 — the peak-window
checkpoints drift directionally (decay), they are not noise around a flat
optimum, so SWA-style averaging mixes in the decline. Checkpoint selection by
scored evaluation stands.

Read-out: any run beating 13.4 by >0.5 px on val (block spread ±1.6 — needs
confirmation seeds before adoption) graduates to a replication + 300-epoch
annealed run with D stabilization (TTUR/R1) to test "longer finally helps".
Failure modes to watch: hallucination (false rate up) for advms0/lamb2;
early divergence for gnorm/skip.

## Results (val, checkpoint-scored; baseline winner seeds 13.4/14.1/14.1)

### advms0_r1 — `--adv_ms 0` (completed first, scored on GPU 6)

| epoch | psnr | corr | z_res | false | recall |
|---|---|---|---|---|---|
| 10 | 22.16 | 0.780 | 14.9 | 0.027 | 0.969 |
| 20 | 22.46 | 0.796 | 13.6 | 0.025 | 0.977 |
| 30 | 22.53 | 0.800 | 13.6 | 0.024 | 0.980 |
| 40 | 22.61 | 0.804 | 13.3 | 0.025 | 0.982 |
| 50 | 22.62 | 0.804 | 13.2 | 0.025 | 0.983 |
| 60 | 22.61 | 0.804 | 13.1 | 0.025 | 0.983 |
| 70 | 22.64 | 0.805 | **12.9** | 0.024 | 0.983 |
| 80 | 22.62 | 0.804 | 13.1 | 0.024 | 0.983 |
| 90 | 22.67 | 0.807 | 13.0 | 0.024 | 0.983 |
| 100 | 22.64 | 0.805 | **12.7** | 0.025 | 0.983 |

**New best: 12.7 at e100 — 0.7 px below the 13.4 baseline best**, meeting
the >0.5 px graduation bar (replication seeds still required; block spread
±1.6). No hallucination penalty: false rate 0.024–0.025 (baseline ~0.028)
and recall 0.983 are both *better* than baseline. Still improving at e100
(12.9→13.1→13.0→12.7 over e70–100) — no decay onset, consistent with the
D-saturation story (all adversarial budget on one D = less fake throughput
per D per step than three-headed baseline... but also simply a better-aimed
gradient). The audit's top-ranked hyperparameter call (`--adv_ms 0`) is
confirmed in direction and magnitude.

### advms0lamb2_r1 — `--adv_ms 0 --lamb 2`

| epoch | psnr | corr | z_res | false | recall |
|---|---|---|---|---|---|
| 10 | 21.72 | 0.753 | 16.6 | 0.034 | 0.967 |
| 40 | 22.33 | 0.789 | 13.7 | 0.027 | 0.979 |
| 80 | 22.44 | 0.795 | **13.3** | 0.025 | 0.982 |
| 100 | 22.42 | 0.794 | 13.5 | 0.026 | 0.981 |

Best 13.3 (e80): matches baseline, clearly behind advms0 alone (12.7).
Slower start (16.6 at e10 vs 14.9) and a lower plateau — lowering `--lamb`
*on top of* adv_ms 0 weakens the PSF anchor without buying anything; the
combination is not additive. `--adv_ms 0` with lamb 5 stands as the recipe.

### lamb2_r1 — `--lamb 2` (multi-scale D kept)

| epoch | psnr | corr | z_res | false | recall |
|---|---|---|---|---|---|
| 10 | 21.63 | 0.747 | 16.6 | 0.026 | 0.960 |
| 40 | 22.24 | 0.784 | 14.8 | 0.027 | 0.977 |
| 70 | 22.32 | 0.789 | **14.2** | 0.026 | 0.979 |
| 100 | 22.25 | 0.785 | 14.9 | 0.026 | 0.980 |

Best 14.2 (e70): within the baseline seed range (13.4–14.1), slower start,
mild drift back after e70. Lowering the PSF weight alone does nothing —
consistent with advms0lamb2: the gain lives entirely in `--adv_ms 0`, and
weakening the PSF anchor is neutral-to-negative.

### gnorm_r1 — `--yaml aisr_gn` (GroupNorm generator, dropout off)

| epoch | psnr | corr | z_res | false | recall |
|---|---|---|---|---|---|
| 10 | 20.98 | 0.698 | 20.3 | 0.038 | 0.955 |
| 50 | 22.03 | 0.772 | 15.7 | 0.027 | 0.979 |
| 90 | 22.41 | 0.794 | **13.6** | 0.025 | 0.981 |
| 100 | 22.41 | 0.794 | 13.7 | 0.024 | 0.982 |

Best 13.6 (e90): matches the baseline best despite a much slower start
(20.3 at e10 — GroupNorm trains slower here), monotone improvement with no
sign of the feared early divergence, and the cleanest false rate of the
slate (0.024). It removes the triple train/eval mismatch for free and is
still improving at e100, so it's the natural *combination* partner:
advms0 + gnorm is the obvious next pairing for the 300-epoch annealed run.

### blurD_r1 — `--netD patchblur_16` (anti-aliased patch D) — negative

| epoch | psnr | corr | z_res | false | recall |
|---|---|---|---|---|---|
| 10 | 21.97 | 0.770 | 15.5 | 0.033 | 0.977 |
| 20 | 22.32 | 0.789 | **14.7** | 0.031 | 0.982 |
| 60 | 22.26 | 0.785 | 15.4 | 0.028 | 0.983 |
| 100 | 22.21 | 0.783 | 15.3 | 0.027 | 0.983 |

Best 14.7 (e20), then drifts to 15.3–15.6 — worse than every baseline seed.
BlurPool's low-pass before each stride-2 downsample removes exactly the
high-frequency sensitivity the D needs in the 8–16 px band: the aliasing the
code comment warns about was apparently *helping* (any band-folding still
provides gradient there), and smoothing it away weakens the only loss that
operates in the win band. Anti-aliasing the D is the wrong direction here.

### skip_r1 — `--residual` (output = trilinear Xup + learned correction) — **failed**

| epoch | psnr | corr | z_res | false | recall |
|---|---|---|---|---|---|
| 10 | 21.58 | 0.768 | 16.7 | 0.063 | 0.987 |
| 50 | 21.43 | 0.754 | 16.7 | 0.105 | 0.990 |
| 100 | 21.14 | 0.737 | 17.3 | 0.140 | 0.989 |

z_res pinned at the interpolation floor (16.5–17.3) for all 100 epochs while
the false rate climbs monotonically (0.063 → 0.140, ~5× baseline) and corr
falls — the predicted failure mode. Reading: with the output anchored at Xup
and the final tanh removed, the PSF data term is already nearly satisfied by
the anchor, so the correction branch gets no data-term shaping and the
adversarial gradient buys texture (hallucinations) instead of resolution.
Residual-over-interpolation in this minimal form (anchor at the output only,
no encoder-side skip) does **not** work; if revisited, it needs the stronger
variant (inject oriX into the decoder stages) and/or a loss re-balance.

## Slate verdict (all six runs scored, 2026-11-25)

| run | best z_res (epoch) | vs baseline 13.4 | trajectory |
|---|---|---|---|
| **advms0_r1** | **12.7 (e100)** | **−0.7 ✓** | still improving at e100 |
| advms0lamb2_r1 | 13.3 (e80) | −0.1 | plateau |
| gnorm_r1 | 13.6 (e90) | +0.2 | still improving at e100 |
| lamb2_r1 | 14.2 (e70) | +0.8 | mild late drift |
| blurD_r1 | 14.7 (e20) | +1.3 ✗ | decays after e20 |
| skip_r1 | 16.5 (e70) | +3.1 ✗ | stuck at interp. floor, false 5× |

- `--adv_ms 0` is the single change that matters: −0.7 px, better false
  rate/recall, and no decay within 100 epochs. Pairing it with `--lamb 2`
  subtracts from it; `--lamb 2` alone is neutral.
- GroupNorm (+ dropout off) matches baseline with the cleanest false rate
  and a still-rising curve — the right robustness partner for longer runs.
- Anti-aliasing the D and output-level residual both hurt; neither of the
  "design" changes beats the loss-reallocation hyperparameter.
- Next (per read-out): replication seeds of advms0 (advms0_r2/r3) and a
  300-epoch advms0 + gnorm run with lr annealing (`--n_epochs 300`) and
  TTUR, to test whether "still improving at e100" extends.

## Graduation slate (launched 2026-11-25, 2 GPU × 3 runs)

| run | description | GPUs | epochs | expected time |
|---|---|---|---|---|
| advms0_r2 | replication seed of `--adv_ms 0` (eff. batch 2) | 2,3 | 100 | ~3.5 h |
| advms0_e300 | `--adv_ms 0` + lr annealing (`--n_epochs 300`) | 4,5 | 300 | ~10.5 h |
| advms0gn_e300 | advms0 + GroupNorm/no-dropout (`--yaml aisr_gn`) + annealing | 6,7 | 300 | ~10.5 h |

All DDP pairs avoid GPUs 0+1 (NCCL hang). 200 steps/epoch at ~1.6 it/s
confirmed healthy at launch. Training-log metrics are unreliable under DDP
(known scrambling) — judge by checkpoint scoring only. Note the 2-GPU setup
means effective batch 2, so advms0_r2 is a replication of the *effect*, not
a bit-exact replication of r1's batch-1 config; batch 2 was the stable best
cell of the crop×batch 2×2.

### advms0_r2 — replication seed (2 GPU, eff. batch 2) — partial replication

| epoch | psnr | corr | z_res | false | recall |
|---|---|---|---|---|---|
| 10 | 21.34 | 0.728 | 19.7 | 0.040 | 0.968 |
| 50 | 22.59 | 0.803 | 13.8 | 0.025 | 0.981 |
| 90 | 22.61 | 0.804 | **13.7** | 0.025 | 0.982 |
| 100 | 22.60 | 0.803 | 13.7 | 0.025 | 0.982 |

Best 13.7 — does **not** reproduce r1's 12.7. It still edges the baseline
seed band (13.4/14.1/14.1) on its stability (flat 13.7–14.2 from e30, no
decay, false 0.025), but the honest read is: advms0's expected value is
~13.5±0.5, and r1's 12.7 was a good draw (or a batch-1 effect — r2 ran at
eff. batch 2, a confound we accepted for speed). The >0.5 px claim is not
confirmed yet; the two 300-epoch advms0 runs add two more draws at e100.

### 300-epoch annealed runs (scored 2026-11-26; last checkpoint is e290 — with `--n_epochs 300` training ends at epoch 299, saves at multiples of 10)

**advms0gn_e300** (`--adv_ms 0` + GroupNorm/no-dropout + annealing) — **new best 12.4**:

| epoch | psnr | corr | z_res | false | recall |
|---|---|---|---|---|---|
| 10 | 20.96 | 0.698 | 19.3 | 0.031 | 0.936 |
| 50 | 22.50 | 0.800 | 13.3 | 0.019 | 0.972 |
| 80 | 22.67 | 0.809 | 12.7 | 0.020 | 0.976 |
| 100 | 22.62 | 0.806 | 12.8 | 0.021 | 0.976 |
| 120 | 22.61 | 0.806 | 12.5 | 0.022 | 0.977 |
| 140 | 22.61 | 0.805 | **12.4** | 0.022 | 0.977 |
| 200 | 22.50 | 0.801 | 12.8 | 0.025 | 0.980 |
| 240 | 22.43 | 0.797 | 13.1 | 0.026 | 0.980 |
| 290 | 22.32 | 0.791 | 13.5 | 0.029 | 0.981 |

**advms0_e300** (annealing only): best 13.2 (e100), plateau 13.2–13.8
through e220, mild decay to 14.8 by e290.

Conclusions:

1. **advms0 + GroupNorm + annealing is the new recipe: z_res 12.4 (e140),
   1.0 px below the old 13.4 best**, with the lowest false rates seen
   (0.019–0.022 around the peak). It is the second independent run to reach
   ≤12.7, so the gain is real, not r1 seed luck.
2. **"Longer finally helps" — partly.** The gn run improves past e100
   (12.8 → 12.4 at e140) and its late decay is mild and slow (13.5 at e290)
   instead of the old collapse to 16.5+. Annealing alone (advms0_e300)
   does not extend improvement but does soften the decay (14.8 at e290 vs
   the unannealed long run's 17+). GroupNorm/no-dropout is what buys the
   extra headroom; the stabilization stack worked.
3. Four advms0 draws at e100 — 12.7 (r1, b1), 13.7 (r2), 13.2 (e300),
   12.8 (gn_e300) — mean ≈ 13.1 vs the baseline band 13.4–14.1: `--adv_ms 0`
   is worth ~0.5–0.7 px on average, and gnorm adds the rest.
4. Checkpoint to carry forward: **advms0gn_e300 e140** (12.4 / corr 0.805 /
   false 0.022). A confirmation seed of advms0+gn and the move to real
   THX10 data (stage 5) are the natural next steps.
