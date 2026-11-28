# Stage 4 status (2026-11-28) — under the compression / latent-only framing

The current state of the synthetic-validation campaign, written for the
framing the project commits to: **the model is a compressive
representation — the isotropic volume is decoded from the quantized
multi-scale latent alone.** The measured volume may shape training
(losses, discriminators) but never enters the decode path at inference.
History and per-run tables: 2026-11-26-stage4-next-models.md (campaign
log), 2026-11-26-stage4-summary.md (chronology), earlier docs for the
PSF-sweep and audit eras.

## The claim and its numbers

- Representation: per-slice 2D encoder → residual multi-scale VQ, 4
  stages (code grids 2², 4², 8², 16² per 128² slice; embed_dim 4,
  codebook 256 each). Nominal rate ≈ 2.7 kbit/slice ≈ 43 kbit per
  128×128×16 crop vs ~2.1 Mbit raw → ~50× compression. Measured
  codebook perplexities 3.1 / 4.3 / 10 / 20 → effective rate roughly
  half nominal (codebooks underutilized — capacity is not the binding
  constraint).
- Decoder: 3D nearest-upsample + conv stages (netG=ed023emsfpnu uses
  the use_upsample branch of deconv3d_bn_block, not ConvTranspose)
  decode the summed quantized latent (16³ grid) to the 8×-Z-upsampled
  volume. Codes-only.
- Multi-scale structure: kept. It is additional rate plus a residual
  decomposition (coarse = structure, fine = detail) — a fidelity aid at
  fixed codebook size and the stronger compression story. The
  multi-scale ADVERSARIAL loss was dropped (`--adv_ms 0`, a ~0.5 px
  win); the multi-scale reconstruction losses and all four quantizer
  stages remain active.

## Scoreboard (val; z_res = 0.5/fz50 px, lower is better)

Reference ladder: truth ~1 · lateral ~2 (ceiling) · 8 sampling
half-period (milestone) · ~14–18.5 per-bin SNR=1 support edge and
~15–16 Wiener information bound *(erratum 2026-11-28, independent
re-audit (two-agent validation): the earlier "11–13.5, nearly flat
across lateral frequency" edge is refuted — bit-exact degradation
replay shows the noise is colored (corner floor was the spectrum
minimum, not conservative) and the edge falls ~2× across fr; per-bin
SNR=1 edge z_res ≈ 14.0–18.5, best-linear-reweighting (Wiener) bound
≈ 14.7–15.8, pooled raw-data coherence crosses 0.5 at ≈ 17.0–17.2 =
the cubic score. Everything below ~15–16, including 12.4 and 11.1, is
prior-driven reconstruction; see 2026-11-28-trackA-validation.md)* ·
16 Nyquist period · 17.2 interpolation floor.

**Track A — claimed model (codes-only decode):**

| rung | z_res | lever |
|---|---|---|
| interpolation | 17.2 | — |
| PSF replay loss | 13.4 | data term (the old "axial transfer ~1e-3" was the fr=0 worst case — see audit round 2. *Erratum 2026-11-28, independent re-audit: "carried to the measured SNR support edge ≈ 13.5" relied on the refuted 11–13.5 edge; 13.4 is already below the ~15–16 Wiener information bound, i.e. partly prior-driven — see 2026-11-28-trackA-validation.md*) |
| adv_ms 0 | ~13.2 | concentrate adversarial weight at full res |
| + GroupNorm / no dropout / annealing | 12.4 | train–eval consistency, pacing |
| + R1 γ=1 + TTUR | **11.1** | D regularization (r1ttur_r1 e260: 11.1 / corr 0.841 / false 0.0125; band 11.1–11.5 across two UNSEEDED re-runs — both seed=None, so this is an unseeded re-run, not a seeded replication; per-audit 2026-11-28, headline is min-of-scored-epochs, plateau mean ~11.3) |

**Carry-forward claim-intact checkpoint: MScleanQDPSFX/r1ttur_r1 e260.**

**Track B — conditioned ablation (decoder additionally sees the raw
measurement via `--inject_x`; prices the bottleneck, NOT the claimed
model):**

| run | z_res | note |
|---|---|---|
| combo (inject_x + R1 + TTUR) | 10.6–10.8 | two seeds; flat 90+ epochs |
| + multi-plane 2.5D D (`--mp_d 5`) | **10.5** | mpd_r1 e140; flat ~150 epochs. **Attribution invalidated 2026-11-28**: the mp real construction (shifted slice copies) is degenerate — the mp game collapsed by ~e15 (axmp 24 / dmp 4e-8 vs the healthy main game's axx 0.71); the 10.6→10.5 step is within the seed band. Score stands, mechanism does not (next-models.md, "mp_d construction audit"). |
| + hinge (`--d_hinge`) | 10.6 | faster early, but false creeps 0.010→0.020 |

**Bottleneck price: ~0.3–0.7 px at matched epochs** (Track B − Track A
with identical D-side recipe). Injection only helps under a healthy D
(alone it decays) — its false rates are the lowest of any sub-11 run
(0.008–0.013), i.e. conditioning converts hallucination into
extrapolation.

## Running now (slate 4, Track B; launched 2026-11-28 ~11:20, ends ~22:30)

mpd_r2 (replication of 10.5) · mpdhinge_r1 (`--mp_d 5 --d_hinge`) ·
mpd9_r1 (`--mp_d 9`). e50 gate passed by all three; **mpdhinge_r1 hit
10.8 at e50** — the old all-time best at a quarter of the epochs — with
false 0.011 (kill threshold 0.025 armed for hinge's creep). mpd9 lags
the 5-plane runs. After the mp_d invalidation (below) these runs no
longer nominate a D recipe for Track A; mpd_r2's final read instead
tests whether the 10.5 was seed noise (combo seed band 10.6–10.8).

**Slate 5 was cancelled by the user on 2026-11-28 before launch.** The
implemented claim-intact flags (band_split, g_noise, refine_hr,
zshuffle, zctx) stay validated in MScleanQDPSFX.py, unlaunched. No new
runs start when slate 4 releases the GPUs.

## Plan

1. **~~Port mp_d to Track A~~ — WITHDRAWN (2026-11-28 audit):** the
   mp_d real construction (shifted copies of one slice) is provably
   degenerate — channel k ≡ channel 0 translated k px, a watermark the
   D keys on; the mp adversarial game collapsed by ~e15 in mpd_r1
   while the main slice game stayed at equilibrium in the same run.
   Genuine 1-px multi-plane reals don't exist in anisotropic data, so
   there is no sound repair that doesn't collapse mp_d into the
   existing slice D. Full analysis: next-models.md, "mp_d construction
   audit". The vacated slate-5 slot goes to the Z-aware quantizer
   (item 3, zctx_r1).
2. **Latent-only generator changes** (from the three-agent architecture
   research; all opt-in in MScleanQDPSFX.py, zero-init): decode-tail
   capacity `--refine_hr` (the current tail has TWO 3×3×3 convs after
   the final upsample op — up1's conv + conv7_k; the band beyond the
   data support is synthesized by them — corrected per audit round 2),
   axial sub-pixel head `--zshuffle`, band-split losses
   `--band_split` (PSF term on the axial low band, adversarial on the
   high band — repairs the failed --residual correctly), decoder noise
   `--g_noise`, axial 1D FFC.
3. **Bottleneck-internal direction (claim-intact) — IMPLEMENTED, in
   slate 5 (user-approved 2026-11-28):** the binding narrowness is
   per-slice-independent quantization destroying inter-slice
   partial-volume gradations (the sub-slice axial cue). `--zctx 3`
   mixes a zero-init slice-group Z-context block into the encoder
   features before VQ; decode stays codes-only, rate unchanged. Takes
   the slot vacated by the withdrawn mp_d port (zctx_r1).
4. **Excluded under the framing** (kept as future ablations only):
   `--inject_x_hr`, `--inject_h`, `--skip_hs` — they condition the
   decode path on the measurement or on un-quantized features.
5. **Paper ablations to schedule later:** codes-only eval of a trained
   Track-B checkpoint (within-model bottleneck price); per-scale code
   masking (does the 2² stage earn its bits); 800-block dataset before
   any 500+-epoch run.

## Ops facts

Scoring: checkpoint eval only (DDP logs scrambled);
`ENV_NAME=brcb CUDA_VISIBLE_DEVICES=<0|1> python -m synthetic.evaluate
run --method <model/run> --epoch E --sets val`. Cadence: gate
e10/30/50 (kill z_res>15 or false>0.05), then e60–100 by 10, e120–200
and e220–290 by 20. 2-GPU DDP ~2.1 min/epoch nominal (~4.5 with three
concurrent runs + eval contention); `--n_epochs 300` → last checkpoint
e290. GPU pairs 2,3 / 4,5 / 6,7 (0+1 unsafe as a pair; 0/1 are the
scoring GPUs). New modules in MScleanQDPSFX.py need the LitEma
re-register + configure_optimizers rebuild or EMA crashes.
