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
- Decoder: 3D ConvTranspose stages decode the summed quantized latent
  (16³ grid) to the 8×-Z-upsampled volume. Codes-only.
- Multi-scale structure: kept. It is additional rate plus a residual
  decomposition (coarse = structure, fine = detail) — a fidelity aid at
  fixed codebook size and the stronger compression story. The
  multi-scale ADVERSARIAL loss was dropped (`--adv_ms 0`, a ~0.5 px
  win); the multi-scale reconstruction losses and all four quantizer
  stages remain active.

## Scoreboard (val; z_res = 0.5/fz50 px, lower is better)

Reference ladder: truth ~1 · lateral ~2 (ceiling) · 8 sampling
half-period (milestone) · ~13–14 edge of PSF data support · 16 Nyquist
period · 17.2 interpolation floor.

**Track A — claimed model (codes-only decode):**

| rung | z_res | lever |
|---|---|---|
| interpolation | 17.2 | — |
| PSF replay loss | 13.4 | data term (blind beyond ~13: axial PSF transfer ~1e-3) |
| adv_ms 0 | ~13.2 | concentrate adversarial weight at full res |
| + GroupNorm / no dropout / annealing | 12.4 | train–eval consistency, pacing |
| + R1 γ=1 + TTUR | **11.1** | D regularization (r1ttur_r1 e260: 11.1 / corr 0.841 / false 0.012; seed band 11.1–11.5, replicated) |

**Carry-forward claim-intact checkpoint: MScleanQDPSFX/r1ttur_r1 e260.**

**Track B — conditioned ablation (decoder additionally sees the raw
measurement via `--inject_x`; prices the bottleneck, NOT the claimed
model):**

| run | z_res | note |
|---|---|---|
| combo (inject_x + R1 + TTUR) | 10.6–10.8 | two seeds; flat 90+ epochs |
| + multi-plane 2.5D D (`--mp_d 5`) | **10.5** | mpd_r1 e140; flat ~150 epochs |
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
the 5-plane runs. These sharpen Track B and tell us which D recipe to
port to Track A.

## Plan

1. **Port the D-side wins to Track A (top priority, claim-intact):**
   `base + R1 + TTUR + mp_d 5 (− inject_x)`, plus hinge if slate 4
   shows its false creep is tamed by mp_d. Expected ~10.8 codes-only if
   the mpd gain transfers. This is the next claimed-model result.
2. **Latent-only generator changes** (from the three-agent architecture
   research; all opt-in in MScleanQDPSFX.py, zero-init): decode-tail
   capacity `--refine_hr` (the current tail has ONE 3×3×3 conv after
   the final upsample — the whole band beyond the PSF data support is
   synthesized by it), axial sub-pixel head `--zshuffle`, band-split losses
   `--band_split` (PSF term on the axial low band, adversarial on the
   high band — repairs the failed --residual correctly), decoder noise
   `--g_noise`, axial 1D FFC.
3. **Bottleneck-internal direction (claim-intact, new):** the binding
   narrowness is per-slice-independent quantization destroying
   inter-slice partial-volume gradations (the sub-slice axial cue).
   Z-aware encoding/quantization (slice-group context before VQ) keeps
   codes-only decode true while letting the codes carry axial
   information. Candidate for a slate-5 slot.
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
