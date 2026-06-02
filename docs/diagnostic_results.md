# Diagnostic Test Results — All Models

This document collects all quantitative diagnostic results from the Python tests run on each training iteration. It is the primary source for the **Results chapter** of the thesis.

## Diagnostic methodology

For every trained model we run two complementary tests:

### Test 1: Encoder/decoder responsiveness on synthetic inputs

Five spectrally distinct synthetic inputs of 2 seconds each at 44100 Hz mono:

| Input | Description |
|---|---|
| `silence` | All-zero waveform |
| `sine 220 Hz` | Pure tone at A3 (guitar 5th string open ≈ 110 Hz × 2) |
| `sine 440 Hz` | Pure tone at A4 (concert pitch) |
| `sine 880 Hz` | Pure tone at A5 (octave above concert pitch) |
| `sweep 100–2000 Hz` | Linear frequency sweep across guitar's primary range |
| `white noise` | Standard Gaussian noise scaled by 0.3 |

For each input we compute:
- **Encoder output (latent) z**: shape, mean, standard deviation
- **Decoder output y = model(x)**: mean absolute amplitude
- **Pairwise differences**: mean absolute difference between latents (encoder responsiveness) and between outputs (decoder responsiveness)

### Test 2: Reconstruction of in-distribution audio

5-second guitar excerpt from GuitarSet `gs__00_BN1-129-Eb_solo_mic.wav` (and similar samples) fed through the autoencoder. We measure:
- **Input vs output amplitude ratio** (1.0 = perfect energy preservation)
- **STFT magnitude correlation** (1.0 = perfect spectral match)

### Health thresholds

- **Encoder diff > 0.5** between distinct inputs → encoder is responsive
- **Forward diff > 0.01** between distinct inputs → decoder is responsive
- **Output amplitude in [0.05, 0.3]** for typical input → sensible energy
- **Silence amplitude near 0** → decoder respects zero input
- **Reconstruction STFT correlation > 0.3** → autoencoder works for in-distribution input

---

## guitar_v1 (epoch 3099, ~step 1.24M, Phase 1 only completed)

**Config:** `c128_r10.gin` (original BRAVE config, `LATENT_SIZE=128`, beta=0.1 constant from step 0).

**Outcome:** Posterior collapse.

**TensorBoard:** `regularization` (KL) decreased steadily from 0.83 → 0.25 throughout training — diagnostic sign of posterior collapse caused by constant high beta from step 0 preventing the encoder/decoder from learning a useful latent space.

**Python diagnostic:**
- Encoder differentiates inputs: encoder diff = 1.25 between silence and 440 Hz sine ✓
- Decoder ignores latent: forward diff = 0.006 (BAD, threshold > 0.01)
- Perceptual: model produces same output regardless of input

**Lesson carried forward:** Beta must start near zero and ramp up. Implemented log-space warmup: `initial_value = 0.0001`, `target_value = 0.1`, `warmup_len = 1000000`.

---

## guitar_v2 (epoch 7799, ~step 3.11M, full training completed)

**Config:** `c128_r10_beta_fixed.gin` (LATENT_SIZE=128 still, beta warmup applied).

**Outcome:** Partial collapse from oversized latent.

**TensorBoard:** KL plateaued at ~0.6 across 128 dimensions = **0.005 nats/dim** — encoder using almost none of its capacity.

**Python diagnostic:**
- Encoder diff = 0.79 (marginal)
- Forward diff = 0.003 (BAD)
- Output amplitude = 0.012 (BAD, ~60× quieter than input)
- 20× gain boost in Pure Data still produced random noise unrelated to input

**Lesson carried forward:** `LATENT_SIZE = 16` (matches IIL pretrained guitar model).

---

## guitar_v3 (epoch 8424, ~step 3.36M, full training)

**Config:** `c16_r10_beta_fixed.gin` (LATENT_SIZE=16). Dataset: mixed 5.79h (GuitarSet + Guitar-TECHS DI).

**Outcome:** Encoder works, generator partially collapses during Phase 2.

**TensorBoard:**
- KL plateaued at 0.65 (0.041 nats/dim — much better than v2)
- pred_real–pred_fake gap ≈ 3 mid-training, growing

**Python diagnostic (best checkpoint, epoch 6574):**
- Encoder diff = 26.4 (excellent)
- Forward diff = 0.042 (good)
- Output amplitude = 0.040 (low but present)
- Silence amplitude = 0.003 (good, ≪ signal)
- z mean = 40.3, std = 26.1 (NOT normalised — should be ≈ N(0,1))

**Perceptual:** Quiet noise with faint input-driven variation; recognisably present but never sounds like guitar.

**Lesson carried forward:** Confirms `LATENT_SIZE=16` fixes per-dim utilisation; reveals that GAN Phase 2 destabilises the model with this dataset.

---

## guitar_v4 (epoch 7511, ~step 3M, full training)

**Config:** Same as v3. Dataset: cleaner 3.05h (GuitarSet `solo_mic` only — DI removed).

**Outcome:** Full GAN discriminator dominance, generator mode collapse.

**TensorBoard:**
- KL = 0.80 (0.05 nats/dim — even better encoder utilisation)
- pred_real = 3.34, pred_fake = −2.31 → **gap = 5.65** (clear discriminator dominance)

**Python diagnostic:**
- Encoder diff = 7.5 (responsive)
- Forward diff = 0.007 (BELOW threshold)
- Output amplitude (signal) = 0.0045
- **Output amplitude (silence) = 0.017 → silence > signal** (textbook adversarial mode collapse)
- z mean = −0.79, std = 7.3

**Perceptual:** Constant low-level noise; mic input has no audible effect.

**Lesson carried forward:** Cleaner data hurt more than helped. With less data, the discriminator memorises faster, so it overpowers the generator. **Quantity + diversity** matters more than purity for adversarial training.

---

## guitar_v5 (epoch 1519, ~step 1.8M, training stopped early after collapse detected)

**Config:** `c16_r10_v5_balanced.gin` — weakened discriminator (capacity 64→32, layers 4→3), feature_matching weight 10→30, target_value 0.1→0.05, PHASE_1_DURATION 1M→1.5M. Dataset: 16h preprocessed (5.79h + IDMT-SMT-Guitar dataset4 added).

**Outcome:** Opposite failure — discriminator collapse from over-weakening.

**TensorBoard:**
- KL = 0.80 (encoder still healthy)
- pred_real = **−0.90** (NEGATIVE — discriminator labels REAL audio as fake!)
- pred_fake = −1.70 → gap = 0.80 (discriminator can't distinguish anything)

**Python diagnostic:**
- Encoder diff = 4.48 (responsive)
- Forward diff = 0.0099 (borderline)
- Output amplitude (signal) = 0.0079
- Output amplitude (silence) = 0.0172 (still silence > signal)

**Lesson carried forward:** No obvious Goldilocks discriminator capacity reachable in time. Halving the discriminator is too aggressive. The fundamental fix must come from architecture, not parameter tuning. Pivot to the official RAVE v2 + causal configs.

---

## drums_v1 (epoch 755)

**Config:** `c16_r10_beta_fixed.gin` (same as v3/v4 — custom config). Dataset: Groove 10.86 h.

**Outcome:** Below-threshold numerical diagnostic with partial transient response on informal listening.

**Python diagnostic (ep755):**
- Encoder diff = 2.40 ✓
- **Forward diff = 0.0045 — below the 0.01 health threshold defined in §3.4**
- Output amplitude (signal) = 0.0028, silence = 0.0039 (signal not louder than silence)

**Perceptual (informal Pure Data listening):** Initial listening tests suggested drum-like sounds tracking voice transients/consonants, and ep755 was originally documented as the working voice-to-drums prototype. Subsequent listening at a later date, with the same test setup, did not reproduce a perceptually compelling voice-to-drum mapping. The numerical diagnostic indicates that a "working voice-to-drums" claim cannot be substantiated on spectral grounds alone; informal listening evidence is mixed. Chapter 5.2 of the memoir is framed accordingly.

**Why drums failed less catastrophically than guitar under the same custom config:** Drum sounds are transient-driven and voice has natural transients (consonants, attacks), so even a limited decoder dynamic range produces some onset-aligned output. Guitar requires sustained tonal content that voice cannot drive.

---

## guitar_v6 (epoch 2511, ~step 2.98M, training completed naturally)

**Config:** Official `v2.gin` + `causal.gin` from `rave/configs/`, plus `guitar_v6_overrides.gin` with only two overrides: `LATENT_SIZE = 16`, `PHASE_1_DURATION = 1500000`. Dataset: 16h.

**Key v2 architectural differences from custom c16_r10:**
- `update_discriminator_every = 4` (D updates 4× slower than G)
- `MultiPeriodDiscriminator` (HiFi-GAN style) + `MultiScaleDiscriminator` combined
- `EncoderV2` / `GeneratorV2` with `amplitude_modulation = True`
- `valid_signal_crop = True`
- Relative feature-matching loss

**TensorBoard trajectory:**

| Step | pred_real | pred_fake | gap | regularization (KL) |
|---|---|---|---|---|
| 463k | — | — | — | 0.71 (smoothed) |
| 1.51M (Phase 2 start) | −1.28 | −2.11 | 0.83 | ~0.60 |
| 1.83M (version_2) | 2.12 | −4.90 | 7.02 | stable |
| 2.05M (version_3) | 7.21 | −4.67 | 11.88 | stable |
| 2.13M (version_4) | 3.46 | −5.09 | 8.55 | **gap reversed!** |
| 2.50M (version_5) | — | — | ~5–8 | stable |
| 2.98M (final)  | — | — | ~5–9 (plateau) | ~0.60 |

**First training in our series where the GAN gap reversed direction** — peaked then decreased. Confirms `update_discriminator_every=4` works on a longer timescale than initially expected.

**Python diagnostic at final epoch (2511):**
- Latent: mean = −0.029, std = 0.74 — **near-Gaussian**, the first model in our series to achieve a normalised posterior
- Output amplitudes by input: silence 0.014, sine 220 0.0085, sine 440 0.015, sine 880 0.0098, sweep 0.0082 (range 0.007)
- Pairwise output diffs (decoder responsiveness): all in range 0.015–0.024
- z mean for noise input = −549, std = 2484 (noise far out of distribution as expected)

**Reconstruction test (guitar input → model → output) — formal cross-checkpoint calibration:**

The STFT magnitude correlation diagnostic was applied to every checkpoint in the iteration sequence (N = 8 GuitarSet `solo_mic` samples × 5 s each; n_fft = 2048, hop = 512). Mean values per checkpoint:

| Checkpoint | Output RMS | STFT corr (raw) | STFT corr (time-aligned) |
|---|---|---|---|
| guitar_v1 (posterior collapse) | 0.005 | 0.16 | 0.16 |
| guitar_v2 (latent underutilisation) | 0.003 | 0.24 | 0.25 |
| guitar_v3 (Phase 2 collapse) | 0.004 | 0.24 | 0.26 |
| guitar_v4 (mode collapse) | 0.021 | 0.25 | 0.26 |
| guitar_v5 (discriminator collapse) | 0.024 | 0.26 | 0.28 |
| guitar_v6 Phase 1 (epoch 1499) | 0.041 | 0.19 | 0.21 |
| **guitar_v6 best (epoch 1935)** | **0.024** | **0.16** | **0.17** |
| guitar_v6 final (epoch 2511) | 0.024 | 0.14 | 0.17 |

**Key finding:** No checkpoint passes the 0.3 heuristic proposed initially in §3.4. Counter-intuitively, collapsed checkpoints v2–v5 score equal to or higher than the working guitar_v6 canonical checkpoint. The reason is structural: collapsed near-silent outputs (RMS ≈ 0.003–0.024) have STFT magnitudes that share the general spectral envelope of any natural audio source (low-frequency dominance, high-frequency attenuation), so the flattened Pearson correlation picks up trivial shared shape rather than genuine reconstruction. Output RMS, by contrast, does discriminate (guitar_v6 Phase 1 = 0.041, ten times the collapsed checkpoints' near-silence). This calibration limitation of the §3.4 diagnostic is documented in §5.1.4 and §6.3 of the memoir.

**Best checkpoint: version_4, epoch 1935** (exported as `guitar_v6_best.ts`). The user listening-tested both this and the final epoch 2511 model: epoch 1935 reconstruction is more recognisable as guitar; final epoch reconstruction "sounds very different from the input." This is consistent with the gap-reversal pattern — the model was best mid-Phase-2 before further GAN training slightly degraded reconstruction.

**Voice as input:** Decoder produces constant low-energy noise regardless of voice content. This is now understood as **decoder mode collapse on out-of-distribution input**, structurally distinct from the GAN-instability failures of v1–v5. The encoder still differentiates voice inputs in latent space (latent diffs 1.27–6.89), but voice latents fall outside the trained guitar manifold; the decoder is not trained to produce meaningful output from such latents.

**Conclusion:** v6 is a **working guitar autoencoder** for in-distribution input. It is not a voice-to-guitar timbre transfer system, and the project's central finding is that voice-to-guitar with BRAVE on small datasets fails at the decoder-manifold step rather than at training stability.

---

## drums_v2 (started 15 May 2026, terminated 24 May 2026)

**Config:** `v2.gin` + `causal.gin` + `drums_v2_overrides.gin` (LATENT_SIZE = 16, PHASE_1_DURATION = 1.5 M). Dataset: Groove MIDI Dataset (10.86 h, same as drums_v1).

**Outcome:** Terminated before completion at step ≈ 2.28 M (~95 wall-clock hours) when Phase 2 discriminator dominance reproduced despite `update_discriminator_every=4`.

**Phase 1 metrics (end of Phase 1, ≈ step 1.47 M):**
- validation loss ≈ 4.1
- multiband_spectral_distance ≈ 3.7
- pred_real / pred_fake gap ≈ 0 (Phase 2 not yet active)

**Phase 2 trajectory (versions 4 through 7, step 1.73 M → 2.28 M):**
- pred_real: 5.54 → 12.87 (increasing)
- pred_fake: −4.22 → −7.47 (decreasing)
- **GAN gap: 9.77 → 20.34** (monotonic widening over 800 k steps; end-of-training gap = 21.88 from full-trajectory analysis)
- validation loss: 6.43 → 7.46 (plateaued around 7.5)
- multiband_spectral_distance: 5.66 → 7.07 (slow degradation)

**Diagnosis:** Mode 3 (discriminator dominance / generator mode collapse) reproduced despite `update_discriminator_every=4`. Linear fit on the last 200 k steps before termination showed the gap widening at +1.91 per 100 k steps with no reversal in sight. Validation plateauing at 7.5 indicates the model has stopped meaningfully improving reconstruction quality.

**Lesson carried forward:** The v2.gin recipe that succeeded for guitar_v6 did not transfer cleanly to drums under the same data scale and training conditions. This is reported in Chapter 5.2 of the memoir; the reason is open and is included in the IRCAM-versus-ours hypothesis discussion (§5.3 and the future-work item on data scale in §6.4).

---

## IRCAM external baseline (ACIDS-IRCAM percussion pretrained model)

**Model:** Official `percussion` model from the ACIDS-IRCAM pretrained-models index (TorchScript, 71 MB). Downloaded May 2026 from `https://play.forum.ircam.fr/rave-vst-api/get_model/percussion`.

**Purpose:** To distinguish whether the OOD limitation observed across our guitar v1–v6 and drums iterations is structural to BRAVE or a consequence of training-recipe choices.

**Methodology:** Same synthetic-input diagnostic as §3.4 applied to the IRCAM model.

**Diagnostic results:**
- Output (silence input) = 0.0009 (≈ 0.001; decoder respects zero input — healthy)
- Output (440 Hz sine input) mean abs = 0.058 (responsive)
- Output (white noise input) mean abs = 1.75 (strongly responsive)
- **Forward diff between silence and white noise = 1.75**
- Compared to our drums_v1's forward diff = 0.0045, this is approximately **390× more responsive**

**Critical observation:** Output amplitude for noise input (1.75) is roughly an order of magnitude *louder* than typical voice input amplitude (≈ 0.1). In informal Pure Data listening tests with the IRCAM model loaded via `nn~` (at 44 100 Hz, the model's training rate), voice input produced audibly modulated percussive output — confirming that a well-trained BRAVE-family model can respond meaningfully to OOD voice input even though that input is technically off-manifold.

**Conclusion:** The OOD limitation observed in our guitar_v6 and drums_v1 iterations is not structural to BRAVE; the architecture is capable of producing strong voice-driven response when trained on more, cleaner, or differently-curated data. The bottleneck is the project's training data and training recipe rather than the BRAVE framework. This conclusion underpins Chapter 5.3 and the framing of Chapter 6.

---

## Summary table — final failure taxonomy

| Version | Failure mode | Diagnostic signature | Root cause |
|---|---|---|---|
| guitar_v1 | Posterior collapse | KL decreasing 0.83→0.25, forward diff = 0.006 | Constant high β from step 0 |
| guitar_v2 | Latent underutilisation | KL/dim ≈ 0.005 nats, output 60× quieter than input | `LATENT_SIZE=128` too large for ~5.4 h corpus |
| guitar_v3 | Phase 2 generator collapse | Forward diff 0.042 peaks then degrades | Default custom discriminator overpowers generator |
| guitar_v4 | Adversarial mode collapse | pred_real = 3.34, pred_fake = −2.31, silence amp > signal amp | Less data → easier discriminator memorisation |
| guitar_v5 | Discriminator collapse | pred_real = −0.90 (negative on real audio), gap = 0.80 | Over-weakening of discriminator |
| guitar_v6 | Decoder OOD behaviour on voice | Encoder differentiates voice (latent diffs 1.27–6.89), decoder produces near-constant output | Voice latents outside trained guitar manifold |
| drums_v1 | Below-threshold response (mixed listening evidence) | Forward diff 0.0045 (below 0.01); listening initially suggested transient response, later not reproduced | Adequate scale for transient task but custom config still inadequate |
| drums_v2 | Discriminator dominance reproduced under v2.gin | GAN gap widened 9.8 → 21.9 over 800 k steps; training terminated | v2.gin recipe did not transfer cleanly from guitar to drums |
| IRCAM percussion (external baseline) | Reference for "working" state | Forward diff 1.75, silent output 0.001, audibly voice-responsive in PD | Larger / more curated training corpus and recipe |
