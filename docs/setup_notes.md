# Environment Setup Notes

## System

| Component | Version |
|-----------|---------|
| OS | Windows 11 |
| Python | 3.14.3 |
| CUDA driver | 13.2 |
| PyTorch | 2.11.0+cu128 |
| acids-rave | 2.3.1 |
| pytorch-lightning | 2.6.1 |

## Compatibility Patches Applied to acids-rave 2.3.1

After installing acids-rave, three patches must be applied to
`venv/Lib/site-packages/rave/pqmf.py` and one to `model.py`.

These patches were needed because acids-rave was written for scipy ~1.11 and
pytorch-lightning 1.9, while we are running newer versions.

### pqmf.py — Patch 1: scipy.signal.kaiser moved to scipy.signal.windows

**File:** `venv/Lib/site-packages/rave/pqmf.py`, line 10

```python
# BEFORE (broken on scipy >= 1.14):
from scipy.signal import firwin, kaiser, kaiser_beta, kaiserord

# AFTER:
from scipy.signal import firwin, kaiser_beta, kaiserord
from scipy.signal.windows import kaiser
```

### pqmf.py — Patch 2: kaiserord argument must be a Python float

**File:** `venv/Lib/site-packages/rave/pqmf.py`, line 67

```python
# BEFORE (broken on numpy 2.x where fmin passes 1-d arrays):
N_, beta = kaiserord(atten, wc / np.pi)

# AFTER:
N_, beta = kaiserord(atten, float(np.asarray(wc).ravel()[0]) / np.pi)
```

### pqmf.py — Patch 3: firwin nyq parameter renamed to fs

**File:** `venv/Lib/site-packages/rave/pqmf.py`, line 70

```python
# BEFORE (nyq removed in scipy >= 1.14):
h = firwin(N, wc, window=('kaiser', beta), scale=False, nyq=np.pi)

# AFTER:
h = firwin(N, wc, window=('kaiser', beta), scale=False, fs=2 * np.pi)
```

### model.py — Patch 4: pytorch-lightning 2.0 API rename

**File:** `venv/Lib/site-packages/rave/model.py`, line 445

```python
# BEFORE (removed in pytorch-lightning 2.0):
def validation_epoch_end(self, out):

# AFTER:
def on_validation_epoch_end(self, out=[]):
```

---

## Training Command

All steps are automated via `scripts/train_guitar.bat` (Windows). It auto-detects and resumes from the latest checkpoint. Manual equivalent:

```bash
# 1. Preprocess audio (custom script handles path/filter logic)
python scripts/preprocess.py --instrument guitar

# 2. Build LMDB (rave's own preprocessor)
rave preprocess --input_path data/processed/guitar --output_path data/rave_ready/guitar

# 3. Train — BRAVE c128_r10 config (25M params, 17.3M without discriminator)
rave train \
  --config C:/Users/Usuario/Documents/BRAVE/configs/c128_r10.gin \
  --name guitar_v1 \
  --db_path data/rave_ready/guitar \
  --channels 1 \
  --gpu 0 \
  --val_every 10000
```

> **Note:** Always pass `--channels 1`. The default is 0, which silently creates a zero-dimension model.
> **Note:** Do NOT use `brave.gin` (BRAVE light, 4.9M params) — use `config/c128_r10_beta_fixed.gin` (25M params) for thesis-quality output.
> **Note:** `rave preprocess` input must be the custom-preprocessed files from `scripts/preprocess.py`, not raw audio directly.
> **Note:** Do NOT use the original `c128_r10.gin` from the BRAVE repo — it has `BetaWarmupCallback initial_value=0.1` (constant, no warmup) which causes posterior collapse. Use `config/c128_r10_beta_fixed.gin`: `initial_value=0.0001, target_value=0.1, warmup_len=1000000`.

## Posterior Collapse — Diagnosis and Fix (guitar_v1)

**Symptom:** After 1.24M training steps, the exported model produced the same audio output regardless of input (mic muting had no effect). Python inference confirmed encoder was working (diff=1.25 between silence and 440Hz sine) but decoder output was nearly identical (diff=0.006).

**Diagnosis:** TensorBoard `regularization` metric showed KL divergence steadily decreasing from 0.5 → 0.25 throughout training. The KL pressure from step 1 (beta=0.1) prevented the encoder/decoder from learning a meaningful latent space before regularization kicked in — a classic VAE posterior collapse.

**Fix:** Created `config/c128_r10_beta_fixed.gin` with:
- `initial_value = 0.0001` — effectively zero KL at step 0 (cannot use exactly 0: BetaWarmupCallback uses log-space interpolation, so math.log(0) crashes immediately)
- `target_value = 0.1` — matches the BRAVE paper's intended constant beta=0.1 for c128_r10, reached at end of Phase 1 and held through Phase 2. (Setting 1.0 would be too aggressive and could destabilize Phase 2 onset; RAVE v2/v3 use 0.05 for reference.)
- `warmup_len = 1000000` — ramps through all of Phase 1; beta is locked at 0.1 from step 1M onward

Beta trajectory: 0.0001 (step 0) → ~0.003 (step 500k) → 0.1 (step 1M+)

Retraining as `guitar_v2` from scratch with fixed config.

**TensorBoard health check for guitar_v2:**
- `regularization` should be near 0 early, gradually rising to ~0.1 range by step 1M. If it collapses toward 0 again → posterior collapse recurring.
- `distance` (reconstruction loss) should decrease steadily through Phase 1.
- `fidelity_95` should stabilize well below 128 (the latent size) — values around 10-30 are healthy.

---

## Training History — What Worked, What Didn't

| Version | Dataset | Latent | Outcome | Lesson |
|---------|---------|--------|---------|--------|
| guitar_v1 | mixed 5.79h | 128 | Posterior collapse | Need beta warmup |
| guitar_v2 | mixed 5.79h | 128 | KL/dim too low (0.005 nats) | LATENT_SIZE too large |
| guitar_v3 | mixed 5.79h | 16 | KL=0.65, quiet noise + faint response | Generator partial collapse |
| guitar_v4 | clean 3.05h mic only | 16 | KL=0.80, silence > signal (gap=5.6) | **Discriminator dominated** — less data made it worse |
| guitar_v5 | mixed 16h, weak D | 16 | pred_real=−0.9, pred_fake=−1.7 (gap=0.8) | **Discriminator collapsed** — weakening went too far |
| **drums_v1** | Groove 10.86h | 16 | _in progress_ | BRAVE-paper defaults; drums already proven at 2.8h |

**Key insights from the guitar series:**
- v3→v4: Cleaner data (removing DI) didn't help — actually made it worse. Less data = discriminator memorizes faster = generator can't compete. **Quantity + diversity matters more than purity.**
- v4→v5: Halving discriminator capacity went too far. v4 had pred_real=+3.34 (D too strong); v5 has pred_real=−0.9 (D can't classify real audio at all). There is no obvious Goldilocks setting and we've run out of training budget for guitar.

**Decision (May 3 2026):** Pivot to drums. The BRAVE paper proved drums works at 2.8h; we have 10.86h (4× more). Default config, no architectural tweaks. Guitar findings become a documented research contribution (failure taxonomy + diagnostic methodology).

---

## Noise-Floor Heterogeneity (guitar dataset analysis)

The professor flagged that the guitar corpus mixed line-out (DI) and microphone recordings, which would introduce inconsistent noise floors. Quantitative confirmation via `scripts/analyze_noise_floor.py` (5th-percentile of 100 ms RMS windows, per source):

| Source | Recording method | N files | Noise floor (dBFS) |
|--------|------------------|---------|---------------------|
| GuitarSet | mic'd acoustic | 360 | −46.5 |
| Guitar-TECHS | line-out DI | 52 | −67.2 |
| IDMT dataset4 / acoustic_mic | mic'd acoustic | 128 | −57.8 |
| IDMT dataset4 / acoustic_pickup | acoustic pickup | 128 | −53.3 |
| IDMT dataset4 / Career SG | electric DI | 123 | −82.4 |
| IDMT dataset4 / Ibanez 2820 | electric DI | 128 | −71.5 |

**Spread: 35.8 dB** — two orders of magnitude in amplitude between cleanest and noisiest. This exceeds the typical dynamic variation within a single guitar performance and is therefore likely encoded as a salient feature of the latent space. Acknowledged as a known limitation; future work could restrict training to a single recording method or apply explicit noise-floor normalisation.

---

## drums_v1 Config (Current Run)

**Dataset:** Groove MIDI Dataset audio, 1090 files, 10.86h, mono 16-bit @ 44100 Hz.
- Consolidated to `data/raw/drums_v1_combined/` after librosa-based stereo→mono conversion
- 991 originally stereo files, 6 originally 24-bit, all normalised in one pass

**Config:** `config/c16_r10_beta_fixed.gin` (BRAVE paper paradigm — no v5 tweaks)
- LATENT_SIZE = 16, RATIOS = [2,2,2,1], CAPACITY = 64
- Discriminator at default (capacity=64, n_layers=4)
- PHASE_1_DURATION = 1M (paper default)
- Beta warmup 0.0001 → 0.1 over 1M steps
- Causal convolutions for streaming/realtime

**Why default config for drums:**
- BRAVE paper proved this exact paradigm at 2.8h drums; we have 4× more data
- Drums is transient/percussive — easier latent space than melodic guitar
- No reason to deviate from a known-working setup

**Expected training time:** ~90h on RTX 5080 (3M total steps).

**Mid-training health checks:**
- `regularization` should rise during Phase 1 and hold during Phase 2
- `pred_real - pred_fake` gap should stay between ~1 and ~3 (signs of GAN balance)
- If gap > 5 → discriminator dominating (v4 mode); if both negative → discriminator collapse (v5 mode)

---

## ffmpeg

Installed via `winget install --id Gyan.FFmpeg`. Path added to `venv/Scripts/activate`.

Location: `C:\Users\Usuario\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\`

If the venv activate script is regenerated (e.g., by re-creating the venv), re-add
the ffmpeg PATH line manually.
