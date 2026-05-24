# Configuration files used in training

This document explains every `.gin` configuration file used across the seven training iterations (six guitar + drums + drums_v2). Source for the **Methodology and Implementation chapters**.

## RAVE/BRAVE config system

RAVE uses Gin-Config for hyperparameter management. Configurations can be passed via the `--config <name_or_path>` flag, and multiple `--config` arguments are merged in order (later overrides earlier). The official RAVE distribution includes several base configurations in `rave/configs/`:

- `v1.gin` — original RAVE v1 architecture
- `v2.gin` — improved RAVE v2 with EncoderV2/GeneratorV2 (includes v1.gin)
- `causal.gin` — sets `cached_conv.get_padding.mode = 'causal'` for streaming
- `wasserstein.gin` — alternative loss
- `discrete.gin` — discrete latent variant
- `descript_discriminator.gin`, `spectral_discriminator.gin` — alternative discriminators

## Our custom configs

### `c128_r10.gin` (original BRAVE-paper config, never modified)

The starting point. We did NOT use this directly because its beta schedule was misconfigured for our setup.

Key parameters:
- `SAMPLING_RATE = 44100`
- `LATENT_SIZE = 128`
- `RATIOS = [2, 2, 2, 1]` (8× temporal compression)
- `N_BAND = 16` (PQMF bands → total 128× compression)
- `CAPACITY = 64`
- `PHASE_1_DURATION = 1000000`
- Beta callback: `initial_value = .1`, `target_value = .1`, `warmup_len = 1` (i.e., constant 0.1 from step 0)
- `cc.get_padding.mode = 'causal'` (streaming)

**Used for:** Not directly. Was the source of guitar_v1's posterior collapse — the constant high beta prevented the latent space from learning.

### `c128_r10_beta_fixed.gin` (our custom fix to v1's posterior collapse)

Identical to `c128_r10.gin` except the beta schedule was replaced with a log-space warmup:

```
rave.BetaWarmupCallback:
    initial_value = 0.0001
    target_value = 0.1
    warmup_len = 1000000
```

Why log-space: `BetaWarmupCallback` interpolates in log space, so `initial_value = 0.` would crash (`math.log(0)`). Starting at 0.0001 effectively zero KL pressure at step 0 while keeping the interpolation valid.

**Used for:** guitar_v2 — fixed posterior collapse but revealed the next problem (LATENT_SIZE=128 too large for our data → 0.005 nats/dim utilisation).

### `c16_r10_beta_fixed.gin` (LATENT_SIZE reduced to 16)

Same as `c128_r10_beta_fixed.gin` but with `LATENT_SIZE = 16`. Rationale: the IIL pretrained guitar model uses z=16, and our v2 showed 0.005 nats/dim utilisation with z=128.

Other parameters unchanged: `CAPACITY = 64`, `RATIOS = [2, 2, 2, 1]`, `N_BAND = 16`, `PHASE_1_DURATION = 1000000`, beta 0.0001 → 0.1 over 1M steps.

Discriminator at default values (capacity=64, n_layers=4, MultiScaleDiscriminator only).

**Used for:** guitar_v3, guitar_v4, drums_v1. All three trained Phase 1 successfully (encoder learns) but the default custom discriminator dominated Phase 2 (gap up to 5.65 in v4, generator collapsed for guitar; for drums, the transient nature of the data masked the issue and a usable checkpoint exists at ep755).

### `c16_r10_v5_balanced.gin` (weakened discriminator experiment, v5)

Targeted attempt to fix the discriminator dominance observed in v3/v4 by weakening the discriminator:

| Parameter | c16_r10 (v3/v4) | v5_balanced |
|---|---|---|
| Discriminator `capacity` | 64 | **32** (half) |
| Discriminator `n_layers` | 4 | **3** |
| `feature_matching` weight | 10 | **30** (stronger perceptual loss) |
| `target_value` (beta) | 0.1 | **0.05** (matches RAVE v2 default) |
| `PHASE_1_DURATION` | 1000000 | **1500000** (longer reconstruction) |

**Used for:** guitar_v5. Outcome: opposite failure — discriminator collapsed (pred_real went negative, gap = 0.80). Confirmed there is no Goldilocks discriminator capacity reachable by simple parameter scaling.

### `guitar_v6_overrides.gin` (final guitar config, on top of official v2 + causal)

After v1–v5 all failed with the custom architecture, we abandoned `c16_r10` entirely and used the official `v2.gin` + `causal.gin` as the base. Only two overrides:

```
LATENT_SIZE = 16
PHASE_1_DURATION = 1500000
```

Why these two:
- `LATENT_SIZE = 16` — v2's default is 128, but v2 (our custom) showed 0.005 nats/dim for our 16h dataset. The IIL pretrained guitar uses 16.
- `PHASE_1_DURATION = 1500000` — extend reconstruction phase to give the model more time to stabilise before adversarial training.

The official `v2.gin` brings architectural features absent from our custom configs:
- `update_discriminator_every = 4` — D updates 4× slower than G, naturally preventing the dominance pattern observed in v3/v4
- `MultiPeriodDiscriminator` (5 periods: 2, 3, 5, 7, 11) + `MultiScaleDiscriminator` combined via `CombineDiscriminators`
- `EncoderV2` and `GeneratorV2` (newer architectures with `amplitude_modulation = True`)
- `valid_signal_crop = True` (different training-sample handling)
- Relative feature-matching loss (`relative = True` in `feature_matching/core.mean_difference`)
- More aggressive temporal compression internally (`RATIOS = [4, 4, 4, 2]` × 16 PQMF bands = same 128× total compression but at different granularities)
- Different dilation pattern (`DILATIONS = [[1,3,9],[1,3,9],[1,3,9],[1,3]]`) vs our `[[3,1],[9,1],...]`
- Beta schedule in v2: `initial = 1e-6`, `target = 5e-2`, `warmup = 20000` (fast warmup, lower target)

**Used for:** guitar_v6. Outcome: **working guitar autoencoder** (in-distribution reconstruction works, voice still out-of-distribution). Phase 2 GAN gap reversed direction for the first time in our series.

### `drums_v2_overrides.gin` (drums final config)

Same approach as `guitar_v6_overrides.gin` applied to drums:

```
LATENT_SIZE = 16
PHASE_1_DURATION = 1500000
```

Base configs: `v2.gin` + `causal.gin`. Dataset: drums_v1 LMDB (10.86h Groove).

**Status:** In progress as of May 15 2026.

## Compatibility patches applied to `acids-rave 2.3.1`

These are not configuration changes but environment patches needed to make the published `acids-rave 2.3.1` run with our newer Python/PyTorch/scipy stack. They are documented in `setup_notes.md` and reproduced here for completeness:

### Patch 1 — `pqmf.py` line 10: `scipy.signal.kaiser` moved

```python
# Before
from scipy.signal import firwin, kaiser, kaiser_beta, kaiserord
# After (scipy ≥ 1.14)
from scipy.signal import firwin, kaiser_beta, kaiserord
from scipy.signal.windows import kaiser
```

### Patch 2 — `pqmf.py` line 67: `kaiserord` argument type

```python
# Before
N_, beta = kaiserord(atten, wc / np.pi)
# After (numpy 2.x compatibility)
N_, beta = kaiserord(atten, float(np.asarray(wc).ravel()[0]) / np.pi)
```

### Patch 3 — `pqmf.py` line 70: `firwin` `nyq` parameter

```python
# Before
h = firwin(N, wc, window=('kaiser', beta), scale=False, nyq=np.pi)
# After (scipy ≥ 1.14)
h = firwin(N, wc, window=('kaiser', beta), scale=False, fs=2 * np.pi)
```

### Patch 4 — `model.py` line 445: PyTorch Lightning 2.0 API

```python
# Before
def validation_epoch_end(self, out):
# After (pytorch-lightning 2.0+)
def on_validation_epoch_end(self, out=[]):
```

These four patches are required to install and run our pipeline on Windows 11 with Python 3.14, PyTorch 2.11, scipy ≥ 1.14, and pytorch-lightning 2.6.

## Augmentation note

`acids-rave 2.3.1` supports `--augment compress / gain / mute` flags, but the implementation calls `torchaudio.sox_effects.apply_effects_tensor` which is removed in newer `torchaudio` versions. We attempted augmentation in guitar_v5 and the training crashed immediately with `AttributeError: module 'torchaudio' has no attribute 'sox_effects'`. We disabled augmentation for all subsequent runs.
