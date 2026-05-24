# Pure Data patches

This document describes the Pure Data (`.pd`) patches built for testing each trained model in real time. Source for the **Implementation chapter (Section 4.3 Real-time deployment)**.

## What `nn~` is

`nn~` is a Pure Data external maintained by ACIDS-IRCAM (the same group that develops RAVE/BRAVE). It loads TorchScript-exported `.ts` model files and calls a named method (`forward`, `encode`, `decode`, etc.) for each audio block. The model receives a tensor of shape `(1, 1, block_size)` per inlet and returns a tensor of the same shape per outlet.

Key properties:
- Block-by-block processing, so latency is `block_size / sample_rate` plus driver buffer plus model forward time.
- No sample-rate conversion — the audio engine's sample rate must match the model's training sample rate.
- The model's TorchScript graph encodes the streaming-causal architecture from training.

## Audio configuration (UPF lab environment)

All patches assume Pure Data is configured with:

| Setting | Value |
|---|---|
| Sample rate | **44100 Hz** (mandatory, matches model training) |
| Block size | 64 samples |
| Input device | mic (HyperX Cloud Flight Wireless was used during testing) |
| Output device | speakers / headphones |

**Critical compatibility note:** During testing we discovered Pure Data was running at 48000 Hz by default, which silently mismatched the 44100 Hz model and produced unusable output. Always verify the sample rate in `Media → Audio settings` before running.

## Common signal chain

All test patches share the same minimal structure:

```
adc~ 1                # mic input (mono)
  ↓
nn~ <model>.ts forward  # neural autoencoder
  ↓
*~ <gain>             # gain compensation (model output amplitudes ~0.005–0.05)
  ↓
dac~                  # stereo output (model output sent to both channels)
```

The `get_methods` message connected to nn~ inlet outputs the model's available method names to the Pure Data console — used to verify the model loaded correctly.

## Why a gain multiplier is needed

BRAVE/RAVE decoder output amplitudes are typically in `[0.005, 0.05]` mean absolute value, much quieter than typical input levels around `0.05–0.3`. Direct connection to `dac~` produces nearly inaudible output. Each test patch applies a fixed multiplier:

- `*~ 10` to `*~ 50` depending on the model
- Tuned per model based on the Python diagnostic's measured output amplitude
- Higher gain (e.g., ×50) for models with smaller output amplitudes (~0.005); lower gain (×20–30) for models with larger amplitudes (~0.02)

## Per-model patches

### `test_drums.pd` — drums_v1 (ep755)

```
adc~ 1 → nn~ models/drums_v1_ep755.ts forward → *~ 50 → dac~
```

Used to demonstrate the working voice-to-drums prototype. With ×50 gain on the typical 0.005 output amplitude, the produced signal reaches typical listening levels (~0.25). Voice consonants and beatbox transients produce drum-like sounds in real time.

### `test_guitar_v6.pd` — guitar_v6 (intermediate epoch)

```
adc~ 1 → nn~ models/guitar_v6.ts forward → *~ 30 → dac~
```

Intermediate checkpoint test patch (used at epochs 1335, 1791, 2103, 2511). Demonstrates the guitar autoencoder. ×30 gain matches the typical 0.015–0.025 output amplitude.

### `test_guitar_v6_final.pd` — guitar_v6 final epoch

Same signal chain as above, loads `models/guitar_v6_final.ts` (epoch 2511). Used for the final listening evaluation, including the reconstruction comparison against in-distribution guitar audio.

### `test_guitar_v6_filtered.pd` — voice pre-processing experiment

A failed experiment in Pure Data domain — added a pre-processing chain to reshape voice towards guitar's spectral envelope:

```
adc~ 1 → hip~ 80 → lop~ 2000 → lop~ 2000 → *~ 0.5 →
  nn~ models/guitar_v6.ts forward → *~ 30 → dac~
```

- `hip~ 80` removes DC and rumble below 80 Hz
- `lop~ 2000` (×2 for ~−12 dB/oct rolloff) removes voice formants above 2 kHz
- `*~ 0.5` attenuates the filtered signal before the model to prevent clipping

**Outcome:** Made no perceptible difference. The model's response to voice is determined by the encoder's mapping of voice spectra to latent space, not by the presence or absence of high-frequency content. This is documented in the failure analysis chapter.

### `test_guitar_phase1.pd` — input-gated patch (also a failed experiment)

A patch built when we believed the guitar model produced a constant "idle hum" that needed gating. Used a mic envelope follower to multiply the model output by a gate signal:

```
adc~ 1 ──── nn~ → *~ 10 ──┐
       └── env~ 1024 → threshold → sig~ ─→ ×→ dac~
```

**Outcome:** Demonstrated the model's voice response is too subtle for ducking to help. The output sounded gated but contained no useful voice-modulated content.

## Latency budget

The BRAVE design target is sub-10 ms end-to-end latency. Components:

| Component | Latency |
|---|---|
| Pure Data block (64 samples @ 44100 Hz) | 1.45 ms |
| Model forward pass (per block, RTX 5080) | < 1 ms |
| Windows WASAPI audio driver buffer | ~5 ms |
| Causal convolutions (no lookahead) | 0 ms |
| **Total end-to-end** | **~7 ms** |

This was verified informally by speaking near the mic and observing no perceptible delay through monitor headphones. A more rigorous loopback measurement was not performed.

## How to reproduce the demo

1. Install Pure Data ≥ 0.55 with the `nn~` external from the ACIDS-IRCAM `nn_tilde` GitHub repository.
2. Verify the `.ts` model exists in `models/`. If not, run `rave export --run runs/<run_dir>` to regenerate.
3. Open the patch in Pure Data.
4. Open `Media → Audio settings`; set sample rate to **44100 Hz**.
5. Enable DSP (`Ctrl+/`).
6. Speak into the mic; output should be produced live with sub-10 ms latency.

If the model fails to load, the `print nn~-info` object will display the error. Common causes: wrong sample rate, model file moved, `nn~` external not installed.
