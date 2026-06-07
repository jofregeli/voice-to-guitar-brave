# Training scripts and data pipeline

This document describes the Python scripts and Windows batch files that orchestrate the training and evaluation pipeline. Source for the **Implementation chapter (Section 4.1 Environment, 4.2 Training pipeline)** and the **Methodology chapter (Section 3.3 Datasets and preprocessing)**.

## Preprocessing pipeline

### Stage 1 — `scripts/preprocess.py` (custom preprocessing)

A Python script using `librosa` and `soundfile` that walks input directories, normalises audio, and writes consolidated mono 16-bit WAV files. Implemented because `rave preprocess` directly silently deadlocks on stereo files, 24-bit audio, and corrupt files. Critical preprocessing decisions documented below.

Per-file processing:

1. Load with `librosa.load(path, sr=44100, mono=True)` — automatically converts to 44100 Hz mono.
2. Trim leading/trailing silence with `librosa.effects.trim(top_db=40)`.
3. Skip files shorter than 1 second after trimming.
4. Save as 16-bit PCM WAV to consolidated output directory.

Source directories per instrument:

- **guitar:** GuitarSet (`data/raw/guitarset`), Guitar-TECHS (`data/raw/guitartechs`), IDMT-SMT-Guitar (`data/raw/idmt_smt_guitar/`)
- **drums:** Groove MIDI Dataset (`data/raw/groove`)

### Stage 2 — `rave preprocess` (LMDB build)

After stage 1 produces a clean consolidated directory, RAVE's own preprocessor builds the Lightning Memory-Mapped Database (LMDB) used during training:

```
rave preprocess --input_path <consolidated_dir> --output_path data/rave_ready/<name> --channels 1 --sampling_rate 44100
```

Important flags:
- `--no-lazy` (default behaviour): pre-decodes all audio into the LMDB so training does not have to decode WAVs on the fly. Trades disk space for training speed.
- `--channels 1`: writes mono. Defaults to 0 which silently produces a zero-channel model — a known footgun.
- `--sampling_rate 44100`: matches our training rate.

LMDB outputs:
- `data.mdb` — the actual audio data (100 GB allocation preallocated, actual content much smaller).
- `lock.mdb` — LMDB lock file.
- `metadata.yaml` — small file with `channels`, `lazy`, `n_seconds`, `sr`. Used to verify preprocessing succeeded.

Final datasets used in training:

| Dataset | LMDB path | `n_seconds` | Note |
|---|---|---|---|
| guitar_v5 corpus | `data/rave_ready/guitar_v5` | 57665 (~16 h) | GuitarSet + Guitar-TECHS + IDMT dataset4 |
| guitar_v4 corpus | `data/rave_ready/guitar_v4` | 19538 (~5.4 h) | GuitarSet mic only (cleaner experiment) |
| guitar baseline | `data/rave_ready/guitar` | 19408 (~5.4 h) | First guitar dataset, mixed sources |
| drums | `data/rave_ready/drums_v1` | ~39000 (~10.86 h) | Groove |

Note that `n_seconds` in the LMDB metadata is the total stored audio duration after RAVE's internal chunking — it does not equal the raw input duration because of overlap and segmenting.

## Noise floor analysis script — `scripts/analyze_noise_floor.py`

A Python script that measures the noise floor of each source directory by computing the 5th percentile of 100 ms RMS windows per file, in dBFS.

Why 5th percentile rather than the minimum: the absolute minimum of RMS in a recording is dominated by digital silence (zeros), which would give −∞ dBFS. The 5th percentile captures the typical quiet-passage level, which is the real signal noise floor.

Method per file:

1. Load WAV with librosa at 44100 Hz mono.
2. Split into non-overlapping 100 ms windows.
3. Compute RMS per window.
4. Filter out exact-zero windows.
5. Return the 5th percentile of remaining RMS values, converted to dBFS via `20 * log10(amplitude + 1e-12)`.

Per source: aggregate median and IQR of the per-file noise floors.

Output (the result presented in the thesis):

| Source | N files | Noise floor median (dBFS) | IQR (dB) |
|---|---|---|---|
| GuitarSet (mic'd acoustic) | 360 | −46.5 | 10.4 |
| Guitar-TECHS (line-out DI) | 52 | −67.2 | 28.1 |
| IDMT dataset4 / acoustic_mic | 128 | −57.8 | 7.1 |
| IDMT dataset4 / acoustic_pickup | 128 | −53.3 | 9.4 |
| IDMT dataset4 / Career SG (electric DI) | 123 | −82.4 | 7.2 |
| IDMT dataset4 / Ibanez 2820 (electric DI) | 128 | −71.5 | 11.3 |

Spread between cleanest (Career SG) and loudest (GuitarSet) sources: **35.8 dB**. Two orders of magnitude in amplitude. Documented in the thesis as a known limitation: this exceeds the typical dynamic variation within a single guitar performance, so the encoder will encode recording method as a latent feature.

## Training automation

Each model has a Windows batch script in `scripts/train_<name>.bat` that:

1. Activates the Python venv.
2. Searches via PowerShell for the latest checkpoint matching the run name.
3. Resumes from that checkpoint if found; otherwise starts fresh.
4. Calls `rave train` with the appropriate configs and dataset path.

The check-for-latest-checkpoint logic uses:

```powershell
Get-ChildItem -Path runs -Recurse -Filter *.ckpt |
  Where-Object { $_.FullName -like '*<run_name>*' } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty FullName
```

This makes the training scripts safely re-runnable: a user can close the training window at any time and re-run the script, and training resumes from the most recent checkpoint. Used several times during the project when sessions were interrupted.

### Run name versioning

Each run gets its own name (`guitar_v1`, `guitar_v2`, …, `guitar_v6`, `drums_v1`, `drums_v2`). RAVE appends a content-hash suffix derived from the configuration (e.g., `guitar_v6_bbcc6d36c3`), so a config change automatically starts a new run directory rather than overwriting the old one.

### Checkpointing strategy

`--val_every 10000` causes a checkpoint every 10,000 training steps (~18 minutes at 9 it/s). Each checkpoint includes:

- `best.ckpt` — best by validation reconstruction loss (auto-selected by RAVE).
- `epoch-epoch=N.ckpt` — the latest epoch.
- `epoch_500000.ckpt` etc. — fixed-step checkpoints saved every 500k steps.

This redundancy turned out to be important: for guitar_v6, the final epoch (2511) had slightly degraded reconstruction compared to mid-Phase-2 (epoch 1935). We selected the mid-training checkpoint as the canonical `guitar_v6_best.ts` based on listening tests.

## Model export

After training, `rave export --run <run_dir> --name <output_name>` produces the TorchScript `.ts` file used by Pure Data's `nn~` external. Internally, export selects `best.ckpt` by default (the lowest validation loss), but we can also point at a specific version subdirectory to export an intermediate checkpoint.

Exports used in the thesis:

- `models/drums_v1_ep755.ts` — drums prototype, demonstrably responds to voice in PD
- `models/guitar_v6_best.ts` — guitar autoencoder (epoch 1935, mid-Phase-2)
- `models/guitar_v6_final.ts` — final-epoch guitar model for comparison
- `models/drums_v2.ts` — pending (drums_v2 training in progress)

## Environment

- **OS:** Windows 11 (build 26200+)
- **GPU:** NVIDIA RTX 5080 (16 GB VRAM)
- **Python:** 3.14.3
- **PyTorch:** 2.11.0+cu128
- **acids-rave:** 2.3.1 (with four compatibility patches — see `configs_summary.md` and `setup_notes.md`)
- **pytorch-lightning:** 2.6.1
- **librosa:** for audio loading and noise-floor measurement
- **soundfile:** for WAV writing
- **ffmpeg:** installed via `winget install --id Gyan.FFmpeg`, on the venv PATH for `librosa` to call when loading non-WAV inputs
- **Pure Data:** 0.55+ with the ACIDS-IRCAM `nn_tilde` external

Estimated training throughput: ~9–12 iterations per second on RTX 5080 with the v2 architecture. A full 3 M-step training takes approximately 90–100 hours wall-clock.

---

## Additional analysis and figure-generation scripts (added during the Final phase)

### `scripts/guitar_v6_stft_correlation.py` — Cross-checkpoint diagnostic calibration

Computes the mean STFT magnitude correlation between input and output for every guitar iteration (v1 through v6, including v6 Phase 1 and final checkpoints) using the same N = 8 GuitarSet `solo_mic` samples (5 s each, n_fft = 2048, hop = 512). Produces the cross-checkpoint calibration table (Table 5.2 in the memoir) that revealed the STFT-correlation metric does not robustly discriminate working from collapsed checkpoints (see §5.1.4 and §6.3 of the memoir). Methodology: time-aligned Pearson correlation (max over ±8 frame shifts) on flattened |STFT| magnitudes. Also reports per-checkpoint output RMS, which does discriminate.

### `scripts/generate_chapter5_figures.py` — Figures for Chapter 5

Generates the five Chapter 5 figures at 300 DPI from existing TensorBoard logs and the canonical model:

- **Figure 5.1** — KL trajectory of guitar_v1 (posterior collapse evidence)
- **Figure 5.2** — Cross-iteration GAN logit gap evolution (guitar_v4 dominance vs guitar_v5 collapse vs guitar_v6 reversal)
- **Figure 5.3** — Input/output spectrogram comparison for guitar_v6 in-distribution reconstruction
- **Figure 5.4** — drums_v2 Phase 2 GAN gap widening (9.8 → 20.3)
- **Figure 5.5** — Per-source noise-floor distribution boxplot (six recording paths, 35.9 dB spread)

Outputs are saved to `figures/figure_5_*.png`. TensorBoard scalar extraction uses `tensorboard.backend.event_processing.event_accumulator`; noise-floor computation replicates `scripts/analyze_noise_floor.py` for direct consistency with §3.3 of the memoir.

### `src/vpt/` + `scripts/vpt_*.py` — Vocal-percussion classifier (Path A preliminary exploration)

A small 1D-CNN classifier (≈30 k parameters) trained on the AVP dataset (Delgado et al., 2019) for three-class onset classification (kick / snare / hi-hat). Deployed as a streaming TorchScript model loadable by Pure Data's `nn~` external. Built as a preliminary exploration of the hybrid pipeline future-work item mentioned in §6.4 of the memoir; not pursued for thesis demonstration after listening evaluation showed phoneme-dependent classification accuracy below demonstration quality.

Scripts:

- `scripts/vpt_train.py` — train the classifier on AVP (speaker-disjoint split, 5 participants held out for validation)
- `scripts/vpt_calibrate.py` — fine-tune on user-recorded calibration data (kick.wav, snare.wav, hh.wav) with class balancing
- `scripts/vpt_export.py` — wrap the classifier in `nn_tilde.Module` and export as streaming TorchScript
- `scripts/vpt_make_samples.py` — synthesise reference kick/snare/hh one-shot WAVs for the PD patch

Pure Data demonstration patch: `examples/test_vpt_drumkit.pd`. AVP corpus must be downloaded to `data/raw/avp/AVP_Dataset` (CC BY 4.0, 230 MB, Zenodo DOI 10.5281/zenodo.3250230).
