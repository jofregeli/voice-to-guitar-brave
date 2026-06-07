# BRAVE-Based Real-Time Voice-to-Instrument Resynthesis

Bachelor's Thesis (TFG) — Grau en Enginyeria de Sistemes Audiovisuals, Universitat Pompeu Fabra.

**Author:** Jofre Geli de Fuenmayor
**Directors:** Lonce Wyse & Xavier Lizarraga
**Submission:** 12 June 2026
**Memoir:** [`docs/thesis/thesis.pdf`](docs/thesis/thesis.pdf) ([source](docs/thesis/thesis.tex))

---

## Project summary

A streaming-causal variational autoencoder (BRAVE) is trained to autoencode either guitar or drum audio at sub-10 ms latency and then driven, at inference time, with vocal input as a cross-modal excitation. The work is presented as a **diagnostic study**: six guitar training iterations and two drum iterations are systematically analysed for failure modes (posterior collapse, latent underutilisation, discriminator dominance, discriminator collapse, OOD decoder behaviour), an external IRCAM percussion checkpoint is used as a baseline, and the methodology is calibrated against measured outcomes rather than asserted.

The submitted contribution is the **diagnostic framework, the failure-mode taxonomy, and a guitar autoencoder that partially reconstructs in-distribution audio**, not a complete voice-to-instrument product. The thesis documents both the successful and the unsuccessful experiments honestly, including a methodological finding that the STFT-magnitude-correlation heuristic used in early evaluation does not robustly discriminate working from collapsed checkpoints.

---

## Repository layout

```
voice-to-guitar-brave/
├── docs/thesis/             # Final memoir: thesis.tex + references.bib + thesis.pdf
├── docs/                    # Auxiliary docs: scripts_and_pipeline, diagnostic_results, timeline_and_decisions
├── figures/                 # Production figures used in Chapter 5 (300 DPI)
├── config/                  # BRAVE .gin training configurations (one per iteration)
├── scripts/                 # Preprocessing, training, evaluation, figure generation
│   ├── preprocess.py
│   ├── analyze_noise_floor.py
│   ├── guitar_v6_stft_correlation.py
│   ├── generate_chapter5_figures.py
│   ├── train_*.bat                # Per-iteration training launchers (Windows)
│   └── vpt_*.py                   # Path-A vocal-percussion classifier (preliminary)
├── src/vpt/                 # VPT classifier source (1D-CNN on AVP dataset)
├── examples/                # Pure Data demo patches (.pd)
├── PDF Thesis Draft/        # Earlier chapter drafts retained for review history
├── References/              # Bibliographic PDFs cited by the memoir
└── runs/, models/, data/    # Training artefacts (gitignored)
```

---

## What was actually produced

| Iteration | Status | Notes |
|---|---|---|
| guitar_v1 | Diagnostic | Posterior collapse identified — KL falls 0.83 → 0.25 |
| guitar_v2 | Diagnostic | Latent underutilisation (KL/dim ≈ 0.005 nats) |
| guitar_v3 | Diagnostic | Phase-2 generator collapse |
| guitar_v4 | Diagnostic | Adversarial mode collapse |
| guitar_v5 | Diagnostic | Discriminator collapse |
| **guitar_v6** | **Canonical model** | Partial in-distribution reconstruction; canonical checkpoint at epoch 1935 (mid Phase 2) |
| drums_v1 | Diagnostic | Partial transient response, below demonstration threshold |
| drums_v2 | Terminated | Discriminator dominance reproduced; terminated before completion |
| VPT (Path A) | Preliminary | Vocal-percussion classifier; explored as future work, not pursued |

Canonical model export: `models/guitar_v6_best.ts` (TorchScript, loadable by Pure Data's `nn~` external; not committed due to size, regenerable via `rave export`).

---

## Reproduction

Setup (one-shot, Windows 11 with NVIDIA RTX-class GPU):

```powershell
git clone https://github.com/jofregeli/voice-to-guitar-brave
cd voice-to-guitar-brave
python scripts/setup.py
.\venv\Scripts\activate
```

Data, preprocessing, training, evaluation, and Pure Data deployment are described in detail in [`docs/scripts_and_pipeline.md`](docs/scripts_and_pipeline.md). The thesis's Chapter 4 (Implementation) lists the four `acids-rave 2.3.1` compatibility patches required for current PyTorch/scipy and the exact Pure Data + `nn~` configuration used.

To compile the memoir from source (XeLaTeX is required for the Times New Roman font via `fontspec`):

```powershell
cd docs\thesis
xelatex thesis ; biber thesis ; xelatex thesis ; xelatex thesis
```

Or build both the digital (one-side) and print (two-side) PDFs in one step:

```powershell
.\scripts\build_thesis.ps1
```

---

## Datasets used

| Dataset | Instrument | Used | License |
|---|---|---|---|
| [GuitarSet](https://zenodo.org/records/3371780) | Acoustic guitar (mic) | ~3.05 h | CC BY 4.0 |
| [Guitar-TECHS](https://zenodo.org/records/14963133) | Electric guitar (DI) | ~1.5 h | CC BY 4.0 |
| [IDMT-SMT-Guitar dataset 4](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html) | Mixed acoustic/electric | ~4.35 h | Research-only |
| [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) | Drums | ~10.86 h | CC BY 4.0 |
| [AVP](https://zenodo.org/records/3250230) | Vocal percussion (Path A) | ~3 h | CC BY 4.0 |

Datasets are downloaded via `scripts/download_data.py`; raw audio is never committed.

---

## Key references

- Caspe, Shier, Sandler, Saitis & McPherson (2025). *Designing Neural Synthesizers for Low-Latency Interaction*. arXiv:2503.11562 (BRAVE).
- Caillon & Esling (2021). *RAVE: A variational autoencoder for fast and high-quality neural audio synthesis*. arXiv:2111.05011.
- Engel, Hantrakul, Gu & Roberts (2020). *DDSP: Differentiable Digital Signal Processing*. ICLR.
- Wessel & Wright (2002). *Problems and Prospects for Intimate Musical Control of Computers*. Computer Music Journal.

Full bibliography in [`docs/thesis/references.bib`](docs/thesis/references.bib).
