# Voice-to-Guitar (and Beatbox-to-Drums) — TFG Project

Real-time voice-to-instrument timbre transfer using [BRAVE](https://github.com/fcaspe/BRAVE) and Pure Data.

**Author:** Jofre Geli de Fuenmayor
**Degree:** Audiovisual Systems Engineerin, UPF
**Deadline:** June 12, 2026

---

## What This Does

- Sings/hums into a microphone → outputs realistic **guitar** sound in real-time
- Beatboxes into a microphone → outputs realistic **drum** sound in real-time (secondary experiment)
- Runs live with ~5ms latency using BRAVE's streaming encoder + Pure Data's `nn~` external

---

## Project Structure

```
voice-to-guitar-brave/
├── config/                  # BRAVE training configuration files
│   ├── guitar_model.yaml    # Config for guitar model
│   └── drums_model.yaml     # Config for drums model
├── data/
│   ├── raw/                 # Downloaded datasets (not committed to git)
│   │   ├── guitarset/
│   │   ├── guitartechs/
│   │   └── groove/
│   └── processed/           # Preprocessed audio ready for BRAVE (not committed)
│       ├── guitar/
│       └── drums/
├── docs/
│   ├── workplan.md          # Week-by-week project plan
│   └── state_of_the_art.md  # Thesis Section 2 draft
├── examples/
│   └── pd/                  # Pure Data patches for real-time demo
├── models/                  # Trained model checkpoints (not committed to git)
│   ├── guitar/
│   └── drums/
├── scripts/
│   ├── download_data.py     # Download all datasets
│   └── preprocess.py        # Preprocess audio for BRAVE training
├── src/
│   ├── preprocessing/       # Audio preprocessing utilities
│   └── evaluation/          # Evaluation metrics (FAD, mel distance, etc.)
├── tests/                   # Unit tests
├── Dockerfile               # Reproducible training environment
├── docker-compose.yml
└── requirements.txt
```

---

## Setup

### Option A — One-command setup (recommended)

```bash
git clone https://github.com/jofregeli/voice-to-guitar-brave
cd voice-to-guitar-brave
python scripts/setup.py
```

The setup script handles everything automatically:
- Creates `venv/` with the correct Python version
- Installs PyTorch with the right CUDA wheel for your GPU (Blackwell/sm_120+ → cu128, older → cu126, no GPU → CPU)
- Installs all dependencies with the correct flags to avoid conflicts
- Patches `acids-rave` for scipy ≥ 1.14 and pytorch-lightning 2.x compatibility
- Detects ffmpeg and adds it to the venv `activate` script if needed

Then activate:
```bash
source venv/Scripts/activate   # Git Bash on Windows
source venv/bin/activate       # Linux / macOS
```

### Option B — Docker (for reproducibility)

```bash
docker-compose up --build
```

---

## Datasets

| Dataset | Instrument | Duration | License | Download |
|---------|-----------|----------|---------|----------|
| [GuitarSet](https://zenodo.org/records/3371780) | Acoustic guitar | ~3h | CC BY 4.0 | `python scripts/download_data.py --dataset guitarset` |
| [Guitar-TECHS](https://zenodo.org/records/14963133) | Electric guitar | ~5.2h | CC BY 4.0 | `python scripts/download_data.py --dataset guitartechs` |
| [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) | Drums | ~13h | CC BY 4.0 | `python scripts/download_data.py --dataset groove` |

---

## Training

```bash
# 1. Preprocess audio
python scripts/preprocess.py --instrument guitar
python scripts/preprocess.py --instrument drums

# 2. Train (uses config files in config/)
# Guitar model
rave train --config config/guitar_model.gin --name guitar_v1 --preprocessed data/processed/guitar

# Drums model
rave train --config config/drums_model.gin --name drums_v1 --preprocessed data/processed/drums
```

Expected training time on RTX 5080: ~12–24 hours per model.

---

## Real-Time Demo (Pure Data)

1. Open Pure Data
2. Install `nn~` external (see [RAVE releases](https://github.com/acids-ircam/RAVE/releases))
3. Open `examples/pd/voice_to_guitar.pd`
4. Load your trained model checkpoint
5. Enable audio (Ctrl+Alt+A), select your microphone

---

## Key References

- Caspe et al. (2025). *BRAVE: Bravely Realtime Audio Variational autoEncoder*. arXiv:2503.11562
- Caillon & Esling (2021). *RAVE: A variational autoencoder for fast and high-quality neural audio synthesis*. arXiv:2111.05011
- Engel et al. (2020). *DDSP: Differentiable Digital Signal Processing*. ICLR 2020.
- Engel et al. (2017). *Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders*. ICML 2017.
- Huang et al. (2018). *TimbreTron: A WaveNet(CycleGAN(CQT(Audio))) Pipeline for Musical Timbre Transfer*. arXiv:1811.09620
