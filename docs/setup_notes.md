# Environment Setup Notes

## System

| Component | Version |
|-----------|---------|
| OS | Windows 11 |
| Python | 3.14.3 |
| GPU | NVIDIA RTX 5080 (sm_120, 16GB VRAM) |
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

```bash
# Activate venv (includes ffmpeg in PATH)
source venv/Scripts/activate

# Preprocess audio
rave preprocess \
  --input_path data/raw/guitarset/audio_mono-mic \
  --output_path data/processed/guitar \
  --sampling_rate 44100

# Train (guitar model, ~12-20h on RTX 5080)
rave train \
  --config configs/BRAVE/brave.gin \
  --name guitar_v1 \
  --db_path data/processed/guitar \
  --channels 1 \
  --gpu 0
```

> **Note:** Always pass `--channels 1` explicitly. The default is 0, which causes
> a silent zero-dimension model initialization error.

---

## ffmpeg

Installed via `winget install --id Gyan.FFmpeg`. Path added to `venv/Scripts/activate`.

Location: `C:\Users\Usuario\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\`

If the venv activate script is regenerated (e.g., by re-creating the venv), re-add
the ffmpeg PATH line manually.
