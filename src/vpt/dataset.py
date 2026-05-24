"""AVP dataset loader for VPT classifier training.

Each AVP recording is one WAV with multiple labelled onsets in a paired CSV
(onset_time_seconds, label). We extract a fixed-length causal window starting
at each onset, since at inference time the classifier sees the first N samples
of a transient (no lookahead allowed for streaming use).
"""
from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import LABEL_MAP, SAMPLE_RATE, WINDOW_SAMPLES


@dataclass
class OnsetExample:
    wav_path: str
    onset_sample: int
    label: int


def _load_csv(csv_path: str) -> List[Tuple[float, str]]:
    out = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row or len(row) < 2:
                continue
            try:
                t = float(row[0])
                lbl = row[1].strip()
                out.append((t, lbl))
            except ValueError:
                continue
    return out


def index_avp(root: str) -> List[OnsetExample]:
    """Walk AVP_Dataset directory and return per-onset examples."""
    examples: List[OnsetExample] = []
    wavs = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    skipped_unknown = 0
    for wav in wavs:
        csv_path = wav[:-4] + ".csv"
        if not os.path.exists(csv_path):
            continue
        for t, lbl in _load_csv(csv_path):
            if lbl not in LABEL_MAP:
                skipped_unknown += 1
                continue
            examples.append(
                OnsetExample(
                    wav_path=wav,
                    onset_sample=int(round(t * SAMPLE_RATE)),
                    label=LABEL_MAP[lbl],
                )
            )
    if skipped_unknown:
        print(f"[avp] skipped {skipped_unknown} onsets with unrecognised labels")
    return examples


def participant_id(wav_path: str) -> str:
    """Return 'Participant_N' from path, for speaker-disjoint splits."""
    parts = wav_path.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("Participant_"):
            return p
    return "unknown"


class AVPWindows(Dataset):
    """Yields (audio_window, label) tensors of fixed length.

    `pre_samples` is how many samples *before* the onset we include; the rest
    of the window is after the onset. For a causal streaming classifier we
    typically use pre_samples = 0 (window starts AT the onset).

    Train-time augmentation: random gain (-6 to +3 dB), and random onset jitter
    (+/- 2 ms) so the model is robust to imprecise onset detection in PD.
    """

    def __init__(
        self,
        examples: List[OnsetExample],
        wav_cache: dict,
        pre_samples: int = 0,
        window_samples: int = WINDOW_SAMPLES,
        augment: bool = False,
        jitter_samples: int = 88,  # ~2 ms @ 44.1 kHz
    ):
        self.examples = examples
        self.wav_cache = wav_cache
        self.pre = pre_samples
        self.win = window_samples
        self.augment = augment
        self.jitter_samples = jitter_samples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        y = self.wav_cache[ex.wav_path]  # np.ndarray, 1D float32

        start = ex.onset_sample - self.pre
        if self.augment and self.jitter_samples > 0:
            start += int(np.random.randint(-self.jitter_samples, self.jitter_samples + 1))
        start = max(0, start)
        end = start + self.win

        if end <= len(y):
            window = y[start:end]
        else:
            window = np.zeros(self.win, dtype=np.float32)
            avail = max(0, len(y) - start)
            if avail > 0:
                window[:avail] = y[start:start + avail]

        if self.augment:
            gain_db = np.random.uniform(-6.0, 3.0)
            window = window * (10.0 ** (gain_db / 20.0))

        x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)  # (1, win)
        return x, ex.label
