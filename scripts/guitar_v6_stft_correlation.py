"""Calibration study of the STFT-magnitude-correlation diagnostic across the
full iteration sequence.

Runs the same in-distribution reconstruction test on every available guitar
checkpoint (v1, v2, v3, v4, v5, v6) using the same N GuitarSet samples,
N_FFT and HOP settings. Reports the mean correlation per checkpoint so the
0.3 threshold of §3.4 can be re-calibrated against the actual measurements.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa

SAMPLE_RATE = 44100
N_SAMPLES = 8
CLIP_SECONDS = 5.0
N_FFT = 2048
HOP_LENGTH = 512

MODELS = [
    ("guitar_v1", "models/guitar_v1.ts"),
    ("guitar_v2", "models/guitar_v2_final.ts"),
    ("guitar_v3", "models/guitar_v3.ts"),
    ("guitar_v4", "models/guitar_v4.ts"),
    ("guitar_v5", "models/guitar_v5.ts"),
    ("guitar_v6 (phase1)", "models/guitar_v6_phase1.ts"),
    ("guitar_v6 (best)", "models/guitar_v6_best.ts"),
    ("guitar_v6 (final)", "models/guitar_v6_final.ts"),
]


def load_clip(path: str, n_samples: int) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(y) <= n_samples:
        return y
    start = (len(y) - n_samples) // 2
    return y[start:start + n_samples]


def whole_clip_pass(model, audio: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
        y = model(x)
        y_np = y[0, 0].cpu().numpy() if y.ndim == 3 else y.cpu().numpy().reshape(-1)
    return y_np[:len(audio)] if len(y_np) >= len(audio) else np.pad(y_np, (0, len(audio) - len(y_np)))


def stft_mag(x: np.ndarray) -> np.ndarray:
    return np.abs(librosa.stft(x.astype(np.float32), n_fft=N_FFT, hop_length=HOP_LENGTH))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel(); b = b.ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def aligned_pearson(X_in: np.ndarray, Y_out: np.ndarray, max_shift_frames: int = 8) -> float:
    best = -1.0
    min_t = min(X_in.shape[1], Y_out.shape[1])
    for shift in range(-max_shift_frames, max_shift_frames + 1):
        if shift >= 0:
            a = X_in[:, :min_t - shift]
            b = Y_out[:, shift:shift + (min_t - shift)]
        else:
            s = -shift
            a = X_in[:, s:s + (min_t - s)]
            b = Y_out[:, :min_t - s]
        if a.shape != b.shape or a.size == 0:
            continue
        c = pearson(a, b)
        if not np.isnan(c) and c > best:
            best = c
    return best


def main():
    torch.set_grad_enabled(False)
    all_wavs = sorted(glob.glob("data/raw/guitarset/*_solo_mic.wav"))
    rng = np.random.default_rng(42)
    selected = list(rng.choice(all_wavs, size=min(N_SAMPLES, len(all_wavs)), replace=False))
    n_samples = int(CLIP_SECONDS * SAMPLE_RATE)
    print(f"Evaluating with N = {len(selected)} GuitarSet samples, {CLIP_SECONDS:.1f} s each")
    print()

    print(f"{'checkpoint':<24} {'mean_raw':<10} {'mean_align':<11} {'mean_out_rms':<13} {'in_rms_mean':<11}")
    print("-" * 80)
    for label, path in MODELS:
        if not Path(path).exists():
            print(f"{label:<24} (missing)")
            continue
        raw_cs = []
        align_cs = []
        rmss = []
        in_rmss = []
        for wav in selected:
            x = load_clip(wav, n_samples)
            m = torch.jit.load(path, map_location="cpu"); m.eval()
            try:
                y = whole_clip_pass(m, x)
            except Exception as e:
                print(f"{label:<24} FAILED ({type(e).__name__})")
                raw_cs = None
                break
            X = stft_mag(x)
            Y = stft_mag(y)
            min_t = min(X.shape[1], Y.shape[1])
            raw_cs.append(pearson(X[:, :min_t], Y[:, :min_t]))
            align_cs.append(aligned_pearson(X[:, :min_t], Y[:, :min_t]))
            rmss.append(float(np.sqrt((y ** 2).mean())))
            in_rmss.append(float(np.sqrt((x ** 2).mean())))
        if raw_cs is None:
            continue
        raw_arr = np.array(raw_cs)
        ali_arr = np.array(align_cs)
        print(f"{label:<24} {raw_arr.mean():<10.4f} {ali_arr.mean():<11.4f} {np.mean(rmss):<13.4f} {np.mean(in_rmss):<11.4f}")
    print()
    print("Interpretation: higher = closer spectral envelope to input.")
    print("Threshold from §3.4: 0.3 (set as 'autoencoder works for in-distribution input').")


if __name__ == "__main__":
    main()
