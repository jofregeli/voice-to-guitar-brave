"""Synthesise three one-shot drum samples for the Path A vocal-percussion demo.

Each sample is generated procedurally so we have no licensing dependency.
Outputs: data/samples/kick.wav, snare.wav, hh.wav  (mono, 44100 Hz, 16-bit PCM)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 44100
OUT_DIR = Path("data/samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def env(n: int, attack_ms: float, release_ms: float) -> np.ndarray:
    a = max(1, int(SR * attack_ms / 1000))
    r = max(1, int(SR * release_ms / 1000))
    out = np.zeros(n, dtype=np.float32)
    out[:a] = np.linspace(0, 1, a, dtype=np.float32)
    rel = np.exp(-np.linspace(0, 6, max(0, n - a)))
    out[a:] = rel[:n - a].astype(np.float32)
    return out


def kick(duration_s: float = 0.35) -> np.ndarray:
    n = int(SR * duration_s)
    t = np.arange(n) / SR
    # Frequency sweep 150 -> 40 Hz over 120 ms
    f = 150 * np.exp(-t / 0.04) + 40
    phase = 2 * np.pi * np.cumsum(f) / SR
    tone = np.sin(phase).astype(np.float32)
    e = env(n, attack_ms=2, release_ms=duration_s * 1000 - 2)
    return (tone * e * 0.9).astype(np.float32)


def snare(duration_s: float = 0.20) -> np.ndarray:
    n = int(SR * duration_s)
    # Body: 180 Hz sine
    t = np.arange(n) / SR
    body = 0.4 * np.sin(2 * np.pi * 180 * t).astype(np.float32)
    # Noise component, band-passed 1-4 kHz
    noise = np.random.randn(n).astype(np.float32) * 0.9
    sos = butter(4, [1000, 4000], btype="bandpass", fs=SR, output="sos")
    noise = sosfilt(sos, noise).astype(np.float32)
    e_body = env(n, attack_ms=2, release_ms=150)
    e_noise = env(n, attack_ms=1, release_ms=130)
    return (body * e_body + noise * e_noise * 0.9).astype(np.float32) * 0.7


def hh(duration_s: float = 0.10) -> np.ndarray:
    n = int(SR * duration_s)
    noise = np.random.randn(n).astype(np.float32)
    sos = butter(4, 6000, btype="highpass", fs=SR, output="sos")
    noise = sosfilt(sos, noise).astype(np.float32)
    e = env(n, attack_ms=1, release_ms=70)
    return (noise * e * 0.6).astype(np.float32)


def normalize(x: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = float(np.abs(x).max())
    if peak == 0:
        return x
    return (x * (target_peak / peak)).astype(np.float32)


def main():
    np.random.seed(42)
    for name, gen in [("kick", kick), ("snare", snare), ("hh", hh)]:
        x = normalize(gen())
        path = OUT_DIR / f"{name}.wav"
        sf.write(str(path), x, SR, subtype="PCM_16")
        print(f"  wrote {path}  ({len(x)} samples, {len(x)/SR*1000:.0f} ms)")


if __name__ == "__main__":
    main()
