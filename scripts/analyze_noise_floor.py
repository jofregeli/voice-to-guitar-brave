"""
Analyse noise floor across guitar dataset sources.

For each source directory, computes the noise floor (RMS during silent regions)
of every audio file and reports per-source statistics. Used to quantify the
recording-method variation noted by supervisor: GuitarSet uses mic'd acoustic
recordings, Guitar-TECHS uses line-out DI, and IDMT-SMT-Guitar dataset4 mixes
acoustic_mic, acoustic_pickup, and electric DI.

Methodology:
  1. Split each file into 100ms windows
  2. Compute RMS per window
  3. Take the 5th percentile of RMS values as the noise floor estimate
     (assumption: at least 5% of any musical recording is between notes,
     during decay tails, or in pre/post-roll silence)
  4. Convert to dBFS for readability

Usage:
    python scripts/analyze_noise_floor.py
"""

from pathlib import Path
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

# Each entry: (display name, directory path)
SOURCES = [
    ("GuitarSet (mic'd acoustic)", "data/raw/guitarset"),
    ("Guitar-TECHS (line-out DI)", "data/raw/guitartechs"),
    ("IDMT-SMT-Guitar/dataset4/acoustic_mic", "data/raw/idmt_smt_guitar/IDMT-SMT-GUITAR_V2/dataset4/acoustic_mic"),
    ("IDMT-SMT-Guitar/dataset4/acoustic_pickup", "data/raw/idmt_smt_guitar/IDMT-SMT-GUITAR_V2/dataset4/acoustic_pickup"),
    ("IDMT-SMT-Guitar/dataset4/Career SG (electric)", "data/raw/idmt_smt_guitar/IDMT-SMT-GUITAR_V2/dataset4/Career SG"),
    ("IDMT-SMT-Guitar/dataset4/Ibanez 2820 (electric)", "data/raw/idmt_smt_guitar/IDMT-SMT-GUITAR_V2/dataset4/Ibanez 2820"),
]

WINDOW_MS = 100
PERCENTILE = 5.0  # 5th percentile of windowed RMS = noise floor estimate
TARGET_SR = 44100


def noise_floor_dbfs(audio: np.ndarray, sr: int) -> float:
    """Estimate noise floor in dBFS from the 5th percentile of windowed RMS."""
    win = int(sr * WINDOW_MS / 1000)
    if len(audio) < win:
        return None
    n_windows = len(audio) // win
    audio = audio[: n_windows * win].reshape(n_windows, win)
    rms = np.sqrt(np.mean(audio ** 2, axis=1))
    rms = rms[rms > 0]  # ignore exact-zero silence
    if len(rms) == 0:
        return None
    floor_amplitude = np.percentile(rms, PERCENTILE)
    return 20 * np.log10(floor_amplitude + 1e-12)


def main():
    print(f"\nNoise floor analysis (5th percentile of {WINDOW_MS}ms RMS windows)\n")
    print(f"{'Source':<55} {'N files':>8} {'Noise floor (dBFS)':>20}")
    print(f"{'':<55} {'':>8} {'median  /  IQR':>20}")
    print("-" * 90)

    summary = []

    for name, dir_str in SOURCES:
        d = Path(dir_str)
        if not d.exists():
            print(f"{name:<55} (directory not found)")
            continue

        wavs = list(d.rglob("*.wav"))
        # Skip __MACOSX/ artefacts and stereo files for fair comparison
        wavs = [w for w in wavs if "__MACOSX" not in str(w)]

        floors = []
        for wav in wavs:
            try:
                y, sr = librosa.load(str(wav), sr=TARGET_SR, mono=True)
                if y is None or len(y) == 0:
                    continue
                f = noise_floor_dbfs(y, sr)
                if f is not None and np.isfinite(f):
                    floors.append(f)
            except Exception:
                continue

        if not floors:
            print(f"{name:<55} {len(wavs):>8} (no measurements)")
            continue

        median = np.median(floors)
        q1 = np.percentile(floors, 25)
        q3 = np.percentile(floors, 75)
        print(f"{name:<55} {len(floors):>8} {median:>10.1f}  /  {q3-q1:>5.1f}")
        summary.append((name, len(floors), median, q1, q3))

    # Range across sources
    if len(summary) >= 2:
        medians = [s[2] for s in summary]
        spread = max(medians) - min(medians)
        print()
        print(f"Spread across sources: {spread:.1f} dB (min {min(medians):.1f}, max {max(medians):.1f})")
        print()
        print("Interpretation:")
        print("  Lower (more negative) = quieter = cleaner recording")
        print("  Higher (less negative) = louder noise floor = mic'd or noisier signal path")
        print("  A spread > 10 dB indicates the model will encounter substantially")
        print("  different background-noise levels between training samples.")


if __name__ == "__main__":
    main()
