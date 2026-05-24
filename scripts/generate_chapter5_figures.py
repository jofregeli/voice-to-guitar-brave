"""Generate the five Chapter 5 figures from existing TensorBoard logs and data.

Output: figures/figure_5_*.png at 300 DPI.

Figures produced:
  5.1  KL trajectory of guitar_v1 showing posterior collapse
  5.2  GAN gap evolution comparing v4 (dominance), v5 (collapse), v6 (reversal)
  5.3  drums_v2 Phase 2 GAN gap widening (9.8 -> 20.3 across version_4..7)
  5.4  Spectrogram input/output comparison for guitar_v6 in-distribution
  5.5  Noise-floor distribution per source (boxplot)
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import librosa
import librosa.display

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_scalars(run_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate a scalar tag across all version_* subdirectories of a run."""
    files = sorted(glob.glob(os.path.join(run_dir, "version_*", "events.out.tfevents.*")))
    steps, values = [], []
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if tag not in ea.Tags()["scalars"]:
            continue
        for e in ea.Scalars(tag):
            steps.append(e.step)
            values.append(e.value)
    if not steps:
        return np.array([]), np.array([])
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(values)[order]


def smooth(values: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    if len(values) == 0:
        return values
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i-1] + (1 - alpha) * values[i]
    return out


# -------- Figure 5.1 --------
def figure_5_1():
    steps, kl = load_scalars("runs/guitar_v1_3415d67484", "regularization")
    if len(steps) == 0:
        print("[5.1] no KL data; skipping")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps / 1e6, kl, color="#888", lw=0.6, alpha=0.4, label="raw")
    ax.plot(steps / 1e6, smooth(kl), color="#c0392b", lw=1.8, label="smoothed")
    ax.set_xlabel("Training step (millions)")
    ax.set_ylabel("KL divergence (regularization)")
    ax.set_title("Figure 5.1. KL trajectory of guitar_v1 — posterior collapse")
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    out = FIG_DIR / "figure_5_1_guitar_v1_posterior_collapse.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[5.1] saved {out}  (steps {steps[0]} -> {steps[-1]}, KL {kl[0]:.3f} -> {kl[-1]:.3f})")


# -------- Figure 5.2 --------
def figure_5_2():
    """GAN gap = pred_real - pred_fake, comparing v4 vs v5 vs v6."""
    runs = [
        ("guitar_v4 (dominance)", "runs/guitar_v4_b994fb6a2a", "#d35400"),
        ("guitar_v5 (collapse)", "runs/guitar_v5_33f9fba0ce", "#2980b9"),
        ("guitar_v6 (reversal)", "runs/guitar_v6_bbcc6d36c3", "#27ae60"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, run, color in runs:
        s_real, v_real = load_scalars(run, "pred_real")
        s_fake, v_fake = load_scalars(run, "pred_fake")
        if len(s_real) == 0 or len(s_fake) == 0:
            print(f"[5.2] no GAN data for {label}; skipping")
            continue
        # Align on common steps
        common = np.intersect1d(s_real, s_fake)
        if len(common) == 0:
            continue
        idx_r = np.searchsorted(s_real, common)
        idx_f = np.searchsorted(s_fake, common)
        gap = v_real[idx_r] - v_fake[idx_f]
        ax.plot(common / 1e6, smooth(gap, alpha=0.97), color=color, lw=1.6, label=label)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Training step (millions)")
    ax.set_ylabel("GAN logit gap (pred_real − pred_fake)")
    ax.set_title("Figure 5.2. Adversarial dynamics across iterations")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    out = FIG_DIR / "figure_5_2_gan_gap_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[5.2] saved {out}")


# -------- Figure 5.3 --------
def figure_5_3():
    """drums_v2 Phase 2 GAN gap widening."""
    s_real, v_real = load_scalars("runs/drums_v2_bbcc6d36c3", "pred_real")
    s_fake, v_fake = load_scalars("runs/drums_v2_bbcc6d36c3", "pred_fake")
    if len(s_real) == 0:
        print("[5.3] no drums_v2 GAN data; skipping")
        return
    common = np.intersect1d(s_real, s_fake)
    idx_r = np.searchsorted(s_real, common)
    idx_f = np.searchsorted(s_fake, common)
    gap = v_real[idx_r] - v_fake[idx_f]

    # Phase 1 ends around step 1.5M
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(common / 1e6, gap, color="#7f8c8d", lw=0.4, alpha=0.3, label="raw")
    ax.plot(common / 1e6, smooth(gap, alpha=0.98), color="#8e44ad", lw=1.8, label="smoothed")
    ax.axvline(1.5, color="grey", lw=0.6, ls="--", label="Phase 2 begins")
    ax.set_xlabel("Training step (millions)")
    ax.set_ylabel("GAN logit gap (pred_real − pred_fake)")
    ax.set_title("Figure 5.3. drums_v2 Phase 2 discriminator dominance reproduction")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    out = FIG_DIR / "figure_5_3_drums_v2_gan_widening.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[5.3] saved {out}  (gap end-of-training = {gap[-1]:.2f})")


# -------- Figure 5.4 --------
def figure_5_4():
    """Input vs guitar_v6 output spectrogram on a GuitarSet sample."""
    model_path = "models/guitar_v6_best.ts"
    if not Path(model_path).exists():
        print(f"[5.4] model missing: {model_path}")
        return
    wavs = sorted(glob.glob("data/raw/guitarset/*_solo_mic.wav"))
    if not wavs:
        print("[5.4] no GuitarSet samples available")
        return
    rng = np.random.default_rng(42)
    sample = wavs[int(rng.integers(0, len(wavs)))]
    sr = 44100
    n = int(5.0 * sr)  # 5s clip
    y, file_sr = sf.read(sample, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    start = max(0, (len(y) - n) // 2)
    y = y[start:start + n]

    torch.set_grad_enabled(False)
    m = torch.jit.load(model_path, map_location="cpu")
    m.eval()
    x = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        z = m(x)
    rec = z[0, 0].numpy() if z.ndim == 3 else z.numpy().reshape(-1)
    rec = rec[:len(y)]

    # Spectrograms (log-magnitude)
    S_in = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
    S_out = librosa.amplitude_to_db(np.abs(librosa.stft(rec.astype(np.float32), n_fft=2048, hop_length=512)), ref=np.max)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im0 = librosa.display.specshow(S_in, sr=sr, hop_length=512, y_axis="log", x_axis="time", ax=axes[0])
    axes[0].set_title("Input: in-distribution guitar (GuitarSet)")
    fig.colorbar(im0, ax=axes[0], format="%+2.0f dB")
    im1 = librosa.display.specshow(S_out, sr=sr, hop_length=512, y_axis="log", x_axis="time", ax=axes[1])
    axes[1].set_title("Output: guitar_v6 canonical reconstruction (epoch 1935)")
    fig.colorbar(im1, ax=axes[1], format="%+2.0f dB")
    fig.suptitle(f"Figure 5.4. In-distribution reconstruction example ({Path(sample).name})", y=1.02)
    out = FIG_DIR / "figure_5_4_spectrogram_in_dist.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[5.4] saved {out}  (sample = {Path(sample).name})")


# -------- Figure 5.5 --------
def figure_5_5():
    """Noise floor distribution per source. Computed via the same method as
    scripts/analyze_noise_floor.py: 5th percentile of 100ms RMS windows, in dBFS."""
    sources = {
        "GuitarSet (mic acoustic)": "data/raw/guitarset/*.wav",
        "Guitar-TECHS (DI electric)": "data/raw/guitartechs/**/*.wav",
        "IDMT acoustic_mic": "data/raw/idmt_smt_guitar/**/acoustic_mic/**/*.wav",
        "IDMT acoustic_pickup": "data/raw/idmt_smt_guitar/**/acoustic_pickup/**/*.wav",
        "IDMT Career SG (DI)": "data/raw/idmt_smt_guitar/**/Career SG/**/*.wav",
        "IDMT Ibanez 2820 (DI)": "data/raw/idmt_smt_guitar/**/Ibanez 2820/**/*.wav",
    }

    sr = 44100
    win = int(0.1 * sr)  # 100 ms

    def noise_floor_db(y: np.ndarray) -> float:
        n_chunks = max(1, len(y) // win)
        if n_chunks < 5:
            return float("nan")
        rms_values = []
        for i in range(n_chunks):
            chunk = y[i * win:(i + 1) * win]
            r = float(np.sqrt(np.mean(chunk * chunk)))
            if r > 0:
                rms_values.append(r)
        if not rms_values:
            return float("nan")
        p5 = float(np.percentile(rms_values, 5))
        return 20.0 * float(np.log10(p5 + 1e-12))

    data = {}
    for label, pattern in sources.items():
        files = glob.glob(pattern, recursive=True)
        if not files:
            print(f"[5.5] no files for {label}")
            continue
        vals = []
        for f in files[:200]:  # cap for speed
            try:
                y, file_sr = sf.read(f, dtype="float32", always_2d=False)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if file_sr != sr:
                    y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
                v = noise_floor_db(y)
                if np.isfinite(v):
                    vals.append(v)
            except Exception:
                continue
        if vals:
            data[label] = vals
            print(f"[5.5] {label}: n={len(vals)}, median={np.median(vals):.1f} dB")

    if not data:
        print("[5.5] no data; skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(data.keys())
    values = [data[k] for k in labels]
    box = ax.boxplot(values, vert=True, patch_artist=True, widths=0.6,
                     boxprops=dict(facecolor="#bdc3c7", color="#34495e"),
                     medianprops=dict(color="#c0392b", linewidth=2))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Noise floor (dBFS, 5th percentile of 100 ms RMS)")
    ax.set_title("Figure 5.5. Noise-floor distribution per recording source")
    ax.grid(True, axis="y", alpha=0.3)
    out = FIG_DIR / "figure_5_5_noise_floor.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[5.5] saved {out}")


def main():
    print("=" * 70)
    print("Generating Chapter 5 figures")
    print("=" * 70)
    figure_5_1()
    figure_5_2()
    figure_5_3()
    figure_5_4()
    figure_5_5()
    print()
    print(f"All done. Figures saved to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
