"""Generate the TensorBoard scalar-dashboard figure for the methodology chapter.

Plots the five primary training scalars tracked during a representative healthy
run (guitar_v6): KL regularisation, multi-band spectral distance, the two
discriminator logits (pred_real / pred_fake) and the total discriminator loss.
Output: figures/figure_tensorboard_scalars.png at 300 DPI.
"""
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN = "runs/guitar_v6_bbcc6d36c3"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def load_scalars(run_dir, tag):
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


def smooth(v, a=0.9):
    if len(v) == 0:
        return v
    out = np.empty_like(v, dtype=float)
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = a * out[i - 1] + (1 - a) * v[i]
    return out


PANELS = [
    ("regularization", "KL divergence (regularisation)", "#c0392b"),
    ("multiband_spectral_distance", "Multi-band spectral distance", "#2c3e50"),
    ("pred_real", "Discriminator logit: pred_real", "#27ae60"),
    ("pred_fake", "Discriminator logit: pred_fake", "#2980b9"),
    ("loss_dis", "Discriminator loss (loss_dis)", "#8e44ad"),
]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
    axes = axes.ravel()
    for ax, (tag, title, color) in zip(axes, PANELS):
        s, v = load_scalars(RUN, tag)
        if len(s) == 0:
            ax.text(0.5, 0.5, f"{tag}\n(no data)", ha="center", va="center")
            ax.set_title(title)
            continue
        ax.plot(s / 1e6, v, color=color, lw=0.4, alpha=0.3)
        ax.plot(s / 1e6, smooth(v), color=color, lw=1.6)
        ax.axvline(1.5, color="grey", ls="--", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("Training step (millions)")
    # hide the unused 6th panel, add a legend/explanation there
    axes[5].axis("off")
    axes[5].text(0.0, 0.8, "Representative run: guitar_v6", fontweight="bold")
    axes[5].text(0.0, 0.62, "Dashed line: Phase 1 → Phase 2\n(adversarial onset, ~1.5 M steps)")
    axes[5].text(0.0, 0.36, "Faint trace: raw scalar\nSolid trace: EMA-smoothed")
    fig.suptitle("Example TensorBoard training scalars (guitar_v6)", fontweight="bold", y=1.0)
    fig.tight_layout()
    out = FIG_DIR / "figure_tensorboard_scalars.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
