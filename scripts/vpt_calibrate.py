"""Personal calibration: fine-tune the VPT classifier on the user's own phonemes.

Workflow:
    1. User records three mono 44.1 kHz WAV files into data/user_calibration/:
         - kick.wav  (~30 deliberate "puh"/"boom"/"bm" hits)
         - snare.wav (~30 deliberate "ka"/"pa"/"k!" hits)
         - hh.wav    (~30 deliberate "shi"/"ts"/"tk" hits)
       Leave ~200 ms between hits. Background should be quiet.
    2. Run: python scripts/vpt_calibrate.py
       This script:
         (a) Detects onsets in each WAV via amplitude threshold.
         (b) Extracts 1024-sample windows starting at each onset.
         (c) Combines with AVP (downweighted) and fine-tunes the classifier.
         (d) Saves runs/vpt/vpt_calibrated.pt.
    3. Run: python scripts/vpt_export.py (after editing CKPT path) — or this
       script can also re-export automatically (controlled by REEXPORT below).

Heavy oversampling of the user's data (10x) makes the classifier strongly
adapt to user-specific phoneme spectra without forgetting general structure.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List

# Allow `from src.vpt import ...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from src.vpt import CLASS_NAMES, N_CLASSES, SAMPLE_RATE, WINDOW_SAMPLES
from src.vpt.dataset import AVPWindows, OnsetExample, index_avp
from src.vpt.model import VPTClassifier, count_params

# ---------- Config ----------
USER_DIR = Path("data/user_calibration")
AVP_ROOT = "data/raw/avp/AVP_Dataset"
BASE_CKPT = Path("runs/vpt/vpt_best.pt")
OUT_CKPT = Path("runs/vpt/vpt_calibrated.pt")

# Onset detection (offline) — env-shifted threshold tuned to match the PD patch.
# env value: PD-scale (0..100). We replicate that scale offline.
ONSET_THRESHOLD = 40.0      # env value above which we declare an onset
ENV_WINDOW = 256            # samples for envelope RMS
MIN_GAP_MS = 250            # debounce in ms (rejects within-hit secondary attacks)
HP_CUTOFF_HZ = 200          # high-pass to match PD chain

# Fine-tune
BALANCE_CLASSES = True      # downsample to the smallest class count per user
USER_OVERSAMPLE = 5         # repeat each user example N times so it dominates
USER_LR = 5e-4
EPOCHS = 12
BATCH_SIZE = 64

REEXPORT = True             # automatically run TorchScript export at the end
# ----------------------------


def env_dbpd(window: np.ndarray) -> float:
    """Mimic PD's env~ output (the '0..100' scale).

    PD's env~ returns an RMS measurement in a unit where ~100 = full-scale
    sinusoid. The conversion is: 100 + 20*log10(RMS) (clipped at 0).
    """
    rms = np.sqrt(np.mean(window * window)) + 1e-12
    db = 100.0 + 20.0 * np.log10(rms)
    return max(0.0, db)


def detect_onsets(y: np.ndarray, sr: int) -> List[int]:
    """Find onset sample indices in y using a PD-like envelope + threshold."""
    # High-pass at HP_CUTOFF_HZ using a simple 1-pole filter (matches PD's hip~).
    rc = 1.0 / (2 * np.pi * HP_CUTOFF_HZ)
    alpha = rc / (rc + 1.0 / sr)
    yh = np.zeros_like(y, dtype=np.float32)
    prev = 0.0
    prev_y = 0.0
    for i, s in enumerate(y):
        prev = alpha * (prev + s - prev_y)
        yh[i] = prev
        prev_y = s
    # Envelope: RMS in non-overlapping ENV_WINDOW chunks.
    n_chunks = len(yh) // ENV_WINDOW
    env = np.zeros(n_chunks, dtype=np.float32)
    for i in range(n_chunks):
        env[i] = env_dbpd(yh[i * ENV_WINDOW:(i + 1) * ENV_WINDOW])
    # Find rising edges (env crosses threshold from below).
    above = env > ONSET_THRESHOLD
    edges = np.where((~above[:-1]) & above[1:])[0] + 1  # +1 because edge is at i+1
    onsets = (edges * ENV_WINDOW).tolist()
    # Debounce: drop edges closer than MIN_GAP_MS apart.
    min_gap_samples = int(MIN_GAP_MS / 1000 * sr)
    out: List[int] = []
    for s in onsets:
        if not out or (s - out[-1]) >= min_gap_samples:
            out.append(s)
    return out


def load_user_examples() -> tuple[List[OnsetExample], dict]:
    """Index user WAVs and return per-onset examples + wav cache."""
    name_to_label = {"kick": 0, "snare": 1, "hh": 2}
    examples: List[OnsetExample] = []
    cache: dict = {}
    for name, label in name_to_label.items():
        wav_path = USER_DIR / f"{name}.wav"
        if not wav_path.exists():
            print(f"  [WARN] missing {wav_path} — skipping {name}")
            continue
        y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SAMPLE_RATE:
            print(f"  [WARN] {wav_path}: SR={sr} (expected {SAMPLE_RATE}); resample externally")
            continue
        onsets = detect_onsets(y, sr)
        print(f"  {wav_path.name}: detected {len(onsets)} {name} onsets ({len(y)/sr:.1f}s)")
        cache[str(wav_path)] = y
        for o in onsets:
            examples.append(OnsetExample(wav_path=str(wav_path), onset_sample=o, label=label))
    return examples, cache


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    if not BASE_CKPT.exists():
        raise SystemExit(f"Base checkpoint missing: {BASE_CKPT}. Run vpt_train.py first.")

    USER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[user] scanning {USER_DIR}")
    user_examples, user_cache = load_user_examples()
    print(f"[user] {len(user_examples)} user-labelled onsets")
    if len(user_examples) == 0:
        raise SystemExit(
            f"No user examples found. Drop kick.wav, snare.wav, hh.wav into {USER_DIR} and retry."
        )

    counts = {0: 0, 1: 0, 2: 0}
    for ex in user_examples:
        counts[ex.label] += 1
    print(f"[user] kick={counts[0]} snare={counts[1]} hh={counts[2]}")
    if min(counts.values()) < 5:
        print("[WARN] one class has <5 onsets — fine-tune will be unreliable for that class.")

    if BALANCE_CLASSES:
        per_class = min(counts.values())
        print(f"[balance] downsampling each class to {per_class} examples")
        rng = np.random.default_rng(0)
        by_class = {0: [], 1: [], 2: []}
        for ex in user_examples:
            by_class[ex.label].append(ex)
        balanced: List[OnsetExample] = []
        for cls in (0, 1, 2):
            idxs = rng.permutation(len(by_class[cls]))[:per_class]
            for i in idxs:
                balanced.append(by_class[cls][i])
        user_examples = balanced
        print(f"[balance] user examples after balancing: {len(user_examples)}")

    print(f"[avp]  indexing {AVP_ROOT}")
    avp_examples = index_avp(AVP_ROOT)
    print(f"[avp]  {len(avp_examples)} labelled onsets")

    # Load WAV caches.
    avp_paths = sorted({ex.wav_path for ex in avp_examples})
    print(f"[cache] loading {len(avp_paths)} AVP WAVs...")
    avp_cache = {}
    for i, p in enumerate(avp_paths):
        y, sr = sf.read(p, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        avp_cache[p] = y
        if (i + 1) % 50 == 0:
            print(f"  loaded {i+1}/{len(avp_paths)}")

    # Combine caches.
    cache = {**avp_cache, **user_cache}

    # Oversample user data.
    user_oversampled = user_examples * USER_OVERSAMPLE
    print(
        f"[mix] AVP={len(avp_examples)} examples + user={len(user_examples)} x{USER_OVERSAMPLE} "
        f"= {len(user_oversampled)} effective user examples"
    )

    train_examples = avp_examples + user_oversampled
    train_ds = AVPWindows(train_examples, cache, augment=True)

    # Hold out an unaugmented copy of user data for validation.
    val_ds = AVPWindows(user_examples, cache, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    model = VPTClassifier().to(device)
    state = torch.load(BASE_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    print(f"[init] loaded base checkpoint ({state.get('val_acc', '?')})")
    print(f"[model] params={count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=USER_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    def evaluate():
        model.eval()
        correct = 0
        total = 0
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.numel()
                for p, t in zip(pred.cpu().numpy(), y.cpu().numpy()):
                    cm[t, p] += 1
        return correct / max(1, total), cm

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        running_n = 0
        running_correct = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
            running_n += y.numel()
            running_correct += (logits.argmax(dim=-1) == y).sum().item()
        sched.step()
        tr_loss = running / running_n
        tr_acc = running_correct / running_n
        val_acc, cm = evaluate()
        dt = time.time() - t0
        print(
            f"epoch {epoch:>2d}  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
            f"val_acc(user)={val_acc:.3f}  lr={sched.get_last_lr()[0]:.2e}  {dt:.1f}s"
        )
        if val_acc >= best_val:
            best_val = val_acc
            torch.save(
                {"state_dict": model.state_dict(), "val_acc": val_acc, "calibrated": True},
                OUT_CKPT,
            )

    val_acc, cm = evaluate()
    print(f"\nFinal user-set accuracy: {val_acc:.3f}")
    print("Confusion matrix (rows = true, cols = pred):")
    print("          " + "  ".join(f"{c:>6s}" for c in CLASS_NAMES))
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:>6s}  " + "  ".join(f"{cm[i,j]:>6d}" for j in range(N_CLASSES)))
    print(f"\nSaved calibrated checkpoint to {OUT_CKPT}")

    if REEXPORT:
        # Make export script pick up the new checkpoint by swapping pointers.
        import shutil
        backup = Path("runs/vpt/vpt_best.preuser.pt")
        if not backup.exists():
            shutil.copy(BASE_CKPT, backup)
            print(f"Backed up base checkpoint to {backup}")
        shutil.copy(OUT_CKPT, BASE_CKPT)
        print(f"Overwrote {BASE_CKPT} with calibrated weights (backup at {backup})")
        print("Now run:  python scripts/vpt_export.py")


if __name__ == "__main__":
    main()
