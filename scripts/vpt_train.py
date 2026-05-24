"""Train the VPT classifier on the AVP dataset.

Speaker-disjoint train/val split: 5 of 28 participants are held out, so the
reported validation accuracy reflects generalisation to unseen voices.

Usage:
    python scripts/vpt_train.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

# Allow `from src.vpt import ...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.vpt import CLASS_NAMES, N_CLASSES, SAMPLE_RATE, WINDOW_SAMPLES
from src.vpt.dataset import AVPWindows, index_avp, participant_id
from src.vpt.model import VPTClassifier, count_params

AVP_ROOT = "data/raw/avp/AVP_Dataset"
OUT_DIR = Path("runs/vpt")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_wav_cache(examples):
    paths = sorted({ex.wav_path for ex in examples})
    print(f"[cache] loading {len(paths)} WAV files into memory...")
    cache = {}
    for i, p in enumerate(paths):
        y, sr = sf.read(p, dtype="float32", always_2d=False)
        assert sr == SAMPLE_RATE, f"{p}: expected SR {SAMPLE_RATE}, got {sr}"
        if y.ndim > 1:
            y = y.mean(axis=1)
        cache[p] = y
        if (i + 1) % 50 == 0:
            print(f"  loaded {i+1}/{len(paths)}")
    total_s = sum(len(y) for y in cache.values()) / SAMPLE_RATE
    print(f"[cache] total audio: {total_s:.1f} s")
    return cache


def split_examples(examples, val_participants):
    val_set = set(val_participants)
    train, val = [], []
    for ex in examples:
        (val if participant_id(ex.wav_path) in val_set else train).append(ex)
    return train, val


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
            for p, t in zip(pred.cpu().numpy(), y.cpu().numpy()):
                cm[t, p] += 1
    return correct / max(1, total), cm


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"[index] scanning {AVP_ROOT}")
    examples = index_avp(AVP_ROOT)
    print(f"[index] {len(examples)} labelled onsets")
    label_counts = Counter(ex.label for ex in examples)
    for lbl, n in sorted(label_counts.items()):
        print(f"  class {lbl} ({CLASS_NAMES[lbl]}): {n}")

    val_participants = [f"Participant_{i}" for i in (5, 11, 17, 22, 28)]
    train_ex, val_ex = split_examples(examples, val_participants)
    print(f"[split] train={len(train_ex)}  val={len(val_ex)} ({val_participants})")

    cache = load_wav_cache(examples)

    train_ds = AVPWindows(train_ex, cache, augment=True)
    val_ds = AVPWindows(val_ex, cache, augment=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    model = VPTClassifier().to(device)
    print(f"[model] params={count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

    best_val = 0.0
    best_path = OUT_DIR / "vpt_best.pt"

    for epoch in range(1, 31):
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
        val_acc, cm = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(
            f"epoch {epoch:>2d}  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
            f"val_acc={val_acc:.3f}  lr={sched.get_last_lr()[0]:.2e}  {dt:.1f}s"
        )
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict(), "val_acc": val_acc}, best_path)
            print(f"  -> saved best ({val_acc:.3f}) to {best_path}")

    # Final report
    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    val_acc, cm = evaluate(model, val_loader, device)
    print(f"\nBest val accuracy: {val_acc:.3f}")
    print("Confusion matrix (rows = true, cols = pred):")
    header = "          " + "  ".join(f"{c:>6s}" for c in CLASS_NAMES)
    print(header)
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:>6s}  " + "  ".join(f"{cm[i,j]:>6d}" for j in range(N_CLASSES)))
    print(f"\nSaved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
