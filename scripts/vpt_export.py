"""Export the trained VPT classifier as a TorchScript model loadable by nn~.

Streaming design:
    The wrapper maintains a 1024-sample circular buffer internally. On every
    audio block, it shifts in the new samples, runs the classifier on the
    latest 1024 samples, and outputs a (1, 3, block) tensor whose three
    channels are the class probabilities held constant across the block.

    PD reads each output channel as a continuous signal; the sample-trigger
    logic in the patch uses the value at the moment an onset fires.

Usage:
    python scripts/vpt_export.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import cached_conv as cc
import nn_tilde

from src.vpt import N_CLASSES, WINDOW_SAMPLES
from src.vpt.model import VPTClassifier


CKPT = Path("runs/vpt/vpt_best.pt")
OUT_TS = Path("models/vpt_classifier_streaming.ts")


class StreamingVPT(nn_tilde.Module):
    """Wraps VPTClassifier with a stateful sliding-window buffer for streaming."""

    def __init__(self, classifier: VPTClassifier, window: int = WINDOW_SAMPLES):
        super().__init__()
        self.classifier = classifier.eval()
        self.window = window
        # Buffer holds last `window` samples; nn~ always calls with B=1.
        self.register_buffer("buf", torch.zeros(1, 1, window))
        self.register_method(
            "forward",
            in_channels=1,
            in_ratio=1,
            out_channels=N_CLASSES,
            out_ratio=1,
            input_labels=["(signal) mic"],
            output_labels=["(signal) p_kick", "(signal) p_snare", "(signal) p_hh"],
            test_buffer_size=2048,
            test_method=False,  # batch>1 testing breaks state buffer; we test below
        )

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T). PD uses B=1; we ignore other batch elements' state.
        T = x.shape[-1]
        x0 = x[:1]  # take first batch element for state update
        if T >= self.window:
            self.buf.copy_(x0[:, :, -self.window:])
        else:
            self.buf.copy_(torch.cat([self.buf[:, :, T:], x0], dim=-1))

        logits = self.classifier(self.buf)        # (1, N_CLASSES)
        probs = F.softmax(logits, dim=-1)         # (1, N_CLASSES)
        # Broadcast to (B, N_CLASSES, T) so PD sees a per-block constant stream.
        return probs.unsqueeze(-1).expand(x.shape[0], -1, T).contiguous()


def main():
    assert CKPT.exists(), f"Missing checkpoint: {CKPT}"

    # Streaming inference must avoid cached_conv's training-only padding.
    cc.use_cached_conv(False)
    torch.set_grad_enabled(False)

    base = VPTClassifier()
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    base.load_state_dict(state["state_dict"])
    base.eval()
    val_acc = state.get("val_acc", None)
    print(f"Loaded checkpoint with val_acc = {val_acc}")

    wrapped = StreamingVPT(base).eval()
    for p in wrapped.parameters():
        p.requires_grad_(False)

    # Smoke test: ensure forward runs at several block sizes.
    for blk in (64, 128, 256, 1024, 2048):
        x = torch.randn(1, 1, blk) * 0.1
        y = wrapped(x)
        assert y.shape == (1, N_CLASSES, blk), f"shape mismatch at blk={blk}: {y.shape}"
        s = y[0, :, -1].sum().item()
        assert abs(s - 1.0) < 1e-3, f"probs do not sum to 1 at blk={blk}: {s}"
    print("Smoke tests passed.")

    OUT_TS.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(wrapped)
    scripted.save(str(OUT_TS))
    print(f"Saved TorchScript model to {OUT_TS}")
    sz_mb = OUT_TS.stat().st_size / 1024 / 1024
    print(f"Model file size: {sz_mb:.2f} MB")


if __name__ == "__main__":
    main()
