"""Small 1D-CNN classifier for vocal percussion (kick/snare/hh).

Architecture goal: <100k params, raw waveform input (1024 samples), streaming
inference in <1 ms on CPU. The model is fully convolutional and the only
non-causal operation is global pooling at the very end, which the streaming
deployment replaces with reading the most recent output frame.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import N_CLASSES, WINDOW_SAMPLES


class VPTClassifier(nn.Module):
    """1D ConvNet: raw audio -> class logits.

    Pipeline:
      Conv1d(1->16, k=32, s=8)  # 1024 -> 128 (analyses ~0.7 ms blocks)
      Conv1d(16->32, k=8, s=4)  # 128 -> 32
      Conv1d(32->64, k=4, s=2)  # 32 -> 16
      Conv1d(64->64, k=4, s=2)  # 16 -> 8
      AvgPool over time -> Linear -> 3 logits
    """

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32, stride=8, padding=12),
            nn.BatchNorm1d(16),
            nn.GELU(),

            nn.Conv1d(16, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) raw audio
        h = self.features(x)              # (B, 64, T')
        h = h.mean(dim=-1)                # global average pool over time
        return self.head(h)               # (B, n_classes) logits


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = VPTClassifier()
    x = torch.zeros(2, 1, WINDOW_SAMPLES)
    y = m(x)
    print(f"params: {count_params(m):,}")
    print(f"input  {tuple(x.shape)} -> output {tuple(y.shape)}")
