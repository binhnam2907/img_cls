from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SimpleCNN(nn.Module):
    """Lightweight CNN for small images (e.g. CIFAR 32x32).

    Architecture: 3 stages of [conv-bn-relu x2 + pool], then linear head.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 32),
            nn.MaxPool2d(2),

            _conv_block(32, 64),
            _conv_block(64, 64),
            nn.MaxPool2d(2),

            _conv_block(64, 128),
            _conv_block(128, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
