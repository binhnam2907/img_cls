from __future__ import annotations

import torch
import torch.nn as nn


def _build_downsample(
    in_ch: int, out_ch: int, stride: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_ch, out_ch,
            kernel_size=1, stride=stride, bias=False,
        ),
        nn.BatchNorm2d(out_ch),
    )


class BasicBlock(nn.Module):
    """Two 3x3 conv residual block (ResNet-18/34)."""

    expansion = 1

    def __init__(
        self, in_ch: int, mid_ch: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(
            in_ch, mid_ch, 3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(
            mid_ch, out_ch, 3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    """1x1 -> 3x3 -> 1x1 bottleneck (ResNet-50/101)."""

    expansion = 4

    def __init__(
        self, in_ch: int, mid_ch: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        expanded = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(
            in_ch, mid_ch, 1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(
            mid_ch, mid_ch, 3,
            stride=stride, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(
            mid_ch, expanded, 1, bias=False,
        )
        self.bn3 = nn.BatchNorm2d(expanded)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


BlockType = type[BasicBlock] | type[Bottleneck]


class ResNet(nn.Module):
    """Full ResNet built from scratch."""

    def __init__(
        self,
        block: BlockType,
        layer_counts: list[int],
        num_classes: int = 1000,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._cur_ch = 64

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, 64,
                kernel_size=7, stride=2,
                padding=3, bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1,
            ),
        )

        self.layer1 = self._make_layer(
            block, 64, layer_counts[0], stride=1,
        )
        self.layer2 = self._make_layer(
            block, 128, layer_counts[1], stride=2,
        )
        self.layer3 = self._make_layer(
            block, 256, layer_counts[2], stride=2,
        )
        self.layer4 = self._make_layer(
            block, 512, layer_counts[3], stride=2,
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512 * block.expansion, num_classes),
        )

        self._init_weights()

    def _make_layer(
        self, block: BlockType,
        mid_ch: int, num_blocks: int, stride: int,
    ) -> nn.Sequential:
        expanded = mid_ch * block.expansion
        downsample = None
        if stride != 1 or self._cur_ch != expanded:
            downsample = _build_downsample(
                self._cur_ch, expanded, stride,
            )

        blocks = [
            block(self._cur_ch, mid_ch, stride, downsample),
        ]
        self._cur_ch = expanded

        for _ in range(1, num_blocks):
            blocks.append(
                block(self._cur_ch, mid_ch),
            )

        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            match m:
                case nn.Conv2d():
                    nn.init.kaiming_normal_(
                        m.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                case nn.BatchNorm2d():
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                case nn.Linear():
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


# ── variant registry ──────────────────────────────

RESNET_VARIANTS: dict[str, tuple[BlockType, list[int]]] = {
    "resnet18": (BasicBlock, [2, 2, 2, 2]),
    "resnet34": (BasicBlock, [3, 4, 6, 3]),
    "resnet50": (Bottleneck, [3, 4, 6, 3]),
    "resnet101": (Bottleneck, [3, 4, 23, 3]),
}


def build_resnet(
    name: str,
    num_classes: int = 10,
    pretrained: bool = False,
    dropout: float = 0.0,
) -> ResNet:
    if pretrained:
        raise NotImplementedError(
            "Pretrained weights are not available for "
            "the from-scratch implementation. "
            "Set pretrained=false in your config."
        )

    if name not in RESNET_VARIANTS:
        raise ValueError(
            f"Unknown ResNet variant '{name}'. "
            f"Choose from {list(RESNET_VARIANTS)}"
        )

    block, layer_counts = RESNET_VARIANTS[name]
    return ResNet(
        block, layer_counts,
        num_classes=num_classes, dropout=dropout,
    )
