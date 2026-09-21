"""The fixed diagnostic backbone.

A small plain 1-D CNN, used unchanged across all five ensemble members.  The
architecture is not a contribution and is not varied: the question is what
model uncertainty does relative to *reference* uncertainty, and the model is
an instrument for producing that uncertainty.

Size is a deliberate choice rather than an accident.  In the superseded study
(see ``archive/beat_relation_study``) scaling this encoder to ResNet1D-18 cost
cross-source confidence ranking while barely moving accuracy, so a small
network is the safer default when the endpoint is uncertainty behaviour rather
than peak discrimination.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_act(ni: int, nf: int, ks: int = 3, stride: int = 1,
                 act: bool = True, zero_bn: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv1d(ni, nf, ks, stride=stride, padding=ks // 2, bias=False),
        nn.BatchNorm1d(nf),
    ]
    nn.init.constant_(layers[1].weight, 0.0 if zero_bn else 1.0)
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class _BasicBlock1d(nn.Module):
    """Two equal-width convolutions plus an identity shortcut."""

    def __init__(self, ni: int, nf: int, stride: int = 1, ks: int = 5):
        super().__init__()
        self.convs = nn.Sequential(
            _conv_bn_act(ni, nf, ks, stride=stride),
            _conv_bn_act(nf, nf, ks, act=False, zero_bn=True),
        )
        self.short = (nn.Sequential(
            nn.Conv1d(ni, nf, 1, stride=stride, bias=False),
            nn.BatchNorm1d(nf))
            if (stride != 1 or ni != nf) else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.convs(x) + self.short(x))


class ResNet1dLite(nn.Module):
    """Small 1-D residual network with independent sigmoid outputs.

    Kernel width 5 rather than 3 keeps the receptive field wide enough to span
    a QRS complex at these sampling rates.  Outputs are never softmaxed: a
    record may carry several simultaneous diagnoses.
    """

    def __init__(self, n_in: int = 12, n_classes: int = 6,
                 widths: tuple[int, ...] = (32, 64, 128, 256),
                 ks: int = 5, stem_ks: int = 7, p_drop: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            _conv_bn_act(n_in, widths[0], stem_ks, stride=2),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        blocks: list[nn.Module] = []
        ni = widths[0]
        for i, nf in enumerate(widths):
            blocks.append(_BasicBlock1d(ni, nf, stride=2 if i > 0 else 1, ks=ks))
            ni = nf
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.BatchNorm1d(ni), nn.Dropout(p_drop), nn.Linear(ni, n_classes),
        )
        self.n_features = ni

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class PlainCNN1d(nn.Module):
    """Four strided convolutions, global average pooling, six sigmoid outputs.

    The primary model.  Architecture is not the contribution here, and a
    reviewer should not have to read about residual bottlenecks to evaluate a
    claim about diagnostic uncertainty.  Stating that even a conventional
    lightweight classifier shows the effect is a stronger position than
    demonstrating it on something bespoke.

    Kernels shrink with depth (15, 11, 7, 5) while the receptive field grows
    through stride, which is the usual arrangement for ECG at a few hundred
    hertz: wide early filters span a QRS complex, later ones combine them.
    """

    def __init__(self, n_in: int = 12, n_classes: int = 6,
                 widths: tuple[int, ...] = (32, 64, 128, 128),
                 kernels: tuple[int, ...] = (15, 11, 7, 5)):
        super().__init__()
        layers: list[nn.Module] = []
        ni = n_in
        for nf, k in zip(widths, kernels):
            layers += [nn.Conv1d(ni, nf, k, stride=2, padding=k // 2,
                                 bias=False),
                       nn.BatchNorm1d(nf), nn.ReLU(inplace=True)]
            ni = nf
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(ni, n_classes))
        self.n_features = ni

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build(n_in: int = 12, n_classes: int = 6,
          kind: str = "plain") -> nn.Module:
    """``plain`` is the study model; ``resnet_lite`` is kept for sensitivity.

    Measured at batch 256 on 12 x 4096 input, forward plus backward:
    plain 11.2 ms, ResNet1D-Lite 17.6 ms, against a ~123 ms observed step --
    so the choice between them moves about 5% of training time. It is made on
    simplicity of exposition, not speed.
    """
    if kind == "plain":
        return PlainCNN1d(n_in=n_in, n_classes=n_classes)
    if kind == "resnet_lite":
        return ResNet1dLite(n_in=n_in, n_classes=n_classes)
    raise ValueError(f"unknown model {kind!r}")
