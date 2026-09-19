"""Classifier backbones.

Deliberately conventional.  The study compares signal representations and
selective-risk behaviour, not architectures, so each pipeline uses a strong,
well-established network and nothing more exotic:

    xresnet1d101   1-D time-series backbone for Raw (12 ch) and VCG (3 ch)
    resnet2d18     2-D backbone for the log-power STFT

Both end in ``K`` independent sigmoid logits -- never a softmax, since a record
may carry several simultaneous diagnoses.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  xresnet1d
# --------------------------------------------------------------------------- #
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


class _Bottleneck1d(nn.Module):
    """ResNet-D bottleneck: the shortcut downsamples by average pooling so no
    spatial information is dropped by a strided 1x1 convolution."""

    expansion = 4

    def __init__(self, ni: int, nh: int, stride: int = 1, ks: int = 5):
        super().__init__()
        nf = nh * self.expansion
        self.convs = nn.Sequential(
            _conv_bn_act(ni, nh, 1),
            _conv_bn_act(nh, nh, ks, stride=stride),
            _conv_bn_act(nh, nf, 1, act=False, zero_bn=True),
        )
        idpath: list[nn.Module] = []
        if stride != 1:
            idpath.append(nn.AvgPool1d(2, stride=stride, ceil_mode=True))
        if ni != nf:
            idpath.append(_conv_bn_act(ni, nf, 1, act=False))
        self.idpath = nn.Sequential(*idpath) if idpath else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.convs(x) + self.idpath(x))


class XResNet1d(nn.Module):
    def __init__(self, layers: list[int], n_in: int, n_classes: int,
                 ks: int = 5, widen: float = 1.0, p_drop: float = 0.5):
        super().__init__()
        stem_szs = [n_in, 32, 32, 64]
        self.stem = nn.Sequential(
            *[_conv_bn_act(stem_szs[i], stem_szs[i + 1], ks,
                           stride=2 if i == 0 else 1) for i in range(3)],
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        block_szs = [int(64 * widen * 2 ** i) for i in range(len(layers))]
        blocks: list[nn.Module] = []
        ni = 64
        for i, (nh, n_blocks) in enumerate(zip(block_szs, layers)):
            for b in range(n_blocks):
                blocks.append(_Bottleneck1d(ni, nh, stride=2 if (i > 0 and b == 0) else 1, ks=ks))
                ni = nh * _Bottleneck1d.expansion
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.BatchNorm1d(ni), nn.Dropout(p_drop), nn.Linear(ni, n_classes),
        )
        self.n_features = ni

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


def xresnet1d101(n_in: int, n_classes: int, **kw) -> XResNet1d:
    return XResNet1d([3, 4, 23, 3], n_in, n_classes, **kw)


def xresnet1d50(n_in: int, n_classes: int, **kw) -> XResNet1d:
    return XResNet1d([3, 4, 6, 3], n_in, n_classes, **kw)


# --------------------------------------------------------------------------- #
#  ResNet1d-18
# --------------------------------------------------------------------------- #
class _BasicBlock1d(nn.Module):
    """Two equal-width convolutions plus an identity shortcut (expansion 1)."""

    expansion = 1

    def __init__(self, ni: int, nf: int, stride: int = 1, ks: int = 5):
        super().__init__()
        self.convs = nn.Sequential(
            _conv_bn_act(ni, nf, ks, stride=stride),
            _conv_bn_act(nf, nf, ks, act=False, zero_bn=True),
        )
        self.short = (nn.Sequential(nn.Conv1d(ni, nf, 1, stride=stride, bias=False),
                                    nn.BatchNorm1d(nf))
                      if (stride != 1 or ni != nf) else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.convs(x) + self.short(x))


class ResNet1d(nn.Module):
    """Standard ResNet-18 topology in 1-D, sized for 10 s of 500 Hz ECG.

    Stem is a single wide stride-2 convolution plus max-pool, so the 5000-sample
    input is reduced to 1250 before the first block. Kernel width 5 in the blocks
    (rather than 3) keeps the receptive field wide enough to span a QRS complex
    at this sampling rate.
    """

    def __init__(self, layers: tuple[int, ...], n_in: int, n_classes: int,
                 ks: int = 5, stem_ks: int = 7, width: int = 64,
                 p_drop: float = 0.5):
        super().__init__()
        self.stem = nn.Sequential(
            _conv_bn_act(n_in, width, stem_ks, stride=2),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        blocks: list[nn.Module] = []
        ni = width
        for i, n_blocks in enumerate(layers):
            nf = width * 2 ** i
            for b in range(n_blocks):
                blocks.append(_BasicBlock1d(ni, nf, stride=2 if (i > 0 and b == 0) else 1,
                                            ks=ks))
                ni = nf
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.BatchNorm1d(ni), nn.Dropout(p_drop), nn.Linear(ni, n_classes),
        )
        self.n_features = ni

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


def resnet1d18(n_in: int, n_classes: int, **kw) -> ResNet1d:
    return ResNet1d((2, 2, 2, 2), n_in, n_classes, **kw)


def resnet1d34(n_in: int, n_classes: int, **kw) -> ResNet1d:
    return ResNet1d((3, 4, 6, 3), n_in, n_classes, **kw)


# --------------------------------------------------------------------------- #
#  2-D ResNet for spectrograms
# --------------------------------------------------------------------------- #
class _BasicBlock2d(nn.Module):
    def __init__(self, ni: int, nf: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(ni, nf, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nf)
        nn.init.constant_(self.bn2.weight, 0.0)
        self.short = (nn.Sequential(nn.Conv2d(ni, nf, 1, stride=stride, bias=False),
                                    nn.BatchNorm2d(nf))
                      if (stride != 1 or ni != nf) else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))
        return self.act(out + self.short(x))


class ResNet2d(nn.Module):
    """ResNet-18 topology, adapted to small spectrograms.

    The stem uses a stride-1 3x3 convolution and no max-pool: the STFT input is
    only ~25 frequency bins tall, so the usual 7x7/stride-2 + pool stem would
    destroy most of the frequency axis before the first block.
    """

    def __init__(self, n_in: int, n_classes: int,
                 layers: tuple[int, ...] = (2, 2, 2, 2), p_drop: float = 0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(n_in, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        blocks: list[nn.Module] = []
        ni = 64
        for i, n_blocks in enumerate(layers):
            nf = 64 * 2 ** i
            for b in range(n_blocks):
                blocks.append(_BasicBlock2d(ni, nf, stride=2 if (i > 0 and b == 0) else 1))
                ni = nf
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.BatchNorm1d(ni), nn.Dropout(p_drop), nn.Linear(ni, n_classes))
        self.n_features = ni

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


def resnet2d18(n_in: int, n_classes: int, **kw) -> ResNet2d:
    return ResNet2d(n_in, n_classes, **kw)


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
# Raw, VCG and FFT are all 1-D and share one backbone, differing only in the
# input channel count -- which is what makes this a representation benchmark
# rather than an architecture benchmark. ``stft`` is 2-D and is kept only as a
# non-study reference.
N_CHANNELS = {"raw": 12, "vcg": 3, "fft": 12, "stft": 12}

# 1-D backbones, for the Raw and VCG representations.
BACKBONES_1D = {
    "resnet1d18": resnet1d18,
    "resnet1d34": resnet1d34,
    "xresnet1d50": xresnet1d50,
    "xresnet1d101": xresnet1d101,
}

# ``resnet1d18`` is the default: it is the 1-D counterpart of the ResNet-18 used
# for the STFT, so all three representations share one depth/width family and
# the comparison isolates the representation rather than model capacity.
DEFAULT_BACKBONE_1D = "resnet1d18"


def build_model(representation: str, n_classes: int,
                backbone: str = DEFAULT_BACKBONE_1D, **kw) -> nn.Module:
    """``representation`` is one of ``raw`` / ``vcg`` / ``fft`` (or ``stft``).

    ``backbone`` selects the 1-D network used for raw/vcg/fft; it is ignored for
    ``stft``, which necessarily uses the 2-D ResNet-18 and is therefore not part
    of the study's representation comparison.
    """
    if representation not in N_CHANNELS:
        raise ValueError(f"unknown representation {representation!r}")
    if representation == "stft":
        return resnet2d18(N_CHANNELS["stft"], n_classes, **kw)
    if backbone not in BACKBONES_1D:
        raise ValueError(f"unknown backbone {backbone!r}; "
                         f"choose from {sorted(BACKBONES_1D)}")
    return BACKBONES_1D[backbone](N_CHANNELS[representation], n_classes, **kw)
