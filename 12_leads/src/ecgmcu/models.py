r"""Lightweight 1-D encoders for the 12-lead benchmark.

Six families, chosen so the comparison says something rather than filling a
table. All share the same contract -- (batch, 12, T) in, (batch, 13) out,
global average pooling, one linear head -- so the only thing that varies is
how the trunk spends its parameters and its activation memory.

    tiny        plain strided convolutions; the lower-complexity reference
    resnet      ResNet1D-Lite, the baseline carried over from the archive
    dscnn       depthwise-separable, the usual MCU-oriented design
    mobilenet   MobileNetV1-style depthwise/pointwise stack
    mbconv      MobileNetV2-style inverted residual with expansion
    tcn         dilated temporal convolutions, no attention

mbconv earns its place by being the likely counter-example: its expansion
layers inflate intermediate tensors, so a model that looks small by
parameter count can still exceed SRAM. A benchmark that reported only
parameters would miss that, which is precisely the failure this study is
meant to surface.
"""

from __future__ import annotations

import torch
from torch import nn

N_LEADS, N_CLASSES = 12, 13


def _bn_relu(c: int) -> nn.Sequential:
    return nn.Sequential(nn.BatchNorm1d(c), nn.ReLU(inplace=True))


class Head(nn.Module):
    """Global average pool to one vector, then a single linear layer."""

    def __init__(self, c: int, n_classes: int = N_CLASSES):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c, n_classes)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


class TinyCNN(nn.Module):
    def __init__(self, n_in=N_LEADS, widths=(16, 32, 48, 64), k=7):
        super().__init__()
        layers, ci = [], n_in
        for co in widths:
            layers += [nn.Conv1d(ci, co, k, stride=2, padding=k // 2,
                                 bias=False), *_bn_relu(co)]
            ci = co
        self.features, self.head = nn.Sequential(*layers), Head(ci)

    def forward(self, x):
        return self.head(self.features(x))


class _Res(nn.Module):
    def __init__(self, ci, co, k=7, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ci, co, k, stride, k // 2, bias=False), *_bn_relu(co),
            nn.Conv1d(co, co, k, 1, k // 2, bias=False), nn.BatchNorm1d(co))
        self.skip = (nn.Sequential(nn.Conv1d(ci, co, 1, stride, bias=False),
                                   nn.BatchNorm1d(co))
                     if (ci != co or stride != 1) else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class ResNet1DLite(nn.Module):
    def __init__(self, n_in=N_LEADS, widths=(32, 64, 128, 128)):
        super().__init__()
        blocks, ci = [], n_in
        for co in widths:
            blocks.append(_Res(ci, co))
            ci = co
        self.features, self.head = nn.Sequential(*blocks), Head(ci)

    def forward(self, x):
        return self.head(self.features(x))


class _DWSep(nn.Module):
    """Depthwise then pointwise: the MCU workhorse."""

    def __init__(self, ci, co, k=9, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(ci, ci, k, stride, k // 2, groups=ci, bias=False),
            *_bn_relu(ci),
            nn.Conv1d(ci, co, 1, bias=False), *_bn_relu(co))

    def forward(self, x):
        return self.block(x)


class DSCNN(nn.Module):
    def __init__(self, n_in=N_LEADS, widths=(32, 64, 64, 96)):
        super().__init__()
        stem = [nn.Conv1d(n_in, widths[0], 9, 2, 4, bias=False),
                *_bn_relu(widths[0])]
        body, ci = [], widths[0]
        for co in widths[1:]:
            body.append(_DWSep(ci, co))
            ci = co
        self.features = nn.Sequential(*stem, *body)
        self.head = Head(ci)

    def forward(self, x):
        return self.head(self.features(x))


class MobileNet1D(nn.Module):
    """V1: a deeper depthwise/pointwise stack, some stages at stride 1."""

    def __init__(self, n_in=N_LEADS, widths=(16, 32, 32, 64, 64, 96)):
        super().__init__()
        stem = [nn.Conv1d(n_in, widths[0], 9, 2, 4, bias=False),
                *_bn_relu(widths[0])]
        body, ci = [], widths[0]
        for i, co in enumerate(widths[1:]):
            body.append(_DWSep(ci, co, stride=2 if i % 2 == 0 else 1))
            ci = co
        self.features = nn.Sequential(*stem, *body)
        self.head = Head(ci)

    def forward(self, x):
        return self.head(self.features(x))


class _MBConv(nn.Module):
    """Inverted residual. The expand ratio is the point of including it."""

    def __init__(self, ci, co, k=9, stride=2, expand=4):
        super().__init__()
        cm = ci * expand
        self.block = nn.Sequential(
            nn.Conv1d(ci, cm, 1, bias=False), *_bn_relu(cm),
            nn.Conv1d(cm, cm, k, stride, k // 2, groups=cm, bias=False),
            *_bn_relu(cm),
            nn.Conv1d(cm, co, 1, bias=False), nn.BatchNorm1d(co))
        self.res = (stride == 1 and ci == co)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.res else y


class MBConv1D(nn.Module):
    def __init__(self, n_in=N_LEADS, widths=(16, 24, 40, 64), expand=4):
        super().__init__()
        stem = [nn.Conv1d(n_in, widths[0], 9, 2, 4, bias=False),
                *_bn_relu(widths[0])]
        body, ci = [], widths[0]
        for co in widths[1:]:
            body += [_MBConv(ci, co, stride=2, expand=expand),
                     _MBConv(co, co, stride=1, expand=expand)]
            ci = co
        self.features = nn.Sequential(*stem, *body)
        self.head = Head(ci)

    def forward(self, x):
        return self.head(self.features(x))


class _TCNBlock(nn.Module):
    def __init__(self, ci, co, k=7, dilation=1):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(ci, co, k, 1, pad, dilation=dilation, bias=False),
            *_bn_relu(co))
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(self.block(x))


class TCNLite(nn.Module):
    """Dilated receptive field without attention or recurrence."""

    def __init__(self, n_in=N_LEADS, widths=(24, 32, 48, 64),
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        blocks, ci = [], n_in
        for co, d in zip(widths, dilations):
            blocks.append(_TCNBlock(ci, co, dilation=d))
            ci = co
        self.features, self.head = nn.Sequential(*blocks), Head(ci)

    def forward(self, x):
        return self.head(self.features(x))


ZOO = {"tiny": TinyCNN, "resnet": ResNet1DLite, "dscnn": DSCNN,
       "mobilenet": MobileNet1D, "mbconv": MBConv1D, "tcn": TCNLite}


DEFAULT_WIDTHS = {
    "tiny": (16, 32, 48, 64), "resnet": (32, 64, 128, 128),
    "dscnn": (32, 64, 64, 96), "mobilenet": (16, 32, 32, 64, 64, 96),
    "mbconv": (16, 24, 40, 64), "tcn": (24, 32, 48, 64),
}


def scaled_widths(name: str, alpha: float) -> tuple:
    """Channel widths scaled by alpha, rounded to a multiple of 4.

    One explicit rule applied to every architecture, so a scaling curve
    measures the architecture rather than six separate hand-tunings. The
    multiple of 4 keeps CMSIS-NN kernels on their fast path; a floor of 4
    stops the narrowest stage collapsing to a single channel.
    """
    return tuple(max(4, int(round(w * alpha / 4)) * 4)
                 for w in DEFAULT_WIDTHS[name])


def build(name: str, n_in: int = N_LEADS, n_classes: int = N_CLASSES,
          alpha: float = 1.0):
    kw = {} if alpha == 1.0 else {"widths": scaled_widths(name, alpha)}
    m = ZOO[name](n_in=n_in, **kw)
    if n_classes != N_CLASSES:
        m.head.fc = nn.Linear(m.head.fc.in_features, n_classes)
    return m


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def peak_activation_bytes(m: nn.Module, length: int, dtype_bytes: int = 1
                          ) -> int:
    """Largest single intermediate tensor, as an SRAM proxy.

    Not the allocator's true peak -- X-CUBE-AI reports that -- but it is
    computed from the graph alone, so it can be checked before any export
    and explains *why* a model does or does not fit.
    """
    peak = 0
    hooks = []

    def hook(_mod, _inp, out):
        nonlocal peak
        if torch.is_tensor(out):
            peak = max(peak, out.numel() * dtype_bytes)

    for mod in m.modules():
        if not list(mod.children()):
            hooks.append(mod.register_forward_hook(hook))
    m.eval()(torch.zeros(1, N_LEADS, length))
    for h in hooks:
        h.remove()
    return peak
