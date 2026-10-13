r"""Faithful 1-D reference implementations of the ten architecture families.

This module replaces the earlier adaptations, which deviated from their
sources in ways that turned out to matter:

  * depth was truncated to 3-4 blocks for most families while
    MobileNetV3 kept all eleven of its published blocks, so the
    benchmark held one family to a different standard than the rest;
  * TCN was non-causal, had no dropout, used BatchNorm instead of weight
    normalisation, and inserted two max-pools -- which contradicted its
    own docstring, since pooling grows the receptive field by stride
    rather than by dilation;
  * MobileNetV4's UIB put the stride on the final pointwise projection
    when there was no middle depthwise, where the reference rule puts it
    on the starting depthwise. A strided 1x1 subsamples without
    aggregating, which aliases.

Every family here keeps its published stage structure, channel widths,
block internals and stride schedule. The 2-D to 1-D adaptation replaces
k x k convolutions with k-tap 1-D convolutions and the square input with
12 x 1000; nothing else is changed.

Both scaling knobs are exposed on every family:

    w   width multiplier, applied to every channel count
    d   depth multiplier, applied to the repeat count of each stage,
        never below one block per published stage

so the uniform method is (w, d=1) and the joint method varies both from
the same definition.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_LEADS = 12
N_CLASSES = 13


def _divisible(v: float, divisor: int = 8) -> int:
    """MobileNet's `_make_divisible`, including the 10% guard.

    The earlier `_round4` rounded to multiples of four with a hard floor
    of four and no guard, so a layer could lose more than a tenth of its
    width to rounding alone. That is what drove MobileNetV3's realised
    multiplier to vary by 50% across its own depth.
    """
    new = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new < 0.9 * v:
        new += divisor
    return int(new)


def _depth(n: int, d: float) -> int:
    """Stage repeat count under depth multiplier `d`, never below one."""
    import math
    return max(1, int(math.ceil(n * d)))


def _stages(cfg, stride_at):
    """Group a flat per-block config into stages, breaking at a stride.

    Depth scaling needs a unit to repeat, and the published tables
    are flat lists of blocks, so a stage is taken to be a maximal run
    that does not downsample.
    """
    out = []
    for spec in cfg:
        if spec[stride_at] == 2 or not out:
            out.append([spec])
        else:
            out[-1].append(spec)
    return out


def _cna(ci, co, k=1, s=1, g=1, act=True, d=1):
    pad = ((k - 1) * d) // 2
    layers = [nn.Conv1d(ci, co, k, s, pad, dilation=d, groups=g, bias=False),
              nn.BatchNorm1d(co)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return layers


class _HardSwish(nn.Module):
    def forward(self, x):
        return x * torch.clamp(x + 3, 0, 6) / 6


class _HardSigmoid(nn.Module):
    def forward(self, x):
        return torch.clamp(x + 3, 0, 6) / 6


class _SE(nn.Module):
    """Squeeze-excitation. MobileNetV3/GhostNet use a hard sigmoid gate."""

    def __init__(self, c, ratio=0.25, hard=True):
        super().__init__()
        h = _divisible(c * ratio)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                nn.Conv1d(c, h, 1), nn.ReLU(inplace=True),
                                nn.Conv1d(h, c, 1),
                                _HardSigmoid() if hard else nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class _Classifier(nn.Module):
    def __init__(self, c, n=N_CLASSES):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c, n)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


# ----------------------------------------------------------- FCN, 2017
class FCN1D(nn.Module):
    """Wang et al. 2017: three convolution blocks, 128-256-128, k 8-5-3.

    Already fully convolutional and 1-D in the original, so this is the
    published network rather than an adaptation of one.
    """

    CH, K = (128, 256, 128), (8, 5, 3)

    def __init__(self, w=1.0, d=1.0):
        super().__init__()
        ci, L = N_LEADS, []
        for c, k in zip(self.CH, self.K):
            co = _divisible(c * w)
            L += _cna(ci, co, k)
            ci = co
        self.features = nn.Sequential(*L)
        self.head = _Classifier(ci)

    def forward(self, x):
        return self.head(self.features(x))


# -------------------------------------------------------- ResNet, 2016
class _BasicBlock(nn.Module):
    def __init__(self, ci, co, s=1, k=7):
        super().__init__()
        self.conv = nn.Sequential(*_cna(ci, co, k, s), *_cna(co, co, k,
                                                             act=False))
        self.down = (nn.Sequential(*_cna(ci, co, 1, s, act=False))
                     if (s != 1 or ci != co) else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.down(x))


class ResNet1D(nn.Module):
    """He et al. 2016, ResNet-18 layout: [2,2,2,2] at 64/128/256/512."""

    CH, N, S = (64, 128, 256, 512), (2, 2, 2, 2), (1, 2, 2, 2)

    def __init__(self, w=1.0, d=1.0):
        super().__init__()
        c0 = _divisible(64 * w)
        self.stem = nn.Sequential(*_cna(N_LEADS, c0, 15, 2),
                                  nn.MaxPool1d(3, 2, 1))
        L, ci = [], c0
        for c, n, s in zip(self.CH, self.N, self.S):
            co = _divisible(c * w)
            for i in range(_depth(n, d)):
                L.append(_BasicBlock(ci, co, s if i == 0 else 1))
                ci = co
        self.features = nn.Sequential(*L)
        self.head = _Classifier(ci)

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# ----------------------------------------------------------- TCN, 2018
class _Chomp(nn.Module):
    """Trim the right-hand padding so the convolution stays causal."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, :, :-self.n] if self.n > 0 else x


class _TemporalBlock(nn.Module):
    """Bai et al. 2018 Fig. 1b, reproduced in full.

    Two dilated *causal* convolutions, each with weight normalisation,
    ReLU and dropout, and a residual connection taking a 1x1 convolution
    when the channel counts differ. No pooling: the receptive field
    grows by dilation alone, which is the design's whole point.
    """

    def __init__(self, ci, co, k, dilation, dropout=0.2):
        super().__init__()
        pad = (k - 1) * dilation
        c1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(ci, co, k, padding=pad, dilation=dilation))
        c2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(co, co, k, padding=pad, dilation=dilation))
        self.net = nn.Sequential(
            c1, _Chomp(pad), nn.ReLU(inplace=True), nn.Dropout(dropout),
            c2, _Chomp(pad), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.down = nn.Conv1d(ci, co, 1) if ci != co else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + (x if self.down is None
                                       else self.down(x)))


class TCN(nn.Module):
    """Bai et al. 2018. Depth sets the receptive field, so it is the
    knob the architecture is defined by; eight levels at 64 channels is
    this benchmark's choice, the paper having no canonical setting."""

    LEVELS, CH, K = 8, 64, 3

    def __init__(self, w=1.0, d=1.0):
        super().__init__()
        c = _divisible(self.CH * w)
        L, ci = [], N_LEADS
        for i in range(_depth(self.LEVELS, d)):
            L.append(_TemporalBlock(ci, c, self.K, 2 ** i))
            ci = c
        self.features = nn.Sequential(*L)
        self.head = _Classifier(ci)

    def forward(self, x):
        return self.head(self.features(x))


# --------------------------------------------------- MobileNetV2, 2018
class _InvertedResidual(nn.Module):
    def __init__(self, ci, co, s, t, k=3):
        super().__init__()
        ce = _divisible(ci * t)
        L = []
        if t != 1:
            L += _cna(ci, ce, 1)
        L += _cna(ce, ce, k, s, g=ce)
        L += _cna(ce, co, 1, act=False)
        self.op = nn.Sequential(*L)
        self.res = s == 1 and ci == co

    def forward(self, x):
        return x + self.op(x) if self.res else self.op(x)


class MobileNetV2(nn.Module):
    """Sandler et al. 2018, Table 2: the (t, c, n, s) specification."""

    CFG = ((1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1))
    LAST = 1280

    def __init__(self, w=1.0, d=1.0):
        super().__init__()
        c0 = _divisible(32 * w)
        self.stem = nn.Sequential(*_cna(N_LEADS, c0, 3, 2))
        L, ci = [], c0
        for t, c, n, s in self.CFG:
            co = _divisible(c * w)
            for i in range(_depth(n, d)):
                L.append(_InvertedResidual(ci, co, s if i == 0 else 1, t))
                ci = co
        last = _divisible(self.LAST * max(w, 1.0))
        L += _cna(ci, last, 1)
        self.features = nn.Sequential(*L)
        self.head = _Classifier(last)

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# ------------------------------------------------- ShuffleNetV2, 2018
def _shuffle(x, groups=2):
    b, c, t = x.shape
    return x.view(b, groups, c // groups, t).transpose(1, 2).reshape(b, c, t)


class _ShuffleBlock(nn.Module):
    """Ma et al. 2018. Stride 1 splits the channels and leaves half
    untouched; stride 2 has no split and both branches downsample."""

    def __init__(self, ci, co, s, k=3):
        super().__init__()
        self.s = s
        br = co // 2
        # torch.split with explicit sizes, never chunk or a Python
        # slice: all three are the same tensor, but only split
        # survives code generation (see scripts/_sliceprobe.py).
        self.sizes = [ci // 2, ci - ci // 2]
        if s == 1:
            self.main = nn.Sequential(*_cna(ci // 2, br, 1),
                                      *_cna(br, br, k, 1, g=br, act=False),
                                      *_cna(br, br, 1))
            self.side = None
        else:
            self.side = nn.Sequential(*_cna(ci, ci, k, s, g=ci, act=False),
                                      *_cna(ci, br, 1))
            self.main = nn.Sequential(*_cna(ci, br, 1),
                                      *_cna(br, br, k, s, g=br, act=False),
                                      *_cna(br, br, 1))

    def forward(self, x):
        if self.s == 1:
            a, b = torch.split(x, self.sizes, dim=1)
            out = torch.cat([a, self.main(b)], 1)
        else:
            out = torch.cat([self.side(x), self.main(x)], 1)
        return _shuffle(out)


class ShuffleNetV2(nn.Module):
    """Ma et al. 2018, 1.0x: stages of 4, 8, 4 at 116/232/464."""

    CH, N, STEM, LAST = (116, 232, 464), (4, 8, 4), 24, 1024

    def __init__(self, w=1.0, d=1.0):
        super().__init__()
        c0 = _divisible(self.STEM * w)
        self.stem = nn.Sequential(*_cna(N_LEADS, c0, 3, 2),
                                  nn.MaxPool1d(3, 2, 1))
        L, ci = [], c0
        for c, n in zip(self.CH, self.N):
            co = _divisible(c * w)
            for i in range(_depth(n, d)):
                L.append(_ShuffleBlock(ci, co, 2 if i == 0 else 1))
                ci = co
        last = _divisible(self.LAST * max(w, 1.0))
        L += _cna(ci, last, 1)
        self.features = nn.Sequential(*L)
        self.head = _Classifier(last)

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# ------------------------------------------------------ GhostNet, 2020
class _Ghost(nn.Module):
    """Han et al. 2020: a primary convolution plus cheap linear maps."""

    def __init__(self, ci, co, k=1, dw=3, ratio=2, act=True):
        super().__init__()
        import math
        prim = math.ceil(co / ratio)
        # Han et al. generate prim*(ratio-1) cheap maps and keep the
        # first co. Every channel count here is a multiple of eight and
        # ratio is two, so the discarded tail is empty; asking for the
        # kept width directly avoids emitting a trailing slice, which
        # the code generator cannot compile.
        self.prim = nn.Sequential(*_cna(ci, prim, k, act=act))
        self.cheap = nn.Sequential(*_cna(prim, co - prim, dw, g=prim,
                                         act=act))

    def forward(self, x):
        y = self.prim(x)
        return torch.cat([y, self.cheap(y)], 1)


class _GhostBottleneck(nn.Module):
    def __init__(self, ci, ce, co, k, s, se):
        super().__init__()
        L = [_Ghost(ci, ce, act=True)]
        if s == 2:
            L += _cna(ce, ce, k, s, g=ce, act=False)
        if se:
            L.append(_SE(ce, se))
        L.append(_Ghost(ce, co, act=False))
        self.op = nn.Sequential(*L)
        self.short = (nn.Identity() if s == 1 and ci == co else
                      nn.Sequential(*_cna(ci, ci, k, s, g=ci, act=False),
                                    *_cna(ci, co, 1, act=False)))

    def forward(self, x):
        return self.op(x) + self.short(x)


class GhostNet(nn.Module):
    """Han et al. 2020, 1.0x: the sixteen-block configuration."""

    CFG = ((3, 16, 16, 0, 1), (3, 48, 24, 0, 2), (3, 72, 24, 0, 1),
           (5, 72, 40, 0.25, 2), (5, 120, 40, 0.25, 1),
           (3, 240, 80, 0, 2), (3, 200, 80, 0, 1), (3, 184, 80, 0, 1),
           (3, 184, 80, 0, 1), (3, 480, 112, 0.25, 1),
           (3, 672, 112, 0.25, 1), (5, 672, 160, 0.25, 2),
           (5, 960, 160, 0, 1), (5, 960, 160, 0.25, 1),
           (5, 960, 160, 0, 1), (5, 960, 160, 0.25, 1))
    LAST = 960

    def __init__(self, w=1.0, d=1.0):
        super().__init__()
        c0 = _divisible(16 * w)
        self.stem = nn.Sequential(*_cna(N_LEADS, c0, 3, 2))
        L, ci = [], c0
        for st in _stages(self.CFG, stride_at=4):
            for i in range(_depth(len(st), d)):
                k, ce, c, se, s = st[min(i, len(st) - 1)]
                co = _divisible(c * w)
                L.append(_GhostBottleneck(ci, _divisible(ce * w), co, k,
                                          s if i == 0 else 1, se))
                ci = co
        last = _divisible(self.LAST * max(w, 1.0))
        L += _cna(ci, last, 1)
        self.features = nn.Sequential(*L)
        self.head = _Classifier(last)

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# -------------------------------------------------- MobileNetV3, 2019
class _V3Bneck(nn.Module):
    def __init__(self, ci, ce, co, k, s, se, hs):
        super().__init__()
        act = _HardSwish if hs else (lambda: nn.ReLU(inplace=True))
        L = []
        if ce != ci:
            L += [nn.Conv1d(ci, ce, 1, bias=False), nn.BatchNorm1d(ce), act()]
        L += [nn.Conv1d(ce, ce, k, s, (k - 1) // 2, groups=ce, bias=False),
              nn.BatchNorm1d(ce), act()]
        if se:
            L.append(_SE(ce))
        L += [nn.Conv1d(ce, co, 1, bias=False), nn.BatchNorm1d(co)]
        self.op = nn.Sequential(*L)
        self.res = s == 1 and ci == co

    def forward(self, x):
        return x + self.op(x) if self.res else self.op(x)


class MobileNetV3(nn.Module):
    """Howard et al. 2019, Table 2 (Small): all eleven bottlenecks.

    The expansion channel is scaled by the width multiplier alongside the
    output channel, as in the reference implementation -- holding it
    fixed would change the expansion ratio at every scale.
    """

    #    k  exp   c  se  hs  s
    CFG = ((3, 16, 16, 1, 0, 2), (3, 72, 24, 0, 0, 2), (3, 88, 24, 0, 0, 1),
           (5, 96, 40, 1, 1, 2), (5, 240, 40, 1, 1, 1), (5, 240, 40, 1, 1, 1),
           (5, 120, 48, 1, 1, 1), (5, 144, 48, 1, 1, 1),
           (5, 288, 96, 1, 1, 2), (5, 576, 96, 1, 1, 1),
           (5, 576, 96, 1, 1, 1))
    LAST = 576

    def __init__(self, w=1.0, d=1.0):
        super().__init__()
        c0 = _divisible(16 * w)
        self.stem = nn.Sequential(nn.Conv1d(N_LEADS, c0, 3, 2, 1, bias=False),
                                  nn.BatchNorm1d(c0), _HardSwish())
        L, ci = [], c0
        for st in _stages(self.CFG, stride_at=5):
            for i in range(_depth(len(st), d)):
                k, ce, c, se, hs, s = st[min(i, len(st) - 1)]
                co = _divisible(c * w)
                L.append(_V3Bneck(ci, _divisible(ce * w), co, k,
                                  s if i == 0 else 1, se, hs))
                ci = co
        last = _divisible(self.LAST * w)
        L += [nn.Conv1d(ci, last, 1, bias=False), nn.BatchNorm1d(last),
              _HardSwish()]
        self.features = nn.Sequential(*L)
        self.head = _Classifier(last)

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# ------------------------------------------------- FasterNet-T0, 2023
class _PConv(nn.Module):
    """Chen et al. 2023, partial convolution.

    Only the first c/n_div channels are convolved; the remainder pass
    through untouched. The published ratio is n_div = 4.
    """

    def __init__(self, c, k=3, n_div=4):
        super().__init__()
        self.cp = c // n_div
        self.sizes = [self.cp, c - self.cp]
        self.conv = nn.Conv1d(self.cp, self.cp, k, 1, (k - 1) // 2,
                              bias=False)

    def forward(self, x):
        a, b = torch.split(x, self.sizes, dim=1)
        return torch.cat([self.conv(a), b], 1)


class _FasterBlock(nn.Module):
    def __init__(self, c, mlp=2.0, n_div=4):
        super().__init__()
        h = int(c * mlp)
        self.op = nn.Sequential(_PConv(c, 3, n_div),
                                nn.Conv1d(c, h, 1, bias=False),
                                nn.BatchNorm1d(h), nn.ReLU(inplace=True),
                                nn.Conv1d(h, c, 1, bias=False))

    def forward(self, x):
        return x + self.op(x)


class FasterNet(nn.Module):
    """Chen et al. 2023, T0: embed 40, depths (1, 2, 8, 2), MLP ratio 2.

    Patch embedding is a stride-4 convolution and each stage transition
    is a stride-2 merging convolution that doubles the width, so the
    channel schedule 40-80-160-320 follows from the embedding dimension
    rather than being tabulated separately.
    """

    DIM, N, MLP, NDIV, LAST = 40, (1, 2, 8, 2), 2.0, 4, 1280

    def __init__(self, w=1.0, d=1.0):
        super().__init__()
        c0 = _divisible(self.DIM * w)
        self.stem = nn.Sequential(nn.Conv1d(N_LEADS, c0, 4, 4, bias=False),
                                  nn.BatchNorm1d(c0))
        L, ci = [], c0
        for si, n in enumerate(self.N):
            if si:
                co = _divisible(self.DIM * (2 ** si) * w)
                L += [nn.Conv1d(ci, co, 2, 2, bias=False),
                      nn.BatchNorm1d(co)]
                ci = co
            for _ in range(_depth(n, d)):
                L.append(_FasterBlock(ci, self.MLP, self.NDIV))
        last = _divisible(self.LAST * max(w, 1.0))
        L += _cna(ci, last, 1)
        self.features = nn.Sequential(*L)
        self.head = _Classifier(last)

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# --------------------------------------------------- RepViT-M0.9, 2024
class _RepViTBlock(nn.Module):
    """Wang et al. 2024, in its deployed (reparameterised) form.

    Training-time RepViT has a multi-branch depthwise token mixer that
    folds into a single depthwise convolution for inference. Every model
    here is quantised and run on the device, so the fused form is the
    one that matters and is what gets built.
    """

    def __init__(self, ci, co, s, se, k=3, mlp=2.0):
        super().__init__()
        mix = [nn.Conv1d(ci, ci, k, s, (k - 1) // 2, groups=ci, bias=False),
               nn.BatchNorm1d(ci)]
        if se:
            mix.append(_SE(ci, hard=False))
        self.mix_res = s == 1 and ci == co
        if not self.mix_res:
            mix += [nn.Conv1d(ci, co, 1, bias=False), nn.BatchNorm1d(co)]
        self.mixer = nn.Sequential(*mix)
        h = int(co * mlp)
        self.ffn = nn.Sequential(nn.Conv1d(co, h, 1, bias=False),
                                 nn.BatchNorm1d(h), nn.ReLU(inplace=True),
                                 nn.Conv1d(h, co, 1, bias=False),
                                 nn.BatchNorm1d(co))

    def forward(self, x):
        x = x + self.mixer(x) if self.mix_res else self.mixer(x)
        return x + self.ffn(x)


class RepViT(nn.Module):
    """Wang et al. 2024, M0.9: depths (2, 2, 14, 2) at 48/96/192/384.

    Squeeze-excitation alternates between blocks within a stage, as in
    the released configuration, rather than appearing in all of them.
    """

    CH, N = (48, 96, 192, 384), (2, 2, 14, 2)

    def __init__(self, w=1.0, d=1.0):
        super().__init__()
        c0 = _divisible(self.CH[0] * w)
        self.stem = nn.Sequential(*_cna(N_LEADS, max(8, c0 // 2), 3, 2),
                                  *_cna(max(8, c0 // 2), c0, 3, 2))
        L, ci = [], c0
        for si, (c, n) in enumerate(zip(self.CH, self.N)):
            co = _divisible(c * w)
            for i in range(_depth(n, d)):
                down = si > 0 and i == 0
                L.append(_RepViTBlock(ci, co, 2 if down else 1, i % 2 == 0))
                ci = co
        self.features = nn.Sequential(*L)
        self.head = _Classifier(ci)

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# ------------------------------------------- MobileNetV4-Conv-S, 2024
class _UIB(nn.Module):
    """Qin et al. 2024, universal inverted bottleneck.

    The stride belongs to the starting depthwise when there is one and
    no middle depthwise, otherwise to the middle depthwise. The final
    pointwise projection is never strided: a strided 1x1 subsamples
    without aggregating, which aliases.
    """

    def __init__(self, ci, co, s, e, ka, km):
        super().__init__()
        ce = _divisible(ci * e)
        L = []
        if ka:
            L += _cna(ci, ci, ka, s if not km else 1, g=ci, act=False)
        L += _cna(ci, ce, 1)
        if km:
            L += _cna(ce, ce, km, s, g=ce)
        L += _cna(ce, co, 1, act=False)
        self.op = nn.Sequential(*L)
        self.res = s == 1 and ci == co

    def forward(self, x):
        return x + self.op(x) if self.res else self.op(x)


class MobileNetV4(nn.Module):
    """Qin et al. 2024, Conv-Small: four fused convolutions then twelve
    UIB blocks, as given by the released architecture definition.

    Each UIB entry is (start dw kernel, mid dw kernel, stride, expand,
    channels, repeats); a kernel of zero means that depthwise is absent,
    which is how the four UIB variants are selected.
    """

    STEM = 32
    #        k  s   c
    FUSED = ((3, 2, 32), (1, 1, 32), (3, 2, 96), (1, 1, 64))
    #      ka km  s    e    c  n
    UIB = ((5, 5, 2, 3.0, 96, 1), (0, 3, 1, 2.0, 96, 4),
           (3, 0, 1, 4.0, 96, 1), (3, 3, 2, 6.0, 128, 1),
           (5, 5, 1, 4.0, 128, 1), (0, 5, 1, 4.0, 128, 1),
           (0, 5, 1, 3.0, 128, 1), (0, 3, 1, 4.0, 128, 2))
    LAST = 960

    def __init__(self, w=1.0, d=1.0):
        super().__init__()
        c0 = _divisible(self.STEM * w)
        self.stem = nn.Sequential(*_cna(N_LEADS, c0, 3, 2))
        L, ci = [], c0
        for k, s, c in self.FUSED:
            co = _divisible(c * w)
            L += _cna(ci, co, k, s)
            ci = co
        for st in _stages(self.UIB, stride_at=2):
            flat = [b for b in st for _ in range(b[5])]
            for i in range(_depth(len(flat), d)):
                ka, km, s, e, c, _ = flat[min(i, len(flat) - 1)]
                co = _divisible(c * w)
                L.append(_UIB(ci, co, s if i == 0 else 1, e, ka, km))
                ci = co
        last = _divisible(self.LAST * max(w, 1.0))
        L += _cna(ci, last, 1)
        self.features = nn.Sequential(*L)
        self.head = _Classifier(last)

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# ------------------------------------------------------------ registry
REFERENCE = {"FCN-1D": FCN1D, "ResNet-1D": ResNet1D, "TCN": TCN,
             "MobileNetV2": MobileNetV2, "ShuffleNetV2": ShuffleNetV2,
             "GhostNet": GhostNet, "MobileNetV3": MobileNetV3,
             "FasterNet": FasterNet, "RepViT": RepViT,
             "MobileNetV4": MobileNetV4}

#: published block count per family, for the depth-fidelity check
PUBLISHED_DEPTH = {"FCN-1D": 3, "ResNet-1D": 8, "TCN": 8, "MobileNetV2": 17,
                   "ShuffleNetV2": 16, "GhostNet": 16, "MobileNetV3": 11,
                   "FasterNet": 13, "RepViT": 20, "MobileNetV4": 12}


def build(name, w=1.0, d=1.0):
    return REFERENCE[name](w=w, d=d)


def fuse(model: nn.Module) -> nn.Module:
    """Fold any weight-norm parametrisation into a plain weight tensor.

    TCN trains with weight normalisation, which stores the weight as a
    magnitude and a direction. The device sees neither: the exported
    graph must carry the product, and leaving the parametrisation in
    place would also put two tensors in the checkpoint where the
    deployment path expects one.
    """
    from torch.nn.utils import parametrize
    for m in model.modules():
        if parametrize.is_parametrized(m):
            for name in list(m.parametrizations):
                parametrize.remove_parametrizations(m, name,
                                                    leave_parametrized=True)
    return model
