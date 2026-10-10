r"""Efficient 1-D encoder families, as they would actually reach an MCU.

Ten architecture families spanning 2016-2024, each reduced to 1-D for a
12 x 1000 input and a 13-way multi-label head. Every family keeps the
block that defines it. That constraint is the point of the study: a
"MobileNetV4" whose Universal Inverted Bottleneck has been quietly
replaced by an ordinary depthwise stack is not MobileNetV4, and a
benchmark built that way measures nothing but the author's channel
budget.

    fcn           Wang 2017       plain wide convolutions, the TSC baseline
    resnet        He 2016         residual learning
    tcn           Bai 2018        dilated temporal convolution
    mobilenetv2   Sandler 2018    inverted residual bottleneck
    shufflenetv2  Ma 2018         channel split and shuffle
    ghostnet      Han 2020        cheap feature generation
    mobilenetv3   Howard 2019     inverted residual, SE, h-swish
    fasternet     Chen 2023       partial convolution
    repvit        Wang 2024       reparameterised token/channel mixing
    mobilenetv4   Qin 2024        universal inverted bottleneck

What varies between them is how they spend two separate budgets that an
MCU enforces independently: parameters, which land in Flash, and the
largest intermediate tensor, which must fit in SRAM alongside everything
else the firmware needs. A design can be cheap in one and ruinous in the
other -- inverted residuals are the classic case, inflating activations
by their expansion factor while looking small by parameter count -- and
only measuring both separates "mobile-efficient" from "MCU-deployable".

Every family shares the same contract: (B, 12, T) in, (B, 13) out,
global average pooling, one linear layer. Width is the only free knob,
and `fit_to_budget` sets it per family so each one is given the largest
version of itself that the device will accept.
"""

from __future__ import annotations

import torch
from torch import nn

N_LEADS, N_CLASSES = 12, 13
STEM_STRIDE = 4          # a 12 x 1000 int8 input is already 11.7 KB


def _bn_act(c: int) -> list[nn.Module]:
    return [nn.BatchNorm1d(c), nn.ReLU(inplace=True)]


def _round4(v: float) -> int:
    """Channel counts a CMSIS-NN kernel can use without ragged tails."""
    return max(4, int(round(v / 4.0)) * 4)


class HardSigmoid(nn.Module):
    """ReLU6(x + 3) / 6, the piecewise-linear sigmoid of Howard 2019."""

    def forward(self, x):
        return torch.nn.functional.relu6(x + 3.0) / 6.0


class HardSwish(nn.Module):
    """x * ReLU6(x + 3) / 6. Cheaper than swish and quantises cleanly."""

    def forward(self, x):
        return x * torch.nn.functional.relu6(x + 3.0) / 6.0


class SE(nn.Module):
    """Squeeze-excitation, as GhostNet and RepViT both use it.

    Cheap by the usual accounting -- a global pool and two 1x1 layers add
    well under 1% of MACs -- and not cheap at all on this part. It is
    +35% parameters for GhostNet and +15% for RepViT, which under a fixed
    Flash budget buys a narrower network. It also keeps the block input
    alive until the rescale, so it costs concurrent liveness that a
    largest-single-tensor proxy scores as free. Omitting it, as an
    earlier version of this zoo did, flatters both families on exactly
    the two axes this study measures.
    """

    def __init__(self, c: int, r: int = 4, hard: bool = False):
        super().__init__()
        h = max(4, _round4(c // r))
        gate = HardSigmoid() if hard else nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(c, h, 1),
            nn.ReLU(inplace=True), nn.Conv1d(h, c, 1), gate)

    def forward(self, x):
        return x * self.fc(x)


class Head(nn.Module):
    def __init__(self, c: int, n_classes: int = N_CLASSES):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c, n_classes)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


class Stem(nn.Module):
    """Shared entry: one strided convolution.

    Identical for every family so the comparison starts from the same
    activation footprint. Without it the early layers dominate SRAM and
    the benchmark would mostly rank stem designs.
    """

    def __init__(self, c_out: int, k: int = 9, stride: int = STEM_STRIDE):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv1d(N_LEADS, c_out, k, stride=stride, padding=k // 2,
                      bias=False), *_bn_act(c_out))

    def forward(self, x):
        return self.op(x)


# --------------------------------------------------------------- FCN 2017
class FCN1D(nn.Module):
    """Wang et al. 2017. Three wide plain convolutions, no bottleneck."""

    def __init__(self, w=1.0):
        super().__init__()
        c = [_round4(32 * w), _round4(64 * w), _round4(32 * w)]
        self.stem = Stem(c[0])
        layers = []
        ci = c[0]
        for co, k, s in zip(c, (8, 5, 3), (2, 2, 2)):
            layers += [nn.Conv1d(ci, co, k, stride=s, padding=k // 2,
                                 bias=False), *_bn_act(co)]
            ci = co
        self.features, self.head = nn.Sequential(*layers), Head(ci)

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# ------------------------------------------------------------ ResNet 2016
class _ResBlock(nn.Module):
    def __init__(self, ci, co, k=7, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ci, co, k, stride=stride, padding=k // 2, bias=False),
            *_bn_act(co),
            nn.Conv1d(co, co, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(co))
        self.skip = (nn.Identity() if stride == 1 and ci == co else
                     nn.Sequential(nn.Conv1d(ci, co, 1, stride=stride,
                                             bias=False),
                                   nn.BatchNorm1d(co)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self, w=1.0):
        super().__init__()
        c = [_round4(v * w) for v in (24, 40, 64)]
        self.stem = Stem(c[0])
        self.features = nn.Sequential(
            _ResBlock(c[0], c[0]),
            _ResBlock(c[0], c[1], stride=2),
            _ResBlock(c[1], c[2], stride=2))
        self.head = Head(c[2])

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# --------------------------------------------------------------- TCN 2018
class _TCNBlock(nn.Module):
    def __init__(self, ci, co, k=5, dilation=1):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(ci, co, k, padding=pad, dilation=dilation,
                      bias=False), *_bn_act(co),
            nn.Conv1d(co, co, k, padding=pad, dilation=dilation,
                      bias=False), nn.BatchNorm1d(co))
        self.skip = (nn.Identity() if ci == co else
                     nn.Conv1d(ci, co, 1, bias=False))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class TCN(nn.Module):
    """Bai et al. 2018. Receptive field grows by dilation, not by stride."""

    def __init__(self, w=1.0):
        super().__init__()
        c = _round4(32 * w)
        self.stem = Stem(c)
        self.features = nn.Sequential(
            _TCNBlock(c, c, dilation=1), nn.MaxPool1d(2),
            _TCNBlock(c, c, dilation=2), nn.MaxPool1d(2),
            _TCNBlock(c, c, dilation=4))
        self.head = Head(c)

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# ------------------------------------------------------- MobileNetV2 2018
class _InvertedResidual(nn.Module):
    """Expand, depthwise, project. The expansion is the SRAM hazard."""

    def __init__(self, ci, co, stride=1, expand=4, k=5):
        super().__init__()
        ce = _round4(ci * expand)
        self.use_res = stride == 1 and ci == co
        self.op = nn.Sequential(
            nn.Conv1d(ci, ce, 1, bias=False), *_bn_act(ce),
            nn.Conv1d(ce, ce, k, stride=stride, padding=k // 2,
                      groups=ce, bias=False), *_bn_act(ce),
            nn.Conv1d(ce, co, 1, bias=False), nn.BatchNorm1d(co))

    def forward(self, x):
        return x + self.op(x) if self.use_res else self.op(x)


class MobileNetV2(nn.Module):
    def __init__(self, w=1.0, expand=4):
        super().__init__()
        c = [_round4(v * w) for v in (24, 32, 48, 64)]
        self.stem = Stem(c[0])
        self.features = nn.Sequential(
            _InvertedResidual(c[0], c[1], 2, expand),
            _InvertedResidual(c[1], c[1], 1, expand),
            _InvertedResidual(c[1], c[2], 2, expand),
            _InvertedResidual(c[2], c[3], 2, expand))
        self.head = Head(c[3])

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# ------------------------------------------------------ ShuffleNetV2 2018
def _channel_shuffle(x, groups=2):
    b, c, t = x.shape
    return (x.view(b, groups, c // groups, t)
             .transpose(1, 2).reshape(b, c, t))


class _ShuffleUnit(nn.Module):
    """Ma et al. 2018. Split the channels, convolve half, shuffle.

    Half the tensor passes through untouched, so the branch carries only
    c/2 channels -- the memory-access argument the paper makes, which is
    exactly the property an MCU should reward.
    """

    def __init__(self, c, k=5):
        super().__init__()
        h = c // 2
        self.branch = nn.Sequential(
            nn.Conv1d(h, h, 1, bias=False), *_bn_act(h),
            nn.Conv1d(h, h, k, padding=k // 2, groups=h, bias=False),
            nn.BatchNorm1d(h),
            nn.Conv1d(h, h, 1, bias=False), *_bn_act(h))

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return _channel_shuffle(torch.cat([a, self.branch(b)], dim=1))


class _ShuffleDown(nn.Module):
    def __init__(self, ci, co, k=5):
        super().__init__()
        h = co // 2
        self.left = nn.Sequential(
            nn.Conv1d(ci, ci, k, stride=2, padding=k // 2, groups=ci,
                      bias=False), nn.BatchNorm1d(ci),
            nn.Conv1d(ci, h, 1, bias=False), *_bn_act(h))
        self.right = nn.Sequential(
            nn.Conv1d(ci, h, 1, bias=False), *_bn_act(h),
            nn.Conv1d(h, h, k, stride=2, padding=k // 2, groups=h,
                      bias=False), nn.BatchNorm1d(h),
            nn.Conv1d(h, h, 1, bias=False), *_bn_act(h))

    def forward(self, x):
        return _channel_shuffle(torch.cat([self.left(x), self.right(x)], 1))


class ShuffleNetV2(nn.Module):
    def __init__(self, w=1.0):
        super().__init__()
        c = [_round4(v * w) for v in (24, 48, 64)]
        self.stem = Stem(c[0])
        self.features = nn.Sequential(
            _ShuffleDown(c[0], c[1]), _ShuffleUnit(c[1]),
            _ShuffleDown(c[1], c[2]), _ShuffleUnit(c[2]))
        self.head = Head(c[2])

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# ---------------------------------------------------------- GhostNet 2020
class _GhostModule(nn.Module):
    """Han et al. 2020. Half the maps are convolved, half are made cheap.

    The intrinsic half comes from a pointwise convolution; the rest are
    produced by a depthwise operation on those, then concatenated. Fewer
    MACs for the same channel count -- a claim worth testing against a
    measured cycle count rather than accepting.
    """

    def __init__(self, ci, co, k=5, ratio=2):
        super().__init__()
        init = co // ratio
        self.primary = nn.Sequential(
            nn.Conv1d(ci, init, 1, bias=False), *_bn_act(init))
        self.cheap = nn.Sequential(
            nn.Conv1d(init, co - init, k, padding=k // 2, groups=init,
                      bias=False), *_bn_act(co - init))

    def forward(self, x):
        y = self.primary(x)
        return torch.cat([y, self.cheap(y)], dim=1)


class _GhostBottleneck(nn.Module):
    def __init__(self, ci, co, stride=1, k=5, use_se=True):
        super().__init__()
        mid = _round4(ci * 2)
        ops = [_GhostModule(ci, mid)]
        if stride > 1:
            ops += [nn.Conv1d(mid, mid, k, stride=stride, padding=k // 2,
                              groups=mid, bias=False), nn.BatchNorm1d(mid)]
        if use_se:
            ops += [SE(mid)]
        ops += [_GhostModule(mid, co)]
        self.op = nn.Sequential(*ops)
        self.use_res = stride == 1 and ci == co

    def forward(self, x):
        return x + self.op(x) if self.use_res else self.op(x)


class GhostNet(nn.Module):
    """SE follows the published pattern, not a blanket application.

    Han et al. apply SE at ratio 0.25 to the wider stages (40, 112, 160
    channels) and omit it from the narrow early ones (16, 24) and from
    the 80-channel stage. Compressing sixteen bottlenecks into three,
    the faithful reading is to skip SE on the first and narrowest block
    and keep it on the two wider ones -- not to attach it everywhere,
    which would overstate its cost by half.
    """

    def __init__(self, w=1.0):
        super().__init__()
        c = [_round4(v * w) for v in (24, 40, 64)]
        self.stem = Stem(c[0])
        self.features = nn.Sequential(
            _GhostBottleneck(c[0], c[1], 2, use_se=False),
            _GhostBottleneck(c[1], c[1], 1, use_se=True),
            _GhostBottleneck(c[1], c[2], 2, use_se=True))
        self.head = Head(c[2])

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# ------------------------------------------------------------ MCUNet 2020
# --------------------------------------------------- MobileNetV3 2019
# Table 2 of Howard et al. 2019, verbatim: (kernel, exp size, out, SE,
# nonlinearity, stride). Expansion is an absolute channel count per
# block, not a ratio -- that is a defining property of the design and is
# preserved here, scaled by width rather than replaced by a ratio.
_V3_SMALL = [
    (3, 16, 16, True, "RE", 2),
    (3, 72, 24, False, "RE", 2),
    (3, 88, 24, False, "RE", 1),
    (5, 96, 40, True, "HS", 2),
    (5, 240, 40, True, "HS", 1),
    (5, 240, 40, True, "HS", 1),
    (5, 120, 48, True, "HS", 1),
    (5, 144, 48, True, "HS", 1),
    (5, 288, 96, True, "HS", 2),
    (5, 576, 96, True, "HS", 1),
    (5, 576, 96, True, "HS", 1),
]


class _V3Bneck(nn.Module):
    """MobileNetV3 bottleneck: expand, depthwise, SE, project.

    The squeeze-excitation sits *after the depthwise inside the
    expansion*, as the paper specifies, so it attends to the largest
    representation in the block -- and therefore to the tensor that
    dominates peak SRAM.
    """

    def __init__(self, ci, ce, co, k, se, nl, stride):
        super().__init__()
        act = HardSwish if nl == "HS" else nn.ReLU
        self.use_res = stride == 1 and ci == co
        ops = []
        if ce != ci:
            ops += [nn.Conv1d(ci, ce, 1, bias=False),
                    nn.BatchNorm1d(ce), act()]
        ops += [nn.Conv1d(ce, ce, k, stride=stride, padding=k // 2,
                          groups=ce, bias=False), nn.BatchNorm1d(ce), act()]
        if se:
            ops += [SE(ce, r=4, hard=True)]
        ops += [nn.Conv1d(ce, co, 1, bias=False), nn.BatchNorm1d(co)]
        self.op = nn.Sequential(*ops)

    def forward(self, x):
        return x + self.op(x) if self.use_res else self.op(x)


class MobileNetV3(nn.Module):
    """Howard et al. 2019, ICCV. The published small configuration.

    Replaces an earlier hand-specified MCU configuration: a published
    architecture with a reference implementation is a better benchmark
    entry than a setting chosen by the author. It also fills the 2019
    slot between MobileNetV2 and GhostNet.

    Kept from the paper: the eleven-block table with its per-block
    absolute expansion sizes, the 3/5 kernel schedule, which blocks carry
    SE, and the h-swish / ReLU assignment. Changed: 1-D rather than 2-D,
    the shared benchmark stem in place of the paper's own, and width
    scaling to the budget -- the same three deviations every family here
    carries.
    """

    def __init__(self, w=1.0):
        super().__init__()
        c0 = _round4(16 * w)
        self.stem = Stem(c0)
        blocks, ci = [], c0
        for k, ce, co, se, nl, st in _V3_SMALL:
            ce_, co_ = _round4(ce * w), _round4(co * w)
            blocks.append(_V3Bneck(ci, ce_, co_, k, se, nl, st))
            ci = co_
        last = _round4(576 * w)
        blocks += [nn.Conv1d(ci, last, 1, bias=False),
                   nn.BatchNorm1d(last), HardSwish()]
        self.features = nn.Sequential(*blocks)
        self.head = Head(last)

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# --------------------------------------------------------- FasterNet 2023
class _PConv(nn.Module):
    """Chen et al. 2023. Convolve a contiguous slice, pass the rest.

    Only 1/n of the channels are touched, so both MACs and memory traffic
    fall. Whether fewer memory accesses actually shorten Cortex-M4
    inference is an open question on this part, and is one of the things
    this benchmark is for.
    """

    def __init__(self, c, k=5, n_div=4):
        super().__init__()
        self.c_conv = c // n_div
        self.c_pass = c - self.c_conv
        self.conv = nn.Conv1d(self.c_conv, self.c_conv, k,
                              padding=k // 2, bias=False)

    def forward(self, x):
        a, b = torch.split(x, [self.c_conv, self.c_pass], dim=1)
        return torch.cat([self.conv(a), b], dim=1)


class _FasterBlock(nn.Module):
    def __init__(self, c, expand=2, n_div=4):
        super().__init__()
        ce = _round4(c * expand)
        self.pconv = _PConv(c, n_div=n_div)
        self.mlp = nn.Sequential(
            nn.Conv1d(c, ce, 1, bias=False), *_bn_act(ce),
            nn.Conv1d(ce, c, 1, bias=False))

    def forward(self, x):
        # Chen et al. place the shortcut across the whole block:
        # out = x + MLP(PConv(x)). An earlier version added the MLP to
        # the PConv output instead, which is close -- PConv passes most
        # channels through -- but is not the published block.
        return x + self.mlp(self.pconv(x))


class FasterNet(nn.Module):
    """Benchmarked at the published partial ratio only.

    n_div remains a constructor argument so the block stays general, but
    the zoo exposes a single FasterNet entry at n_div=4 as Chen et al.
    specify. Reporting a second ratio alongside it would put a tuned
    variant of one family next to the published form of nine others.
    """

    def __init__(self, w=1.0, n_div=4):
        super().__init__()
        c = [_round4(v * w) for v in (24, 40, 64)]
        self.stem = Stem(c[0])
        self.features = nn.Sequential(
            _FasterBlock(c[0], n_div=n_div),
            nn.Conv1d(c[0], c[1], 3, stride=2, padding=1, bias=False),
            *_bn_act(c[1]), _FasterBlock(c[1], n_div=n_div),
            nn.Conv1d(c[1], c[2], 3, stride=2, padding=1, bias=False),
            *_bn_act(c[2]), _FasterBlock(c[2], n_div=n_div))
        self.head = Head(c[2])

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# ------------------------------------------------------------ RepViT 2024
class _RepDW(nn.Module):
    """Multi-branch depthwise at training time, one kernel at inference.

    Structural reparameterisation: a k-wide depthwise, a 1-wide
    depthwise and an identity are summed while training, then folded
    into a single k-wide kernel for deployment. The deployed graph is
    therefore cheaper than the trained one, which is the property that
    has to survive export to mean anything here.
    """

    def __init__(self, c, k=5):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(c, c, k, padding=k // 2, groups=c, bias=False)
        self.conv1 = nn.Conv1d(c, c, 1, groups=c, bias=False)
        self.bn = nn.BatchNorm1d(c)
        self.deployed = False

    def forward(self, x):
        if self.deployed:
            return self.bn(self.conv(x))
        return self.bn(self.conv(x) + self.conv1(x) + x)

    @torch.no_grad()
    def reparameterise(self):
        """Fold the 1-wide branch and the identity into the k-wide kernel."""
        if self.deployed:
            return
        c, k = self.conv.weight.shape[0], self.k
        pad = k // 2
        w = self.conv.weight.clone()
        w[:, :, pad:pad + 1] += self.conv1.weight
        ident = torch.zeros_like(w)
        ident[:, :, pad] = 1.0
        self.conv.weight.copy_(w + ident)
        # The folded branch must actually leave the module, or the
        # deployed parameter count -- which is the Flash budget -- would
        # still be charged for weights no longer on the forward path.
        self.conv1 = nn.Identity()
        self.deployed = True


class _RepViTBlock(nn.Module):
    def __init__(self, c, expand=2, use_se=True):
        super().__init__()
        ce = _round4(c * expand)
        self.token = _RepDW(c)
        self.se = SE(c) if use_se else nn.Identity()
        self.channel = nn.Sequential(
            nn.Conv1d(c, ce, 1, bias=False), *_bn_act(ce),
            nn.Conv1d(ce, c, 1, bias=False), nn.BatchNorm1d(c))

    def forward(self, x):
        x = x + self.token(x)
        x = self.se(x)
        return x + self.channel(x)


class RepViT(nn.Module):
    """SE in every block, which is what the published rule gives here.

    Wang et al. adopt SE "in the 1st, 3rd, 5th ... block in each stage".
    This topology carries a single block per stage, so every block is
    the first of its stage and every block takes SE. The rule is applied
    rather than the count copied.
    """

    def __init__(self, w=1.0):
        super().__init__()
        c = [_round4(v * w) for v in (24, 40, 64)]
        self.stem = Stem(c[0])
        self.features = nn.Sequential(
            _RepViTBlock(c[0]),
            nn.Conv1d(c[0], c[1], 3, stride=2, padding=1, bias=False),
            *_bn_act(c[1]), _RepViTBlock(c[1]),
            nn.Conv1d(c[1], c[2], 3, stride=2, padding=1, bias=False),
            *_bn_act(c[2]), _RepViTBlock(c[2]))
        self.head = Head(c[2])

    def forward(self, x):
        return self.head(self.features(self.stem(x)))

    def reparameterise(self):
        for m in self.modules():
            if isinstance(m, _RepDW):
                m.reparameterise()
        return self


# ------------------------------------------------------- MobileNetV4 2024
class _UIB(nn.Module):
    """Universal Inverted Bottleneck, Qin et al. 2024.

    The generalisation of the inverted residual: an optional depthwise
    *before* the expansion and an optional depthwise *after* it. The four
    on/off combinations recover ConvNext-like, inverted-residual-like and
    ExtraDW variants, which is why one block type can serve every stage.
    Both flags are kept configurable rather than fixed, because
    collapsing them is exactly how this becomes an ordinary depthwise
    stack wearing a 2024 label.
    """

    def __init__(self, ci, co, stride=1, expand=4, k=5,
                 dw_start=True, dw_end=False):
        super().__init__()
        ce = _round4(ci * expand)
        ops: list[nn.Module] = []
        if dw_start:
            ops += [nn.Conv1d(ci, ci, k, stride=1, padding=k // 2,
                              groups=ci, bias=False), nn.BatchNorm1d(ci)]
        ops += [nn.Conv1d(ci, ce, 1, bias=False), *_bn_act(ce)]
        if dw_end:
            ops += [nn.Conv1d(ce, ce, k, stride=stride, padding=k // 2,
                              groups=ce, bias=False), *_bn_act(ce)]
            stride_left = 1
        else:
            stride_left = stride
        ops += [nn.Conv1d(ce, co, 1, stride=stride_left, bias=False),
                nn.BatchNorm1d(co)]
        self.op = nn.Sequential(*ops)
        self.use_res = stride == 1 and ci == co

    def forward(self, x):
        return x + self.op(x) if self.use_res else self.op(x)


class MobileNetV4Conv(nn.Module):
    """Convolution-only MobileNetV4. No attention: it does not belong on
    a Cortex-M4, and the hybrid variant would not be a fair comparator."""

    def __init__(self, w=1.0):
        super().__init__()
        c = [_round4(v * w) for v in (24, 32, 48, 64)]
        self.stem = Stem(c[0])
        self.features = nn.Sequential(
            _UIB(c[0], c[1], 2, 4, dw_start=True, dw_end=False),
            _UIB(c[1], c[1], 1, 4, dw_start=True, dw_end=True),
            _UIB(c[1], c[2], 2, 4, dw_start=False, dw_end=True),
            _UIB(c[2], c[3], 2, 4, dw_start=True, dw_end=False))
        self.head = Head(c[3])

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


ZOO = {
    "fcn": FCN1D, "resnet": ResNet1D, "tcn": TCN,
    "mobilenetv2": MobileNetV2, "shufflenetv2": ShuffleNetV2,
    "ghostnet": GhostNet, "mobilenetv3": MobileNetV3,
    "fasternet": FasterNet,
    "repvit": RepViT, "mobilenetv4": MobileNetV4Conv,
}

NICE = {
    "fcn": "FCN-1D", "resnet": "ResNet-1D", "tcn": "TCN",
    "mobilenetv2": "MobileNetV2", "shufflenetv2": "ShuffleNetV2",
    "ghostnet": "GhostNet", "mobilenetv3": "MobileNetV3",
    "fasternet": "FasterNet", "repvit": "RepViT",
    "mobilenetv4": "MobileNetV4-Conv",
}

YEAR = {"fcn": 2017, "resnet": 2016, "tcn": 2018, "mobilenetv2": 2018,
        "shufflenetv2": 2018, "ghostnet": 2020, "mobilenetv3": 2019,
        "fasternet": 2023, "repvit": 2024, "mobilenetv4": 2024,
        "fasternet": 2023, "repvit": 2024, "mobilenetv4": 2024}

BLOCK = {
    "fcn": "plain convolution",
    "resnet": "residual block",
    "tcn": "dilated convolution",
    "mobilenetv2": "inverted residual",
    "shufflenetv2": "channel split + shuffle",
    "ghostnet": "ghost module",
    "mobilenetv3": "inverted residual + SE + h-swish",
    "fasternet": "partial convolution",
    "repvit": "reparameterised depthwise + FFN",
    "mobilenetv4": "universal inverted bottleneck",
}


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def peak_activation_bytes(m: nn.Module, length: int = 1000,
                          dtype_bytes: int = 1) -> int:
    """Largest single intermediate tensor, in bytes, as an SRAM proxy.

    Not the allocator's true peak -- the vendor tool reports that after
    export -- but it comes from the graph alone, so a design can be
    rejected before anything is compiled, and it explains *why* a model
    does or does not fit rather than only that it did not.
    """
    peak, hooks = 0, []

    def hook(_m, _i, out):
        nonlocal peak
        if torch.is_tensor(out):
            peak = max(peak, out.numel() * dtype_bytes)

    for mod in m.modules():
        if not list(mod.children()):
            hooks.append(mod.register_forward_hook(hook))
    was = m.training
    m.eval()
    m(torch.zeros(1, N_LEADS, length))
    m.train(was)
    for h in hooks:
        h.remove()
    return peak


@torch.no_grad()
def macs(m: nn.Module, length: int = 1000) -> int:
    """Multiply-accumulates for one inference, convolutions and linear."""
    total, hooks = 0, []

    def hook(mod, inp, out):
        nonlocal total
        if isinstance(mod, nn.Conv1d):
            total += (mod.in_channels // mod.groups * mod.out_channels
                      * mod.kernel_size[0] * out.shape[-1])
        elif isinstance(mod, nn.Linear):
            total += mod.in_features * mod.out_features

    for mod in m.modules():
        if isinstance(mod, (nn.Conv1d, nn.Linear)):
            hooks.append(mod.register_forward_hook(hook))
    was = m.training
    m.eval()
    m(torch.zeros(1, N_LEADS, length))
    m.train(was)
    for h in hooks:
        h.remove()
    return total


def build(name: str, w: float = 1.0) -> nn.Module:
    return ZOO[name](w=w)


def fit_to_budget(name: str, max_param_bytes: int, max_act_bytes: int,
                  length: int = 1000, lo: float = 0.02, hi: float = 6.0,
                  tol: float = 1e-3):
    """Largest width multiplier of this family that meets both budgets.

    Found by bisection, not by sweeping a grid. The earlier version
    stepped a linear grid from `hi` down to `lo` and took the first width
    that fit, which is scale-dependent: a fixed step of 0.067 is a 7%
    move for a family sitting near w=1 and a 56% move for one near
    w=0.12. Deep families need small widths, so the grid penalised them,
    and refining it to compensate shifted every other family's width at
    the same time. Adding one architecture could therefore perturb the
    results of all the others, which is indefensible in a benchmark
    whose premise is that only the architecture varies.

    Bisection converges on the same answer at any scale and to any
    tolerance, so each family's width depends on that family alone.

    Each family is given the biggest version of itself the device will
    take, rather than a common width: a fixed width would hand the
    comparison to whichever family happens to be densest at that
    setting, and the question is what a design buys for a fixed budget.

    Returns (width, model, stats), or (None, None, stats-at-lo) when even
    the smallest admissible width does not fit.
    """
    def fits(w):
        m = build(name, w)
        return (n_params(m) <= max_param_bytes
                and peak_activation_bytes(m, length) <= max_act_bytes), m

    ok_lo, m_lo = fits(lo)
    if not ok_lo:
        return None, None, {"params": n_params(m_lo),
                            "peak_act": peak_activation_bytes(m_lo, length),
                            "macs": macs(m_lo, length)}
    ok_hi, _ = fits(hi)
    if ok_hi:
        best_w, best_m = hi, build(name, hi)
    else:
        a, b, best_w, best_m = lo, hi, lo, m_lo
        while b - a > tol:
            mid = (a + b) / 2
            ok, m = fits(mid)
            if ok:
                a, best_w, best_m = mid, mid, m
            else:
                b = mid
    return (round(best_w, 4), best_m,
            {"params": n_params(best_m),
             "peak_act": peak_activation_bytes(best_m, length),
             "macs": macs(best_m, length)})
