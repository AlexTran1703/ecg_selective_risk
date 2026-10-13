r"""Reference topologies and two scaling methods, for all ten families.

For each family: the published stage structure adapted to 1-D, then the
width needed to reach each INT8 budget under

  uniform  -- published depth kept, channels scaled by w
  joint    -- depth and width scaled together, fixed-resolution compound
              scaling (EfficientNet alpha=1.2, beta=1.1, resolution term
              dropped because the input length is fixed), with at least
              one block retained per published stage

Flash is INT8 weight bytes (one per parameter). SRAM is the graph peak
activation -- a proxy, and Finding 9 shows it is rank-uncorrelated with
what the device actually allocates, so it is an ordering hint only.

Blocks are our 1-D adaptations, so these estimate the 1-D reference
rather than the published 2-D network.
"""
import math
import sys
from pathlib import Path

import torch.nn as nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE.parent / "12_leads" / "src"))
import mcubench.models as M  # noqa: E402

A, B = 1.2, 1.1                      # EfficientNet depth / width coefficients


def _conv(ci, co, s):
    return nn.Sequential(nn.Conv1d(ci, co, 7, stride=s, padding=3,
                                   bias=False), *M._bn_act(co))


def _fb(ci, co, s):
    if ci == co and s == 1:
        return M._FasterBlock(co)
    return nn.Sequential(nn.Conv1d(ci, co, 1, stride=s, bias=False),
                         *M._bn_act(co), M._FasterBlock(co))


def _rv(ci, co, s):
    if ci == co and s == 1:
        return M._RepViTBlock(co)
    return nn.Sequential(nn.Conv1d(ci, co, 1, stride=s, bias=False),
                         *M._bn_act(co), M._RepViTBlock(co))


def _tcn(ci, co, s, i=[0]):
    i[0] += 1
    return M._TCNBlock(ci, co, dilation=2 ** min(i[0], 6))


BLK = {"conv": _conv,
       "res": lambda ci, co, s: M._ResBlock(ci, co, stride=s),
       "tcn": _tcn,
       "ir": lambda ci, co, s: M._InvertedResidual(ci, co, s, 6),
       "sh": lambda ci, co, s: (M._ShuffleDown(ci, co) if s == 2
                                else M._ShuffleUnit(co)),
       "gb": lambda ci, co, s: M._GhostBottleneck(ci, co, s),
       "v3": lambda ci, co, s: M._V3Bneck(ci, co * 4, co, 5, True, "HS", s),
       "fb": _fb, "rv": _rv,
       "uib": lambda ci, co, s: M._UIB(ci, co, s, 4)}

# (stem, [(channels, repeats, stride)], final 1x1 expansion, block type)
REF = {
 "FCN-1D":       (128, [(128, 1, 1), (256, 1, 1), (128, 1, 1)], None, "conv"),
 "ResNet-1D":    (64, [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)],
                  None, "res"),
 "TCN":          (64, [(64, 8, 1)], None, "tcn"),
 "MobileNetV2":  (32, [(16, 1, 1), (24, 2, 2), (32, 3, 2), (64, 4, 2),
                       (96, 3, 1), (160, 3, 2), (320, 1, 1)], 1280, "ir"),
 "ShuffleNetV2": (24, [(116, 4, 2), (232, 8, 2), (464, 4, 2)], 1024, "sh"),
 "GhostNet":     (16, [(16, 1, 1), (24, 2, 2), (40, 2, 2), (80, 4, 2),
                       (112, 2, 1), (160, 5, 2)], 960, "gb"),
 "MobileNetV3":  (16, [(16, 1, 2), (24, 2, 2), (40, 3, 2), (48, 2, 1),
                       (96, 3, 2)], 576, "v3"),
 "FasterNet":    (40, [(40, 1, 1), (80, 2, 2), (160, 8, 2), (320, 2, 2)],
                  1280, "fb"),
 "RepViT":       (48, [(48, 2, 1), (96, 2, 2), (192, 14, 2), (384, 2, 2)],
                  None, "rv"),
 "MobileNetV4":  (32, [(96, 6, 2), (128, 6, 2)], 960, "uib"),
}
NOTE = {"TCN": "8 levels at 64ch -- our choice, no canonical depth"}


def build(name, w=1.0, dmul=1.0):
    stem, stages, final, kind = REF[name]
    f = BLK[kind]
    if kind == "tcn":
        _tcn.__defaults__ = (1, [0])
    L, ci = [], M._round4(stem * w)
    for c, n, s in stages:
        co = M._round4(c * w)
        for i in range(max(1, round(n * dmul))):
            L.append(f(ci, co, s if i == 0 else 1))
            ci = co
    if final:
        co = M._round4(final * w)
        L += [nn.Conv1d(ci, co, 1, bias=False), *M._bn_act(co)]
        ci = co

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = M.Stem(M._round4(stem * w))
            self.features = nn.Sequential(*L)
            self.head = M.Head(ci)

        def forward(self, x):
            return self.head(self.features(self.stem(x)))
    return Net()


def fit(name, cap, joint):
    """Largest model of this family meeting `cap` INT8 weight bytes."""
    lo, hi = 0.005, 1.0
    for _ in range(45):
        mid = (lo + hi) / 2
        if joint:
            phi = math.log(mid ** 2) / math.log(A * B * B)
            w, d = B ** phi, A ** phi
        else:
            w, d = mid, 1.0
        try:
            ok = M.n_params(build(name, w, d)) <= cap
        except Exception:                                   # noqa: BLE001
            ok = False
        if ok:
            lo = mid
        else:
            hi = mid
    if joint:
        phi = math.log(lo ** 2) / math.log(A * B * B)
        return B ** phi, A ** phi
    return lo, 1.0


def rep(name, w, d):
    mo = build(name, w, d)
    p = M.n_params(mo)
    nb = sum(max(1, round(n * d)) for _, n, _ in REF[name][1])
    chans = [M._round4(c * w) for c, _, _ in REF[name][1]]
    clamp = sum(1 for c in chans if c < 8)
    return p / 1024, M.peak_activation_bytes(mo, 1000) / 1024, nb, clamp, min(chans)



for cap_kb in (32, 64, 128):
    print("=== budget {} KB ===".format(cap_kb))
    print("{:<13} | {:>22} | {:>22}".format("", "uniform", "joint"))
    print("{:<13} | {:>5} {:>5} {:>4} {:>5} | {:>5} {:>5} {:>4} {:>5}".format(
        "family", "w", "cmin", "clmp", "SRAM", "w", "cmin", "clmp", "SRAM"))
    tot = {False: 0, True: 0}
    for name in REF:
        row = "{:<13} |".format(name)
        for joint in (False, True):
            w, d = fit(name, cap_kb * 1024, joint)
            f2, s2, b2, cl, cmin = rep(name, w, d)
            tot[joint] += cl
            row += " {:>5.2f} {:>5} {:>4} {:>5.0f} |".format(w, cmin, cl, s2)
        print(row)
    print("   channels below 8:  uniform {}   joint {}".format(
        tot[False], tot[True]))
    print()
