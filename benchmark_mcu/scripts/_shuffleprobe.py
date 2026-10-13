"""Which part of the ShuffleNetV2 block defeats code generation?

Replacing chunk with torch.split was not enough: generate still fails
with an INTERNAL ERROR naming a Conv. The block has three candidate
constructs -- the channel shuffle's reshape/transpose/reshape, a
depthwise convolution applied to a split half, and the stride-2 branch
that convolves the whole input twice. Each is isolated here.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "benchmark_mcu"
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))
import b3_deploy as T                                           # noqa: E402
from mcubench.reference import (_Ghost, _ShuffleBlock,          # noqa
                                _shuffle)

WORK = HERE / "results" / "_shuffleprobe"
C = 32


class ShuffleOnly(nn.Module):
    """conv -> channel shuffle -> conv."""

    def __init__(self):
        super().__init__()
        self.a = nn.Conv1d(C, C, 1, bias=False)
        self.b = nn.Conv1d(C, C, 3, padding=1, bias=False)

    def forward(self, x):
        return self.b(_shuffle(self.a(x)))


class SplitDepthwise(nn.Module):
    """split -> depthwise conv on one half -> cat."""

    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(C // 2, C // 2, 3, padding=1, groups=C // 2,
                           bias=False)

    def forward(self, x):
        a, b = torch.split(x, [C // 2, C // 2], dim=1)
        return torch.cat([a, self.c(b)], 1)


class TransposeOnly(nn.Module):
    """The shuffle's reshape/transpose/reshape with no convolution after."""

    def __init__(self):
        super().__init__()
        self.a = nn.Conv1d(C, C, 1, bias=False)

    def forward(self, x):
        return _shuffle(self.a(x))


class BlockS1(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = _ShuffleBlock(C, C, 1)

    def forward(self, x):
        return self.b(x)


class BlockS2(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = _ShuffleBlock(C, C * 2, 2)

    def forward(self, x):
        return self.b(x)


class CatConv(nn.Module):
    """Two convolutions concatenated, then convolved.

    Never tested by the earlier probe, which ended at the concat. Both
    remaining failures feed a Conv from a Concat.
    """

    def __init__(self):
        super().__init__()
        self.a = nn.Conv1d(C, C // 2, 1, bias=False)
        self.b = nn.Conv1d(C, C // 2, 1, bias=False)
        self.c = nn.Conv1d(C, C, 3, padding=1, bias=False)

    def forward(self, x):
        return self.c(torch.cat([self.a(x), self.b(x)], 1))


class GhostConv(nn.Module):
    """A ghost module followed by a convolution."""

    def __init__(self):
        super().__init__()
        self.g = _Ghost(C, C)
        self.c = nn.Conv1d(C, C, 1, bias=False)

    def forward(self, x):
        return self.c(self.g(x))


class DepthwiseOdd(nn.Module):
    """Depthwise convolution on a channel count that is not a power of two."""

    def __init__(self):
        super().__init__()
        self.a = nn.Conv1d(C, 12, 1, bias=False)
        self.b = nn.Conv1d(12, 12, 3, padding=1, groups=12, bias=False)

    def forward(self, x):
        return self.b(self.a(x))


class R:
    def __init__(self, n=16):
        self.d = [np.random.randn(1, C, 1000).astype("float32")
                  for _ in range(n)]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.d):
            return None
        self.i += 1
        return {"x": self.d[self.i - 1]}


def main():
    from onnxruntime.quantization import (quantize_static, QuantFormat,
                                          QuantType, CalibrationMethod)
    WORK.mkdir(parents=True, exist_ok=True)
    print("{:<14} {:>7} {:>6}".format("variant", "gen", "link"))
    for name, cls in (("cat_conv", CatConv), ("ghost", GhostConv),
                      ("dw_odd", DepthwiseOdd), ("shuffle", ShuffleOnly),
                      ("split_dw", SplitDepthwise),
                      ("transpose", TransposeOnly), ("block_s1", BlockS1),
                      ("block_s2", BlockS2)):
        m = cls()
        # Untrained batch-norm folds to an identical all-zero bias in
        # every layer, and the exporter deduplicates those into one
        # initializer aliased by Identity nodes -- which is what the
        # code generator actually choked on. Trained models never look
        # like this, so the probe perturbs the norm parameters to match.
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm1d):
                nn.init.normal_(mod.weight, 1.0, 0.1)
                nn.init.normal_(mod.bias, 0.0, 0.1)
                mod.running_mean.normal_(0.0, 0.1)
                mod.running_var.uniform_(0.8, 1.2)
        m = m.eval()
        work = WORK / name
        work.mkdir(parents=True, exist_ok=True)
        f32, i8 = work / "m.onnx", work / "m_int8.onnx"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(m, torch.zeros(1, C, 1000), str(f32),
                              input_names=["x"], output_names=["y"],
                              opset_version=13, dynamo=False)
            quantize_static(str(f32), str(i8), R(),
                            quant_format=QuantFormat.QDQ,
                            activation_type=QuantType.QInt8,
                            weight_type=QuantType.QInt8,
                            calibrate_method=CalibrationMethod.Percentile,
                            extra_options={"CalibPercentile": 99.99,
                                           "CalibNumBins": 2048},
                            per_channel=True)
        g = T.generate(i8, work / "gen")
        ok = False
        if g.get("generate_ok"):
            ok, _ = T.build(work / "gen", work / "fw")
        print("{:<14} {:>7} {:>6}".format(
            name, "ok" if g.get("generate_ok") else "FAIL",
            "ok" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
