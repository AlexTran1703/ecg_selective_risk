"""Which way of convolving a channel subset survives code generation?

ShuffleNetV2 and FasterNet both failed `stedgeai generate` with an
INTERNAL ERROR naming a Conv node, and both convolve part of their
input: one splits the channels in half, the other convolves the first
quarter. The earlier implementations of both deployed successfully, and
differed only in how the subset was taken. This isolates that choice.
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
sys.path.insert(0, str(HERE / "scripts"))
import b3_deploy as T                                           # noqa: E402

WORK = HERE / "results" / "_sliceprobe"
C = 32


class Basic(nn.Module):
    """No subsetting at all: the control."""

    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(C, C, 3, padding=1, bias=False)

    def forward(self, x):
        return self.c(x)


class Index(nn.Module):
    """Python slice -> ONNX Slice."""

    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(C // 4, C // 4, 3, padding=1, bias=False)

    def forward(self, x):
        return torch.cat([self.c(x[:, :C // 4]), x[:, C // 4:]], 1)


class Split(nn.Module):
    """torch.split with explicit sizes -> ONNX Split."""

    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(C // 4, C // 4, 3, padding=1, bias=False)

    def forward(self, x):
        a, b = torch.split(x, [C // 4, C - C // 4], dim=1)
        return torch.cat([self.c(a), b], 1)


class Chunk(nn.Module):
    """tensor.chunk -> ONNX Split, equal parts."""

    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(C // 2, C // 2, 3, padding=1, bias=False)

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.cat([a, self.c(b)], 1)


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
    print("{:<8} {:>9} {:>7} {:>6}".format("variant", "weights_b", "gen",
                                           "link"))
    for name, cls in (("basic", Basic), ("index", Index),
                      ("split", Split), ("chunk", Chunk)):
        m = cls().eval()
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
        a = T.analyze(i8, work)
        g = T.generate(i8, work / "gen")
        ok = False
        if g.get("generate_ok"):
            ok, _ = T.build(work / "gen", work / "fw")
        print("{:<8} {:>9} {:>7} {:>6}".format(
            name, a.get("weights_b") or 0,
            "ok" if g.get("generate_ok") else "FAIL",
            "ok" if ok else "FAIL"))


if __name__ == "__main__":
    main()
