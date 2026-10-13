"""Are dilated convolutions quantisable in this pipeline?

Same conv at several dilations, taken through the real path: export,
ONNX Runtime static INT8, then stedgeai analyze. Parameter count is
identical in every case, so any growth in weight bytes means the layer
fell back to float.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "src"))
import b3_deploy as D                                        # noqa: E402
from onnxruntime.quantization import (quantize_static, QuantFormat,  # noqa
                                      QuantType, CalibrationMethod)

WORK = HERE / "results" / "_dilprobe"
WORK.mkdir(parents=True, exist_ok=True)


class Reader:
    def __init__(self, n=16):
        self.d = [{"i": np.random.randn(1, 32, 1000).astype("float32")}
                  for _ in range(n)]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.d):
            return None
        self.i += 1
        return self.d[self.i - 1]


print("{:>9} {:>9} {:>11} {:>12}".format("dilation", "params", "weights_b",
                                         "bytes/param"))
for dil in (1, 2, 4, 8, 32, 128):
    m = nn.Sequential(nn.Conv1d(32, 32, 3, padding=dil, dilation=dil,
                                bias=False), nn.ReLU()).eval()
    p = sum(q.numel() for q in m.parameters())
    f32, i8 = WORK / f"q{dil}.onnx", WORK / f"q{dil}_int8.onnx"
    torch.onnx.export(m, torch.randn(1, 32, 1000), str(f32),
                      opset_version=13, input_names=["i"],
                      output_names=["o"], dynamo=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantize_static(str(f32), str(i8), Reader(),
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.Percentile,
                        extra_options={"CalibPercentile": 99.99,
                                       "CalibNumBins": 2048},
                        per_channel=True)
    w = D.analyze(i8, WORK / f"qw{dil}").get("weights_b")
    print("{:>9} {:>9} {:>11} {:>12}".format(
        dil, p, w if w is not None else "-", f"{w / p:.2f}" if w else "-"))
