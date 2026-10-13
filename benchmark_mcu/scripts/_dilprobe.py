"""Does the toolchain store a dilated kernel densely?

One conv, 32->32 channels, k=3, at a range of dilations. The parameter
count is identical in every case; if weight bytes grow with dilation,
the kernel is being expanded with zeros and TCN's budget cannot be
enforced on parameter count.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import b3_deploy as D                                        # noqa: E402

WORK = HERE / "results" / "_dilprobe"
WORK.mkdir(parents=True, exist_ok=True)
x = torch.randn(1, 32, 1000)

print("{:>9} {:>10} {:>11} {:>12}".format("dilation", "params", "weights_b",
                                          "bytes/param"))
for dil in (1, 2, 4, 8, 16, 32, 64, 128):
    m = nn.Sequential(nn.Conv1d(32, 32, 3, padding=dil, dilation=dil,
                                bias=False), nn.ReLU()).eval()
    p = sum(q.numel() for q in m.parameters())
    f = WORK / f"d{dil}.onnx"
    torch.onnx.export(m, x, str(f), opset_version=13,
                      input_names=["i"], output_names=["o"], dynamo=False)
    r = D.analyze(f, WORK / f"w{dil}")
    w = r.get("weights_b")
    print("{:>9} {:>10} {:>11} {:>12}".format(
        dil, p, w if w is not None else "-",
        f"{w / p:.2f}" if w else "-"))
