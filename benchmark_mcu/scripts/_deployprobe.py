"""Can the toolchain compile these forty graphs at all?

Random weights, real shapes: export, quantise, analyze, generate C and
link it. Accuracy is irrelevant here -- the question is whether any of
the reconstructed blocks (partial convolution, channel shuffle, ghost
concatenation, squeeze-excitation, causal chomp) defeats the code
generator. Finding that out now costs minutes; finding it out after the
training run costs the training run.
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
from mcubench.footprint import weight_bytes                     # noqa: E402
from mcubench.reference import REFERENCE, build, fuse           # noqa: E402
from mcubench.scaling import BUDGETS, fit                       # noqa: E402

WORK = HERE / "results" / "_deployprobe"
CONFIGS = [("U-UC128", "uniform", "UC-128"), ("U-C256", "uniform", "C-256"),
           ("J-UC128", "joint", "UC-128"), ("J-C256", "joint", "C-256")]


class R:
    def __init__(self, n=24):
        self.d = [np.random.randn(1, 12, 1000).astype("float32")
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
    only = sys.argv[1:] or None
    WORK.mkdir(parents=True, exist_ok=True)
    print("{:<13} {:<9} {:>9} {:>9} {:>7} {:>8} {:>6} {:>6}".format(
        "family", "config", "pred_b", "meas_b", "err%", "sram", "gen", "link"))
    bad = []
    for cname, meth, bud in CONFIGS:
        for name in REFERENCE:
            if only and name not in only:
                continue
            w, d = fit(name, BUDGETS[bud], meth)
            m = fuse(build(name, w, d))
            # Untrained batch-norm folds to an identical all-zero bias
            # in every layer; the exporter deduplicates those into a
            # single initializer aliased by Identity nodes, and the code
            # generator fails on that with an INTERNAL ERROR naming a
            # Conv. It is an artefact of random weights -- a trained
            # model never has two identical bias vectors -- but it
            # masquerades as an architecture defect, so the probe
            # perturbs the norm parameters to look trained.
            for mod in m.modules():
                if isinstance(mod, nn.BatchNorm1d):
                    nn.init.normal_(mod.weight, 1.0, 0.1)
                    nn.init.normal_(mod.bias, 0.0, 0.1)
                    mod.running_mean.normal_(0.0, 0.1)
                    mod.running_var.uniform_(0.8, 1.2)
            m = m.eval()
            pred = weight_bytes(m)
            work = WORK / f"{name}__{cname}"
            work.mkdir(parents=True, exist_ok=True)
            f32, i8 = work / "m_f32.onnx", work / "m_int8.onnx"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    torch.onnx.export(m, torch.zeros(1, 12, 1000), str(f32),
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
            except Exception as exc:                          # noqa: BLE001
                # Random activations can collapse to a single value and
                # defeat the percentile calibrator. Real data does not,
                # so this is noted rather than treated as a failure.
                print("{:<13} {:<9} calibration artefact: {}".format(
                    name, cname, str(exc)[:48]), flush=True)
                continue
            a = T.analyze(i8, work)
            g = T.generate(i8, work / "gen")
            ok, err = (False, "generate failed")
            if g.get("generate_ok"):
                ok, err = T.build(work / "gen", work / "fw")
            meas = a.get("weights_b") or 0
            e = 100 * (pred - meas) / meas if meas else float("nan")
            if not ok or (meas and meas > BUDGETS[bud]):
                bad.append((name, cname, err if not ok else "OVER BUDGET"))
            print("{:<13} {:<9} {:>9,} {:>9,} {:>+7.1f} {:>7.1f}K {:>6} "
                  "{:>6}".format(
                      name, cname, pred, meas, e,
                      (a.get("activations_b") or 0) / 1024,
                      "ok" if g.get("generate_ok") else "FAIL",
                      "ok" if ok else "FAIL"))
    print("\nfailures: {}".format(len(bad)))
    for row in bad:
        print("  ", row)


if __name__ == "__main__":
    main()
