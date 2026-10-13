r"""Freeze the forty deployed configurations, each verified on the toolchain.

    .\.venv\Scripts\python.exe benchmark_mcu\scripts\r0_manifest.py

The analytic fit in `mcubench.scaling` predicts deployed weight bytes
accurately -- within +17% on every one of the twenty earlier hardware
measurements, and conservative in all of them. Its activation estimate
is weaker. For a graph the int8 path fully accepts the two agree; for
one it does not, they diverge badly. TCN's faithful eight-level form is
predicted at 78 KB and measured at 110 KB, which is the difference
between linking on a 128 KB part and overrunning RAM by 17 KB.

So the activation ceiling is settled by building rather than by
modelling. Each configuration starts at the analytic fit and, while the
firmware fails to link, the ceiling is tightened and the family refitted
until it does. The result is written once, and every later stage reads
it instead of recomputing the search -- which also means the width a
checkpoint was trained at can never drift away from the width that gets
deployed.

Configurations the toolchain gate has already linked are taken as
verified and are not rebuilt.
"""

from __future__ import annotations

import argparse
import json
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

import b3_deploy as T                                          # noqa: E402
from mcubench.footprint import (float_fallback_frac,           # noqa: E402
                                peak_activation_bytes, weight_bytes)
from mcubench.reference import REFERENCE, build, fuse          # noqa: E402
from mcubench.scaling import ACT_CAP, BUDGETS, fit             # noqa: E402

GATE = HERE / "results" / "_deployprobe"
WORK = HERE / "results" / "_manifest"
OUT = HERE / "results" / "config_manifest.json"
CONFIGS = [("U-UC128", "uniform", "UC-128"), ("U-C256", "uniform", "C-256"),
           ("J-UC128", "joint", "UC-128"), ("J-C256", "joint", "C-256")]
SHRINK, TRIES = 0.85, 6


class R:
    """Calibration on noise: this stage asks whether the graph links, a
    question the weights do not enter into."""

    def __init__(self, n=16):
        self.d = [np.random.randn(1, 12, 1000).astype("float32")
                  for _ in range(n)]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.d):
            return None
        self.i += 1
        return {"x": self.d[self.i - 1]}


def _as_trained(m):
    """Give batch-norm trained-looking parameters.

    Untrained norms all fold to the same all-zero bias; the exporter
    deduplicates those into one initializer aliased by Identity nodes
    and code generation fails on it with an error naming a Conv. It
    looks exactly like an unsupported operator and is not one.
    """
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            nn.init.normal_(mod.weight, 1.0, 0.1)
            nn.init.normal_(mod.bias, 0.0, 0.1)
            mod.running_mean.normal_(0.0, 0.1)
            mod.running_var.uniform_(0.8, 1.2)
    return m.eval()


def links(name, w, d, work) -> tuple[bool, dict]:
    """Export, quantise, generate and link. Does the firmware fit?"""
    from onnxruntime.quantization import (quantize_static, QuantFormat,
                                          QuantType, CalibrationMethod)
    work.mkdir(parents=True, exist_ok=True)
    m = _as_trained(fuse(build(name, w, d)))
    f32, i8 = work / "m.onnx", work / "m_int8.onnx"
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
    except Exception as exc:                                   # noqa: BLE001
        return False, {"stage": "quantise", "error": str(exc)[:80]}
    a = T.analyze(i8, work)
    g = T.generate(i8, work / "gen")
    if not g.get("generate_ok"):
        return False, {"stage": "generate", **a}
    ok, err = T.build(work / "gen", work / "fw")
    return ok, {"stage": "link" if not ok else "ok",
                "error": None if ok else err[:80], **a}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(REFERENCE))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    man = json.loads(OUT.read_text()) if OUT.exists() else {}
    WORK.mkdir(parents=True, exist_ok=True)

    for cname, meth, bud in CONFIGS:
        for name in args.models:
            key = f"{name}__{cname}"
            if key in man and not args.force:
                continue
            cap = BUDGETS[bud]
            w, d = fit(name, cap, meth)

            # The gate already linked this exact configuration.
            if (GATE / key / "fw" / "fw.elf").exists() and not args.force:
                m = fuse(build(name, w, d))
                man[key] = {"model": name, "config": cname, "budget": bud,
                            "method": meth, "width": w, "depth": d,
                            "act_cap": ACT_CAP, "tries": 0,
                            "verified": True, "source": "gate",
                            "weight_bytes": weight_bytes(m),
                            "pred_act_b": peak_activation_bytes(m, 1000),
                            "fp32_frac": float_fallback_frac(m)}
                print(f"[{key:28s}] verified by gate  w={w:.3f} d={d:.3f}",
                      flush=True)
                OUT.write_text(json.dumps(man, indent=1))
                continue

            act = ACT_CAP
            rec = None
            for attempt in range(TRIES):
                w, d = fit(name, cap, meth, act_cap=act)
                ok, info = links(name, w, d, WORK / key)
                if ok:
                    rec = {"tries": attempt + 1, "verified": True}
                    break
                print(f"[{key:28s}] attempt {attempt + 1}: "
                      f"{info.get('stage')} failed at w={w:.3f} "
                      f"act_cap={act / 1024:.0f}K "
                      f"(tool says {(info.get('activations_b') or 0) / 1024:.0f}K)",
                      flush=True)
                act = int(act * SHRINK)
            if rec is None:
                rec = {"tries": TRIES, "verified": False}
            m = fuse(build(name, w, d))
            man[key] = {"model": name, "config": cname, "budget": bud,
                        "method": meth, "width": w, "depth": d,
                        "act_cap": act, "source": "measured",
                        "weight_bytes": weight_bytes(m),
                        "pred_act_b": peak_activation_bytes(m, 1000),
                        "fp32_frac": float_fallback_frac(m), **rec}
            OUT.write_text(json.dumps(man, indent=1))
            print(f"[{key:28s}] {'VERIFIED' if rec['verified'] else 'FAILED'}"
                  f"  w={w:.3f} d={d:.3f} act_cap={act / 1024:.0f}K",
                  flush=True)

    bad = [k for k, v in man.items() if not v["verified"]]
    print(f"\n-> {OUT}  ({len(man)} configurations, {len(bad)} unverified)")
    for k in bad:
        print("   ", k)


if __name__ == "__main__":
    main()
