r"""Study IIIB: real int8 deployment, not a division by four.

    .\.venv\Scripts\python.exe 12_leads\scripts\s3b_int8.py

ST Edge AI imports quantised models; it does not quantise. So the int8
model is built here with ONNX Runtime static post-training quantisation in
QDQ form, and only then handed to the analyser. Two numbers come out that
the float32 study could not produce:

    int8 macro AUPRC   -- what quantisation costs in accuracy
    int8 Flash / SRAM  -- what the allocator really needs, not params / 4

Calibration draws only from the training sources of each fold. Calibrating
on the held-out source would leak the target distribution into the
deployed model and quietly flatter every cross-source number.

Both precisions are scored on the identical record set, so the difference
is attributable to quantisation alone.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from ecgmcu.data import (Cohort, SOURCES, LENGTH, N_LEADS,      # noqa: E402
                         FS_OUT)
from ecgmcu.models import ZOO, build, n_params                 # noqa: E402

STEDGEAI = Path("C:/Users/Alith/STM32Cube/Repository/Packs/STMicroelectronics"
                "/X-CUBE-AI/10.2.0/Utilities/windows/stedgeai.exe")
_TAG = "" if FS_OUT == 250 else f"_{FS_OUT}hz"
S2 = HERE / "results" / f"study2{_TAG}"
OUT = HERE / "results" / f"study3b{_TAG}"
SRAM_KB, FLASH_KB = 128, 512
N_CALIB = 256


def macro_auprc(y, p):
    from sklearn.metrics import average_precision_score
    v = [average_precision_score(y[:, c], p[:, c])
         for c in range(y.shape[1]) if 0 < y[:, c].sum() < len(y)]
    return float(np.mean(v)) if v else float("nan")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def export_onnx(name, state, path, batch=None):
    m = build(name).eval()
    m.load_state_dict(state)
    dyn = None if batch else {"input": {0: "N"}, "output": {0: "N"}}
    torch.onnx.export(m, torch.zeros(batch or 1, N_LEADS, LENGTH), str(path),
                      input_names=["input"], output_names=["output"],
                      opset_version=13, dynamic_axes=dyn, dynamo=False)


class Reader:
    """Calibration batches, drawn from training sources only."""

    def __init__(self, sig, rows, mean, std):
        self.data = [{"input": ((np.asarray(sig[r:r + 1], np.float32)
                                 - mean) / std)} for r in rows]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        self.i += 1
        return self.data[self.i - 1]

    def rewind(self):
        self.i = 0


def quantize(src: Path, dst: Path, reader, calib: str = "percentile") -> None:
    """Fold BatchNorm, then calibrate activation ranges.

    The calibration rule is not a detail. An ECG is a flat baseline with a
    QRS spike on it, and some leads are clipped outright, so a MinMax range
    is set by the most extreme sample in the calibration set and spends
    almost all of the 256 int8 levels on amplitudes that essentially never
    occur. Clipping at the 99.99th percentile instead costs a handful of
    saturated QRS peaks and recovers nearly all of the accuracy -- see the
    `--calib minmax` ablation.
    """
    from onnxruntime.quantization import (quantize_static, QuantFormat,
                                          QuantType, CalibrationMethod)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = src.with_name(src.stem + "_pre.onnx")
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    extra = {"ActivationSymmetric": False, "WeightSymmetric": True}
    if calib == "percentile":
        method = CalibrationMethod.Percentile
        extra |= {"CalibPercentile": 99.99, "CalibNumBins": 2048}
    else:
        method = CalibrationMethod.MinMax
    quantize_static(str(pre), str(dst), reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True, reduce_range=False,
                    calibrate_method=method, extra_options=extra)


def run_onnx(path: Path, sig, rows, mean, std, batch=64):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so,
                                providers=["CPUExecutionProvider"])
    out = []
    for s in range(0, len(rows), batch):
        idx = np.sort(rows[s:s + batch])
        x = (np.asarray(sig[idx], np.float32) - mean) / std
        out.append(sess.run(None, {"input": x})[0])
    return sigmoid(np.concatenate(out).astype(np.float64))


def analyse(onnx: Path, workdir: Path) -> dict:
    r = subprocess.run(
        [str(STEDGEAI), "analyze", "--model", str(onnx), "--target", "stm32",
         "--input-data-type", "int8", "--output-data-type", "float32",
         "--allocate-inputs", "--allocate-outputs", "--verbosity", "1",
         "--workspace", str(workdir / "ws"), "--output", str(workdir)],
        capture_output=True, text=True)
    txt = r.stdout + r.stderr
    (workdir / "stedgeai.log").write_text(txt, encoding="utf-8")

    def grab(pat, cast=float):
        m = re.search(pat, txt, re.I)
        return cast(m.group(1).replace(",", "")) if m else None

    return {"ok": r.returncode == 0,
            "weights_b": grab(r"weights\s*\(ro\)\s*:\s*([\d,]+)", int),
            "act_b": grab(r"activations\s*\(rw\)\s*:\s*([\d,]+)", int),
            "ram_total_b": grab(r"ram\s*\(total\)\s*:\s*([\d,]+)", int),
            "macc": grab(r"macc\s*:\s*([\d,]+)", int)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ZOO))
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--max-eval", type=int, default=0,
                    help="0 = full held-out set")
    ap.add_argument("--calib", choices=("percentile", "minmax"),
                    default="percentile")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    coh = Cohort()
    sig = np.load(coh.sig_path, mmap_mode="r")
    rng = np.random.default_rng(0)
    rows_out = []
    tag = "" if args.calib == "percentile" else f"_{args.calib}"
    dst_json = OUT / f"int8{tag}.json"
    done = {}
    if dst_json.exists() and not args.force:
        done = {(d["model"], d["source"]): d
                for d in json.loads(dst_json.read_text())}

    for name in args.models:
        for src in args.sources:
            key = (name, src)
            if key in done:
                rows_out.append(done[key])
                print(f"[{name}__{src}] cached")
                continue
            ckpt = S2 / f"{name}__{src}.pt"
            if not ckpt.exists():
                continue
            t0 = time.time()
            wd = OUT / f"{name}__{src}{tag}"
            wd.mkdir(parents=True, exist_ok=True)
            tr, _, te = coh.loso(src)
            mean, std = coh.stats(tr)
            if args.max_eval and te.size > args.max_eval:
                te = np.sort(rng.choice(te, args.max_eval, replace=False))
            y = coh.y[te]

            state = torch.load(ckpt, map_location="cpu")
            f32 = wd / "model_f32.onnx"
            i8 = wd / "model_int8.onnx"
            export_onnx(name, state, f32)
            calib = np.sort(rng.choice(tr, N_CALIB, replace=False))
            quantize(f32, i8, Reader(sig, calib, mean, std), args.calib)

            p32 = run_onnx(f32, sig, te, mean, std)
            p8 = run_onnx(i8, sig, te, mean, std)
            rec = {"model": name, "source": src, "n_eval": int(te.size),
                   "auprc_f32": macro_auprc(y, p32),
                   "auprc_int8": macro_auprc(y, p8),
                   "calib": args.calib,
                   "minutes": (time.time() - t0) / 60}
            rec["delta"] = rec["auprc_int8"] - rec["auprc_f32"]
            np.savez_compressed(wd / "int8_prob.npz",
                                prob=p8.astype(np.float32),
                                prob_f32=p32.astype(np.float32),
                                rows=te, y=y.astype(np.int8))
            rows_out.append(rec)
            print(f"[{name}__{src}] n={te.size}  f32 {rec['auprc_f32']:.4f}"
                  f"  int8 {rec['auprc_int8']:.4f}"
                  f"  d {rec['delta']:+.4f}  ({rec['minutes']:.1f} min)",
                  flush=True)
            dst_json.write_text(json.dumps(rows_out, indent=1))

    # ---- static cost: identical for every fold, so analyse once -------
    dep = []
    dep_json = OUT / "deployment_int8.json"
    if dep_json.exists() and not args.force:
        dep = json.loads(dep_json.read_text())
    seen = {d["model"] for d in dep}
    for name in args.models:
        if name in seen:
            continue
        ck = next((S2 / f"{name}__{s}.pt" for s in SOURCES
                   if (S2 / f"{name}__{s}.pt").exists()), None)
        if ck is None:
            continue
        wd = OUT / f"deploy_{name}"
        wd.mkdir(parents=True, exist_ok=True)
        f32 = wd / "b1_f32.onnx"
        i8 = wd / "b1_int8.onnx"
        export_onnx(name, torch.load(ck, map_location="cpu"), f32, batch=1)
        calib = np.sort(rng.choice(np.arange(sig.shape[0]), N_CALIB, False))
        quantize(f32, i8, Reader(sig, calib, 0.0, 1.0), args.calib)
        a = analyse(i8, wd)
        flash_kb = (a["weights_b"] or 0) / 1024
        sram_kb = (a["ram_total_b"] or 0) / 1024
        d = {"model": name, "params": n_params(build(name)),
             "flash_kb": round(flash_kb, 1), "sram_kb": round(sram_kb, 1),
             "act_kb": round((a["act_b"] or 0) / 1024, 1),
             "macc": a["macc"],
             "fits": bool(flash_kb < FLASH_KB and sram_kb < SRAM_KB),
             "analyzer_ok": a["ok"]}
        dep.append(d)
        print(f"[deploy {name}] flash {d['flash_kb']} KB  SRAM {d['sram_kb']}"
              f" KB  fits={d['fits']}", flush=True)
        dep_json.write_text(json.dumps(dep, indent=1))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
