r"""int8 post-training quantisation of every trained encoder.

    .\.venv\Scripts\python.exe benchmark_mcu\scripts\b2_int8.py

ST Edge AI imports quantised graphs; it does not quantise. So the int8
model is built here with ONNX Runtime static PTQ in QDQ form, which is
what the device will actually run, and both precisions are scored on the
same held-out records so the cost of quantisation is attributable.

Calibration uses 256 records drawn **only from each fold's training
sources**. Calibrating on the held-out source would quietly leak target
statistics into a cross-source result, and the whole benchmark is about
what survives contact with an unseen source.

Percentile calibration at 99.99 rather than MinMax: a single outlier
sample otherwise stretches the activation range and costs more accuracy
than the bit width does. Which calibrator is used matters more than the
architecture for several of these families, so it is fixed and stated
rather than left to a default.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "benchmark_mcu"
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(ROOT / "12_leads" / "src"))

from ecgmcu.data import Cohort, SOURCES, FS_OUT                  # noqa: E402
from mcubench.models import ZOO, NICE, build, n_params           # noqa: E402

BUDGETS = {"compact": (32 * 1024, 96 * 1024),
           "standard": (128 * 1024, 96 * 1024)}
TRAIN = HERE / "results" / f"train_{FS_OUT}hz"
OUT = HERE / "results" / f"int8_{FS_OUT}hz"
LENGTH, N_LEADS = 1000, 12
CALIB_N = 256


def macro_auprc(y, p):
    from sklearn.metrics import average_precision_score
    v = [average_precision_score(y[:, c], p[:, c])
         for c in range(y.shape[1]) if 0 < y[:, c].sum() < len(y)]
    return float(np.mean(v)) if v else float("nan")


def macro_auroc(y, p):
    from sklearn.metrics import roc_auc_score
    v = [roc_auc_score(y[:, c], p[:, c])
         for c in range(y.shape[1]) if 0 < y[:, c].sum() < len(y)]
    return float(np.mean(v)) if v else float("nan")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def export_onnx(model, path: Path, batch: int = 1,
                dynamic: bool = True) -> None:
    """Dynamic batch for host scoring; the deploy export pins it to 1.

    ST Edge AI wants a fixed single-sample graph, but scoring 20k
    records one at a time is pointless, so the two exports differ only
    in that axis.
    """
    model.eval()
    dyn = {"x": {0: "n"}, "y": {0: "n"}} if dynamic else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, torch.zeros(batch, N_LEADS, LENGTH), str(path),
            input_names=["x"], output_names=["y"], opset_version=13,
            dynamic_axes=dyn, dynamo=False)


class Reader:
    """Calibration feed: training-source records only, normalised as trained."""

    def __init__(self, sig_path, rows, mean, std):
        sig = np.load(sig_path, mmap_mode="r")
        take = np.sort(rows)
        self.data = ((np.asarray(sig[take], dtype=np.float32) - mean) / std)
        self.i = 0

    def get_next(self):
        if self.i >= self.data.shape[0]:
            return None
        x = self.data[self.i:self.i + 1]
        self.i += 1
        return {"x": x}

    def rewind(self):
        self.i = 0


def quantize(src: Path, dst: Path, reader) -> None:
    from onnxruntime.quantization import (quantize_static, QuantFormat,
                                          QuantType, CalibrationMethod)
    reader.rewind()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantize_static(
            str(src), str(dst), reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.Percentile,
            extra_options={"CalibPercentile": 99.99, "CalibNumBins": 2048},
            per_channel=True)


def run_onnx(path: Path, sig_path, rows, mean, std, batch=64):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so,
                                providers=["CPUExecutionProvider"])
    sig = np.load(sig_path, mmap_mode="r")
    out = []
    order = np.argsort(rows)
    srt = rows[order]
    for i in range(0, srt.size, batch):
        chunk = srt[i:i + batch]
        x = ((np.asarray(sig[chunk], dtype=np.float32) - mean) / std)
        out.append(sess.run(None, {"x": x})[0])
    y = np.concatenate(out)
    back = np.empty_like(y)
    back[order] = y
    return sigmoid(back)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ZOO))
    ap.add_argument("--budgets", nargs="+", default=list(BUDGETS))
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    coh = Cohort()
    rng = np.random.default_rng(args.seed)
    rows_out = []

    for budget in args.budgets:
        pb, ab = BUDGETS[budget]
        for name in args.models:
            for src in args.sources:
                tag = f"{name}__{budget}__{src}"
                ck = TRAIN / f"{tag}.pt"
                run = TRAIN / f"{tag}.npz"
                if not (ck.exists() and run.exists()):
                    continue
                d = np.load(run)
                f32p = OUT / f"{tag}_f32.onnx"
                i8p = OUT / f"{tag}_int8.onnx"
                probp = OUT / f"{tag}_prob.npz"
                if probp.exists() and not args.force:
                    rows_out.append(json.loads(
                        (OUT / f"{tag}.json").read_text()))
                    continue

                # Rebuild from the width recorded by the training run,
                # never by recomputing the budget fit. The checkpoint
                # defines the model; recomputing means any later change
                # to the search silently desynchronises the two, which
                # presents as a state_dict shape mismatch halfway
                # through a long job.
                width = float(d["width"])
                model = build(name, width)
                # Checkpoints are saved folded, so the module has to be
                # folded before the load or the keys will not line up.
                if hasattr(model, "reparameterise"):
                    model.reparameterise()
                assert n_params(model) == int(d["params"]), (
                    f"{name}/{budget}/{src}: rebuilt model has "
                    f"{n_params(model)} parameters, checkpoint has "
                    f"{int(d['params'])}")
                model.load_state_dict(torch.load(ck, map_location="cpu"))
                export_onnx(model, f32p)

                tr, _, te = coh.loso(src, seed=args.seed)
                mean, std = coh.stats(tr)
                calib = rng.choice(tr, min(CALIB_N, tr.size), replace=False)
                quantize(f32p, i8p, Reader(coh.sig_path, calib, mean, std))

                y = d["test_y"].astype(np.int8)
                p32 = run_onnx(f32p, coh.sig_path, te, mean, std)
                p8 = run_onnx(i8p, coh.sig_path, te, mean, std)
                np.savez_compressed(probp, y=y, prob_f32=p32.astype(np.float32),
                                    prob=p8.astype(np.float32))

                rec = {
                    "model": name, "budget": budget, "source": src,
                    "width": float(width),
                    "onnx_f32_kb": f32p.stat().st_size / 1024,
                    "onnx_int8_kb": i8p.stat().st_size / 1024,
                    "auprc_f32": macro_auprc(y, p32),
                    "auprc_int8": macro_auprc(y, p8),
                    "auroc_f32": macro_auroc(y, p32),
                    "auroc_int8": macro_auroc(y, p8)}
                rec["d_auprc"] = rec["auprc_int8"] - rec["auprc_f32"]
                (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=1))
                rows_out.append(rec)
                print(f"[{tag}] f32 {rec['auprc_f32']:.4f} -> int8 "
                      f"{rec['auprc_int8']:.4f} ({rec['d_auprc']:+.4f})  "
                      f"{rec['onnx_int8_kb']:.1f} KB", flush=True)

    # Rebuild from every per-model record on disk, not from this run's
    # rows. A scoped run (--models ...) would otherwise overwrite the
    # aggregate with only the subset it touched -- the same truncation
    # bug that silently cut the deployment table to a third.
    allrec = []
    for f in sorted(OUT.glob("*__*__*.json")):
        try:
            allrec.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    (OUT / "int8.json").write_text(json.dumps(allrec, indent=1))
    print(f"\n-> {OUT}  ({len(rows_out)} entries)")


if __name__ == "__main__":
    main()
