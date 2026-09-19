"""Run inference once, cache the logits, then do the selective analysis.

    python scripts/04_calibrate_eval.py --run artifacts/runs/raw__PTBXL

Inference is the only GPU work here, and its output is written to
``predictions.npz``.  Every later question -- a different confidence score, a
different risk level, adapted thresholds, the oracle bound -- is answered from
that file by ``scripts/08_selective_sweep.py`` in seconds, with no ECG loading
and no forward pass.

Order of operations is enforced:

  1. thresholds ``t_k`` and the per-class margin scale -- fitted on the
     development split of the three *training* sources, then frozen;
  2. confidence ``s(x)`` -- normalised mean margin using those frozen values;
  3. ``tau*``  -- certified on the 20% target calibration subset (Learn-then-Test);
  4. report    -- the remaining 80% target test subset, read once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgsr.analysis import ALPHA_LEVELS, Predictions, evaluate_run   # noqa: E402
from ecgsr.datasets import (ECGCacheDataset, Representation, Split,  # noqa: E402
                            load_index)
from ecgsr.models import DEFAULT_BACKBONE_1D, build_model            # noqa: E402
from ecgsr.scores import PRIMARY_SCORE                               # noqa: E402
from ecgsr.train import predict                                      # noqa: E402
from torch.utils.data import DataLoader                              # noqa: E402


def load_run(run_dir: Path, cache: Path, meta: Path, batch_size: int,
             workers: int, device: torch.device):
    index = load_index(meta)
    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    split_npz = np.load(run_dir / "split.npz", allow_pickle=True)
    split = Split(target=str(split_npz["target"]), train=split_npz["train"],
                  dev=split_npz["dev"], cal=split_npz["cal"], test=split_npz["test"])

    rep_kind = ckpt["config"]["representation"]
    rep = Representation(rep_kind).to(device)
    rep.mean = ckpt["rep_mean"].to(device)
    rep.std = ckpt["rep_std"].to(device)

    backbone = ckpt["config"].get("backbone", DEFAULT_BACKBONE_1D)
    model = build_model(rep_kind, index.n_classes, backbone=backbone).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def loader(rows):
        return DataLoader(ECGCacheDataset(cache, rows, index.y[rows]),
                          batch_size=batch_size, shuffle=False,
                          num_workers=workers, pin_memory=True)

    return index, split, rep, model, rep_kind, backbone, loader


def cache_predictions(run_dir: Path, cache: Path, meta: Path, batch_size: int,
                      workers: int, device: torch.device) -> Predictions:
    """Single forward pass over dev/cal/test; writes ``predictions.npz``."""
    index, split, rep, model, rep_kind, backbone, loader = load_run(
        run_dir, cache, meta, batch_size, workers, device)

    logits, labels, records = {}, {}, {}
    for name, rows in (("dev", split.dev), ("cal", split.cal), ("test", split.test)):
        probs, y = predict(model, rep, loader(rows), device)
        # ``predict`` returns probabilities; store logits so the cache is
        # independent of any later squashing choice.
        p = np.clip(probs.astype(np.float64), 1e-7, 1 - 1e-7)
        logits[name] = np.log(p / (1 - p)).astype(np.float32)
        labels[name] = y.astype(np.int8)
        records[name] = index.record[rows]

    pred = Predictions(logits=logits, labels=labels, records=records,
                       meta={"representation": rep_kind, "backbone": backbone,
                             "target": split.target, "classes": index.classes,
                             "abbreviations": index.abbreviations})
    pred.save(run_dir / "predictions.npz")
    (run_dir / "predictions_meta.json").write_text(
        json.dumps(pred.meta, indent=2), encoding="utf-8")
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "cache" / "signals.npy")
    ap.add_argument("--meta", type=Path, default=ROOT / "data" / "meta")
    ap.add_argument("--score", default=PRIMARY_SCORE)
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reuse", action="store_true",
                    help="skip inference if predictions.npz already exists")
    args = ap.parse_args()

    pred_path = args.run / "predictions.npz"
    if args.reuse and pred_path.exists():
        pred = Predictions.load(pred_path)
        meta = json.loads((args.run / "predictions_meta.json").read_text(encoding="utf-8"))
        pred.meta = meta
        print(f"  reusing cached logits from {pred_path}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred = cache_predictions(args.run, args.cache, args.meta,
                                 args.batch_size, args.workers, device)

    m = pred.meta
    print(f"=== {m['representation'].upper()} | {m['backbone']} | "
          f"target = {m['target']} ===")

    result = evaluate_run(pred, score_name=args.score, n_grid=args.grid)
    curve = result.pop("curve")
    result.update({"representation": m["representation"], "backbone": m["backbone"],
                   "target": m["target"], "classes": m["classes"],
                   "abbreviations": m["abbreviations"]})

    c, s = result["classification"], result["selective"]
    print(f"  dev  n={result['n']['dev']:5d}  (thresholds + margin scale frozen here)")
    print(f"  cal  n={result['n']['cal']:5d}  tau* {s['tau']:.4f}"
          f"  cov {s['cal_coverage']:.3f}  risk {s['cal_risk']:.4f}")
    print(f"  test n={result['n']['test']:5d}  macro-AUROC {c['macro_auroc']:.4f}"
          f"  AUPRC {c['macro_auprc']:.4f}  F1 {c['macro_f1']:.4f}")
    print(f"       full-coverage risk {c['full_coverage_risk']:.4f}"
          f"   AURC {s['aurc']:.4f}   (oracle AURC {s['oracle_aurc']:.4f})")
    print(f"  {'alpha':>7}{'cov':>8}{'risk':>8}{'best':>8}{'oracle':>8}   ok")
    for e in result["alpha_sweep"]:
        star = "*" if e["alpha"] in ALPHA_LEVELS else " "
        print(f"  {e['alpha']:>6.2f}{star}{e['test_coverage']:>8.3f}"
              f"{e['test_risk']:>8.4f}{e['best_possible_coverage']:>8.3f}"
              f"{e['oracle_coverage']:>8.3f}   {'yes' if e['risk_respected'] else 'NO'}")

    (args.run / "eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(args.run / "risk_coverage.npz",
                        coverage=curve["coverage"], risk=curve["risk"],
                        oracle_risk=curve["oracle_risk"], tau=np.array(s["tau"]))
    print(f"  wrote {args.run / 'eval.json'}")


if __name__ == "__main__":
    main()
