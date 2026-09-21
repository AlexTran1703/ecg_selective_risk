r"""Train a 5-member deep ensemble and cache its predictions.

    .\.venv\Scripts\python.exe scripts\02_train_ensemble.py --dataset code15
    .\.venv\Scripts\python.exe scripts\02_train_ensemble.py --dataset ptbxl

Members share the frozen patient-level split and every hyperparameter; only
the seed differs, which sets initialisation and batch order.  That is what
makes the spread across members interpretable as model uncertainty rather
than as a mixture of model and data-split variance.

CODE-test is touched exactly once, at the end, to write predictions.  It is
never used for training, validation, threshold fitting or any choice made
here -- the decision thresholds come from the CODE-15% validation partition
and are frozen before the test set is read.

Resumable: a seed whose predictions already exist is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecguq import model as M                                       # noqa: E402
from ecguq.data import LABELS                                      # noqa: E402
from ecguq.metrics import macro_auprc, macro_auroc                 # noqa: E402
from ecguq.feed import BatchFeeder, train_statistics               # noqa: E402
from ecguq.train import (load_prepared, positive_weights,          # noqa: E402
                         predict_probs, train_member)
from ecguq.uncertainty import fit_thresholds                       # noqa: E402

N_MEMBERS = 5


def rows_for(prep, which: str) -> np.ndarray:
    meta = prep.meta
    if "is_val" in meta:                              # CODE-15%
        present = meta["present"] if "present" in meta else np.ones(
            len(meta["is_val"]), bool)
        val = meta["is_val"].astype(bool)
        return np.flatnonzero(present & (val if which == "val" else ~val))
    split = meta["split"].astype(str)                 # PTB-XL
    return np.flatnonzero(split == which)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="code15", choices=["code15", "ptbxl"])
    ap.add_argument("--prepared", type=Path, default=ROOT / "data" / "prepared")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "ensembles")
    ap.add_argument("--members", type=int, default=N_MEMBERS)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--prefetch", type=int, default=3,
                    help="batches buffered by the single reader thread")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model", default="plain",
                    choices=["plain", "resnet_lite"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    prep = load_prepared(args.prepared / args.dataset)
    tr, va = rows_for(prep, "train"), rows_for(prep, "val")
    print(f"{args.dataset}: train {tr.size}  val {va.size}  "
          f"({prep.manifest['n_records']} records, {device})")

    mean, std = train_statistics(prep.path / "signals.npy", tr)
    print(f"  train-partition normalisation: mean {mean:.4f}  sd {std:.4f}")

    def feeder(rows, shuffle, seed=0, source=None):
        """Block-shuffled batches read straight off the memmap.

        No worker processes: one prefetch thread overlaps the NumPy gather
        with the GPU, which is all the overlap available when the gather is
        the only expensive host-side step, and costs a few batches of memory
        instead of six processes mapping a 34 GB file.
        """
        src = source if source is not None else prep
        return BatchFeeder(src.path / "signals.npy", src.y, rows,
                           args.batch_size, mean, std, device,
                           shuffle=shuffle, seed=seed)

    val_feeder = feeder(va, False)
    pw = positive_weights(prep.y[tr])
    print("  positive weights: "
          + ", ".join(f"{l}={w:.0f}" for l, w in zip(LABELS, pw.tolist())))

    # Test set is assembled once and never influences training.
    test_prep = load_prepared(args.prepared / "code_test") \
        if args.dataset == "code15" else None
    test_rows = (np.arange(test_prep.signals.shape[0])
                 if test_prep is not None else rows_for(prep, "test"))
    test_feeder = (feeder(test_rows, False, source=test_prep)
                   if test_rows.size else None)

    t0 = time.time()
    for seed in range(args.members):
        dst = out / f"member{seed}.npz"
        if dst.exists():
            print(f"[{seed + 1}/{args.members}] seed {seed}: already done")
            continue
        print(f"[{seed + 1}/{args.members}] seed {seed} "
              f"({(time.time() - t0) / 60:.0f} min elapsed)", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = M.build(n_in=12, n_classes=len(LABELS), kind=args.model)
        net = train_member(net, feeder(tr, True, seed), val_feeder, device,
                           args.epochs, pw, lr=args.lr)
        p_val = predict_probs(net, val_feeder, device)
        p_test = (predict_probs(net, test_feeder, device)
                  if test_feeder is not None else np.zeros((0, len(LABELS))))
        np.savez_compressed(dst, val=p_val.astype(np.float32),
                            test=p_test.astype(np.float32), seed=seed)
        torch.save(net.state_dict(), out / f"member{seed}.pt")
        print(f"    val macro-AUROC {macro_auroc(prep.y[va], p_val):.4f}  "
              f"macro-AUPRC {macro_auprc(prep.y[va], p_val):.4f}", flush=True)

    # Ensemble, and thresholds fitted on validation only.
    members = sorted(out.glob("member*.npz"))
    pv = np.stack([np.load(f)["val"] for f in members])
    pt = np.stack([np.load(f)["test"] for f in members])
    thresholds = fit_thresholds(prep.y[va], pv.mean(axis=0))
    np.savez_compressed(out / "ensemble.npz", val=pv, test=pt,
                        thresholds=thresholds, val_rows=va,
                        y_val=prep.y[va], mean=mean, std=std)
    (out / "config.json").write_text(json.dumps({
        "dataset": args.dataset, "members": len(members),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "split": prep.manifest.get("split"),
        "split_seed": prep.manifest.get("split_seed"),
        "normalisation": {"mean": mean, "std": std,
                          "source": "train partition only"},
        "thresholds": dict(zip(LABELS, thresholds.tolist())),
        "threshold_rule": "max F1 on the CODE-15% validation partition, "
                          "frozen before the test set is read",
        "labels": list(LABELS),
    }, indent=2), encoding="utf-8")

    pbar = pv.mean(axis=0)
    print(f"\nensemble of {len(members)}: val macro-AUROC "
          f"{macro_auroc(prep.y[va], pbar):.4f}  macro-AUPRC "
          f"{macro_auprc(prep.y[va], pbar):.4f}")
    print("  thresholds: "
          + ", ".join(f"{l}={t:.3f}" for l, t in zip(LABELS, thresholds)))
    print(f"  wrote {out / 'ensemble.npz'}  ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
