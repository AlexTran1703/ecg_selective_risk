r"""Study II: leave-one-source-out benchmark of the six lightweight encoders.

    .\.venv\Scripts\python.exe 12_leads\scripts\s2_train.py
    .\.venv\Scripts\python.exe 12_leads\scripts\s2_train.py --models tiny --sources PTBXL

Everything except the encoder is held fixed: the same 250 Hz whole-record
input, labels, partitions, loss, optimiser, schedule and epoch budget. The
point of the benchmark is to attribute differences to architecture, which
only works if nothing else moves.

Predictions are cached per run, so every downstream question -- quality
stratification, selective risk, bootstrap intervals -- is answered from
stored probabilities and never costs a retrain.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from ecgmcu.data import Cohort, Feeder, SOURCES                  # noqa: E402
from ecgmcu.models import ZOO, build, n_params                   # noqa: E402

from ecgmcu.data import FS_OUT                                  # noqa: E402

OUT = HERE / "results" / ("study2" if FS_OUT == 250
                         else f"study2_{FS_OUT}hz")
POS_CAP = 50.0


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


def pos_weight(y):
    pos = y.sum(0).astype(np.float64)
    neg = y.shape[0] - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(np.clip(w, 1.0, POS_CAP), dtype=torch.float32)


@torch.no_grad()
def predict(model, feeder, device, amp=True):
    model.eval()
    out = []
    for x, _ in feeder:
        with torch.autocast("cuda", torch.bfloat16,
                            enabled=amp and device == "cuda"):
            out.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 13))


def train_one(name, held_out, coh, args, device):
    tag = (f"{name}__{held_out}" if args.alpha == 1.0
           else f"{name}a{args.alpha:g}__{held_out}")
    dst = OUT / f"{tag}.npz"
    if dst.exists() and not args.force:
        print(f"[{tag}] cached, skipping")
        return

    tr, va, te = coh.loso(held_out, seed=args.seed)
    mean, std = coh.stats(tr)
    def feed(rows, shuffle, seed=0):
        return Feeder(coh.sig_path, coh.y, rows, args.batch_size, mean, std,
                      device, shuffle=shuffle, seed=seed)

    f_tr, f_va, f_te = feed(tr, True, args.seed), feed(va, False), feed(te, False)
    y_va, y_te = f_va.labels(), f_te.labels()

    torch.manual_seed(args.seed)
    model = build(name, alpha=args.alpha).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(f_tr)))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight(coh.y[tr]).to(device))

    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(args.epochs):
        model.train()
        run = seen = 0
        for x, y in f_tr:
            with torch.autocast("cuda", torch.bfloat16,
                                enabled=device == "cuda"):
                loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        score = macro_auprc(y_va, predict(model, f_va, device))
        star = ""
        if score > best:
            best, star = score, "  *"
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"[{tag}] epoch {ep:2d}  loss {run / max(seen, 1):.4f}"
              f"  val mAUPRC {score:.4f}{star}", flush=True)

    model.load_state_dict(best_state)
    p_te = predict(model, f_te, device)
    np.savez_compressed(
        dst, test_prob=p_te.astype(np.float32), test_y=y_te.astype(np.int8),
        test_rows=te, model=name, held_out=held_out, alpha=args.alpha,
        params=n_params(model), val_auprc=best,
        minutes=(time.time() - t0) / 60)
    torch.save(model.state_dict(), OUT / f"{tag}.pt")
    print(f"[{tag}] test macro-AUPRC {macro_auprc(y_te, p_te):.4f}"
          f"  AUROC {macro_auroc(y_te, p_te):.4f}"
          f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ZOO))
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="channel width multiplier (scaling study)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    coh = Cohort()
    print(f"cohort {coh.y.shape[0]:,} records, {len(coh.classes)} classes, "
          f"{device}")
    t0 = time.time()
    for name in args.models:
        for src in args.sources:
            train_one(name, src, coh, args, device)
    print(f"\nall runs done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
