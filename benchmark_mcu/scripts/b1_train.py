r"""Leave-one-source-out training of ten efficient families, two budgets.

    .\.venv\Scripts\python.exe benchmark_mcu\scripts\b1_train.py
    .\.venv\Scripts\python.exe benchmark_mcu\scripts\b1_train.py --budgets compact

The comparison is only meaningful if the architecture is the only thing
that moves. Everything else is pinned: the same four-source cohort, the
same harmonised 13-label space, the same leave-one-source-out
partitions, the same 12 x 1000 input, the same loss, optimiser,
schedule, epoch budget and seed. The data layer is imported from the
cross-source study rather than re-implemented, so the two papers cannot
drift apart on cohort definition.

What is *not* held fixed is channel width, deliberately. Each family is
scaled to the largest version of itself that meets the budget
(`fit_to_budget`), because the question is what a design buys for a
fixed Flash and SRAM allowance, not how the families rank at some
arbitrary common width that suits one of them.

Two budgets, both with the same SRAM ceiling:

    compact    int8 weights <=  32 KB, peak activation <= 96 KB
    standard   int8 weights <= 128 KB, peak activation <= 96 KB

96 KB leaves room on a 128 KB part for stack, I/O buffers and firmware.
It is a design target, not a measurement: a model counts as deployed
only after it runs correctly on the physical board (b4_deploy.py).
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

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "benchmark_mcu"
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(ROOT / "12_leads" / "src"))

from ecgmcu.data import Cohort, Feeder, SOURCES, FS_OUT          # noqa: E402
from mcubench.models import (ZOO, NICE, fit_to_budget, n_params,  # noqa: E402
                             peak_activation_bytes, macs)

BUDGETS = {"compact": (32 * 1024, 96 * 1024),
           "standard": (128 * 1024, 96 * 1024)}
OUT = HERE / "results" / f"train_{FS_OUT}hz"
LENGTH = 1000


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


def pos_weight(y, cap=50.0):
    pos = y.sum(0).astype(np.float64)
    neg = y.shape[0] - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(np.clip(w, 1.0, cap), dtype=torch.float32)


@torch.no_grad()
def predict(model, feeder, device):
    model.eval()
    out = []
    for x, _ in feeder:
        with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
            out.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 13))


def train_one(name, budget, held_out, coh, args, device):
    tag = f"{name}__{budget}__{held_out}"
    dst = OUT / f"{tag}.npz"
    if dst.exists() and not args.force:
        print(f"[{tag}] cached")
        return

    pb, ab = BUDGETS[budget]
    width, model, stats = fit_to_budget(name, pb, ab, LENGTH)
    if width is None:
        print(f"[{tag}] does not fit the budget, skipped")
        return

    tr, va, te = coh.loso(held_out, seed=args.seed)
    mean, std = coh.stats(tr)

    def feed(rows, shuffle, seed=0):
        return Feeder(coh.sig_path, coh.y, rows, args.batch_size, mean, std,
                      device, shuffle=shuffle, seed=seed)

    f_tr, f_va, f_te = feed(tr, True, args.seed), feed(va, False), feed(te, False)
    y_va, y_te = f_va.labels(), f_te.labels()

    torch.manual_seed(args.seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(f_tr)))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight(coh.y[tr]).to(device))

    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(args.epochs):
        model.train()
        for x, y in f_tr:
            with torch.autocast("cuda", torch.bfloat16,
                                enabled=device == "cuda"):
                loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        score = macro_auprc(y_va, predict(model, f_va, device))
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # RepViT is folded before scoring, so the reported predictions come
    # from the graph that would actually be exported, not the wider
    # training-time one.
    if hasattr(model, "reparameterise"):
        model.reparameterise()
        stats = {"params": n_params(model),
                 "peak_act": peak_activation_bytes(model.cpu(), LENGTH),
                 "macs": macs(model.cpu(), LENGTH)}
        model = model.to(device)

    p_te = predict(model, f_te, device)
    np.savez_compressed(
        dst, test_prob=p_te.astype(np.float32), test_y=y_te.astype(np.int8),
        test_rows=te, model=name, budget=budget, held_out=held_out,
        width=width, params=stats["params"], peak_act=stats["peak_act"],
        macs=stats["macs"], val_auprc=best,
        minutes=(time.time() - t0) / 60)
    torch.save(model.state_dict(), OUT / f"{tag}.pt")
    print(f"[{tag}] w={width:.2f} params={stats['params']} "
          f"peak={stats['peak_act'] / 1024:.1f}KB  "
          f"AUPRC {macro_auprc(y_te, p_te):.4f} "
          f"AUROC {macro_auroc(y_te, p_te):.4f} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ZOO))
    ap.add_argument("--budgets", nargs="+", default=list(BUDGETS))
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    coh = Cohort()
    total = len(args.models) * len(args.budgets) * len(args.sources)
    print(f"{total} runs on {device}, {FS_OUT} Hz input, "
          f"{LENGTH} samples\n")
    for budget in args.budgets:
        for name in args.models:
            for src in args.sources:
                train_one(name, budget, src, coh, args, device)

    rows = []
    for f in sorted(OUT.glob("*.npz")):
        d = np.load(f)
        rows.append({k: (d[k].item() if d[k].ndim == 0 else None)
                     for k in ("model", "budget", "held_out", "width",
                               "params", "peak_act", "macs", "val_auprc",
                               "minutes")})
    (OUT / "index.json").write_text(json.dumps(rows, indent=1))
    print(f"\n-> {OUT}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
