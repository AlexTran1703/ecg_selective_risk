r"""Leave-one-source-out training of the reconstructed benchmark.

    .\.venv\Scripts\python.exe benchmark_mcu\scripts\r1_train.py
    .\.venv\Scripts\python.exe benchmark_mcu\scripts\r1_train.py --configs REF

Fifty configurations in two tiers, four LOSO rotations each.

reference tier (10 runs x 4)
    Each family at its published topology -- published depth, published
    channel schedule, published block internals -- adapted to 1-D and
    trained in float32 with no device constraint at all. This tier asks
    what the designs are worth as designs, which is a question about
    architecture rather than about a 512 KB part.

deployed tier (40 runs x 4)
    Two scaling methods crossed with two INT8 weight budgets:

        U-UC128   uniform width,       <= 128 KB
        U-C256    uniform width,       <= 256 KB
        J-UC128   joint width-depth,   <= 128 KB
        J-C256    joint width-depth,   <= 256 KB

    Both budgets also carry a 96 KB peak-activation ceiling, which on a
    128 KB part leaves room for stack, I/O buffers and firmware.

Everything outside the architecture is pinned: the same four-source
cohort, the same harmonised 13-label space, the same LOSO partitions,
the same 12 x 1000 input, loss, optimiser, schedule, epoch budget and
seed. Results go to their own directory, so the earlier benchmark's
cached predictions are left untouched.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
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

from ecgmcu.data import Cohort, Feeder, SOURCES, FS_OUT         # noqa: E402
from mcubench.footprint import (weight_bytes,                   # noqa: E402
                                float_fallback_frac)
from mcubench.models import (n_params, peak_activation_bytes,   # noqa: E402
                             macs)
from mcubench.reference import REFERENCE, build, fuse           # noqa: E402
from mcubench.scaling import BUDGETS, fit                       # noqa: E402

LENGTH = 1000
OUT = HERE / "results" / f"recon_{FS_OUT}hz"

#: config name -> (scaling method, budget name); REF is unconstrained
CONFIGS = {"REF": (None, None),
           "U-UC128": ("uniform", "UC-128"),
           "U-C256": ("uniform", "C-256"),
           "J-UC128": ("joint", "UC-128"),
           "J-C256": ("joint", "C-256")}

SLUG = {n: re.sub(r"[^a-z0-9]", "", n.lower()) for n in REFERENCE}


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


MANIFEST = HERE / "results" / "config_manifest.json"


@functools.lru_cache(maxsize=None)
def shape(name, config):
    """Width and depth multipliers for one configuration.

    The frozen manifest wins where it has an entry, because its
    activation ceilings were settled by building the firmware rather
    than by predicting its size, and a configuration that does not link
    is not a configuration. Falling back to the analytic search keeps
    this runnable before the manifest exists.

    Memoised: the bisection rebuilds the model fifty times and the
    answer does not depend on which source is held out, so without a
    cache the search would run once per fold for nothing.
    """
    method, budget = CONFIGS[config]
    if method is None:
        return 1.0, 1.0
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text())
        rec = man.get(f"{name}__{config}")
        if rec:
            return rec["width"], rec["depth"]
    return fit(name, BUDGETS[budget], method)


def train_one(name, config, held_out, coh, args, device):
    tag = f"{SLUG[name]}__{config}__{held_out}"
    dst = OUT / f"{tag}.npz"
    if dst.exists() and not args.force:
        print(f"[{tag}] cached", flush=True)
        return

    w, d = shape(name, config)
    model = fuse(build(name, w, d))
    stats = {"params": n_params(model),
             "peak_act": peak_activation_bytes(model, LENGTH),
             "macs": macs(model, LENGTH),
             "weight_bytes": weight_bytes(model),
             "fp32_frac": float_fallback_frac(model)}

    tr, va, te = coh.loso(held_out, seed=args.seed)
    mean, std = coh.stats(tr)
    # FCN keeps full temporal resolution throughout, so its activations
    # are an order of magnitude larger than any other family here; the
    # batch is cut for that case rather than cut for everybody.
    big = stats["peak_act"] > 64 * 1024
    bs = args.batch_size // 4 if big else args.batch_size

    def feed(rows, shuffle, seed=0):
        return Feeder(coh.sig_path, coh.y, rows, bs, mean, std,
                      device, shuffle=shuffle, seed=seed)

    f_tr, f_va, f_te = (feed(tr, True, args.seed), feed(va, False),
                        feed(te, False))
    y_va, y_te = f_va.labels(), f_te.labels()

    torch.manual_seed(args.seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(f_tr)))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight(coh.y[tr]).to(device))

    best, best_state, t0 = -1.0, None, time.time()
    for _ in range(args.epochs):
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

    p_te = predict(model, f_te, device)
    np.savez_compressed(
        dst, test_prob=p_te.astype(np.float32), test_y=y_te.astype(np.int8),
        test_rows=te, model=name, config=config, held_out=held_out,
        width=w, depth=d, val_auprc=best,
        minutes=(time.time() - t0) / 60, **stats)
    torch.save(model.state_dict(), OUT / f"{tag}.pt")
    print(f"[{tag}] w={w:.3f} d={d:.3f} params={stats['params']:,} "
          f"wb={stats['weight_bytes'] / 1024:.0f}KB "
          f"peak={stats['peak_act'] / 1024:.0f}KB  "
          f"AP {macro_auprc(y_te, p_te):.4f} "
          f"AUROC {macro_auroc(y_te, p_te):.4f} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(REFERENCE))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
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
    total = len(args.models) * len(args.configs) * len(args.sources)
    print(f"{total} runs on {device}, {FS_OUT} Hz, {LENGTH} samples\n",
          flush=True)
    for config in args.configs:
        for name in args.models:
            for src in args.sources:
                train_one(name, config, src, coh, args, device)

    rows = []
    for f in sorted(OUT.glob("*.npz")):
        d = np.load(f)
        rows.append({k: (d[k].item() if d[k].ndim == 0 else None)
                     for k in ("model", "config", "held_out", "width",
                               "depth", "params", "weight_bytes", "peak_act",
                               "macs", "fp32_frac", "val_auprc", "minutes")})
    (OUT / "index.json").write_text(json.dumps(rows, indent=1))
    print(f"\n-> {OUT}  ({len(rows)} runs)", flush=True)


if __name__ == "__main__":
    main()
