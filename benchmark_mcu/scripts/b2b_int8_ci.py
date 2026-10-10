r"""Paired bootstrap confidence intervals for the INT8 - FP32 difference.

    .\.venv\Scripts\python.exe benchmark_mcu\scripts\b2b_int8_ci.py

Figure 4 previously drew the range of the per-source differences as its
interval. That is the wrong quantity. The minimum and maximum over four
sources describe how much the effect varies between those four
datasets; they say nothing about how precisely the mean effect is
estimated, and they cannot shrink with more records.

What the figure needs is an interval on

    mean_j [ AUPRC_int8(m, b, j) - AUPRC_fp32(m, b, j) ]

so the estimand is the equal-source mean difference. The resampling has
to preserve the pairing: FP32 and INT8 scored the *same* records, and
most of the variance in each arm is shared. Resampling them
independently would discard that and inflate the interval enormously.

So each replicate draws one set of record indices per held-out source,
scores both precisions on those same indices, takes the difference, and
averages the four sources. Percentiles 2.5 and 97.5 of the replicate
distribution give the interval.

Runs from the cached probability pairs written by b2_int8.py, so it
needs no requantisation and no device.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "benchmark_mcu"
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(ROOT / "12_leads" / "src"))

from ecgmcu.data import SOURCES, FS_OUT                          # noqa: E402
from mcubench.models import ZOO, NICE                            # noqa: E402

INT8 = HERE / "results" / f"int8_{FS_OUT}hz"
BUDGETS = ["compact", "standard"]


def ap_fast(y, s):
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    tp = np.cumsum(y)
    if not tp[-1]:
        return np.nan
    end = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    te = tp[end]
    return float((np.diff(np.r_[0.0, te / tp[-1]]) * (te / (end + 1.0))).sum())


def macro_ap(y, p):
    v = [ap_fast(y[:, c], p[:, c]) for c in range(y.shape[1])
         if 0 < y[:, c].sum() < y.shape[0]]
    return float(np.mean(v)) if v else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=list(ZOO))
    ap.add_argument("--budgets", nargs="+", default=list(BUDGETS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    dst = INT8 / "int8_ci.json"
    if dst.exists() and not args.force:
        print(f"cached: {dst}")
        return

    # Each configuration's interval depends only on its own cached
    # probability pairs, so a scoped run recomputes those and keeps the
    # rest. Overwriting the file with just the subset would discard the
    # others -- the same data-loss bug b2_int8 and b3_deploy had.
    keep = {}
    if dst.exists():
        try:
            keep = {(r["model"], r["budget"]): r
                    for r in json.loads(dst.read_text())}
        except json.JSONDecodeError:
            keep = {}

    rows = []
    for budget in BUDGETS:
        for m in ZOO:
            if m not in args.models or budget not in args.budgets:
                continue
            folds = []
            for src in SOURCES:
                f = INT8 / f"{m}__{budget}__{src}_prob.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                folds.append((d["y"].astype(np.int8),
                              d["prob_f32"].astype(np.float64),
                              d["prob"].astype(np.float64)))
            if len(folds) != len(SOURCES):
                continue

            point = float(np.mean([macro_ap(y, p8) - macro_ap(y, p32)
                                   for y, p32, p8 in folds]))
            per_src = [macro_ap(y, p8) - macro_ap(y, p32)
                       for y, p32, p8 in folds]
            rng = np.random.default_rng(args.seed)
            reps = np.empty(args.reps)
            for b in range(args.reps):
                d_ = []
                for y, p32, p8 in folds:
                    # One index set per source, used for both precisions:
                    # the pairing is the point of the design.
                    i = rng.integers(0, y.shape[0], y.shape[0])
                    d_.append(macro_ap(y[i], p8[i]) - macro_ap(y[i], p32[i]))
                reps[b] = float(np.mean(d_))
            rows.append({
                "model": m, "budget": budget,
                "d_auprc": point,
                "lo": float(np.percentile(reps, 2.5)),
                "hi": float(np.percentile(reps, 97.5)),
                "src_min": float(np.min(per_src)),
                "src_max": float(np.max(per_src)),
                "reps": args.reps})
            r = rows[-1]
            print(f"  [{NICE[m]:20s} {budget:8s}] {r['d_auprc']:+.4f}  "
                  f"CI ({r['lo']:+.4f}, {r['hi']:+.4f})  "
                  f"source range ({r['src_min']:+.4f}, {r['src_max']:+.4f})",
                  flush=True)

    keep.update({(r["model"], r["budget"]): r for r in rows})
    allrec = [keep[(m, b)] for b in BUDGETS for m in ZOO if (m, b) in keep]
    dst.write_text(json.dumps(allrec, indent=1))
    n_sig = sum(1 for r in allrec if r["hi"] < 0)
    print(f"\n{len(rows)} recomputed, {len(allrec)} configurations on file; "
          f"{n_sig} have a 95% interval entirely below zero")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
