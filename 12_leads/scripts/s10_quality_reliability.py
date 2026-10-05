r"""RQ2c: does selective reliability survive signal-quality degradation?

    .\.venv\Scripts\python.exe 12_leads\scripts\s10_quality_reliability.py

RQ2 establishes two things separately: impaired records are harder
(s6_quality_robustness.py), and confidence ranks errors to some degree
(s9_reliability.py). Neither says whether those two facts interact, and
the interaction is the one a deployment actually cares about.

    dE-AURC = E-AURC(impaired) - E-AURC(clean)

Two readings, both informative:

    dE-AURC > 0   degradation produces errors the model is ALSO worse at
                  recognising -- a compounding reliability failure, and
                  the case where abstention cannot be relied on
    dE-AURC ~ 0   discrimination falls but the confidence ordering holds,
                  so abstaining on low-confidence records still works

Restricted to the three structural indicators, because those are the ones
RQ2b found consistently associated with lower discrimination; running it
on indicators with no established association would be asking whether an
effect that is not there behaves in some particular way.

Strata are formed within held-out source, as in s6. E-AURC is scale-free
in a way AUPRC is not -- it is an excess over the stratum's own oracle --
so prevalence matching is not required here, but the stratum sizes are
reported so the reader can see what each contrast rests on.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from ecgmcu.data import SOURCES, FS_OUT                        # noqa: E402

S1D = HERE / "results" / "study1"
S2D = HERE / "results" / ("study2" if FS_OUT == 250
                          else f"study2_{FS_OUT}hz")
ALPHA_RE = re.compile(r"a[0-9.]+__")
ORDER = ["tiny", "dscnn", "mobilenet", "mbconv", "tcn", "resnet"]
STRUCTURAL = [
    ("flat_leads", "flat/clipped lead"),
    ("qrs_fail", "QRS failure"),
    ("rr_implaus", "RR implausibility"),
]
QRS_LO, QRS_HI = 5, 30
MIN_N = 100          # records per stratum before a contrast is attempted


def jaccard_loss(y, p, thr=0.5):
    yh = (p >= thr).astype(np.int8)
    inter = (yh & y).sum(1)
    union = (yh | y).sum(1)
    return 1.0 - np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def eaurc(loss: np.ndarray, conf: np.ndarray) -> float:
    """Excess area under the risk-coverage curve, over the own oracle."""
    n = loss.size
    k = np.arange(1, n + 1)
    risk = np.cumsum(loss[np.argsort(-conf, kind="stable")]) / k
    oracle = np.cumsum(np.sort(loss)) / k
    return float(risk.mean() - oracle.mean())


def load_runs():
    runs = {}
    for f in sorted(S2D.glob("*.npz")):
        if "__" not in f.stem or ALPHA_RE.search(f.stem):
            continue
        d = np.load(f, allow_pickle=True)
        p = d["test_prob"].astype(np.float64)
        y = d["test_y"].astype(np.int8)
        runs[(str(d["model"]), str(d["held_out"]))] = {
            "loss": jaccard_loss(y, p),
            "conf": np.abs(p - 0.5).min(1),
            "rows": d["test_rows"]}
    return runs


def quality_frame() -> pd.DataFrame:
    q = pd.read_parquet(S1D / "quality.parquet")
    q["qrs_fail"] = ((q.n_qrs < QRS_LO) | (q.n_qrs > QRS_HI)).astype(int)
    idx = pd.read_csv(HERE / "data" / "index.csv")
    cols = ["source", "record", "qrs_fail", "flat_leads", "rr_implaus"]
    return idx.merge(q[cols], on=["source", "record"], how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    dst = S2D / "quality_reliability.json"
    if dst.exists() and not args.force:
        print(f"cached: {dst}")
        return

    runs = load_runs()
    if not runs:
        print(f"no runs in {S2D}")
        return
    qf = quality_frame()
    models = [m for m in ORDER if any(k[0] == m for k in runs)]
    rows = []

    for col, label in STRUCTURAL:
        rng = np.random.default_rng(0)
        # Strata are identical across models (same test rows per source),
        # so one draw per replicate serves all six and the pooled effect
        # stays paired.
        setup = {}
        for s in SOURCES:
            key = (models[0], s)
            if key not in runs:
                continue
            v = qf.iloc[runs[key]["rows"]][col].to_numpy(dtype=float)
            clean = np.flatnonzero(v <= 0)
            imp = np.flatnonzero(v > 0)
            if clean.size >= MIN_N and imp.size >= MIN_N:
                setup[s] = (clean, imp)
        if not setup:
            print(f"  [{label}] no usable strata")
            continue

        per_model = {m: np.full(args.reps, np.nan) for m in models}
        clean_e = {m: np.full(args.reps, np.nan) for m in models}
        imp_e = {m: np.full(args.reps, np.nan) for m in models}
        pooled = np.full(args.reps, np.nan)

        for b in range(args.reps):
            draws = {}
            for s, (clean, imp) in setup.items():
                draws[s] = (clean[rng.integers(0, clean.size, clean.size)],
                            imp[rng.integers(0, imp.size, imp.size)])
            vals = {}
            for m in models:
                dd, ce, ie = [], [], []
                for s, (ci, ii) in draws.items():
                    r = runs[(m, s)]
                    e_c = eaurc(r["loss"][ci], r["conf"][ci])
                    e_i = eaurc(r["loss"][ii], r["conf"][ii])
                    if np.isfinite(e_c) and np.isfinite(e_i):
                        dd.append(e_i - e_c)
                        ce.append(e_c)
                        ie.append(e_i)
                if dd:
                    vals[m] = float(np.mean(dd))
                    per_model[m][b] = vals[m]
                    clean_e[m][b] = float(np.mean(ce))
                    imp_e[m][b] = float(np.mean(ie))
            if vals:
                pooled[b] = float(np.mean(list(vals.values())))
            if (b + 1) % 250 == 0:
                print(f"    {label}: replicate {b + 1}/{args.reps}",
                      flush=True)

        for m in models:
            ok = per_model[m][np.isfinite(per_model[m])]
            if ok.size < args.reps // 4:
                continue
            rows.append({
                "model": m, "factor": col, "label": label,
                "delta_eaurc": float(ok.mean()),
                "lo": float(np.percentile(ok, 2.5)),
                "hi": float(np.percentile(ok, 97.5)),
                "eaurc_clean": float(np.nanmean(clean_e[m])),
                "eaurc_impaired": float(np.nanmean(imp_e[m])),
                "sources_used": len(setup),
                "n_clean": int(sum(c.size for c, _ in setup.values())),
                "n_impaired": int(sum(i.size for _, i in setup.values()))})
        okp = pooled[np.isfinite(pooled)]
        if okp.size:
            rows.append({
                "model": "__pooled__", "factor": col, "label": label,
                "delta_eaurc": float(okp.mean()),
                "lo": float(np.percentile(okp, 2.5)),
                "hi": float(np.percentile(okp, 97.5)),
                "sources_used": len(setup),
                "n_clean": int(sum(c.size for c, _ in setup.values())),
                "n_impaired": int(sum(i.size for _, i in setup.values()))})
            print(f"  [{label:18s}] dE-AURC {okp.mean():+.4f} "
                  f"({np.percentile(okp, 2.5):+.4f}, "
                  f"{np.percentile(okp, 97.5):+.4f})   "
                  f"{len(setup)} sources", flush=True)
    dst.write_text(json.dumps(rows, indent=1))
    print(f"\n-> {dst}  ({len(rows)} rows, B={args.reps}, seed=0)")


if __name__ == "__main__":
    main()
