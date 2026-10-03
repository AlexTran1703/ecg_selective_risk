r"""Study II-B / RQ3: association between signal quality and discrimination.

    .\.venv\Scripts\python.exe 12_leads\scripts\s6_quality_robustness.py

Study I says how the four sources differ. On its own that is dataset
trivia. This asks which of the six measured impairments are associated
with lower diagnostic discrimination, and for which architectures.

    dAUPRC = macro AUPRC(higher impairment) - macro AUPRC(lower impairment)

    dAUPRC < 0  higher impairment associated with worse discrimination
    dAUPRC ~ 0  little observed difference
    dAUPRC > 0  the higher-impairment stratum happened to score better

The last case is NOT evidence that an impairment helps. Nothing here is
causal: the strata differ in whatever else travels with the indicator.

Four things make the contrast defensible.

Direction is pre-specified from signal physics, never from the outcome.
More HF power, more baseline drift, more mains power, a flat or clipped
lead, a failed QRS detection, an implausible RR interval -- each is the
higher-impairment side by definition of what the metric measures. Letting
the data choose which side is "impaired" would be marking our own exam.

Strata are formed within each held-out source. A single global threshold
would put most of the high-impairment group in one dataset and recreate
source shift inside RQ3.

Prevalence is matched per diagnosis. AUPRC moves with positive-class
prevalence, so for every class the two strata are sampled to a common
n+ = min(N+_H, N+_L) and n- = min(N-_H, N-_L). Matching only the number
of labels a record carries would leave the diagnostic *mix* free to
differ, which is the same confound in a thinner disguise.

Sources are weighted equally, inside each bootstrap replicate, so the
largest source cannot dominate the pooled effect.
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
OUT = S2D
ALPHA_RE = re.compile(r"a[0-9.]+__")
ORDER = ["tiny", "dscnn", "mobilenet", "mbconv", "tcn", "resnet"]

# (column, label, kind). "higher impairment" is always the larger value,
# fixed here from what the metric physically means.
FACTORS = [
    ("hf_rel", "HF noise", "continuous"),
    ("bw_rel", "baseline wander", "continuous"),
    ("pli_rel", "mains", "continuous"),
    ("flat_leads", "flat/clipped lead", "binary"),
    ("qrs_fail", "QRS failure", "binary"),
    ("rr_implaus", "RR implausibility", "binary"),
]
# A 10 s record holds 5-30 QRS complexes at 30-180 min-1. Outside that the
# detector has failed, or the record is unusable for rhythm anyway.
QRS_LO, QRS_HI = 5, 30
# Support rule, fixed before reading any contrast: a diagnosis enters
# only if both strata can supply this many positives and negatives.
# AP from a handful of positives is a decimal, not an estimate.
MIN_POS, MIN_NEG = 20, 20
MIN_CLASSES = 4                    # classes needed before a macro is taken
NEG_CAP = 4000                     # bound per class; applied to BOTH arms


def ap_fast(y: np.ndarray, s: np.ndarray) -> float:
    o = np.argsort(-s, kind="mergesort")
    y = y[o]
    s = s[o]
    tp = np.cumsum(y)
    n_pos = tp[-1]
    if not n_pos:
        return np.nan
    end = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    tp_e = tp[end]
    return float((np.diff(np.r_[0.0, tp_e / n_pos]) * (tp_e / (end + 1.0))
                  ).sum())


def load_runs():
    runs = {}
    for f in sorted(S2D.glob("*.npz")):
        if "__" not in f.stem or ALPHA_RE.search(f.stem):
            continue
        d = np.load(f, allow_pickle=True)
        runs[(str(d["model"]), str(d["held_out"]))] = {
            "p": d["test_prob"].astype(np.float64),
            "y": d["test_y"].astype(np.int8), "rows": d["test_rows"]}
    return runs


def quality_frame() -> pd.DataFrame:
    q = pd.read_parquet(S1D / "quality.parquet")
    q["qrs_fail"] = ((q.n_qrs < QRS_LO) | (q.n_qrs > QRS_HI)).astype(int)
    idx = pd.read_csv(HERE / "data" / "index.csv")
    cols = ["source", "record", "n_qrs", "qrs_fail",
            *[f for f, _, _ in FACTORS if f != "qrs_fail"]]
    return idx.merge(q[cols], on=["source", "record"], how="left")


def split(v: np.ndarray, kind: str):
    """(lower-impairment index, higher-impairment index) within a source.

    Continuous indicators are cut at the within-source quartiles, so the
    contrast is the cleanest quarter against the most impaired quarter.
    Binary indicators contrast absent against present.
    """
    v = np.asarray(v, dtype=float)
    if kind == "binary":
        return np.flatnonzero(v <= 0), np.flatnonzero(v > 0)
    ok = np.isfinite(v)
    if ok.sum() < 8:
        return np.array([], int), np.array([], int)
    q25, q75 = np.nanpercentile(v, [25, 75])
    if not np.isfinite(q25) or q25 == q75:
        return np.array([], int), np.array([], int)
    return np.flatnonzero(ok & (v <= q25)), np.flatnonzero(ok & (v >= q75))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--min-support", type=int, default=20,
                    help="positives and negatives required per class "
                         "per stratum; 20 is primary, 10 is the "
                         "sensitivity analysis")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    min_sup = args.min_support
    tag = "" if args.min_support == 20 else f"_s{args.min_support}"
    dst = OUT / f"quality_robustness{tag}.json"
    if dst.exists() and not args.force:
        print(f"cached: {dst}")
        return
    runs = load_runs()
    if not runs:
        print(f"no runs in {S2D}")
        return
    qf = quality_frame()
    models = [m for m in ORDER if any(k[0] == m for k in runs)]
    n_cls = runs[(models[0], SOURCES[0])]["y"].shape[1]
    rows = []

    for col, label, kind in FACTORS:
        rng = np.random.default_rng(0)
        # Strata and per-class pools depend only on labels and quality, so
        # they are shared by every architecture: the same records are
        # compared for all six, which is what makes the pooled effect
        # paired rather than six independent experiments.
        pools = {}
        for s in SOURCES:
            key = (models[0], s)
            if key not in runs:
                continue
            r = runs[key]
            sub = qf.iloc[r["rows"]]
            lo, hi = split(sub[col].to_numpy(), kind)
            if lo.size < min_sup or hi.size < min_sup:
                continue
            y = r["y"]
            per_class = []
            for c in range(n_cls):
                yc = y[:, c]
                Lp, Ln = lo[yc[lo] == 1], lo[yc[lo] == 0]
                Hp, Hn = hi[yc[hi] == 1], hi[yc[hi] == 0]
                npos = min(Lp.size, Hp.size)
                nneg = min(Ln.size, Hn.size, NEG_CAP)
                if npos >= min_sup and nneg >= min_sup:
                    per_class.append((c, Lp, Ln, Hp, Hn, npos, nneg))
            if len(per_class) >= MIN_CLASSES:
                pools[s] = per_class
        if not pools:
            print(f"  [{label}] no usable strata")
            continue

        per_model = {m: np.full(args.reps, np.nan) for m in models}
        base_model = {m: np.full(args.reps, np.nan) for m in models}
        pooled = np.full(args.reps, np.nan)

        for b in range(args.reps):
            # One draw per source and class, reused by every model.
            draws = {}
            for s, per_class in pools.items():
                dd = []
                for (c, Lp, Ln, Hp, Hn, npos, nneg) in per_class:
                    dd.append((
                        c,
                        Lp[rng.integers(0, Lp.size, npos)],
                        Ln[rng.integers(0, Ln.size, nneg)],
                        Hp[rng.integers(0, Hp.size, npos)],
                        Hn[rng.integers(0, Hn.size, nneg)],
                        npos, nneg))
                draws[s] = dd

            vals_by_model = {}
            for m in models:
                d_src, base_src = [], []
                for s, dd in draws.items():
                    p = runs[(m, s)]["p"]
                    dl, bl = [], []
                    for (c, li, lj, hi_, hj, npos, nneg) in dd:
                        yv = np.r_[np.ones(npos, np.int8),
                                   np.zeros(nneg, np.int8)]
                        apL = ap_fast(yv, np.r_[p[li, c], p[lj, c]])
                        apH = ap_fast(yv, np.r_[p[hi_, c], p[hj, c]])
                        if np.isfinite(apL) and np.isfinite(apH):
                            dl.append(apH - apL)
                            bl.append(apL)
                    if dl:
                        d_src.append(float(np.mean(dl)))
                        base_src.append(float(np.mean(bl)))
                if d_src:
                    # Equal weight per source, not per record.
                    vals_by_model[m] = float(np.mean(d_src))
                    per_model[m][b] = vals_by_model[m]
                    base_model[m][b] = float(np.mean(base_src))
            if vals_by_model:
                pooled[b] = float(np.mean(list(vals_by_model.values())))
            if (b + 1) % 200 == 0:
                print(f"    {label}: replicate {b + 1}/{args.reps}",
                      flush=True)

        for m in models:
            ok = per_model[m][np.isfinite(per_model[m])]
            if ok.size < args.reps // 4:
                continue
            base = float(np.nanmean(base_model[m]))
            rows.append({
                "model": m, "factor": col, "label": label, "kind": kind,
                "delta": float(ok.mean()),
                "lo": float(np.percentile(ok, 2.5)),
                "hi": float(np.percentile(ok, 97.5)),
                "baseline": base,
                "relative": float(ok.mean() / base) if base else np.nan,
                "n_low": int(sum(c[1].size + c[2].size
                                 for pc in pools.values() for c in pc)),
                "n_high": int(sum(c[3].size + c[4].size
                                  for pc in pools.values() for c in pc)),
                "classes_used": int(np.mean([len(pc)
                                             for pc in pools.values()])),
                "contrasts": int(sum(len(pc) for pc in pools.values())),
                "sources_used": len(pools),
                "sources": ",".join(sorted(pools)),
                "reps_used": int(ok.size)})
        okp = pooled[np.isfinite(pooled)]
        if okp.size:
            rows.append({
                "model": "__pooled__", "factor": col, "label": label,
                "kind": kind, "delta": float(okp.mean()),
                "lo": float(np.percentile(okp, 2.5)),
                "hi": float(np.percentile(okp, 97.5)),
                "sources_used": len(pools),
                "contrasts": int(sum(len(pc) for pc in pools.values())),
                "sources": ",".join(sorted(pools)),
                "reps_used": int(okp.size)})
            print(f"  [{label:18s}] pooled {okp.mean():+.4f} "
                  f"({np.percentile(okp, 2.5):+.4f}, "
                  f"{np.percentile(okp, 97.5):+.4f})   "
                  f"{len(pools)} sources", flush=True)
    dst.write_text(json.dumps(rows, indent=1))
    print(f"\n-> {dst}  ({len(rows)} rows, {FS_OUT} Hz, B={args.reps}, "
          f"seed=0, support {min_sup}+/{min_sup}-)")


if __name__ == "__main__":
    main()
