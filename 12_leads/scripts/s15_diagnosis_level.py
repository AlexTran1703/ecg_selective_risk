r"""RQ2, at the diagnosis level rather than the source level.

    .\.venv\Scripts\python.exe 12_leads\scripts\s15_diagnosis_level.py

The source-level analysis has four points. Four points cannot carry an
explanatory claim, and the two Zheng databases (Chapman, Ningbo) share a
labelling convention closely enough that the effective number is nearer
three. So the unit of analysis moves down one level, to the 13 x 4
diagnosis-by-source cells.

Three things are computed, in this order, because each one answers an
objection to the one before it.

1.  AP and AUROC for every cell. Average precision has a prevalence
    floor, so a low AP can mean a rare diagnosis rather than a weak
    model. AUROC is rank-only and has no such floor. Reporting both
    separates "this diagnosis is rare here" from "the model cannot
    discriminate it here".

2.  Each diagnosis's prevalence in the held-out source against its
    prevalence in that fold's training pool. This is the shift the
    model actually faces for that diagnosis.

3.  Whether shift tracks discrimination, with the dependence taken
    seriously. 52 cells are not 52 independent observations: the 13
    diagnoses share recordings within a source, and the 4 sources share
    nothing but are only 4. So the pooled estimate is a mixed model with
    a random intercept per diagnosis, and it is reported beside the 13
    separate within-diagnosis slopes and a sign test over them, which
    needs no distributional assumption at all.

AUROC is the outcome for (3) rather than AP, deliberately. Regressing AP
on a prevalence difference would partly regress prevalence on itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from ecgmcu.data import SOURCES, FS_OUT                        # noqa: E402

TAG = "" if FS_OUT == 250 else f"_{FS_OUT}hz"
S2D = HERE / "results" / f"study2{TAG}"
ORDER = ["tiny", "dscnn", "mobilenet", "mbconv", "tcn", "resnet"]


def ap_fast(y, s):
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    tp = np.cumsum(y)
    n_pos = tp[-1]
    if not n_pos:
        return np.nan
    end = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    tp_e = tp[end]
    return float((np.diff(np.r_[0.0, tp_e / n_pos]) * (tp_e / (end + 1.0))
                  ).sum())


def auroc(y, s):
    o = np.argsort(s, kind="mergesort")
    y = y[o]
    s = s[o]
    n1 = int(y.sum())
    n0 = y.size - n1
    if not n1 or not n0:
        return np.nan
    rr = np.arange(1, y.size + 1).astype(float)
    i = 0
    while i < y.size:                       # mid-ranks for ties
        j = i
        while j + 1 < y.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            rr[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    return float((rr[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def main() -> None:
    cls = json.loads((HERE.parent / "data" / "meta" / "classes.json")
                     .read_text())["abbreviations"]
    data = {}
    for m in ORDER:
        for s in SOURCES:
            f = S2D / f"{m}__{s}.npz"
            if f.exists():
                d = np.load(f)
                data[(m, s)] = (d["test_y"].astype(np.int8),
                                d["test_prob"].astype(np.float64))
    srcs = [s for s in SOURCES if all((m, s) in data for m in ORDER)]
    Y = {s: data[(ORDER[0], s)][0] for s in srcs}
    n = {s: Y[s].shape[0] for s in srcs}

    rows = []
    for s in srcs:
        tr = [t for t in srcs if t != s]
        w = np.array([n[t] for t in tr], dtype=float)
        for c, nm in enumerate(cls):
            pi = float(Y[s][:, c].mean())
            pool = float(np.average([Y[t][:, c].mean() for t in tr],
                                    weights=w))
            if not (0 < Y[s][:, c].sum() < n[s]):
                continue
            ap = [ap_fast(data[(m, s)][0][:, c], data[(m, s)][1][:, c])
                  for m in ORDER]
            ro = [auroc(data[(m, s)][0][:, c], data[(m, s)][1][:, c])
                  for m in ORDER]
            rows.append({
                "source": s, "class": nm,
                "n_pos": int(Y[s][:, c].sum()),
                "prev": pi, "prev_pool": pool,
                "log2_shift": float(np.log2(max(pi, 1e-9)
                                            / max(pool, 1e-9))),
                "abs_log2_shift": float(abs(np.log2(max(pi, 1e-9)
                                                    / max(pool, 1e-9)))),
                "ap": float(np.mean(ap)),
                "ap_norm": float((np.mean(ap) - pi) / (1 - pi)),
                "auroc": float(np.mean(ro))})
    t = pd.DataFrame(rows)

    print("=== 1. does AP's prevalence floor explain the source ranking? ===")
    g = t.groupby("source")[["prev", "ap", "ap_norm", "auroc"]].mean()
    g = g.reindex(srcs)
    print(g.to_string(float_format=lambda v: f"{v:.3f}"))
    print("  AP and prevalence-standardised AP rank the sources "
          f"identically: {list(g.ap.rank()) == list(g.ap_norm.rank())}")

    print()
    print("=== 2. cells where AUROC is high but AP is low ===")
    sus = t[(t.auroc > 0.85) & (t.ap < 0.35)].sort_values("ap")
    print(f"{'source':9s}{'class':8s}{'prev':>8s}{'pool':>8s}"
          f"{'AP':>8s}{'AUROC':>8s}")
    for _, r in sus.iterrows():
        print(f"{r['source']:9s}{r['class']:8s}{r['prev']:8.3f}"
              f"{r['prev_pool']:8.3f}{r['ap']:8.3f}{r['auroc']:8.3f}")

    print()
    print("=== 3. does prevalence shift track discrimination? ===")
    import statsmodels.formula.api as smf
    md = smf.mixedlm("auroc ~ abs_log2_shift", t, groups=t["class"]).fit()
    beta = float(md.params["abs_log2_shift"])
    ci = md.conf_int().loc["abs_log2_shift"].tolist()
    print(f"  mixed model, random intercept per diagnosis (n={len(t)} "
          f"cells, {t['class'].nunique()} groups)")
    print(f"    slope {beta:+.4f}  95% CI ({ci[0]:+.4f}, {ci[1]:+.4f})"
          f"  p = {md.pvalues['abs_log2_shift']:.4f}")

    slopes = {}
    for nm, grp in t.groupby("class"):
        if len(grp) >= 3 and grp.abs_log2_shift.std() > 0:
            slopes[nm] = float(np.polyfit(grp.abs_log2_shift,
                                          grp.auroc, 1)[0])
    neg = sum(v < 0 for v in slopes.values())
    from scipy.stats import binomtest
    pb = binomtest(neg, len(slopes), 0.5).pvalue
    print(f"  within-diagnosis slopes: {neg}/{len(slopes)} negative, "
          f"sign test p = {pb:.3f}")

    dst = S2D / "diagnosis_level.json"
    dst.write_text(json.dumps({
        "cells": rows, "sources": srcs, "classes": cls,
        "mixed_slope": beta, "mixed_lo": ci[0], "mixed_hi": ci[1],
        "mixed_p": float(md.pvalues["abs_log2_shift"]),
        "slopes": slopes, "sign_neg": neg, "sign_n": len(slopes),
        "sign_p": float(pb)}, indent=1))
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
