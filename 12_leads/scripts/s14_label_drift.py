r"""RQ2: is the cross-source gap signal shift or label-definition drift?

    .\.venv\Scripts\python.exe 12_leads\scripts\s14_label_drift.py

RQ1 finds a large and perfectly consistent spread across held-out
sources. The natural reading, and the one the signal-quality audit
invites, is that some sources are noisier. The data does not support it:
the cleanest source is the worst performer and the dirtiest is the best.

The alternative is that the sources do not agree on what the labels
mean. Two signatures separate that from signal shift, and they point in
opposite directions:

    AUROC stays high, AUPRC collapses
        the ranking is intact, precision is not -- the model is finding
        the right records and being told they are negative

    predicted positive rate >> labelled positive rate
        the model, trained on the other sources, applies their labelling
        convention and finds far more positives than this source marks

Genuine signal degradation does not do this. It lowers AUROC, because a
corrupted recording is ranked wrongly, not just scored against a
different convention.

So for every class x held-out source this reports AUROC, AUPRC and the
ratio of predicted to labelled positive rate, then asks how much of each
source's deficit survives dropping the classes that look like convention
mismatches rather than failures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from ecgmcu.data import SOURCES, FS_OUT                        # noqa: E402

TAG = "" if FS_OUT == 250 else f"_{FS_OUT}hz"
S2D = HERE / "results" / f"study2{TAG}"
ORDER = ["tiny", "dscnn", "mobilenet", "mbconv", "tcn", "resnet"]
# An absolute predicted-to-labelled ratio cannot be used as the bar.
# These are multi-label models on classes with ~3% prevalence, and at a
# 0.5 threshold they over-predict every rare class in every source, so
# an absolute cut flags almost everything and separates nothing.
#
# What is comparable is each source's labelling propensity *relative to
# its peers on the same class*, which cancels the class-level
# calibration offset:
#
#     propensity(s, c) = labelled(s, c) / predicted(s, c)
#     drift(s, c)      = log2 propensity(s, c)
#                        - log2 median propensity(s', c), s' != s
#
# The model supplies the denominator, so case mix is controlled for: it
# reads the same signal everywhere. What is left is how readily this
# source puts the label on, against how readily the others do.
DRIFT_ABS = 1.0          # one doubling relative to the peer median


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
    """Rank-only, so prevalence differences do not move it."""
    o = np.argsort(s, kind="mergesort")
    y = y[o]
    n1 = y.sum()
    n0 = y.size - n1
    if not n1 or not n0:
        return np.nan
    r = np.empty(y.size)
    r[o] = np.arange(1, y.size + 1)
    # Mid-ranks for ties, else quantisation shifts the statistic.
    s_sorted = s[o]
    i = 0
    rr = np.arange(1, y.size + 1).astype(float)
    while i < y.size:
        j = i
        while j + 1 < y.size and s_sorted[j + 1] == s_sorted[i]:
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

    rows = []
    for s in srcs:
        for c, nm in enumerate(cls):
            y0 = data[(ORDER[0], s)][0]
            if not (0 < y0[:, c].sum() < y0.shape[0]):
                continue
            lab = float(y0[:, c].mean())
            ro, ra, pr = [], [], []
            for m in ORDER:
                y, p = data[(m, s)]
                ro.append(auroc(y[:, c], p[:, c]))
                ra.append(ap_fast(y[:, c], p[:, c]))
                pr.append(float((p[:, c] > 0.5).mean()))
            rows.append({"source": s, "class": nm,
                         "labelled": lab, "predicted": float(np.mean(pr)),
                         "ratio": float(np.mean(pr)) / max(lab, 1e-9),
                         "auroc": float(np.mean(ro)),
                         "auprc": float(np.mean(ra))})

    # propensity and its deviation from the peer median, per class.
    for r in rows:
        r["propensity"] = r["labelled"] / max(r["predicted"], 1e-9)
    for nm in {r["class"] for r in rows}:
        grp = [r for r in rows if r["class"] == nm]
        for r in grp:
            peers = [np.log2(max(o["propensity"], 1e-9))
                     for o in grp if o is not r]
            r["drift"] = float(np.log2(max(r["propensity"], 1e-9))
                               - np.median(peers)) if peers else 0.0
            r["flagged"] = bool(abs(r["drift"]) > DRIFT_ABS)

    print("=== convention mismatch: labelling propensity vs peer sources ===")
    print(f"{'source':9s}{'class':8s}{'labelled':>10s}{'predicted':>11s}"
          f"{'propens':>9s}{'drift':>8s}{'AUROC':>8s}{'AUPRC':>8s}")
    for r in sorted([r for r in rows if r["flagged"]],
                    key=lambda r: r["drift"]):
        print(f"{r['source']:9s}{r['class']:8s}{r['labelled']:10.3f}"
              f"{r['predicted']:11.3f}{r['propensity']:9.3f}"
              f"{r['drift']:+8.2f}{r['auroc']:8.3f}{r['auprc']:8.3f}")

    # How much of each source's deficit is carried by the flagged classes?
    flagged = {s: {r["class"] for r in rows
                   if r["source"] == s and r["flagged"]} for s in srcs}
    keep = [nm for nm in cls
            if not any(nm in flagged[s] for s in srcs)]
    print(f"\nclasses clean in every source: {len(keep)}/{len(cls)} {keep}")

    summary = []
    for s in srcs:
        allc = [r["auprc"] for r in rows if r["source"] == s]
        kept = [r["auprc"] for r in rows
                if r["source"] == s and r["class"] in keep]
        summary.append({"source": s, "mAP_all": float(np.mean(allc)),
                        "mAP_clean": float(np.mean(kept)),
                        "n_flagged": len(flagged[s])})
    a = np.array([x["mAP_all"] for x in summary])
    k = np.array([x["mAP_clean"] for x in summary])
    print("\n=== source spread before and after dropping flagged classes ===")
    print(f"{'source':9s}{'flagged':>9s}{'mAP all':>10s}{'mAP clean':>11s}")
    for x in summary:
        print(f"{x['source']:9s}{x['n_flagged']:9d}"
              f"{x['mAP_all']:10.3f}{x['mAP_clean']:11.3f}")
    print(f"\nspread across sources: {a.max() - a.min():.3f} (all classes) "
          f"-> {k.max() - k.min():.3f} (convention-clean classes)")
    print(f"reduction: {1 - (k.max() - k.min()) / (a.max() - a.min()):.1%} "
          f"of the cross-source spread is carried by classes the sources "
          f"label differently")

    dst = S2D / "label_drift.json"
    dst.write_text(json.dumps(
        {"per_class": rows, "clean_classes": keep, "summary": summary,
         "spread_all": float(a.max() - a.min()),
         "spread_clean": float(k.max() - k.min()),
         "drift_abs": DRIFT_ABS}, indent=1))
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
