r"""RQ3: what the int8 confidence lattice lets a deployment actually do.

    .\.venv\Scripts\python.exe 12_leads\scripts\s12_confidence_lattice.py

Quantisation leaves discrimination intact -- s11 measures that -- but it
also collapses the number of distinct confidence values from roughly one
per record to a few dozen. Selective prediction needs a threshold, and a
threshold can only land between two adjacent levels, so the achievable
coverages form a coarse lattice. A target like "withhold the least
confident tenth" is not guaranteed to exist on that lattice.

This measures the lattice instead of interpolating across it. The earlier
treatment read selective risk at exactly 0.90 with np.interp, which
silently draws a straight line across a tie block that may span 35% of
the records, and reports a value at a coverage the device cannot select.

Reported per model x held-out source:

    levels        distinct confidence values
    max_tie       largest fraction of records sharing one value
    achievable    the coverage lattice near each target
    cov_error     |realised - target|, the thing that was hidden

Both precisions are measured the same way so the contrast is like for
like; float32 is effectively continuous and acts as the control.
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
S3B = HERE / "results" / f"study3b{TAG}"
ORDER = ["tiny", "dscnn", "mobilenet", "mbconv", "tcn", "resnet"]
TARGETS = (0.90, 0.80, 0.70)


def jaccard_loss(y, p, thr=0.5):
    yh = (p >= thr).astype(np.int8)
    inter = (yh & y).sum(1)
    union = (yh | y).sum(1)
    return 1.0 - np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def lattice(conf, loss, targets=TARGETS):
    """Coverages a threshold on `conf` can actually select, and risk there.

    Records are retained from most to least confident. A threshold can
    only fall between distinct values, so the attainable coverages are
    the cumulative sizes of the tied blocks -- nothing in between.
    """
    n = conf.size
    vals, cnt = np.unique(conf, return_counts=True)       # ascending
    cov = np.cumsum(cnt[::-1]) / n                        # descending conf
    # Mean loss over each attainable prefix, tie blocks kept intact.
    o = np.argsort(-conf, kind="stable")
    csum = np.cumsum(loss[o])
    k = np.round(cov * n).astype(int)
    risk = csum[k - 1] / k

    out = {}
    for t in targets:
        i = int(np.argmin(np.abs(cov - t)))
        lo = cov[cov <= t].max() if (cov <= t).any() else float("nan")
        hi = cov[cov >= t].min() if (cov >= t).any() else float("nan")
        out[f"{t:.2f}"] = {
            "nearest_cov": float(cov[i]),
            "risk_at_nearest": float(risk[i]),
            "cov_error": float(abs(cov[i] - t)),
            "bracket_lo": float(lo), "bracket_hi": float(hi),
            "gap": float(hi - lo) if np.isfinite(lo) and np.isfinite(hi)
            else float("nan")}
    return {
        "n": int(n),
        "levels": int(vals.size),
        "max_tie": float(cnt.max() / n),
        "risk_full": float(risk[-1]),
        "targets": out}


def main() -> None:
    rows = []
    for m in ORDER:
        for s in SOURCES:
            f = S3B / f"{m}__{s}" / "int8_prob.npz"
            if not f.exists():
                continue
            d = np.load(f)
            y = d["y"].astype(np.int8)
            for prec, key in (("float32", "prob_f32"), ("int8", "prob")):
                p = d[key].astype(np.float64)
                r = lattice(np.abs(p - 0.5).min(1), jaccard_loss(y, p))
                rows.append({"model": m, "source": s, "precision": prec,
                             **r})
        got = [r for r in rows if r["model"] == m and r["precision"] == "int8"]
        if got:
            lv = [r["levels"] for r in got]
            mt = max(r["max_tie"] for r in got)
            e90 = max(r["targets"]["0.90"]["cov_error"] for r in got)
            print(f"  [{m:10s}] int8 levels {min(lv):4d}-{max(lv):4d}  "
                  f"max tie {mt:5.1%}  worst 90% coverage error {e90:5.1%}",
                  flush=True)

    dst = S2D / "confidence_lattice.json"
    dst.write_text(json.dumps(rows, indent=1))
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
