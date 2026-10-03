r"""Selective reliability under source shift: excess-risk curves and E-AURC.

    .\.venv\Scripts\python.exe 12_leads\scripts\s9_reliability.py

E-AURC is a single number that most readers have to take on trust. The
curve it integrates is not:

    excess(c) = R(c) - R*(c)

where R(c) is the selective risk when the most confident fraction c of
records is retained, and R*(c) is the same model's oracle -- the risk if
the retained set had been chosen by the true loss instead of by
confidence. The oracle is per-model, so the difference isolates how well
confidence *ranks* errors from how many errors there are, and

    E-AURC = integral of excess(c) over c.

Plotting the curve makes the summary legible rather than merely cited.

Curves are averaged over the four held-out sources, with record-level
bootstrap intervals formed by resampling within each source.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from ecgmcu.data import SOURCES, FS_OUT                        # noqa: E402

S2D = HERE / "results" / ("study2" if FS_OUT == 250
                          else f"study2_{FS_OUT}hz")
ALPHA_RE = re.compile(r"a[0-9.]+__")
ORDER = ["tiny", "dscnn", "mobilenet", "mbconv", "tcn", "resnet"]
GRID = np.linspace(0.02, 1.0, 99)


def jaccard_loss(y, p, thr=0.5):
    yh = (p >= thr).astype(np.int8)
    inter = (yh & y).sum(1)
    union = (yh | y).sum(1)
    return 1.0 - np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def curves(loss: np.ndarray, conf: np.ndarray):
    """(selective risk, oracle risk) on the common coverage grid."""
    n = loss.size
    k = np.arange(1, n + 1)
    risk = np.cumsum(loss[np.argsort(-conf, kind="stable")]) / k
    oracle = np.cumsum(np.sort(loss)) / k
    cov = k / n
    return np.interp(GRID, cov, risk), np.interp(GRID, cov, oracle)


def excess_curve(loss: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """excess(c) on the common coverage grid, for one model and source."""
    r, o = curves(loss, conf)
    return r - o


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
            "conf": np.abs(p - 0.5).min(1)}
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    dst = S2D / "reliability_curves.json"
    if dst.exists() and not args.force:
        print(f"cached: {dst}")
        return

    runs = load_runs()
    if not runs:
        print(f"no runs in {S2D}")
        return
    models = [m for m in ORDER if any(k[0] == m for k in runs)]
    out = {"grid": GRID.tolist(), "models": {}}

    for m in models:
        srcs = [s for s in SOURCES if (m, s) in runs]
        point = np.mean([excess_curve(runs[(m, s)]["loss"],
                                      runs[(m, s)]["conf"]) for s in srcs],
                        axis=0)
        rng = np.random.default_rng(0)
        reps = np.empty((args.reps, GRID.size))
        for b in range(args.reps):
            cur = []
            for s in srcs:
                r = runs[(m, s)]
                n = r["loss"].size
                i = rng.integers(0, n, n)
                cur.append(excess_curve(r["loss"][i], r["conf"][i]))
            reps[b] = np.mean(cur, axis=0)
        lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
        # Integrate inside each replicate, so the interval on E-AURC comes
        # from resampling records rather than from treating the four
        # sources as four independent observations.
        eaurc_reps = reps.mean(axis=1)
        per_src = {s: float(np.mean(excess_curve(runs[(m, s)]["loss"],
                                                 runs[(m, s)]["conf"])))
                   for s in srcs}
        rc = {}
        for s in srcs:
            r_, o_ = curves(runs[(m, s)]["loss"], runs[(m, s)]["conf"])
            rc[s] = {"risk": r_.tolist(), "oracle": o_.tolist()}
        out["models"][m] = {
            "excess": point.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
            "eaurc": float(point.mean()),
            "eaurc_lo": float(np.percentile(eaurc_reps, 2.5)),
            "eaurc_hi": float(np.percentile(eaurc_reps, 97.5)),
            "eaurc_by_source": per_src, "risk_coverage": rc}
        print(f"  [{m:10s}] E-AURC {point.mean():.4f} "
              f"({np.percentile(eaurc_reps, 2.5):.4f}-"
              f"{np.percentile(eaurc_reps, 97.5):.4f})  "
              + "  ".join(f"{s}={v:.3f}" for s, v in per_src.items()),
              flush=True)

    dst.write_text(json.dumps(out))
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
