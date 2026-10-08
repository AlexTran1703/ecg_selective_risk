r"""RQ3a: does int8 preserve discrimination *and* selective reliability?

    .\.venv\Scripts\python.exe 12_leads\scripts\s11_int8_preservation.py

There was a conceptual break between RQ2 and RQ3. RQ2 establishes that
confidence ranking is what makes a deployment safe to defer on; RQ3 then
checked only whether AUPRC survived quantisation. If quantisation left
accuracy intact but scrambled the confidence ordering, every selective
conclusion in RQ2 would be void on the device that actually runs.

So both are compared, on the same records, from the same cached float32
and int8 probability pairs:

    dAUPRC  = AUPRC(int8)  - AUPRC(float32)       negative = worse
    dE-AURC = E-AURC(int8) - E-AURC(float32)      positive = worse ranking

Also computed here, because it needs the same curves: selective risk at
fixed coverage. E-AURC is the rigorous summary but it is not
interpretable at a glance, whereas

    dR_90 = R(1.00) - R(0.90)

says what withholding the least-confident tenth of recordings buys.
"""

from __future__ import annotations

import argparse
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
COVERAGES = (1.00, 0.90, 0.80, 0.70)


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


def macro_ap(y, p):
    v = [ap_fast(y[:, c], p[:, c]) for c in range(y.shape[1])
         if 0 < y[:, c].sum() < y.shape[0]]
    return float(np.mean(v)) if v else np.nan


def jaccard_loss(y, p, thr=0.5):
    yh = (p >= thr).astype(np.int8)
    inter = (yh & y).sum(1)
    union = (yh | y).sum(1)
    return 1.0 - np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def selective(loss, conf):
    """(E-AURC, {coverage: selective risk}) for one model and source."""
    n = loss.size
    k = np.arange(1, n + 1)
    risk = np.cumsum(loss[np.argsort(-conf, kind="stable")]) / k
    oracle = np.cumsum(np.sort(loss)) / k
    cov = k / n
    at = {c: float(np.interp(c, cov, risk)) for c in COVERAGES}
    return float(risk.mean() - oracle.mean()), at


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    dst = S2D / "int8_preservation.json"
    if dst.exists() and not args.force:
        print(f"cached: {dst}")
        return

    rows = []
    for m in ORDER:
        folds = []
        for s in SOURCES:
            f = S3B / f"{m}__{s}" / "int8_prob.npz"
            if not f.exists():
                continue
            d = np.load(f)
            folds.append((d["y"].astype(np.int8),
                          d["prob_f32"].astype(np.float64),
                          d["prob"].astype(np.float64)))
        if not folds:
            continue

        def summarise(idx_list):
            """Mean over sources of the four quantities, one resample."""
            a32, a8, e32, e8 = [], [], [], []
            r32 = {c: [] for c in COVERAGES}
            r8 = {c: [] for c in COVERAGES}
            for (y, p32, p8), i in zip(folds, idx_list):
                ys, f32, f8 = y[i], p32[i], p8[i]
                a32.append(macro_ap(ys, f32))
                a8.append(macro_ap(ys, f8))
                e, at = selective(jaccard_loss(ys, f32),
                                  np.abs(f32 - 0.5).min(1))
                e32.append(e)
                for c in COVERAGES:
                    r32[c].append(at[c])
                e, at = selective(jaccard_loss(ys, f8),
                                  np.abs(f8 - 0.5).min(1))
                e8.append(e)
                for c in COVERAGES:
                    r8[c].append(at[c])
            return (np.mean(a32), np.mean(a8), np.mean(e32), np.mean(e8),
                    {c: float(np.mean(r32[c])) for c in COVERAGES},
                    {c: float(np.mean(r8[c])) for c in COVERAGES})

        point = summarise([np.arange(y.shape[0]) for y, _, _ in folds])
        rng = np.random.default_rng(0)
        d_ap = np.empty(args.reps)
        d_ea = np.empty(args.reps)
        for b in range(args.reps):
            idx = [rng.integers(0, y.shape[0], y.shape[0])
                   for y, _, _ in folds]
            a32, a8, e32, e8, _, _ = summarise(idx)
            d_ap[b] = a8 - a32
            d_ea[b] = e8 - e32
            if (b + 1) % 250 == 0:
                print(f"    {m}: replicate {b + 1}/{args.reps}", flush=True)

        rows.append({
            "model": m,
            "auprc_f32": point[0], "auprc_int8": point[1],
            "eaurc_f32": point[2], "eaurc_int8": point[3],
            "d_auprc": point[1] - point[0],
            "d_auprc_lo": float(np.percentile(d_ap, 2.5)),
            "d_auprc_hi": float(np.percentile(d_ap, 97.5)),
            "d_eaurc": point[3] - point[2],
            "d_eaurc_lo": float(np.percentile(d_ea, 2.5)),
            "d_eaurc_hi": float(np.percentile(d_ea, 97.5)),
            "risk_f32": {str(c): v for c, v in point[4].items()},
            "risk_int8": {str(c): v for c, v in point[5].items()},
            "sources": len(folds)})
        print(f"  [{m:10s}] dAUPRC {rows[-1]['d_auprc']:+.4f}  "
              f"dE-AURC {rows[-1]['d_eaurc']:+.4f}  "
              f"R(1.00)={point[4][1.0]:.3f} -> R(0.90)="
              f"{point[4][0.9]:.3f}", flush=True)

    dst.write_text(json.dumps(rows, indent=1))
    print(f"\n-> {dst}  (B={args.reps}, seed=0)")


if __name__ == "__main__":
    main()
