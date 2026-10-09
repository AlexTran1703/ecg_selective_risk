r"""RQ1: does the source or the architecture decide external performance?

    .\.venv\Scripts\python.exe 12_leads\scripts\s13_variance_decomposition.py

The benchmark produces a 6 x 4 table of external macro AUPRC and the
paper reads it one row at a time, ranking architectures. Read down the
columns instead and the columns differ far more than the rows do: every
architecture orders the four held-out sources identically, and the
spread across sources is several times the spread across models.

So the table is decomposed rather than ranked. For encoder m on held-out
source s,

    AP(m, s) = mu + alpha_s + beta_m + eps(m, s)

and the variance of the fitted table is attributed to the source term,
the architecture term, and the interaction. The interaction share is the
part that would justify matching an architecture to a deployment site;
if it is small, architecture choice is close to separable from where the
device is used.

Intervals come from the same record-level bootstrap used elsewhere:
records are resampled within each source, the same resampled records for
every architecture, so the paired structure is preserved and the whole
6 x 4 table is recomputed per replicate.
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


def decompose(a):
    """Variance shares of a models x sources table of scores."""
    gm = a.mean()
    src = a.mean(0) - gm                       # over models -> per source
    mod = a.mean(1) - gm                       # over sources -> per model
    resid = a - gm - src[None, :] - mod[:, None]
    ss_s = a.shape[0] * (src ** 2).sum()
    ss_m = a.shape[1] * (mod ** 2).sum()
    ss_r = (resid ** 2).sum()
    tot = ss_s + ss_m + ss_r
    return {"share_source": ss_s / tot, "share_model": ss_m / tot,
            "share_interaction": ss_r / tot,
            "range_source": float(src.max() - src.min()),
            "range_model": float(mod.max() - mod.min()),
            "ratio": float((src.max() - src.min())
                           / max(mod.max() - mod.min(), 1e-12))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    dst = S2D / "variance_decomposition.json"
    if dst.exists() and not args.force:
        print(f"cached: {dst}")
        return

    data = {}
    for m in ORDER:
        for s in SOURCES:
            f = S2D / f"{m}__{s}.npz"
            if f.exists():
                d = np.load(f)
                data[(m, s)] = (d["test_y"].astype(np.int8),
                                d["test_prob"].astype(np.float64))
    srcs = [s for s in SOURCES if all((m, s) in data for m in ORDER)]
    n_cls = data[(ORDER[0], srcs[0])][0].shape[1]
    # Only classes scorable in every source, so the macro average is over
    # one class set and the columns are comparable.
    cls = [c for c in range(n_cls)
           if all(0 < data[(ORDER[0], s)][0][:, c].sum()
                  < data[(ORDER[0], s)][0].shape[0] for s in srcs)]
    print(f"sources {srcs}")
    print(f"classes scorable in all sources: {len(cls)}/{n_cls}")

    def table(idx):
        a = np.empty((len(ORDER), len(srcs)))
        for j, s in enumerate(srcs):
            i = idx[s]
            for k, m in enumerate(ORDER):
                y, p = data[(m, s)]
                ys, ps = y[i], p[i]
                a[k, j] = np.mean([ap_fast(ys[:, c], ps[:, c]) for c in cls])
        return a

    ident = {s: np.arange(data[(ORDER[0], s)][0].shape[0]) for s in srcs}
    obs = table(ident)
    point = decompose(obs)
    print("\nobserved table (models x sources), macro AUPRC on the "
          "common class set")
    for k, m in enumerate(ORDER):
        print(f"  {m:10s}" + "".join(f"{v:8.3f}" for v in obs[k]))
    print(f"\n  source share {point['share_source']:.1%}   "
          f"model share {point['share_model']:.1%}   "
          f"interaction {point['share_interaction']:.1%}")
    print(f"  source effect range {point['range_source']:.3f} vs "
          f"model {point['range_model']:.3f}  ({point['ratio']:.1f}x)")

    rng = np.random.default_rng(0)
    keys = ["share_source", "share_model", "share_interaction",
            "range_source", "range_model", "ratio"]
    boot = {k: np.empty(args.reps) for k in keys}
    for b in range(args.reps):
        idx = {s: rng.integers(0, data[(ORDER[0], s)][0].shape[0],
                               data[(ORDER[0], s)][0].shape[0]) for s in srcs}
        d = decompose(table(idx))
        for k in keys:
            boot[k][b] = d[k]
        if (b + 1) % 50 == 0:
            print(f"    replicate {b + 1}/{args.reps}", flush=True)

    out = {"sources": srcs, "models": ORDER, "n_classes": len(cls),
           "table": obs.tolist(), "reps": args.reps}
    for k in keys:
        out[k] = point[k]
        out[f"{k}_lo"] = float(np.percentile(boot[k], 2.5))
        out[f"{k}_hi"] = float(np.percentile(boot[k], 97.5))
    dst.write_text(json.dumps(out, indent=1))
    print(f"\nsource share {point['share_source']:.1%} "
          f"({out['share_source_lo']:.1%}, {out['share_source_hi']:.1%})")
    print(f"model  share {point['share_model']:.1%} "
          f"({out['share_model_lo']:.1%}, {out['share_model_hi']:.1%})")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
