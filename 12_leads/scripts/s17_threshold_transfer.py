r"""Does an abstention threshold transfer to an unseen source?

    .\.venv\Scripts\python.exe 12_leads\scripts\s17_threshold_transfer.py

Figure 4 shows that confidence ranks errors on every held-out source.
That is a statement about ordering, and ordering is all E-AURC and the
risk-coverage curves use. A deployment cannot use an ordering. It has to
fix a number in advance, and then live with whatever coverage that
number produces on a source it has never seen.

The two are not the same claim, and the gap between them is the whole
operational question. So the threshold is chosen the only way a real
deployment could choose it:

    tau   : the confidence quantile on the *training-side validation*
            split that would retain a target coverage, e.g. 90%. The
            held-out source contributes nothing -- no labels, no
            predictions, no recalibration.

    then  : tau is applied unchanged to the held-out source, and the
            coverage and selective risk it actually achieves there are
            recorded.

Achieved coverage is the number that matters. If it lands far from the
target, the abstention budget a clinical workflow was planned around is
not the budget it gets, whatever the risk-coverage curve looked like.

Validation predictions are not cached by the training run -- only test
predictions are -- so each saved checkpoint is re-scored on its own
validation split. That is inference only; nothing is retrained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

from ecgmcu.data import Cohort, Feeder, SOURCES, FS_OUT          # noqa: E402
from ecgmcu.models import build                                  # noqa: E402
from s2_train import predict                                     # noqa: E402

TAG = "" if FS_OUT == 250 else f"_{FS_OUT}hz"
S2D = HERE / "results" / f"study2{TAG}"
ORDER = ["tiny", "dscnn", "mobilenet", "mbconv", "tcn", "resnet"]
TARGETS = (0.90, 0.80)


def confidence(p):
    """Distance of the least-decided class from the 0.5 boundary."""
    return np.abs(p - 0.5).min(1)


def jaccard_loss(y, p, thr=0.5):
    yh = (p >= thr).astype(np.int8)
    inter = (yh & y).sum(1)
    union = (yh | y).sum(1)
    return 1.0 - np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ORDER)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dst = S2D / "threshold_transfer.json"
    if dst.exists() and not args.force:
        print(f"cached: {dst}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    coh = Cohort()
    rows = []

    for src in args.sources:
        tr, va, te = coh.loso(src, seed=args.seed)
        mean, std = coh.stats(tr)
        f_va = Feeder(coh.sig_path, coh.y, va, args.batch_size, mean, std,
                      device, shuffle=False)
        print(f"\n=== {src}: validation {va.size} records (training "
              f"sources only), test {te.size} ===", flush=True)

        for name in args.models:
            ck = S2D / f"{name}__{src}.pt"
            run = S2D / f"{name}__{src}.npz"
            if not (ck.exists() and run.exists()):
                continue
            model = build(name).to(device)
            model.load_state_dict(torch.load(ck, map_location=device))
            p_va = predict(model, f_va, device).astype(np.float64)
            d = np.load(run)
            p_te = d["test_prob"].astype(np.float64)
            y_te = d["test_y"].astype(np.int8)

            c_va, c_te = confidence(p_va), confidence(p_te)
            loss_te = jaccard_loss(y_te, p_te)
            risk_full = float(loss_te.mean())

            for t in TARGETS:
                # tau is the validation quantile that would retain t of
                # the validation records. Nothing from the target is used.
                tau = float(np.quantile(c_va, 1.0 - t))
                keep = c_te >= tau
                cov = float(keep.mean())
                risk = float(loss_te[keep].mean()) if keep.any() else np.nan
                # What the same coverage would have cost if the threshold
                # had been free to be set on the target itself -- the
                # oracle-coverage comparison the curves implicitly assume.
                k = max(int(round(t * c_te.size)), 1)
                idx = np.argsort(-c_te, kind="stable")[:k]
                risk_oracle = float(loss_te[idx].mean())
                rows.append({
                    "source": src, "model": name, "target": t,
                    "tau": tau, "coverage": cov, "cov_error": cov - t,
                    "risk": risk, "risk_full": risk_full,
                    "risk_at_target_coverage": risk_oracle,
                    "n_test": int(y_te.shape[0])})

            got = [r for r in rows if r["model"] == name
                   and r["source"] == src]
            msg = "  ".join(f"t={r['target']:.2f} -> cov {r['coverage']:.3f}"
                            f" ({r['cov_error']:+.3f})" for r in got)
            print(f"  [{name:10s}] {msg}", flush=True)

    dst.write_text(json.dumps(rows, indent=1))
    t = pd.DataFrame(rows)
    print("\n=== achieved coverage when tau is set on the training side ===")
    for tg in TARGETS:
        g = t[t.target == tg]
        print(f"  target {tg:.2f}:  achieved {g.coverage.min():.3f}-"
              f"{g.coverage.max():.3f}  mean abs error "
              f"{g.cov_error.abs().mean():.3f}  worst "
              f"{g.cov_error.abs().max():.3f}")
    print("\n  by held-out source, mean |coverage error|")
    print(t.groupby("source").cov_error.apply(lambda v: v.abs().mean())
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
