"""Diagnose why selective risk is not controllable, on an already-trained run.

Two independent things can make ``C@R<=alpha`` collapse to zero, and they call
for different fixes, so this script separates them:

  1. *Level* -- the full-coverage risk is simply too high, e.g. because the
     frozen per-class thresholds do not transfer to the target source.  Measured
     by refitting thresholds on the target calibration split (a diagnostic only;
     the study's protocol forbids it) and seeing how far the risk falls.

  2. *Ranking* -- the confidence score does not order records by loss, so
     abstaining on the least confident records removes no risk.  Measured by
     AURC against the full-coverage risk: if they are equal, the score is
     uninformative.  Several alternative scores are compared on the same frozen
     predictions.

Nothing here changes the study's protocol; it produces the evidence needed to
decide what to change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgsr.metrics import fit_thresholds, macro_f1          # noqa: E402
from ecgsr.selective import (EPS, aurc, coverage_at_risk,    # noqa: E402
                             jaccard_loss, predict_sets)
from ecgsr.train import predict                              # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module                          # noqa: E402

_eval_mod = None


def _load_run(*args, **kw):
    global _eval_mod
    if _eval_mod is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "calib_eval", ROOT / "scripts" / "04_calibrate_eval.py")
        _eval_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_eval_mod)
    return _eval_mod.load_run(*args, **kw)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------- #
#  Candidate confidence scores (higher = more confident)
# --------------------------------------------------------------------------- #
def scores_all(probs: np.ndarray, thr: np.ndarray) -> dict[str, np.ndarray]:
    margin = np.abs(_logit(probs) - _logit(thr)[None, :])
    p = np.clip(probs, EPS, 1 - EPS)

    # Expected Jaccard loss if the classes were independent Bernoulli(p_k).
    # This is a self-estimate of the very quantity being controlled, so it is
    # the natural score for a set-valued loss -- unlike a per-class margin, it
    # weights each class by how much it can actually move the loss.
    yhat = probs >= thr[None, :]
    exp_inter = np.where(yhat, p, 0.0).sum(axis=1)
    exp_union = np.where(yhat, 1.0, p).sum(axis=1)
    exp_jaccard = exp_inter / np.maximum(exp_union, EPS)

    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    return {
        "min_margin": margin.min(axis=1),
        "mean_margin": margin.mean(axis=1),
        "sum_margin": margin.sum(axis=1),
        "kth_margin_2": np.partition(margin, 1, axis=1)[:, 1],
        "exp_jaccard": exp_jaccard,
        "neg_total_entropy": -ent.sum(axis=1),
        "neg_max_entropy": -ent.max(axis=1),
        "max_prob": probs.max(axis=1),
    }


def report(tag: str, losses: np.ndarray, score_sets: dict[str, np.ndarray],
           alphas: tuple[float, ...]) -> list[dict]:
    print(f"\n  {tag}: full-coverage risk {losses.mean():.4f}")
    print(f"    {'score':<20}{'AURC':>8}" +
          "".join(f"{'C@' + format(a, '.2f'):>9}" for a in alphas))
    out = []
    for name, s in score_sets.items():
        covs = [coverage_at_risk(losses, s, a) for a in alphas]
        a = aurc(losses, s)
        out.append({"tag": tag, "score": name, "aurc": a,
                    "oracle_coverage": dict(zip(map(str, alphas), covs))})
        print(f"    {name:<20}{a:>8.4f}" + "".join(f"{c:>9.3f}" for c in covs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "cache" / "signals.npy")
    ap.add_argument("--meta", type=Path, default=ROOT / "data" / "meta")
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[0.10, 0.20, 0.30, 0.40])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    alphas = tuple(args.alphas)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index, split, rep, model, rep_kind, backbone, loader = _load_run(
        args.run, args.cache, args.meta, args.batch_size, args.workers, device)

    print(f"=== {rep_kind} | {backbone} | target = {split.target} ===")

    dev_probs, dev_y = predict(model, rep, loader(split.dev), device)
    test_probs, test_y = predict(model, rep, loader(split.test), device)
    cal_probs, cal_y = predict(model, rep, loader(split.cal), device)

    thr_source = fit_thresholds(dev_y, dev_probs)
    thr_target = fit_thresholds(cal_y, cal_probs)      # diagnostic only

    print(f"  macro-F1 on target test:"
          f"  source-fitted thresholds {macro_f1(test_y, test_probs, thr_source):.4f}"
          f"   target-fitted {macro_f1(test_y, test_probs, thr_target):.4f}")

    abbrev = index.abbreviations
    print(f"\n  {'class':<8}{'t_source':>10}{'t_target':>10}"
          f"{'prev_dev':>10}{'prev_tgt':>10}{'pred_rate':>11}")
    pred_source = predict_sets(test_probs, thr_source)
    for k, name in enumerate(abbrev):
        print(f"  {name:<8}{thr_source[k]:>10.3f}{thr_target[k]:>10.3f}"
              f"{dev_y[:, k].mean():>10.3f}{test_y[:, k].mean():>10.3f}"
              f"{pred_source[:, k].mean():>11.3f}")

    print(f"\n  mean |Y| true {test_y.sum(1).mean():.2f}   "
          f"predicted (source thr) {pred_source.sum(1).mean():.2f}   "
          f"predicted (target thr) {predict_sets(test_probs, thr_target).sum(1).mean():.2f}")

    records = []
    for label, thr in (("source-fitted thresholds (protocol)", thr_source),
                       ("target-fitted thresholds (diagnostic)", thr_target)):
        losses = jaccard_loss(test_y, predict_sets(test_probs, thr))
        records += report(label, losses, scores_all(test_probs, thr), alphas)

    out = args.run / "diagnosis.json"
    out.write_text(json.dumps({
        "target": split.target, "representation": rep_kind, "backbone": backbone,
        "macro_f1_source_thr": macro_f1(test_y, test_probs, thr_source),
        "macro_f1_target_thr": macro_f1(test_y, test_probs, thr_target),
        "thresholds_source": thr_source.tolist(),
        "thresholds_target": thr_target.tolist(),
        "results": records,
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
