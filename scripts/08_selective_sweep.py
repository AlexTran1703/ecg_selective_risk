"""Every selective-risk experiment, offline from cached logits.

    python scripts/08_selective_sweep.py

Reads only ``predictions.npz`` from each run -- no ECG loading, no forward pass,
no GPU.  Answers the three questions that decide the study's design:

  1. Is ``alpha = 0.10`` reachable at all?  Compare the achieved coverage with
     the *oracle* coverage.  If the oracle is also near zero the classifier
     never produces enough near-perfect records and no score can help; if the
     oracle is high, the ranking is what is failing.

  2. Which confidence score ranks records best?  AURC for the normalised mean
     margin, the 25% margin quantile, negative mean entropy, and -- as the
     reported negative result -- the minimum margin and max-probability.

  3. Do the frozen source thresholds hurt?  Refit them on target data and see
     how much risk falls.  Reported here as a *design diagnostic*; using it in
     the study proper requires splitting the target into disjoint adaptation and
     risk-calibration subsets, which ``--adapt`` does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgsr.analysis import (ALPHA_LEVELS, HEADLINE_ALPHA,            # noqa: E402
                            Predictions,
                            compare_scores, evaluate_run,
                            fit_decision_thresholds)
from ecgsr.scores import PRIMARY_SCORE, SCORE_NAMES                  # noqa: E402

REPRESENTATIONS = ("raw", "vcg", "fft")
TARGETS = ("PTBXL", "Georgia", "Chapman", "Ningbo")


def load_all(runs: Path) -> dict[tuple[str, str], Predictions]:
    out = {}
    for rep in REPRESENTATIONS:
        for target in TARGETS:
            p = runs / f"{rep}__{target}" / "predictions.npz"
            if not p.exists():
                continue
            pred = Predictions.load(p)
            meta_path = runs / f"{rep}__{target}" / "predictions_meta.json"
            if meta_path.exists():
                pred.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            out[(rep, target)] = pred
    return out


def split_target(pred: Predictions, adapt_frac: float, seed: int = 0) -> Predictions:
    """Carve an adaptation subset out of the target calibration split.

    The study's 20% target calibration subset becomes 10% adaptation (threshold
    refitting) + 10% risk calibration.  They must stay disjoint: reusing the
    same records to learn the thresholds and to certify the risk would void the
    finite-sample guarantee.
    """
    n = len(pred.labels["cal"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = max(1, int(round(adapt_frac * n)))
    a, c = perm[:k], perm[k:]
    return Predictions(
        logits={"dev": pred.logits["dev"], "adapt": pred.logits["cal"][a],
                "cal": pred.logits["cal"][c], "test": pred.logits["test"]},
        labels={"dev": pred.labels["dev"], "adapt": pred.labels["cal"][a],
                "cal": pred.labels["cal"][c], "test": pred.labels["test"]},
        meta=pred.meta)


# --------------------------------------------------------------------------- #
def question_one_and_two(preds: dict, out_dir: Path) -> dict:
    """Score comparison and the oracle bound, per representation and target."""
    print("\n" + "=" * 78)
    print("Q1/Q2  score ranking vs the oracle bound  (protocol thresholds, "
          "fitted on dev)")
    print("=" * 78)

    report = {}
    for rep in REPRESENTATIONS:
        got = [(t, preds[(rep, t)]) for t in TARGETS if (rep, t) in preds]
        if not got:
            continue
        print(f"\n-- {rep.upper()} " + "-" * 60)
        head = (f"  {'target':<9}{'full-risk':>10}" +
                "".join(f"{n[:16]:>17}" for n in ("AURC " + PRIMARY_SCORE,)) +
                f"{'oracle AURC':>13}")
        print(head)
        per_target = {}
        for target, pred in got:
            thr, scaler = fit_decision_thresholds(pred, "dev")
            cmp = compare_scores(pred, thr, scaler, "test")
            per_target[target] = cmp
            print(f"  {target:<9}{cmp['full_coverage_risk']:>10.4f}"
                  f"{cmp['scores'][PRIMARY_SCORE]['aurc']:>17.4f}"
                  f"{cmp['oracle']['aurc']:>13.4f}")

        print(f"\n  {'score':<20}{'mean AURC':>11}" +
              "".join(f"{'C@' + f'{a:.2f}':>10}" for a in ALPHA_LEVELS))
        for name in SCORE_NAMES:
            aurcs = [per_target[t]["scores"][name]["aurc"] for t, _ in got]
            covs = [np.mean([per_target[t]["scores"][name]["coverage"][f"{a:.2f}"]
                             for t, _ in got]) for a in ALPHA_LEVELS]
            tag = "  <- primary" if name == PRIMARY_SCORE else ""
            print(f"  {name:<20}{np.mean(aurcs):>11.4f}"
                  + "".join(f"{c:>10.3f}" for c in covs) + tag)
        oa = [per_target[t]["oracle"]["aurc"] for t, _ in got]
        oc = [np.mean([per_target[t]["oracle"]["coverage"][f"{a:.2f}"]
                       for t, _ in got]) for a in ALPHA_LEVELS]
        print(f"  {'ORACLE ranking':<20}{np.mean(oa):>11.4f}"
              + "".join(f"{c:>10.3f}" for c in oc))
        report[rep] = per_target

    (out_dir / "score_comparison.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")
    return report


def question_three(preds: dict, adapt_frac: float, out_dir: Path) -> dict:
    """Frozen source thresholds vs target-adapted thresholds."""
    # The target's 20% calibration split is divided in two; as a fraction of the
    # whole target source that is 20*adapt_frac % for adaptation and the rest for
    # risk calibration. The 80% test split is untouched either way.
    adapt_pct, cal_pct = 20.0 * adapt_frac, 20.0 * (1.0 - adapt_frac)
    print("\n" + "=" * 86)
    print(f"Q3  frozen source thresholds vs target-adapted "
          f"({adapt_pct:.0f}% of target adapts thresholds, "
          f"{cal_pct:.0f}% certifies risk, 80% test untouched)")
    print("=" * 86)
    print(f"  {'pipeline':<18}{'risk_src':>10}{'risk_tgt':>10}"
          f"{'AURC_src':>10}{'AURC_tgt':>10}{'orcAURC':>9}"
          f"{'Csrc':>7}{'Ctgt':>7}{'best_s':>8}{'best_t':>8}")

    report = {}
    for rep in REPRESENTATIONS:
        for target in TARGETS:
            if (rep, target) not in preds:
                continue
            pred = preds[(rep, target)]
            three = split_target(pred, adapt_frac)

            src = evaluate_run(pred, PRIMARY_SCORE, adapt_on="dev")
            tgt = evaluate_run(three, PRIMARY_SCORE, adapt_on="adapt")
            report[f"{rep}__{target}"] = {
                "source_thresholds": {k: src[k] for k in
                                      ("classification", "selective")},
                "target_thresholds": {k: tgt[k] for k in
                                      ("classification", "selective")},
            }
            def at_alpha(res, key):
                return next(e[key] for e in res["alpha_sweep"]
                            if abs(e["alpha"] - HEADLINE_ALPHA) < 1e-9)

            print(f"  {rep + '/' + target:<18}"
                  f"{src['classification']['full_coverage_risk']:>10.4f}"
                  f"{tgt['classification']['full_coverage_risk']:>10.4f}"
                  f"{src['selective']['aurc']:>10.4f}"
                  f"{tgt['selective']['aurc']:>10.4f}"
                  f"{tgt['selective']['oracle_aurc']:>9.4f}"
                  f"{at_alpha(src, 'test_coverage'):>7.3f}"
                  f"{at_alpha(tgt, 'test_coverage'):>7.3f}"
                  f"{at_alpha(src, 'best_possible_coverage'):>8.3f}"
                  f"{at_alpha(tgt, 'best_possible_coverage'):>8.3f}")

    (out_dir / "threshold_adaptation.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\n  Csrc/Ctgt = certified coverage at alpha={HEADLINE_ALPHA:.2f}; "
          "best_s/best_t = the same with a label-peeking\n"
          "  threshold, so a gap between C and best is calibration "
          "conservatism while a gap between\n  best and the oracle is a "
          "ranking limitation.")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "artifacts" / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "tables")
    ap.add_argument("--adapt-frac", type=float, default=0.5,
                    help="fraction of the 20%% target calibration split used for "
                         "threshold adaptation (0.5 => 10%% adapt + 10%% risk-cal)")
    ap.add_argument("--skip-q3", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    preds = load_all(args.runs)
    if not preds:
        raise SystemExit(f"no predictions.npz found under {args.runs}; "
                         "run 04_calibrate_eval.py first")
    print(f"Loaded cached logits for {len(preds)}/"
          f"{len(REPRESENTATIONS) * len(TARGETS)} pipelines "
          "(no GPU, no ECG decoding).")

    question_one_and_two(preds, args.out)
    if not args.skip_q3:
        question_three(preds, args.adapt_frac, args.out)
    print(f"\nWrote JSON summaries to {args.out}")


if __name__ == "__main__":
    main()
