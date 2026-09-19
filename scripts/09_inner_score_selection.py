"""Choose the confidence score using training-source data only.

The outer results must not pick the score: announcing "entropy is primary
because it won on the held-out sources" is selection on the test sets.  This
script runs the same source-shift experiment *inside* the training sources.

For an outer target ``T`` the three training sources ``S = {A,B,C}`` give the
pseudo-target folds ``(A,B) -> C``, ``(A,C) -> B``, ``(B,C) -> A``.  The outer
target never appears in any of them.

A model trained on the pair ``{a,b}`` serves **both** sources it has not seen,
so only the 6 two-source pairs need training per representation (18 models, not
36).  The second fold reuses the first fold's train/dev partition and swaps in
the other source's calibration/test records -- refitting the partition would let
threshold fitting see records the network trained on.

Candidates are limited to three, each a one-line definition with no fitted
parameters beyond the class thresholds:

    mean_margin        mean_k |logit p_k - logit t_k|
    neg_mean_entropy   -mean_k H(p_k)
    neg_tc_entropy     -mean_k H(sigmoid(logit p_k - logit t_k))

Selection uses AURC only, averaged over inner folds.  The winner is written to
``artifacts/tables/score_selection.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgsr.analysis import Predictions, fit_decision_thresholds     # noqa: E402
from ecgsr.datasets import (REPRESENTATIONS, SOURCES, _stable_seed,  # noqa: E402
                            load_index)
from ecgsr.scores import INNER_CANDIDATES, SCORE_NAMES, all_scores  # noqa: E402
from ecgsr.selective import aurc, jaccard_loss, predict_sets        # noqa: E402

PY = sys.executable


def training_pairs() -> list[tuple[str, ...]]:
    """The 6 two-source training sets; each serves the 2 sources it omits."""
    return [tuple(sorted(p)) for p in itertools.combinations(SOURCES, 2)]


def run_dir_for(runs: Path, rep: str, pair: tuple[str, ...], target: str) -> Path:
    return runs / f"inner__{rep}__{'-'.join(pair)}__{target}"


def make_sibling_fold(index, first: Path, second: Path, target: str,
                      cal_frac: float = 0.20, seed: int = 0) -> None:
    """Build the second inner fold from the first fold's checkpoint.

    Keeps ``train``/``dev`` exactly as trained -- so thresholds are still fitted
    on held-out training-source records -- and replaces cal/test with the other
    unseen source.
    """
    second.mkdir(parents=True, exist_ok=True)
    s0 = np.load(first / "split.npz", allow_pickle=True)
    rows = np.flatnonzero(index.source == target)
    rng = np.random.default_rng(_stable_seed(target, seed))
    perm = rng.permutation(len(rows))
    n_cal = max(1, int(round(cal_frac * len(rows))))
    np.savez(second / "split.npz", train=s0["train"], dev=s0["dev"],
             cal=rows[perm[:n_cal]], test=rows[perm[n_cal:]],
             target=np.array(target))
    shutil.copy2(first / "model.pt", second / "model.pt")


def evaluate(run: Path, workers: int) -> bool:
    return subprocess.call([str(c) for c in [
        PY, ROOT / "scripts" / "04_calibrate_eval.py", "--run", run,
        "--workers", workers]]) == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "artifacts" / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "tables")
    ap.add_argument("--meta", type=Path, default=ROOT / "data" / "meta")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--representations", nargs="*", default=list(REPRESENTATIONS))
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    index = load_index(args.meta)
    pairs = training_pairs()
    total = len(pairs) * len(args.representations)
    print(f"Inner score selection: {len(pairs)} pairs x "
          f"{len(args.representations)} representations = {total} models, "
          "each serving 2 inner folds.\n")

    folds: list[tuple[str, tuple[str, ...], str, str, Path]] = []
    for rep in args.representations:
        for pair in pairs:
            rest = [s for s in SOURCES if s not in pair]
            for i, inner in enumerate(rest):
                outer = rest[1 - i]
                folds.append((rep, pair, inner, outer,
                              run_dir_for(args.runs, rep, pair, inner)))

    if not args.analyse_only:
        done = 0
        for rep in args.representations:
            for pair in pairs:
                rest = [s for s in SOURCES if s not in pair]
                done += 1
                first = run_dir_for(args.runs, rep, pair, rest[0])
                second = run_dir_for(args.runs, rep, pair, rest[1])
                print(f"[{done}/{total}] {rep}: train {'+'.join(pair)} -> "
                      f"folds {rest[0]}, {rest[1]}", flush=True)

                if not (first / "predictions.npz").exists():
                    cmd = [PY, ROOT / "scripts" / "03_train.py",
                           "--representation", rep, "--target", rest[0],
                           "--train-sources", *pair,
                           "--epochs", args.epochs, "--batch-size", args.batch_size,
                           "--workers", args.workers, "--runs", args.runs,
                           "--run-name", first.name]
                    if subprocess.call([str(c) for c in cmd]) != 0:
                        print(f"    TRAIN FAILED {rep} {pair}", flush=True)
                        continue
                    if not evaluate(first, args.workers):
                        print(f"    EVAL FAILED {first.name}", flush=True)
                else:
                    print(f"    cached {first.name}", flush=True)

                if not (second / "predictions.npz").exists():
                    if not (first / "model.pt").exists():
                        continue
                    make_sibling_fold(index, first, second, rest[1])
                    if not evaluate(second, args.workers):
                        print(f"    EVAL FAILED {second.name}", flush=True)
                else:
                    print(f"    cached {second.name}", flush=True)

    # --- analyse -------------------------------------------------------------
    records: list[dict] = []
    for rep, pair, inner, outer, d in folds:
        path = d / "predictions.npz"
        if not path.exists():
            continue
        pred = Predictions.load(path)
        thr, scaler = fit_decision_thresholds(pred, "dev")
        probs, labels = pred.probs("test"), pred.labels["test"]
        losses = jaccard_loss(labels, predict_sets(probs, thr))
        scores = all_scores(probs, thr, scaler)
        for name in SCORE_NAMES:
            records.append({"representation": rep, "train_pair": list(pair),
                            "inner_target": inner, "outer_target": outer,
                            "score": name, "aurc": aurc(losses, scores[name])})

    if not records:
        raise SystemExit("no inner runs found; run without --analyse-only first")

    n_folds = len({(r["representation"], r["inner_target"], tuple(r["train_pair"]))
                   for r in records})
    print("\n" + "=" * 76)
    print(f"Inner-fold AURC over {n_folds} folds "
          "(training sources only; outer targets never used)")
    print("=" * 76)

    by: dict[tuple[str, str], list[float]] = {}
    for r in records:
        by.setdefault((r["representation"], r["score"]), []).append(r["aurc"])

    print(f"\n  {'score':<20}" + "".join(f"{rep:>12}" for rep in args.representations)
          + f"{'mean':>10}   candidate")
    ranking = {}
    for name in SCORE_NAMES:
        per_rep = [float(np.mean(by.get((rep, name), [np.nan])))
                   for rep in args.representations]
        ranking[name] = float(np.nanmean(per_rep))
        print(f"  {name:<20}" + "".join(f"{v:>12.4f}" for v in per_rep)
              + f"{ranking[name]:>10.4f}   "
              + ("yes" if name in INNER_CANDIDATES else ""))

    eligible = {k: v for k, v in ranking.items() if k in INNER_CANDIDATES}
    winner = min(eligible, key=eligible.get)
    print(f"\n  selection over {INNER_CANDIDATES}:")
    for k in sorted(eligible, key=eligible.get):
        print(f"    {k:<20}{eligible[k]:.4f}"
              + ("   <- selected" if k == winner else ""))

    print("\n  winner per outer target (using only folds that exclude it):")
    per_outer = {}
    for outer in SOURCES:
        means = {}
        for name in INNER_CANDIDATES:
            vals = [r["aurc"] for r in records
                    if r["outer_target"] == outer and r["score"] == name]
            if vals:
                means[name] = float(np.mean(vals))
        if means:
            w = min(means, key=means.get)
            per_outer[outer] = {"winner": w, "aurc": means}
            print(f"    {outer:<9}{w:<20}"
                  + "  ".join(f"{k}={v:.4f}" for k, v in sorted(means.items())))

    unanimous = len({v["winner"] for v in per_outer.values()}) == 1
    print(f"\n  unanimous across outer targets: {unanimous}")
    print(f"  selected score: {winner}")

    (args.out / "score_selection.json").write_text(json.dumps({
        "candidates": list(INNER_CANDIDATES), "selected": winner,
        "unanimous_across_outer_targets": unanimous,
        "inner_mean_aurc": ranking, "per_outer_target": per_outer,
        "n_folds": n_folds, "folds": records,
        "note": "Selected on training-source folds only; outer targets unused.",
    }, indent=2), encoding="utf-8")
    print(f"  wrote {args.out / 'score_selection.json'}")
    print("\n  Next: set scores.PRIMARY_SCORE to this, then re-run "
          "04_calibrate_eval.py --reuse for every outer run.")


if __name__ == "__main__":
    main()
