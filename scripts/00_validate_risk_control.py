"""Monte-Carlo check that the calibration procedure really controls risk.

The paper's central claim is that a threshold chosen on a small target-source
calibration subset keeps the selective risk at or below ``alpha`` on the
untouched test subset.  That claim rests entirely on ``ltt.calibrate_threshold``
being a valid procedure, so it is checked here directly on synthetic data with a
known risk structure, before any ECG result is produced.

Success criterion: over many independent calibration/test draws, the fraction of
runs whose *test* risk exceeds ``alpha`` must be at most ``delta``.  The naive
alternative -- picking the threshold where the empirical calibration risk first
drops below ``alpha`` -- is run alongside to show what the guarantee buys.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgsr.ltt import calibrate_threshold                    # noqa: E402
from ecgsr.selective import selective_risk, threshold_grid   # noqa: E402


def synthetic(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Confidence scores and bounded losses with a realistic, noisy coupling.

    Loss falls with confidence but never reaches zero, so no threshold achieves
    arbitrarily low risk -- the regime where a sloppy calibration procedure
    actually fails.
    """
    scores = rng.gamma(shape=2.0, scale=1.5, size=n)
    p_bad = 1.0 / (1.0 + np.exp(1.2 * (scores - 2.2)))
    losses = rng.beta(2.0, 3.0, size=n) * (p_bad > rng.uniform(size=n))
    return scores, np.clip(losses, 0.0, 1.0)


def naive_threshold(losses: np.ndarray, scores: np.ndarray,
                    grid: np.ndarray, alpha: float) -> float:
    """Lowest grid threshold whose empirical calibration risk is <= alpha."""
    for tau in np.sort(grid):
        _, risk = selective_risk(losses, scores, tau)
        if risk <= alpha:
            return float(tau)
    return float(np.max(grid))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--n-cal", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=8000)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = {"LTT (calibrated)": [], "naive (empirical)": []}

    for _ in range(args.trials):
        s_cal, l_cal = synthetic(rng, args.n_cal)
        s_test, l_test = synthetic(rng, args.n_test)
        grid = threshold_grid(s_cal, n_grid=200)

        cal = calibrate_threshold(l_cal, s_cal, grid, args.alpha, args.delta)
        cov, risk = selective_risk(l_test, s_test, cal.tau)
        rows["LTT (calibrated)"].append((cov, risk, cal.controlled))

        tau_naive = naive_threshold(l_cal, s_cal, grid, args.alpha)
        cov_n, risk_n = selective_risk(l_test, s_test, tau_naive)
        rows["naive (empirical)"].append((cov_n, risk_n, True))

    print(f"alpha = {args.alpha}   delta = {args.delta}   "
          f"n_cal = {args.n_cal}   trials = {args.trials}\n")
    print(f"{'procedure':<20}{'mean cov':>10}{'mean risk':>11}"
          f"{'P(test risk > alpha)':>23}")

    violation = {}
    for name, recs in rows.items():
        cov = np.array([r[0] for r in recs])
        risk = np.array([r[1] for r in recs])
        viol = float((risk > args.alpha).mean())
        violation[name] = viol
        print(f"{name:<20}{cov.mean():>10.3f}{risk.mean():>11.4f}{viol:>23.3f}")

    ltt_viol = violation["LTT (calibrated)"]
    print()
    if ltt_viol <= args.delta:
        print(f"PASS: LTT violation rate {ltt_viol:.3f} <= delta = {args.delta}")
    else:
        print(f"FAIL: LTT violation rate {ltt_viol:.3f} > delta = {args.delta}")
        raise SystemExit(1)

    naive_viol = violation["naive (empirical)"]
    print(f"      naive violation rate {naive_viol:.3f} "
          f"({'also within' if naive_viol <= args.delta else 'exceeds'} delta) "
          "-- the gap is what the guarantee is paying for.")


if __name__ == "__main__":
    main()
