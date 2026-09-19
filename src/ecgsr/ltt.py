"""Finite-sample risk control for the abstention threshold (Learn then Test).

The point of this module is the difference between

    picking tau such that the *empirical* calibration risk is <= alpha,

which is optimistically biased -- the same data chose tau and measured it --
and

    picking tau such that the risk constraint is *certified* at level delta,

which is what actually transfers to the untouched test split.

Two complications are handled explicitly.

1.  Selective risk is a ratio, ``R(tau) = E[L . g_tau] / E[g_tau]``, not a plain
    mean, so a bound for a bounded mean does not apply to it directly.  Rather
    than bounding numerator and denominator separately (valid but very lossy),
    we use the equivalent linear form

        R(tau) <= alpha   <=>   E[(L - alpha) . g_tau] <= 0

    and test the shifted variable ``v = (L - alpha) . g_tau + alpha``, which is
    supported on ``[0, 1]`` exactly when ``L`` is.  That turns the constraint
    into a single bounded-mean test over the *whole* calibration set, with no
    union bound and no loss of sample size.

2.  The threshold is chosen from a grid, which is multiple testing.  Selection
    is therefore made family-wise valid at level ``delta`` by Bonferroni across
    the grid.  Fixed sequence testing is *not* usable here: at the conservative
    end of the grid coverage approaches zero, the constraint holds with
    equality, no hypothesis can be rejected, and a fixed sequence would abort
    before reaching any useful threshold.

Reference: Angelopoulos et al., "Learn then Test: Calibrating Predictive
Algorithms to Achieve Risk Control"; Bates et al., "Distribution-Free,
Risk-Controlling Prediction Sets" for the Hoeffding-Bentkus bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, binom

_E = float(np.e)


# --------------------------------------------------------------------------- #
#  Bounds for a bounded mean
# --------------------------------------------------------------------------- #
def _h1(a: float, b: float) -> float:
    """KL divergence between Bernoulli(a) and Bernoulli(b)."""
    a = min(max(a, 1e-12), 1 - 1e-12)
    b = min(max(b, 1e-12), 1 - 1e-12)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def hoeffding_bentkus_pvalue(r_hat: float, r_null: float, n: int) -> float:
    """p-value for ``H0: true mean >= r_null``, given empirical mean ``r_hat``.

    The minimum of the Hoeffding and Bentkus tail bounds; valid for any loss
    supported on ``[0, 1]``.
    """
    if n <= 0:
        return 1.0
    if r_hat >= r_null:
        return 1.0
    hoeffding = float(np.exp(-n * _h1(r_hat, r_null)))
    bentkus = float(_E * binom.cdf(np.ceil(n * r_hat), n, r_null))
    return float(min(1.0, hoeffding, bentkus))


def mean_ucb(r_hat: float, n: int, delta: float) -> float:
    """One-sided upper confidence bound on the mean of ``[0, 1]``-valued data."""
    if n <= 0 or r_hat >= 1.0:
        return 1.0
    lo, hi = r_hat, 1.0
    for _ in range(60):                  # the p-value is decreasing in r_null
        mid = 0.5 * (lo + hi)
        if hoeffding_bentkus_pvalue(r_hat, mid, n) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def binomial_lcb(successes: int, n: int, delta: float) -> float:
    """Clopper-Pearson one-sided lower confidence bound on a proportion."""
    if n <= 0 or successes <= 0:
        return 0.0
    if successes >= n:
        return float(delta ** (1.0 / n))
    return float(beta.ppf(delta, successes, n - successes + 1))


# --------------------------------------------------------------------------- #
#  Selective-risk hypothesis test
# --------------------------------------------------------------------------- #
def _shifted_loss(losses: np.ndarray, keep: np.ndarray, alpha: float) -> np.ndarray:
    """``v = (L - alpha) . g + alpha``, supported on ``[0, 1]``.

    ``E[v] <= alpha`` is exactly equivalent to ``E[L | g] <= alpha`` whenever
    coverage is positive, so a bounded-mean test on ``v`` certifies the
    selective-risk constraint without any ratio approximation.
    """
    return np.where(keep, losses, alpha).astype(np.float64)


# Why the conditional test is the default
# --------------------------------------
# The linear form above is valid and uses the whole calibration set, but its test
# statistic is ``vbar = alpha - C(tau) . (alpha - R(tau))``: the signal is the
# *product* of coverage and the risk margin. Near the feasibility boundary that
# product is tiny -- measured at most ~0.008 on this data -- so it cannot be
# separated from ``alpha`` at any realistic calibration size, and the procedure
# certifies zero coverage even where the true risk-coverage curve satisfies the
# constraint comfortably.
#
# The conditional test instead bounds ``E[L | s >= tau]`` directly, using only
# the ``m = n_sel`` accepted records. That is valid despite ``m`` being random:
# the score function is fitted on the development split of the *training*
# sources, so it is fixed before the calibration data are seen, which makes the
# pairs ``(s_i, L_i)`` i.i.d. For a fixed ``tau``, conditional on ``n_sel = m``
# the accepted losses are i.i.d. draws from ``P(L | s >= tau)``; a Hoeffding-
# Bentkus bound with ``n = m`` is therefore valid conditionally on every value of
# ``m``, hence valid unconditionally.
PVALUE_MODES = ("conditional", "linear")
DEFAULT_PVALUE_MODE = "conditional"


def selective_risk_pvalue(losses: np.ndarray, scores: np.ndarray, tau: float,
                          alpha: float,
                          mode: str = DEFAULT_PVALUE_MODE) -> float:
    """p-value for ``H0: R(tau) > alpha``; small values certify the threshold."""
    losses = np.asarray(losses, dtype=np.float64)
    keep = np.asarray(scores, dtype=np.float64) >= tau
    if keep.size == 0 or not keep.any():
        return 1.0                       # nothing accepted: nothing to certify

    if mode == "conditional":
        accepted = losses[keep]
        return hoeffding_bentkus_pvalue(float(accepted.mean()), alpha,
                                        int(accepted.size))
    if mode == "linear":
        v = _shifted_loss(losses, keep, alpha)
        return hoeffding_bentkus_pvalue(float(v.mean()), alpha, int(v.size))
    raise ValueError(f"unknown mode {mode!r}; choose from {PVALUE_MODES}")


def selective_risk_ucb(losses: np.ndarray, scores: np.ndarray, tau: float,
                       delta: float) -> float:
    """Reporting-only upper bound on ``E[L | s(X) >= tau]``.

    Bounds ``E[L . g]`` above and coverage below, each at ``delta / 2``.  This is
    strictly more conservative than the test used for selection, and is reported
    as an interpretable certified risk level rather than used to choose ``tau``.
    """
    losses = np.asarray(losses, dtype=np.float64)
    keep = np.asarray(scores, dtype=np.float64) >= tau
    n = keep.size
    if n == 0 or not keep.any():
        return 1.0
    num_ucb = mean_ucb(float(np.where(keep, losses, 0.0).mean()), n, delta / 2.0)
    cov_lcb = binomial_lcb(int(keep.sum()), n, delta / 2.0)
    if cov_lcb <= 0.0:
        return 1.0
    return float(min(1.0, num_ucb / cov_lcb))


# --------------------------------------------------------------------------- #
#  Calibration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Calibration:
    tau: float
    coverage_cal: float
    risk_cal: float
    pvalue: float
    risk_ucb: float            # reporting-only certified risk level
    alpha: float
    delta: float
    n_cal: int
    n_grid: int
    n_certified: int           # grid thresholds that passed the test
    controlled: bool           # False => no threshold could be certified


def calibrate_threshold(losses: np.ndarray, scores: np.ndarray,
                        grid: np.ndarray, alpha: float = 0.10,
                        delta: float = 0.05,
                        mode: str = DEFAULT_PVALUE_MODE) -> Calibration:
    """Maximise coverage subject to a certified risk constraint.

    Every threshold on the pre-specified grid is tested at level ``delta / M``
    (Bonferroni).  Among those certified, the one with the highest calibration
    coverage is returned; because the family-wise error rate is controlled, that
    data-dependent choice is still valid at level ``delta``.

    ``grid`` must not be derived from ``scores``: a grid built from calibration
    quantiles makes the candidate set itself data-dependent.  Callers should
    build it from the development split (see ``analysis.evaluate_run``).
    """
    losses = np.asarray(losses, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    grid = np.unique(np.asarray(grid, dtype=np.float64))
    m = max(1, grid.size)
    level = delta / m

    best: Calibration | None = None
    n_certified = 0
    for tau in grid:
        p = selective_risk_pvalue(losses, scores, tau, alpha, mode)
        if p > level:
            continue
        n_certified += 1
        keep = scores >= tau
        coverage = float(keep.mean())
        if best is not None and coverage <= best.coverage_cal:
            continue
        best = Calibration(
            tau=float(tau), coverage_cal=coverage,
            risk_cal=float(losses[keep].mean()) if keep.any() else 0.0,
            pvalue=float(p), risk_ucb=float("nan"),
            alpha=alpha, delta=delta, n_cal=int(scores.size),
            n_grid=m, n_certified=0, controlled=True)

    if best is None:
        # Nothing on the grid is certifiable at this alpha: abstain on
        # everything, and report that honestly rather than silently falling back
        # to an uncertified threshold.
        return Calibration(tau=float("inf"), coverage_cal=0.0, risk_cal=0.0,
                           pvalue=1.0, risk_ucb=1.0, alpha=alpha, delta=delta,
                           n_cal=int(scores.size), n_grid=m, n_certified=0,
                           controlled=False)

    return Calibration(**{**best.__dict__,
                          "n_certified": n_certified,
                          "risk_ucb": selective_risk_ucb(losses, scores,
                                                         best.tau, delta)})
