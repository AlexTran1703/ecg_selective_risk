"""Selective prediction for multi-label ECG.

Confidence
    For a frozen classifier with per-class decision thresholds ``t_k``, the
    record-level score is the smallest distance of any class decision from its
    own boundary, in logit space:

        m_k(x) = |logit p_k(x) - logit t_k|,      s(x) = min_k m_k(x)

    A record is confident only when *every* diagnostic decision it makes is far
    from flipping.  ``max_k p_k`` is not used: in a multi-label problem it says
    nothing about the classes the model declared negative.

Risk
    Record-wise Jaccard loss between the true and predicted label sets,

        L_i = 1 - |Y_i n Y^_i| / |Y_i u Y^_i|,    L_i in [0, 1]

    which degrades gracefully when only part of a multi-label set is recovered,
    unlike exact-match accuracy, and is not dominated by the many true negatives
    the way Hamming loss is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def predict_sets(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """``(N, K)`` probabilities -> ``(N, K)`` binary predictions."""
    return (np.asarray(probs) >= np.asarray(thresholds)[None, :]).astype(np.int8)


def margin_score(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Record-level confidence ``s(x) = min_k |logit p_k - logit t_k|``."""
    margins = np.abs(_logit(probs) - _logit(thresholds)[None, :])
    return margins.min(axis=1)


def jaccard_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Record-wise Jaccard loss in ``[0, 1]``.

    A record whose true and predicted label sets are both empty counts as a
    perfect prediction (``L = 0``), which is the sensible reading of 0/0 here:
    the model correctly declared no retained diagnosis.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    inter = np.logical_and(y_true, y_pred).sum(axis=1).astype(np.float64)
    union = np.logical_or(y_true, y_pred).sum(axis=1).astype(np.float64)
    # union == 0 leaves the initialised 1.0 in place, i.e. J = 1 and L = 0.
    jaccard = np.divide(inter, union, out=np.ones_like(union), where=union > 0)
    return 1.0 - jaccard


# --------------------------------------------------------------------------- #
#  Risk-coverage analysis
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskCoverage:
    thresholds: np.ndarray      # tau, ascending
    coverage: np.ndarray        # C(tau)
    risk: np.ndarray            # R(tau) = E[L | s >= tau]


def selective_risk(losses: np.ndarray, scores: np.ndarray, tau: float) -> tuple[float, float]:
    """Return ``(coverage, risk)`` at a single threshold."""
    keep = np.asarray(scores) >= tau
    cov = float(keep.mean())
    risk = float(np.asarray(losses)[keep].mean()) if keep.any() else 0.0
    return cov, risk


def risk_coverage_curve(losses: np.ndarray, scores: np.ndarray) -> RiskCoverage:
    """Exact risk-coverage curve over every achievable coverage level."""
    losses = np.asarray(losses, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")          # most confident first
    sorted_losses = losses[order]
    n = len(losses)

    cum = np.cumsum(sorted_losses)
    k = np.arange(1, n + 1)
    return RiskCoverage(thresholds=scores[order], coverage=k / n, risk=cum / k)


def oracle_curve(losses: np.ndarray) -> RiskCoverage:
    """Best risk-coverage curve any confidence score could achieve.

    Ranks records by their *true* loss, so it is not a usable selector -- it is
    the diagnostic that separates two very different failures:

      * oracle ``C@R<=alpha`` is also near zero  -> the classifier itself never
        produces enough near-perfect records, and no confidence score can help;
      * oracle is high but the achieved coverage is low -> the predictions are
        good and the *ranking* is what is failing.
    """
    losses = np.asarray(losses, dtype=np.float64)
    best = np.sort(losses)
    k = np.arange(1, len(best) + 1)
    return RiskCoverage(thresholds=best, coverage=k / len(best),
                        risk=np.cumsum(best) / k)


def oracle_coverage_at_risk(losses: np.ndarray, alpha: float) -> float:
    """Largest coverage achievable at risk ``<= alpha`` by *any* ranking."""
    rc = oracle_curve(losses)
    ok = rc.risk <= alpha
    return float(rc.coverage[ok].max()) if ok.any() else 0.0


def aurc(losses: np.ndarray, scores: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better).

    Averaged over the ``n`` achievable coverage levels, so it is comparable
    across pipelines and across held-out sources of different size.
    """
    return float(risk_coverage_curve(losses, scores).risk.mean())


def coverage_at_risk(losses: np.ndarray, scores: np.ndarray, alpha: float) -> float:
    """Largest coverage whose *empirical* selective risk stays at or below
    ``alpha``.  This is the oracle operating point -- it peeks at the labels of
    the set it is evaluated on, so it is reported only as an upper reference for
    the calibrated result, never as the headline number.
    """
    rc = risk_coverage_curve(losses, scores)
    ok = rc.risk <= alpha
    return float(rc.coverage[ok].max()) if ok.any() else 0.0


def threshold_grid(scores: np.ndarray, n_grid: int = 200) -> np.ndarray:
    """Candidate thresholds, fixed before looking at any risk value.

    Score quantiles are used so the grid spans the achievable coverage range
    evenly rather than the (unbounded) logit-margin range.
    """
    qs = np.linspace(0.0, 1.0, n_grid)
    grid = np.unique(np.quantile(np.asarray(scores, dtype=np.float64), qs))
    return np.sort(grid)
