"""Record-level confidence scores for multi-label selective prediction.

All scores are built from per-class *decision margins* in logit space,

    d_k(x) = logit p_k(x) - logit t_k

so a score measures distance from the multi-label decision boundary rather than
raw probability.  ``max_k p_k`` is deliberately included only as a baseline: in
a multi-label problem it says nothing about the classes the model called
negative, and it measures worse than useless in practice.

Why the margins are normalised
------------------------------
Diagnoses differ widely in difficulty, so ``|d_k|`` has a very different spread
per class.  Averaging raw margins lets one naturally high-margin class dominate
a naturally hard one.  Each class is therefore divided by a robust scale ``s_k``
estimated **on the development split of the training sources only** -- never on
the target -- so the score itself introduces no target leakage.

Why not the minimum
-------------------
``min_k`` asks whether *every* one of K binary decisions is far from flipping.
The chance that at least one output sits near its boundary grows quickly with K,
so with K = 13 the minimum saturates and stops ranking records. It is kept here
only so that failure can be reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-7
_MAD_TO_SIGMA = 1.4826          # makes MAD a consistent estimator of sigma

# The score used by the study.  Chosen by the nested, training-source-only
# selection in ``scripts/09_inner_score_selection.py`` and then frozen, so it is
# never picked on the strength of held-out-source test results.
#
# Result of that selection (36 inner folds, mean AURC):
#     neg_tc_entropy    0.3191  <- selected
#     neg_mean_entropy  0.3234
#     mean_margin       0.3388
# It won on 3 of the 4 outer targets; Chapman preferred plain entropy by 0.003,
# which is why the choice is recorded as non-unanimous rather than clean.
PRIMARY_SCORE = "neg_tc_entropy"

# The three candidates the inner selection is allowed to choose between.
# Deliberately short: each is a one-line definition with no fitted parameters
# beyond the class thresholds themselves.
INNER_CANDIDATES = ("mean_margin", "neg_mean_entropy", "neg_tc_entropy")

# Everything computed for reporting, including the scores kept as negative
# results (see docstring for why ``min`` and ``max_prob`` are here).
SCORE_NAMES = (
    "mean_margin",
    "neg_mean_entropy",
    "neg_tc_entropy",
    "mean_norm_margin",
    "q25_norm_margin",
    "min_norm_margin",
    "max_prob",
)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def signed_margins(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """``d_k(x) = logit p_k(x) - logit t_k``, shape ``(N, K)``."""
    return logit(probs) - logit(thresholds)[None, :]


@dataclass(frozen=True)
class MarginScaler:
    """Per-class robust margin scale, fitted on development data only."""

    scale: np.ndarray            # (K,)

    @classmethod
    def fit(cls, probs: np.ndarray, thresholds: np.ndarray,
            floor: float = 1e-3) -> "MarginScaler":
        d = signed_margins(probs, thresholds)
        mad = np.median(np.abs(d - np.median(d, axis=0, keepdims=True)), axis=0)
        scale = _MAD_TO_SIGMA * mad
        # A degenerate class (constant logit) would otherwise divide by zero and
        # swamp the average; fall back to 1.0 so it contributes unscaled.
        scale = np.where(scale > floor, scale, 1.0)
        return cls(scale=scale)

    def normalised(self, probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        return np.abs(signed_margins(probs, thresholds)) / self.scale[None, :]


def binary_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def threshold_centred_probs(probs: np.ndarray,
                            thresholds: np.ndarray) -> np.ndarray:
    """Re-centre each class probability on its own decision threshold.

    ``q_k = sigmoid(logit p_k - logit t_k)``, so ``q_k = 0.5`` exactly at the
    diagnostic decision boundary.  Plain entropy treats ``p_k = 0.5`` as the
    point of maximum uncertainty, which is wrong whenever the fitted threshold
    ``t_k`` is not 0.5 -- and for the rarer classes here it is far from it.
    """
    return 1.0 / (1.0 + np.exp(-signed_margins(probs, thresholds)))


def all_scores(probs: np.ndarray, thresholds: np.ndarray,
               scaler: MarginScaler) -> dict[str, np.ndarray]:
    """Every candidate score, higher meaning more confident."""
    raw = np.abs(signed_margins(probs, thresholds))
    norm = scaler.normalised(probs, thresholds)
    q = threshold_centred_probs(probs, thresholds)
    return {
        # --- the three inner-selection candidates ---
        "mean_margin": raw.mean(axis=1),
        "neg_mean_entropy": -binary_entropy(probs).mean(axis=1),
        "neg_tc_entropy": -binary_entropy(q).mean(axis=1),
        # --- reported for comparison ---
        "mean_norm_margin": norm.mean(axis=1),
        "q25_norm_margin": np.quantile(norm, 0.25, axis=1),
        "min_norm_margin": norm.min(axis=1),
        "max_prob": np.asarray(probs).max(axis=1),
    }


def score(name: str, probs: np.ndarray, thresholds: np.ndarray,
          scaler: MarginScaler) -> np.ndarray:
    scores = all_scores(probs, thresholds, scaler)
    if name not in scores:
        raise ValueError(f"unknown score {name!r}; choose from {sorted(scores)}")
    return scores[name]
