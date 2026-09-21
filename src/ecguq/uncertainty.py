"""Model uncertainty, decision confidence, and reference uncertainty.

Three quantities, deliberately kept apart because they answer different
questions and are easy to conflate.

**One score throughout.** Confidence is the threshold-aware decision margin

    C = |logit(pbar) - logit(t_c)|          large C = confident

and uncertainty is simply its negation, ``U = -C``, so that plots can be
labelled "uncertainty" without introducing a second quantity. Only the
ordering matters for referral, so no bounded transformation is imposed;
inventing one would add a definition the reader has to check without changing
any ranking. Normalised entropy and ensemble variance remain available as
sensitivity checks and are not reported.

The margin is threshold-aware, which matters here: at 2% prevalence the
operating points sit far from 0.5, and a score centred on 0.5 would call a
prediction "confident" that is in fact sitting on its own decision boundary.

**Reference uncertainty** ``D = 1[reader 1 disagrees with reader 2]`` is a
property of the label, not the model.  It is not available at inference time,
so it may never select records; it is applied only to stratify results after
the retained set is fixed.  Selecting on it would measure an oracle rather
than a referral policy.

PTB-XL statement likelihood is retained in the loaders for later work but is
not part of this study.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-7
# PTB-XL stores 0 for "likelihood not specified", which is not the same as
# "0% likely". Treating it as a value would invert its meaning.
LIKELIHOOD_UNSPECIFIED = 0.0
LIKELIHOOD_SCALE = (15.0, 35.0, 50.0, 80.0, 100.0)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def ensemble_mean(probs: np.ndarray) -> np.ndarray:
    """``(M, N, C)`` member probabilities -> ``(N, C)`` ensemble probability."""
    return np.asarray(probs, dtype=np.float64).mean(axis=0)


def model_uncertainty(probs: np.ndarray) -> np.ndarray:
    """``U = Var_m p^(m)``, the disagreement between ensemble members.

    Population variance over the M members, not the sample variance: the
    members are the ensemble, not a sample from a larger pool.
    """
    return np.asarray(probs, dtype=np.float64).var(axis=0)


def predictive_uncertainty(pbar: np.ndarray) -> np.ndarray:
    """``U``: normalised binary predictive entropy of the ensemble mean.

        U = H(pbar) / log 2,    0 = certain, 1 = maximally uncertain

    This is *the* uncertainty of the study. One quantity does both jobs -- it
    is the value compared against cardiologist disagreement, and the value
    predictions are ranked by for referral -- so the association result and
    the referral result cannot come from two differently-behaved scores.

    Bounded in [0, 1] and comparable across diagnoses, which is what allows a
    single pooled ranking over all (record, diagnosis) pairs.
    """
    p = np.clip(np.asarray(pbar, dtype=np.float64), EPS, 1.0 - EPS)
    h = -(p * np.log(p) + (1.0 - p) * np.log1p(-p))
    return h / np.log(2.0)


def decision_confidence(pbar: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """``C = |logit(pbar) - logit(t_c)|``, distance from the operating point."""
    return np.abs(logit(pbar) - logit(np.asarray(thresholds))[None, :])


def uncertainty(pbar: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """``U = -C``: the reported uncertainty, larger meaning less certain.

    A sign flip on the decision margin, nothing more. Referral ranks by it,
    Panel A compares it across reader agreement, and both therefore use
    literally the same numbers.
    """
    return -decision_confidence(pbar, thresholds)


def predict(pbar: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (np.asarray(pbar) >= np.asarray(thresholds)[None, :]).astype(np.int8)


def fit_thresholds(y: np.ndarray, pbar: np.ndarray,
                   grid: int = 199) -> np.ndarray:
    """Per-class threshold maximising F1 on the validation split.

    Fitted on validation only, then frozen. Choosing a threshold on the data it
    is later evaluated on would make every downstream confidence number
    optimistic, since ``C`` is defined relative to that threshold.
    """
    y = np.asarray(y)
    pbar = np.asarray(pbar, dtype=np.float64)
    out = np.full(y.shape[1], 0.5)
    cand = np.linspace(1.0 / (grid + 1), grid / (grid + 1), grid)
    for c in range(y.shape[1]):
        yc, pc = y[:, c], pbar[:, c]
        if yc.sum() == 0:
            continue
        best, best_f1 = 0.5, -1.0
        for t in cand:
            pred = pc >= t
            tp = float((pred & (yc == 1)).sum())
            if tp == 0:
                continue
            prec = tp / float(pred.sum())
            rec = tp / float((yc == 1).sum())
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best, best_f1 = float(t), f1
        out[c] = best
    return out


def reader_disagreement(y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """``D = 1[y1 != y2]`` per (record, diagnosis)."""
    return (np.asarray(y1) != np.asarray(y2)).astype(np.int8)


def likelihood_specified(likelihood: np.ndarray) -> np.ndarray:
    """Mask of entries carrying an actual likelihood.

    Guards the most dangerous mistake available in PTB-XL: a stored 0 means
    the annotator did not qualify the statement, so including it as a low
    value would manufacture confident-looking disagreement that is really
    missing metadata.
    """
    return np.asarray(likelihood, dtype=np.float64) > LIKELIHOOD_UNSPECIFIED
