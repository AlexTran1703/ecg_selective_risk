"""Classification metrics and per-class decision thresholds.

The decision thresholds ``t_k`` are fitted **once**, on the development split of
the three *training* sources, and then frozen.  They are never re-fitted on the
target source, and never touched by the test split -- otherwise the selective
risk reported later would be measuring threshold tuning rather than abstention.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _valid_classes(y: np.ndarray) -> np.ndarray:
    """Classes with both a positive and a negative example (AUROC is undefined
    otherwise; such a class is skipped rather than silently scored as 0.5)."""
    pos = y.sum(axis=0)
    return (pos > 0) & (pos < len(y))


def _sanitise(probs: np.ndarray) -> np.ndarray:
    """Replace non-finite scores with a neutral 0.5.

    A metric is a monitoring tool and must not be able to kill a multi-hour
    training run: if the model emits NaN, that should surface as a degraded
    score (and the non-finite batch counter in the training loop), not as an
    exception from deep inside scikit-learn.
    """
    probs = np.asarray(probs, dtype=np.float64)
    return np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)


def macro_auroc(y: np.ndarray, probs: np.ndarray) -> float:
    ok = _valid_classes(y)
    if not ok.any():
        return float("nan")
    return float(roc_auc_score(y[:, ok], _sanitise(probs)[:, ok], average="macro"))


def macro_auprc(y: np.ndarray, probs: np.ndarray) -> float:
    ok = _valid_classes(y)
    if not ok.any():
        return float("nan")
    return float(average_precision_score(y[:, ok], _sanitise(probs)[:, ok],
                                         average="macro"))


def per_class_auroc(y: np.ndarray, probs: np.ndarray) -> np.ndarray:
    probs = _sanitise(probs)
    out = np.full(y.shape[1], np.nan)
    for k in np.flatnonzero(_valid_classes(y)):
        out[k] = roc_auc_score(y[:, k], probs[:, k])
    return out


def fit_thresholds(y: np.ndarray, probs: np.ndarray,
                   n_grid: int = 199, floor: float = 0.01,
                   ceil: float = 0.99) -> np.ndarray:
    """Per-class threshold maximising that class's F1 on the development split.

    Ties are broken towards the *larger* threshold, i.e. the more conservative
    decision, which keeps the resulting label sets from being inflated by
    near-degenerate classes.
    """
    y = np.asarray(y)
    probs = _sanitise(probs)
    grid = np.linspace(floor, ceil, n_grid)
    thresholds = np.full(probs.shape[1], 0.5)

    for k in range(probs.shape[1]):
        yk, pk = y[:, k].astype(bool), probs[:, k]
        if yk.sum() == 0:
            continue
        preds = pk[None, :] >= grid[:, None]             # (G, N)
        tp = (preds & yk[None, :]).sum(axis=1)
        fp = (preds & ~yk[None, :]).sum(axis=1)
        fn = (~preds & yk[None, :]).sum(axis=1)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn,
                       out=np.zeros(len(grid)), where=(2 * tp + fp + fn) > 0)
        best = f1.max()
        thresholds[k] = grid[np.flatnonzero(f1 >= best - 1e-12)[-1]]
    return thresholds


def macro_f1(y: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> float:
    preds = probs >= thresholds[None, :]
    scores = []
    for k in range(y.shape[1]):
        yk = y[:, k].astype(bool)
        pk = preds[:, k]
        denom = 2 * (yk & pk).sum() + (pk & ~yk).sum() + (~pk & yk).sum()
        if denom:
            scores.append(2 * (yk & pk).sum() / denom)
    return float(np.mean(scores)) if scores else float("nan")
