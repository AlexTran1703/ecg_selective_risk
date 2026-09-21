"""Clustered bootstrap at the ECG level.

Every statistic in this study is computed over (record, diagnosis) pairs, but
the six diagnoses on one ECG are not independent: they share a patient, a
recording, and a reader.  Resampling pairs would therefore understate the
confidence intervals.  Resampling whole ECGs -- all six labels travelling
together -- preserves the multilabel correlation structure.

Records rather than patients are the cluster here because both CODE-test and
the PTB-XL test fold contain one ECG per patient; if that ever stops being
true, pass explicit ``groups``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

N_REPLICATES = 1000


def clustered_ci(statistic: Callable[[np.ndarray], float], n: int,
                 replicates: int = N_REPLICATES, alpha: float = 0.05,
                 seed: int = 0, groups: np.ndarray | None = None
                 ) -> tuple[float, float, float]:
    """``(point estimate, lower, upper)`` from a percentile bootstrap.

    ``statistic`` receives an index array selecting rows (ECGs) and returns a
    scalar.  Replicates that are undefined -- a resample containing no
    disputed label, say -- are dropped rather than counted as zero, and the
    interval is taken over what remains.
    """
    rng = np.random.default_rng(seed)
    point = float(statistic(np.arange(n)))

    if groups is None:
        draw = lambda: rng.integers(0, n, n)                     # noqa: E731
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        index = {g: np.flatnonzero(groups == g) for g in uniq}

        def draw():
            picked = rng.choice(uniq, uniq.size, replace=True)
            return np.concatenate([index[g] for g in picked])

    vals = []
    for _ in range(replicates):
        v = statistic(draw())
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def fmt(point: float, lo: float, hi: float, nd: int = 3) -> str:
    if not np.isfinite(point):
        return "--"
    if not np.isfinite(lo):
        return f"{point:.{nd}f}"
    return f"{point:.{nd}f} ({lo:.{nd}f} to {hi:.{nd}f})"
