"""Per-class selective risk, and residual risk conditioned on the reference.

Selection is always by model confidence alone.  Reference uncertainty is
applied only *after* the retained set is fixed, because a deployed model
cannot know whether two cardiologists will disagree about a record it is
about to see.  Using disagreement to choose what to keep would measure an
oracle, not a referral policy.

Risk is computed per diagnosis and then macro-averaged, rather than pooling
all (record, diagnosis) pairs into one number.  With six labels at 2-3%
prevalence, a pooled risk is dominated by true negatives and would barely move
regardless of what the model does on the positives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskCoverage:
    coverage: np.ndarray        # (n,) fraction retained, ascending
    risk: np.ndarray            # (n,) error rate among retained


def class_risk_coverage(loss: np.ndarray, conf: np.ndarray) -> RiskCoverage:
    """Risk-coverage curve for one diagnosis, ranked by decreasing confidence."""
    loss = np.asarray(loss, dtype=np.float64)
    order = np.argsort(-np.asarray(conf, dtype=np.float64), kind="stable")
    k = np.arange(1, loss.size + 1)
    return RiskCoverage(coverage=k / loss.size,
                        risk=np.cumsum(loss[order]) / k)


def _at_coverage(rc: RiskCoverage, q: float) -> float:
    i = min(int(np.searchsorted(rc.coverage, q)), rc.risk.size - 1)
    return float(rc.risk[i])


def macro_risk_at(loss: np.ndarray, conf: np.ndarray, q: float) -> float:
    """Mean over diagnoses of the selective risk at coverage ``q``."""
    loss, conf = np.asarray(loss), np.asarray(conf)
    return float(np.mean([
        _at_coverage(class_risk_coverage(loss[:, c], conf[:, c]), q)
        for c in range(loss.shape[1])]))


def macro_curve(loss: np.ndarray, conf: np.ndarray,
                grid: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.02, 1.0, 50) if grid is None else grid
    return grid, np.asarray([macro_risk_at(loss, conf, q) for q in grid])


def class_aurc(loss: np.ndarray, conf: np.ndarray) -> float:
    """Area under one diagnosis's risk-coverage curve."""
    return float(class_risk_coverage(loss, conf).risk.mean())


def macro_aurc(loss: np.ndarray, conf: np.ndarray) -> float:
    loss, conf = np.asarray(loss), np.asarray(conf)
    return float(np.mean([class_aurc(loss[:, c], conf[:, c])
                          for c in range(loss.shape[1])]))


def retained_mask(conf: np.ndarray, q: float) -> np.ndarray:
    """Per-diagnosis mask of the most confident fraction ``q``.

    Thresholded independently per column, so every diagnosis retains the same
    *proportion* of records; a shared threshold would retain wildly different
    fractions across classes whose confidence distributions differ.
    """
    conf = np.asarray(conf, dtype=np.float64)
    n = conf.shape[0]
    keep_n = max(1, int(round(q * n)))
    out = np.zeros_like(conf, dtype=bool)
    for c in range(conf.shape[1]):
        idx = np.argsort(-conf[:, c], kind="stable")[:keep_n]
        out[idx, c] = True
    return out


def stratified_residual_risk(loss: np.ndarray, conf: np.ndarray,
                             ref_uncertain: np.ndarray, q: float
                             ) -> tuple[float, float]:
    """``(risk on consensus labels, risk on disputed labels)`` among retained.

    Both strata come from the *same* confidence-ranked retained set, so the
    contrast answers: after the model has referred what it is unsure of, does
    the error that survives still land on labels the readers themselves
    disputed?
    """
    loss = np.asarray(loss, dtype=np.float64)
    ref = np.asarray(ref_uncertain).astype(bool)
    keep = retained_mask(conf, q)
    agree, disagree = [], []
    for c in range(loss.shape[1]):
        k = keep[:, c]
        a = k & ~ref[:, c]
        d = k & ref[:, c]
        if a.any():
            agree.append(loss[a, c].mean())
        if d.any():
            disagree.append(loss[d, c].mean())
    return (float(np.mean(agree)) if agree else float("nan"),
            float(np.mean(disagree)) if disagree else float("nan"))


def stratified_aurc(loss: np.ndarray, conf: np.ndarray,
                    ref_uncertain: np.ndarray,
                    grid: np.ndarray | None = None) -> tuple[float, float]:
    """AURC restricted to consensus and to disputed labels, same ranking."""
    grid = np.linspace(0.02, 1.0, 50) if grid is None else grid
    a, d = zip(*[stratified_residual_risk(loss, conf, ref_uncertain, q)
                 for q in grid])
    return (float(np.nanmean(a)), float(np.nanmean(d)))


def disputed_error_share(loss: np.ndarray, conf: np.ndarray,
                         ref_uncertain: np.ndarray, q: float) -> float:
    """``rho_D``: of the errors surviving referral, the fraction on disputed labels.

    The most directly interpretable number in the study -- if disputed labels
    are a small share of the data but a large share of the residual error,
    then part of what looks like model failure is reference disagreement.
    """
    loss = np.asarray(loss, dtype=np.float64)
    ref = np.asarray(ref_uncertain).astype(bool)
    keep = retained_mask(conf, q)
    err = loss * keep
    total = err.sum()
    if total == 0:
        return float("nan")
    return float((err * ref).sum() / total)


def disputed_prevalence(ref_uncertain: np.ndarray) -> float:
    """Baseline share of all (record, diagnosis) pairs the readers disputed."""
    return float(np.asarray(ref_uncertain).astype(bool).mean())


def retained_disputed_prevalence(conf: np.ndarray, ref_uncertain: np.ndarray,
                                 q: float) -> float:
    """``pi_D(q)``: share of the *retained* predictions that were disputed.

    The correct reference for enrichment. Referral itself removes disputed
    labels -- they tend to be the uncertain ones -- so the disputed share of
    the population changes as coverage falls. Comparing the disputed share of
    residual errors against the fixed full-cohort prevalence would credit the
    referral policy with an enrichment it did not produce.
    """
    keep = retained_mask(conf, q)
    ref = np.asarray(ref_uncertain).astype(bool)
    n = keep.sum()
    return float((keep & ref).sum() / n) if n else float("nan")


def disagreement_enrichment(loss: np.ndarray, conf: np.ndarray,
                            ref_uncertain: np.ndarray, q: float) -> float:
    """``E_D(q) = rho_D(q) / pi_D(q)`` -- the headline quantity.

        E_D = 1   disputed labels are no more common among residual errors
                  than among retained predictions generally
        E_D > 1   disputed labels are over-represented among the errors that
                  survive uncertainty-based referral

    Both numerator and denominator are computed on the same retained set, so
    the ratio isolates whether disagreement concentrates in the residual
    error rather than merely surviving referral.
    """
    rho = disputed_error_share(loss, conf, ref_uncertain, q)
    pi = retained_disputed_prevalence(conf, ref_uncertain, q)
    if not np.isfinite(rho) or not np.isfinite(pi) or pi == 0:
        return float("nan")
    return float(rho / pi)


def _ecg_rank(conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-ECG confidence and the order that retains the most confident first.

    An ECG is only as confident as its least confident label. Referral in
    practice is by tracing, not by label: if any prediction on an ECG is
    referred, a reader looks at the ECG and the remaining predictions come
    with it. Ranking on the weakest label is what that policy implies.
    """
    conf = np.asarray(conf, dtype=np.float64)
    ecg_conf = conf.min(axis=1)
    return ecg_conf, np.argsort(-ecg_conf, kind="stable")


def _retained_ecgs(conf: np.ndarray, q: float) -> np.ndarray:
    _, order = _ecg_rank(conf)
    k = max(1, int(round(q * order.size)))
    return order[:k]


def patient_risk_at(loss: np.ndarray, conf: np.ndarray, q: float) -> float:
    """Share of retained ECGs carrying at least one error.

    The clinically actionable counterpart to ``macro_risk_at``. Per-label
    referral would credit the policy with declining one prediction on a
    tracing while keeping another from the same tracing, which no reader
    workflow delivers.
    """
    err = np.asarray(loss).astype(bool).any(axis=1)
    return float(err[_retained_ecgs(conf, q)].mean())


def patient_errors_at(loss: np.ndarray, conf: np.ndarray, q: float) -> int:
    """Count of retained ECGs carrying at least one error."""
    err = np.asarray(loss).astype(bool).any(axis=1)
    return int(err[_retained_ecgs(conf, q)].sum())


def _as_ecg_flag(group: np.ndarray) -> np.ndarray:
    g = np.asarray(group).astype(bool)
    return g.any(axis=1) if g.ndim > 1 else g


def patient_risk_in_group(loss: np.ndarray, conf: np.ndarray,
                          group: np.ndarray, q: float,
                          in_group: bool = True) -> float:
    """Error prevalence among retained ECGs of one stratum.

    Ranking is global -- the same whole-ECG confidence order that produces
    overall selective risk -- and the split into strata happens only *after*
    retention. Ranking within each stratum separately would silently make
    coverage mean a different thing in each curve, so the two lines would no
    longer be comparable at a given x.
    """
    err = np.asarray(loss).astype(bool).any(axis=1)
    g = _as_ecg_flag(group)
    keep = _retained_ecgs(conf, q)
    sel = keep[g[keep]] if in_group else keep[~g[keep]]
    return float(err[sel].mean()) if sel.size else float("nan")


def patient_group_size(conf: np.ndarray, group: np.ndarray, q: float,
                       in_group: bool = True) -> int:
    """Number of retained ECGs in one stratum, the support behind the curve."""
    g = _as_ecg_flag(group)
    keep = _retained_ecgs(conf, q)
    return int(g[keep].sum() if in_group else (~g[keep]).sum())
