"""Selective-risk analysis that runs entirely from cached logits.

Everything downstream of the network -- decision thresholds, confidence scores,
risk-coverage curves, AURC, finite-sample calibration, the alpha sweep and the
oracle bound -- is a function of ``(logits, labels)`` alone.  Keeping it here,
separate from any GPU code, means a new confidence score or risk level costs a
few seconds of NumPy instead of a full inference pass.

Primary metric is **AURC**: it summarises the whole risk-coverage ranking and
does not depend on an arbitrary risk level.  Fixed-risk operating points
``C@R<=alpha`` are reported as interpretable secondary numbers at risk levels
declared in ``ALPHA_LEVELS`` before any test split is read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ltt import calibrate_threshold
from .metrics import fit_thresholds, macro_auprc, macro_auroc, macro_f1
from .scores import PRIMARY_SCORE, MarginScaler, all_scores
from .selective import (aurc, coverage_at_risk, jaccard_loss,
                        oracle_coverage_at_risk, oracle_curve, predict_sets,
                        risk_coverage_curve, selective_risk, threshold_grid)

# Pre-declared selective-risk levels, in reporting order. alpha = 0.20 is the
# headline operating point: 0.10 is stringent enough that most source and
# representation combinations collapse towards zero certified coverage (so it
# describes the extreme tail rather than normal selective behaviour), while a
# mean Jaccard loss of 0.30 is already permissive. Note this is selective
# *Jaccard* risk, not a patient-harm threshold.
HEADLINE_ALPHA = 0.20
ALPHA_LEVELS = (0.20, 0.10, 0.30)
ALPHA_SWEEP = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
DELTA = 0.05


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


@dataclass
class Predictions:
    """Cached network outputs for one run, split by role."""

    logits: dict[str, np.ndarray]        # split -> (N, K)
    labels: dict[str, np.ndarray]        # split -> (N, K)
    records: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def probs(self, split: str) -> np.ndarray:
        return sigmoid(self.logits[split])

    @classmethod
    def load(cls, path: Path) -> "Predictions":
        d = np.load(path, allow_pickle=False)
        splits = sorted({k.split("__")[0] for k in d.files if "__" in k})
        return cls(
            logits={s: d[f"{s}__logits"] for s in splits},
            labels={s: d[f"{s}__labels"] for s in splits},
            records={s: d[f"{s}__records"] for s in splits
                     if f"{s}__records" in d.files},
        )

    def save(self, path: Path) -> None:
        payload: dict[str, np.ndarray] = {}
        for split, lg in self.logits.items():
            payload[f"{split}__logits"] = np.asarray(lg, dtype=np.float32)
            payload[f"{split}__labels"] = np.asarray(self.labels[split], dtype=np.int8)
            if split in self.records:
                payload[f"{split}__records"] = np.asarray(self.records[split])
        np.savez_compressed(path, **payload)


# --------------------------------------------------------------------------- #
#  Threshold fitting
# --------------------------------------------------------------------------- #
def fit_decision_thresholds(pred: Predictions, adapt_on: str = "dev"
                            ) -> tuple[np.ndarray, MarginScaler]:
    """Per-class thresholds plus the margin scaler, fitted on one split.

    ``adapt_on='dev'`` is the study protocol: thresholds come from the training
    sources and transfer unchanged to the target.  ``adapt_on='cal'`` is the
    target-adapted variant, only legitimate when the target split used here is
    disjoint from the one used for risk calibration.
    """
    probs = pred.probs(adapt_on)
    labels = pred.labels[adapt_on]
    thresholds = fit_thresholds(labels, probs)
    scaler = MarginScaler.fit(probs, thresholds)
    return thresholds, scaler


# --------------------------------------------------------------------------- #
#  Score comparison
# --------------------------------------------------------------------------- #
def compare_scores(pred: Predictions, thresholds: np.ndarray,
                   scaler: MarginScaler, split: str = "test",
                   alphas: tuple[float, ...] = ALPHA_LEVELS) -> dict:
    """AURC and oracle-relative coverage for every candidate score."""
    probs = pred.probs(split)
    labels = pred.labels[split]
    losses = jaccard_loss(labels, predict_sets(probs, thresholds))

    out = {
        "full_coverage_risk": float(losses.mean()),
        "oracle": {"aurc": float(oracle_curve(losses).risk.mean()),
                   "coverage": {f"{a:.2f}": oracle_coverage_at_risk(losses, a)
                                for a in alphas}},
        "scores": {},
    }
    for name, s in all_scores(probs, thresholds, scaler).items():
        out["scores"][name] = {
            "aurc": aurc(losses, s),
            "coverage": {f"{a:.2f}": coverage_at_risk(losses, s, a) for a in alphas},
        }
    return out


# --------------------------------------------------------------------------- #
#  Per-diagnosis selective behaviour
# --------------------------------------------------------------------------- #
def per_class_selective(pred: Predictions, thresholds: np.ndarray,
                        scores: np.ndarray, tau: float,
                        split: str = "test") -> list[dict]:
    """How each diagnosis fares under abstention.

    A multi-class confusion matrix is not defined here: a record may carry
    several diagnoses at once, so "AF was confused with RBBB" has no meaning
    when both can be simultaneously correct.  Each diagnosis instead gets its
    own binary outcome, reported as

        sens_full = P(yhat_k = 1 | y_k = 1)                   before abstention
        sens_acc  = P(yhat_k = 1 | y_k = 1, accepted)         after abstention
        retention = P(accepted | y_k = 1)                     coverage of positives

    Retention is the number that matters for the selective-risk claim: a
    selector can post a respectable average risk simply by rejecting the
    diagnoses it finds hard, and only ``retention`` exposes that.
    """
    probs = pred.probs(split)
    labels = pred.labels[split]
    preds = predict_sets(probs, thresholds)
    accepted = np.asarray(scores) >= tau

    out = []
    for k in range(labels.shape[1]):
        pos = labels[:, k].astype(bool)
        n_pos = int(pos.sum())
        pos_acc = pos & accepted
        out.append({
            "class_index": k,
            "n_positive": n_pos,
            "prevalence": float(pos.mean()),
            "sens_full": float(preds[pos, k].mean()) if n_pos else float("nan"),
            "sens_accepted": (float(preds[pos_acc, k].mean())
                              if pos_acc.any() else float("nan")),
            "retention": float(pos_acc.sum() / n_pos) if n_pos else float("nan"),
        })
    return out


# --------------------------------------------------------------------------- #
#  Full evaluation for one run
# --------------------------------------------------------------------------- #
def evaluate_run(pred: Predictions, score_name: str = PRIMARY_SCORE,
                 adapt_on: str = "dev", cal_split: str = "cal",
                 test_split: str = "test", n_grid: int = 50,
                 delta: float = DELTA,
                 alpha_sweep: tuple[float, ...] = ALPHA_SWEEP) -> dict:
    """Freeze thresholds, calibrate on ``cal_split``, report ``test_split`` once."""
    thresholds, scaler = fit_decision_thresholds(pred, adapt_on)

    def parts(split):
        probs = pred.probs(split)
        labels = pred.labels[split]
        losses = jaccard_loss(labels, predict_sets(probs, thresholds))
        s = all_scores(probs, thresholds, scaler)[score_name]
        return probs, labels, losses, s

    cal_probs, cal_y, cal_losses, cal_s = parts(cal_split)
    test_probs, test_y, test_losses, test_s = parts(test_split)

    # Candidate thresholds come from the development split of the training
    # sources, so the grid is fixed before any target calibration record is
    # seen. Building it from calibration quantiles would make the candidate set
    # itself data-dependent, which the Bonferroni correction does not cover.
    _, _, _, dev_s = parts(adapt_on)
    grid = threshold_grid(dev_s, n_grid=n_grid)
    headline = calibrate_threshold(cal_losses, cal_s, grid, HEADLINE_ALPHA, delta)
    cov, risk = selective_risk(test_losses, test_s, headline.tau)

    sweep = []
    for a in sorted(set(alpha_sweep) | set(ALPHA_LEVELS)):
        c = calibrate_threshold(cal_losses, cal_s, grid, a, delta)
        cv, rk = selective_risk(test_losses, test_s, c.tau)
        sweep.append({
            "alpha": a, "tau": c.tau, "controlled": bool(c.controlled),
            "cal_coverage": c.coverage_cal, "cal_risk": c.risk_cal,
            "test_coverage": cv, "test_risk": rk,
            "risk_respected": bool(rk <= a),
            "oracle_coverage": oracle_coverage_at_risk(test_losses, a),
            "best_possible_coverage": coverage_at_risk(test_losses, test_s, a),
        })

    rc = risk_coverage_curve(test_losses, test_s)
    return {
        "per_class": per_class_selective(pred, thresholds, test_s,
                                         headline.tau, test_split),
        "score": score_name,
        "adapt_on": adapt_on,
        "delta": delta,
        "n_grid": int(grid.size),
        "n": {k: int(len(v)) for k, v in pred.labels.items()},
        "classification": {
            "macro_auroc": macro_auroc(test_y, test_probs),
            "macro_auprc": macro_auprc(test_y, test_probs),
            "macro_f1": macro_f1(test_y, test_probs, thresholds),
            "full_coverage_risk": float(test_losses.mean()),
        },
        "selective": {
            "aurc": aurc(test_losses, test_s),
            "oracle_aurc": float(oracle_curve(test_losses).risk.mean()),
            "tau": headline.tau,
            "alpha": HEADLINE_ALPHA,
            "controlled": bool(headline.controlled),
            "test_coverage": cov,
            "test_risk": risk,
            "risk_respected": bool(risk <= HEADLINE_ALPHA),
            "cal_coverage": headline.coverage_cal,
            "cal_risk": headline.risk_cal,
            "cal_risk_ucb": headline.risk_ucb,
        },
        "alpha_sweep": sweep,
        "thresholds": thresholds.tolist(),
        "margin_scale": scaler.scale.tolist(),
        "curve": {"coverage": rc.coverage, "risk": rc.risk,
                  "oracle_risk": oracle_curve(test_losses).risk},
    }
