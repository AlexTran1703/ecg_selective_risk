r"""Tests for model uncertainty, confidence, and threshold fitting.

    .\.venv\Scripts\python.exe -m pytest tests/test_uncertainty.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecguq.uncertainty import (decision_confidence, ensemble_mean,   # noqa: E402
                               fit_thresholds, likelihood_specified,
                               model_uncertainty, predict,
                               predictive_uncertainty, reader_disagreement)


def test_uncertainty_is_zero_when_members_agree():
    p = np.full((5, 20, 6), 0.3)
    assert np.allclose(model_uncertainty(p), 0.0)


def test_uncertainty_grows_with_member_spread():
    tight = np.stack([np.full((10, 3), 0.5 + d) for d in (-.01, 0, .01)])
    wide = np.stack([np.full((10, 3), 0.5 + d) for d in (-.4, 0, .4)])
    assert model_uncertainty(wide).mean() > model_uncertainty(tight).mean()


def test_reported_uncertainty_is_normalised_entropy_of_the_ensemble_mean():
    """U is bounded in [0, 1] with the endpoints where they should be."""
    assert predictive_uncertainty(np.array([[0.5]]))[0, 0] == pytest.approx(1.0)
    assert predictive_uncertainty(np.array([[0.999999]]))[0, 0] < 0.01
    assert predictive_uncertainty(np.array([[1e-6]]))[0, 0] < 0.01
    u = predictive_uncertainty(np.random.default_rng(0).random((50, 4)))
    assert np.all((u >= 0) & (u <= 1))


def test_entropy_and_member_variance_answer_different_questions():
    """Why only one of them is reported.

    Members unanimously predicting 0.5 carry maximal entropy but zero
    disagreement; members split between 0 and 1 average to 0.5 with the same
    entropy and maximal disagreement. The study reports entropy of the
    ensemble mean, so this distinction is a documented limitation rather than
    a hidden one, and member variance stays available as a sensitivity check.
    """
    agree = np.stack([np.full((8, 2), 0.5)] * 4)
    split = np.stack([np.zeros((8, 2)), np.ones((8, 2))] * 2)
    assert np.allclose(predictive_uncertainty(ensemble_mean(agree)),
                       predictive_uncertainty(ensemble_mean(split)), atol=1e-9)
    assert model_uncertainty(agree).mean() == pytest.approx(0.0)
    assert model_uncertainty(split).mean() > 0.2


def test_confidence_is_zero_exactly_at_the_threshold():
    pbar = np.array([[0.3, 0.7]])
    t = np.array([0.3, 0.5])
    c = decision_confidence(pbar, t)
    assert c[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert c[0, 1] > 0


def test_confidence_is_symmetric_about_the_threshold():
    """A prediction just above and just below the operating point is equally
    close to flipping, so both must rank the same."""
    t = np.array([0.5])
    a = decision_confidence(np.array([[0.6]]), t)[0, 0]
    b = decision_confidence(np.array([[0.4]]), t)[0, 0]
    assert a == pytest.approx(b, rel=1e-9)


def test_confidence_and_uncertainty_can_disagree():
    """They are different questions, which is why both are reported.

    The ensemble can be far from the threshold on average while its members
    disagree sharply about the probability.
    """
    members = np.stack([np.array([[0.55]]), np.array([[0.999]])])
    pbar = ensemble_mean(members)
    conf = decision_confidence(pbar, np.array([0.5]))[0, 0]
    unc = model_uncertainty(members)[0, 0]
    assert conf > 1.0 and unc > 0.04


def test_thresholds_are_fitted_per_class_and_beat_a_fixed_half():
    """A 0.5 default is wrong for rare classes; F1 fitting must improve on it."""
    rng = np.random.default_rng(0)
    n = 4000
    y = np.zeros((n, 2), dtype=int)
    y[:80, 0] = 1                      # 2% prevalence
    y[:2000, 1] = 1                    # 50% prevalence
    p = np.clip(0.15 * y + rng.normal(0, 0.12, (n, 2)) + 0.1, 0.01, 0.99)
    t = fit_thresholds(y, p)

    def f1(pred, yc):
        tp = (pred & (yc == 1)).sum()
        if tp == 0:
            return 0.0
        return 2 * tp / (pred.sum() + (yc == 1).sum())

    for c in (0, 1):
        assert f1(p[:, c] >= t[c], y[:, c]) >= f1(p[:, c] >= 0.5, y[:, c])


def test_predict_matches_the_fitted_thresholds():
    pbar = np.array([[0.4, 0.6], [0.2, 0.9]])
    t = np.array([0.3, 0.7])
    assert np.array_equal(predict(pbar, t), np.array([[1, 0], [0, 1]]))


def test_reader_disagreement_is_per_label():
    y1 = np.array([[1, 0, 1]])
    y2 = np.array([[1, 1, 0]])
    assert np.array_equal(reader_disagreement(y1, y2), np.array([[0, 1, 1]]))


def test_unspecified_likelihood_is_excluded_not_treated_as_zero():
    """PTB-XL stores 0 for 'not specified'. Reading it as 0% would invert it."""
    L = np.array([0.0, 15.0, 50.0, 100.0])
    keep = likelihood_specified(L)
    assert keep.tolist() == [False, True, True, True]
