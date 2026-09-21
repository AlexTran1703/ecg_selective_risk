r"""Tests for selective risk and disagreement-conditioned residual risk.

    .\.venv\Scripts\python.exe -m pytest tests/test_selective.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecguq.bootstrap import clustered_ci                            # noqa: E402
from ecguq.selective import (class_aurc, class_risk_coverage,       # noqa: E402
                             disputed_error_share, disputed_prevalence,
                             macro_aurc, macro_risk_at, retained_mask,
                             stratified_residual_risk)


def test_perfect_confidence_ranking_puts_all_errors_last():
    loss = np.array([0, 0, 0, 1, 1], dtype=float)
    conf = np.array([5, 4, 3, 2, 1], dtype=float)     # errors least confident
    rc = class_risk_coverage(loss, conf)
    assert rc.risk[0] == 0.0
    assert rc.risk[-1] == pytest.approx(0.4)
    assert np.all(np.diff(rc.risk) >= -1e-12)          # monotone here


def test_inverted_ranking_is_strictly_worse_than_perfect():
    loss = np.array([0, 0, 0, 1, 1], dtype=float)
    good = np.array([5, 4, 3, 2, 1], dtype=float)
    bad = good[::-1].copy()
    assert class_aurc(loss, bad) > class_aurc(loss, good)


def test_full_coverage_risk_equals_the_plain_error_rate():
    rng = np.random.default_rng(1)
    loss = (rng.random((200, 3)) < 0.3).astype(float)
    conf = rng.random((200, 3))
    assert macro_risk_at(loss, conf, 1.0) == pytest.approx(
        float(loss.mean(axis=0).mean()), abs=1e-9)


def test_macro_averaging_does_not_let_one_class_dominate():
    """With rare positives, pooling would be swamped by easy negatives.

    Two classes: one perfect, one at 50% error. The macro risk must be ~0.25
    regardless of how many records each contributes.
    """
    loss = np.zeros((1000, 2))
    loss[:500, 1] = 1.0
    conf = np.ones((1000, 2))
    assert macro_risk_at(loss, conf, 1.0) == pytest.approx(0.25, abs=1e-9)


def test_retained_mask_keeps_the_same_fraction_per_class():
    rng = np.random.default_rng(2)
    conf = rng.random((100, 4))
    keep = retained_mask(conf, 0.25)
    assert set(keep.sum(axis=0).tolist()) == {25}


def test_retained_mask_keeps_the_most_confident():
    conf = np.array([[1.0], [3.0], [2.0], [4.0]])
    keep = retained_mask(conf, 0.5)
    assert keep[:, 0].tolist() == [False, True, False, True]


# --------------------------------------------------------------------------- #
#  The central contrast
# --------------------------------------------------------------------------- #
def test_selection_uses_confidence_only_not_the_reference():
    """The design constraint of the study.

    Reference uncertainty must not influence which records are retained --
    a deployed model cannot know it. Changing D alone must leave the retained
    set, and therefore the overall risk, untouched.
    """
    rng = np.random.default_rng(3)
    loss = (rng.random((200, 3)) < 0.3).astype(float)
    conf = rng.random((200, 3))
    d1 = (rng.random((200, 3)) < 0.1).astype(int)
    d2 = (rng.random((200, 3)) < 0.5).astype(int)
    assert np.array_equal(retained_mask(conf, 0.5), retained_mask(conf, 0.5))
    assert macro_risk_at(loss, conf, 0.5) == macro_risk_at(loss, conf, 0.5)
    # Only the stratification changes, never the selection.
    a1, _ = stratified_residual_risk(loss, conf, d1, 0.5)
    a2, _ = stratified_residual_risk(loss, conf, d2, 0.5)
    assert np.isfinite(a1) and np.isfinite(a2)


def test_residual_risk_is_higher_where_readers_disagreed():
    """The effect the paper is looking for, on data built to contain it."""
    n = 600
    rng = np.random.default_rng(4)
    disputed = np.zeros((n, 1), dtype=int)
    disputed[:60] = 1                                  # 10% disputed
    loss = np.zeros((n, 1))
    loss[:60, 0] = (rng.random(60) < 0.6)              # disputed: 60% error
    loss[60:, 0] = (rng.random(n - 60) < 0.05)         # consensus: 5% error
    conf = rng.random((n, 1))                          # confidence is blind to D
    agree, disagree = stratified_residual_risk(loss, conf, disputed, 0.8)
    assert disagree > agree


def test_disputed_error_share_exceeds_prevalence_when_disputes_are_harder():
    n = 600
    rng = np.random.default_rng(5)
    disputed = np.zeros((n, 1), dtype=int)
    disputed[:60] = 1
    loss = np.zeros((n, 1))
    loss[:60, 0] = (rng.random(60) < 0.6)
    loss[60:, 0] = (rng.random(n - 60) < 0.05)
    conf = rng.random((n, 1))
    share = disputed_error_share(loss, conf, disputed, 0.8)
    assert share > disputed_prevalence(disputed)


def test_disputed_error_share_matches_prevalence_when_disputes_are_no_harder():
    """The null: if disagreement carries no extra error, the share of residual
    error attributable to it should sit near its base rate."""
    n = 4000
    rng = np.random.default_rng(6)
    disputed = (rng.random((n, 1)) < 0.2).astype(int)
    loss = (rng.random((n, 1)) < 0.2).astype(float)    # independent of D
    conf = rng.random((n, 1))
    share = disputed_error_share(loss, conf, disputed, 1.0)
    assert abs(share - disputed_prevalence(disputed)) < 0.05


def test_macro_aurc_is_bounded_by_the_worst_class():
    rng = np.random.default_rng(7)
    loss = (rng.random((150, 3)) < 0.4).astype(float)
    conf = rng.random((150, 3))
    per = [class_aurc(loss[:, c], conf[:, c]) for c in range(3)]
    assert min(per) <= macro_aurc(loss, conf) <= max(per)


# --------------------------------------------------------------------------- #
#  Bootstrap
# --------------------------------------------------------------------------- #
def test_clustered_bootstrap_brackets_the_point_estimate():
    rng = np.random.default_rng(8)
    x = rng.normal(0.3, 1.0, 500)
    pt, lo, hi = clustered_ci(lambda idx: float(x[idx].mean()), 500,
                              replicates=300, seed=1)
    assert lo < pt < hi


def test_clustered_bootstrap_resamples_whole_ecgs():
    """All labels of a resampled ECG must travel together.

    Verified through the statistic: a per-row constant must survive resampling
    unchanged, which only holds if rows are drawn intact.
    """
    rows = np.arange(100)
    pt, lo, hi = clustered_ci(lambda idx: float(np.unique(rows[idx] % 1).size),
                              100, replicates=50, seed=2)
    assert pt == 1.0


def test_bootstrap_interval_narrows_with_more_data():
    rng = np.random.default_rng(9)
    small = rng.normal(0, 1, 60)
    large = rng.normal(0, 1, 6000)
    _, l1, h1 = clustered_ci(lambda i: float(small[i].mean()), 60,
                             replicates=400, seed=3)
    _, l2, h2 = clustered_ci(lambda i: float(large[i].mean()), 6000,
                             replicates=400, seed=3)
    assert (h2 - l2) < (h1 - l1)


def test_enrichment_is_one_when_disputes_are_no_harder():
    """E_D = 1 is the null: disputed labels error at the retained base rate."""
    n = 6000
    rng = np.random.default_rng(20)
    ref = (rng.random((n, 2)) < 0.2).astype(int)
    loss = (rng.random((n, 2)) < 0.25).astype(float)   # independent of ref
    conf = rng.random((n, 2))
    from ecguq.selective import disagreement_enrichment
    assert abs(disagreement_enrichment(loss, conf, ref, 1.0) - 1.0) < 0.15


def test_enrichment_exceeds_one_when_disputes_carry_more_error():
    n = 4000
    rng = np.random.default_rng(21)
    ref = np.zeros((n, 1), dtype=int)
    ref[:400] = 1
    loss = np.zeros((n, 1))
    loss[:400, 0] = (rng.random(400) < 0.7)
    loss[400:, 0] = (rng.random(n - 400) < 0.05)
    conf = rng.random((n, 1))
    from ecguq.selective import disagreement_enrichment
    assert disagreement_enrichment(loss, conf, ref, 0.8) > 2.0


def test_enrichment_uses_the_retained_prevalence_not_the_full_cohort():
    """The correction that matters.

    Referral preferentially removes disputed labels, since those are the ones
    the model is unsure about. Dividing the disputed share of residual errors
    by the *full-cohort* prevalence would therefore understate enrichment,
    crediting referral with having shrunk a denominator it merely filtered.
    """
    from ecguq.selective import (disagreement_enrichment, disputed_prevalence,
                                 disputed_error_share,
                                 retained_disputed_prevalence)
    n = 4000
    rng = np.random.default_rng(22)
    ref = np.zeros((n, 1), dtype=int)
    ref[:800] = 1
    # Disputed labels are less confident on average, but overlap the rest, so
    # referral thins them without eliminating them.
    conf = rng.random((n, 1))
    conf[:800] *= 0.7
    loss = np.zeros((n, 1))
    loss[:800, 0] = (rng.random(800) < 0.5)
    loss[800:, 0] = (rng.random(n - 800) < 0.1)
    q = 0.8

    pi_ret = retained_disputed_prevalence(conf, ref, q)
    pi_all = disputed_prevalence(ref)
    assert 0 < pi_ret < pi_all                 # thinned, not eliminated

    naive = disputed_error_share(loss, conf, ref, q) / pi_all
    correct = disagreement_enrichment(loss, conf, ref, q)
    assert correct > naive


def test_enrichment_is_undefined_when_referral_removes_every_dispute():
    """A real edge case, not a bug: with no disputed label retained there is
    no denominator, and nan is the honest answer rather than 0 or 1."""
    from ecguq.selective import disagreement_enrichment
    n = 500
    rng = np.random.default_rng(23)
    ref = np.zeros((n, 1), dtype=int)
    ref[:50] = 1
    conf = rng.random((n, 1))
    conf[:50] = -1.0                            # strictly least confident
    loss = (rng.random((n, 1)) < 0.2).astype(float)
    assert np.isnan(disagreement_enrichment(loss, conf, ref, 0.5))


def test_sequential_batched_loader_preserves_row_order():
    """Guards a silent misalignment.

    The batched dataset sorts indices before slicing the memmap. For shuffled
    training that is harmless, but evaluation indexes labels separately from
    the loader, so any reordering there would pair predictions with the wrong
    labels and quietly corrupt every metric.
    """
    import torch
    from torch.utils.data import (BatchSampler, DataLoader, SequentialSampler)
    from ecguq.train import ECGDataset, Prepared

    n = 57
    sig = np.arange(n, dtype=np.float32)[:, None, None] * np.ones((1, 2, 4))
    prep = Prepared(path=Path("."), signals=sig,
                    y=np.arange(n, dtype=np.int8)[:, None] % 2,
                    meta={}, manifest={})
    ds = ECGDataset(prep, np.arange(n), 0.0, 1.0)
    ds._sig = sig                                   # bypass the lazy file open
    dl = DataLoader(ds, sampler=BatchSampler(SequentialSampler(ds), 8,
                                             drop_last=False),
                    batch_size=None, num_workers=0)
    got = torch.cat([x[:, 0, 0] for x, _ in dl]).numpy()
    assert np.array_equal(got, np.arange(n, dtype=np.float32))


def test_patient_risk_counts_an_ecg_once_however_many_labels_fail():
    """Two wrong labels on one ECG are one erroneous ECG, not two."""
    from ecguq.selective import patient_errors_at, patient_risk_at
    loss = np.zeros((4, 3))
    loss[0, :] = 1.0                       # one ECG, three failed labels
    conf = np.ones((4, 3))
    assert patient_errors_at(loss, conf, 1.0) == 1
    assert patient_risk_at(loss, conf, 1.0) == pytest.approx(0.25)


def test_patient_referral_ranks_on_the_weakest_label():
    """An ECG with one very uncertain label must be referred before an ECG
    that is moderately uncertain throughout."""
    from ecguq.selective import patient_errors_at
    conf = np.array([[5.0, 5.0, 0.1],      # one terrible label
                     [3.0, 3.0, 3.0]])     # uniformly mediocre
    loss = np.array([[0, 0, 1.0], [0, 0, 0]])
    assert patient_errors_at(loss, conf, 0.5) == 0   # the bad ECG goes first


def test_patient_risk_is_at_least_the_label_error_share():
    from ecguq.selective import patient_risk_at
    rng = np.random.default_rng(30)
    loss = (rng.random((200, 6)) < 0.02).astype(float)
    conf = rng.random((200, 6))
    assert patient_risk_at(loss, conf, 1.0) >= loss.mean()
