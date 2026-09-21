r"""Tests for the memmap batch feeder.

    .\.venv\Scripts\python.exe -m pytest tests/test_feed.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecguq.feed import BatchFeeder, train_statistics                # noqa: E402


@pytest.fixture
def store(tmp_path):
    n, c, t = 97, 3, 8
    x = (np.arange(n, dtype=np.float32)[:, None, None]
         * np.ones((1, c, t), dtype=np.float32))
    np.save(tmp_path / "signals.npy", x.astype(np.float16))
    y = np.zeros((n, 2), dtype=np.int8)
    y[::3, 0] = 1
    return tmp_path / "signals.npy", y, n


def test_unshuffled_feeding_preserves_row_order(store):
    """Evaluation indexes labels separately, so any reordering here would
    pair predictions with the wrong labels and corrupt every metric."""
    path, y, n = store
    rows = np.arange(n)
    f = BatchFeeder(path, y, rows, 16, 0.0, 1.0, "cpu", shuffle=False)
    got = torch.cat([x[:, 0, 0] for x, _ in f]).numpy()
    assert np.array_equal(got, np.arange(n, dtype=np.float32))


def test_unshuffled_labels_match_the_yielded_batches(store):
    path, y, n = store
    rows = np.arange(n)
    f = BatchFeeder(path, y, rows, 16, 0.0, 1.0, "cpu", shuffle=False)
    yielded = torch.cat([b for _, b in f]).numpy()
    assert np.array_equal(yielded, f.labels().astype(np.float32))


def test_unshuffled_feeding_rejects_unsorted_rows(store):
    """Fail loudly rather than silently misalign."""
    path, y, _ = store
    with pytest.raises(ValueError):
        BatchFeeder(path, y, np.array([5, 2, 9]), 2, 0.0, 1.0, "cpu",
                    shuffle=False)


def test_shuffled_feeding_covers_every_row_exactly_once(store):
    path, y, n = store
    rows = np.arange(n)
    f = BatchFeeder(path, y, rows, 16, 0.0, 1.0, "cpu", shuffle=True,
                    drop_last=False)
    got = np.sort(torch.cat([x[:, 0, 0] for x, _ in f]).numpy())
    assert np.array_equal(got, np.arange(n, dtype=np.float32))


def test_block_order_changes_between_epochs_but_content_does_not(store):
    """Stochasticity comes from block order; the data must be identical."""
    path, y, n = store
    f = BatchFeeder(path, y, np.arange(n), 16, 0.0, 1.0, "cpu", shuffle=True,
                    seed=0, drop_last=False)
    e1 = torch.cat([x[:, 0, 0] for x, _ in f]).numpy()
    e2 = torch.cat([x[:, 0, 0] for x, _ in f]).numpy()
    assert not np.array_equal(e1, e2)
    assert np.array_equal(np.sort(e1), np.sort(e2))


def test_shuffled_blocks_are_locally_contiguous(store):
    """The point of block shuffling: reads stay clustered in the file.

    Within a batch the rows must be an ascending run, which is what keeps the
    memmap gather local rather than scattered across 34 GB.
    """
    path, y, n = store
    f = BatchFeeder(path, y, np.arange(n), 16, 0.0, 1.0, "cpu", shuffle=True)
    for x, _ in f:
        v = x[:, 0, 0].numpy()
        assert np.all(np.diff(v) == 1.0)


def test_scattered_rows_are_sorted_before_blocking(store):
    """Training rows are scattered by the patient split; sorting them once is
    what makes the blocks clustered at all."""
    path, y, n = store
    rows = np.array([90, 3, 44, 7, 61, 12, 30, 55])
    f = BatchFeeder(path, y, rows, 4, 0.0, 1.0, "cpu", shuffle=True)
    for x, _ in f:
        v = x[:, 0, 0].numpy()
        assert np.all(np.diff(v) > 0)


def test_normalisation_is_applied_on_the_yielded_tensor(store):
    path, y, n = store
    f = BatchFeeder(path, y, np.arange(8), 8, mean=2.0, std=4.0,
                    device="cpu", shuffle=False)
    x, _ = next(iter(f))
    assert x[:, 0, 0].numpy() == pytest.approx(
        (np.arange(8, dtype=np.float32) - 2.0) / 4.0)


def test_drop_last_matches_reported_length(store):
    path, y, n = store
    for drop in (True, False):
        f = BatchFeeder(path, y, np.arange(n), 16, 0.0, 1.0, "cpu",
                        shuffle=False, drop_last=drop)
        assert sum(1 for _ in f) == len(f)


def test_train_statistics_use_only_the_rows_given(store):
    """Normalisation must never see validation or test rows."""
    path, y, n = store
    lo = np.arange(0, 10)
    hi = np.arange(n - 10, n)
    m_lo, _ = train_statistics(path, lo)
    m_hi, _ = train_statistics(path, hi)
    assert m_lo < m_hi
    assert m_lo == pytest.approx(float(np.arange(10).mean()), abs=1e-3)
