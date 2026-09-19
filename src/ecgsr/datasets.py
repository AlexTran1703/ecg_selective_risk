"""Splits, cache-backed dataset, and on-GPU representation transforms.

One filtered signal cache serves all three representations: the dataset yields
raw 12-lead millivolts and the transform stage derives VCG or STFT on the GPU.
That guarantees Raw / VCG / STFT see byte-identical underlying signals, which is
what makes the representation comparison controlled.
"""

from __future__ import annotations

import csv
import zlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .representations import (INVERSE_DOWER, _INDEPENDENT_LEAD_ROWS,
                              fft_torch, stft_torch)

# Representations in the study. ``stft`` remains implemented but is excluded:
# being 2-D it would need a different backbone and so confound representation
# with architecture.
REPRESENTATIONS = ("raw", "vcg", "fft")
ALL_REPRESENTATIONS = ("raw", "vcg", "fft", "stft")

SOURCES = ("PTBXL", "Georgia", "Chapman", "Ningbo")


# --------------------------------------------------------------------------- #
#  Index and splits
# --------------------------------------------------------------------------- #
@dataclass
class RecordIndex:
    source: np.ndarray          # (N,) str
    patient: np.ndarray         # (N,) str
    record: np.ndarray          # (N,) str
    y: np.ndarray               # (N, K) int8
    classes: list[str]
    abbreviations: list[str]

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def load_index(meta_dir: Path) -> RecordIndex:
    meta_dir = Path(meta_dir)
    spec = json.loads((meta_dir / "classes.json").read_text(encoding="utf-8"))
    classes = spec["classes"]

    with (meta_dir / "records.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    y = np.zeros((len(rows), len(classes)), dtype=np.int8)
    for i, r in enumerate(rows):
        if r["y"]:
            y[i, [int(j) for j in r["y"].split()]] = 1

    return RecordIndex(
        source=np.array([r["source"] for r in rows]),
        patient=np.array([r["patient"] for r in rows]),
        record=np.array([r["record"] for r in rows]),
        y=y, classes=classes, abbreviations=spec["abbreviations"])


@dataclass(frozen=True)
class Split:
    """One leave-one-source-out experiment.

    ``train``/``dev`` come from the three training sources; ``dev`` selects the
    per-class decision thresholds and stops training.  ``cal``/``test`` come
    from the held-out target source; network weights never see either.
    """
    target: str
    train: np.ndarray
    dev: np.ndarray
    cal: np.ndarray
    test: np.ndarray


def _stable_seed(target: str, seed: int) -> int:
    """Reproducible per-target seed.

    ``hash()`` on a str is salted per interpreter process, so using it here would
    give every pipeline a *different* split of the same held-out source -- which
    would silently destroy the controlled Raw/VCG/STFT comparison.  CRC32 is
    stable across processes and runs.
    """
    return (seed * 1_000_003 + zlib.crc32(target.encode("utf-8"))) % (2 ** 32)


def make_split(index: RecordIndex, target: str, dev_frac: float = 0.10,
               cal_frac: float = 0.20, seed: int = 0,
               train_sources: tuple[str, ...] | None = None) -> Split:
    """Leave-one-source-out split.

    By default the three non-target sources are used for training.  Passing
    ``train_sources`` restricts training to a subset, which is what the nested
    inner score selection needs: it trains on two sources and treats a third as
    a pseudo-target, never touching the real outer target.
    """
    rng = np.random.default_rng(_stable_seed(target, seed))

    is_target = index.source == target
    target_rows = np.flatnonzero(is_target)
    if train_sources is None:
        source_rows = np.flatnonzero(~is_target)
    else:
        allowed = np.isin(index.source, list(train_sources))
        if not allowed.any():
            raise ValueError(f"no records for train_sources={train_sources}")
        if target in train_sources:
            raise ValueError(f"target {target!r} must not be in train_sources")
        source_rows = np.flatnonzero(allowed)

    # Dev split is grouped by patient: PTB-XL contains repeat recordings of the
    # same subject, so a record-level split would leak across train/dev.
    patients = np.unique(index.patient[source_rows])
    rng.shuffle(patients)
    n_dev = max(1, int(round(dev_frac * len(patients))))
    dev_patients = set(patients[:n_dev].tolist())
    in_dev = np.array([p in dev_patients for p in index.patient[source_rows]])

    perm = rng.permutation(len(target_rows))
    n_cal = max(1, int(round(cal_frac * len(target_rows))))

    return Split(
        target=target,
        train=source_rows[~in_dev],
        dev=source_rows[in_dev],
        cal=target_rows[perm[:n_cal]],
        test=target_rows[perm[n_cal:]],
    )


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
class ECGCacheDataset(Dataset):
    """Serves raw 12-lead millivolts straight from the memmapped cache."""

    def __init__(self, cache_path: Path, rows: np.ndarray, y: np.ndarray,
                 time_shift: int = 0, rng_seed: int = 0):
        self.cache_path = Path(cache_path)
        self.rows = np.asarray(rows)
        self.y = np.asarray(y, dtype=np.float32)
        self.time_shift = int(time_shift)
        self.rng_seed = rng_seed
        self._cache: np.ndarray | None = None       # opened lazily per worker

    def __len__(self) -> int:
        return len(self.rows)

    def _signals(self) -> np.ndarray:
        if self._cache is None:
            self._cache = np.load(self.cache_path, mmap_mode="r")
        return self._cache

    def __getitem__(self, i: int):
        x = np.asarray(self._signals()[self.rows[i]], dtype=np.float32)
        if self.time_shift:
            rng = np.random.default_rng((self.rng_seed, int(self.rows[i])))
            x = np.roll(x, int(rng.integers(-self.time_shift, self.time_shift + 1)), axis=1)
        return torch.from_numpy(x), torch.from_numpy(self.y[i])


# --------------------------------------------------------------------------- #
#  Representation transforms (GPU side)
# --------------------------------------------------------------------------- #
class Representation(torch.nn.Module):
    """Derive Raw / VCG / STFT from cached millivolts and standardise.

    ``mean``/``std`` are always estimated on the training split only.  The VCG
    transform is applied to physical amplitudes *before* standardisation, so the
    inverse Dower coefficients keep their physical meaning.
    """

    def __init__(self, kind: str, mean: torch.Tensor | None = None,
                 std: torch.Tensor | None = None, fs: float = 500.0):
        super().__init__()
        if kind not in ALL_REPRESENTATIONS:
            raise ValueError(f"unknown representation {kind!r}")
        self.kind = kind
        self.fs = fs
        dower = torch.from_numpy(INVERSE_DOWER.copy())
        self.register_buffer("dower", dower)
        self.register_buffer("lead_rows", torch.tensor(_INDEPENDENT_LEAD_ROWS,
                                                       dtype=torch.long))
        self.register_buffer("mean", mean if mean is not None else torch.zeros(1))
        self.register_buffer("std", std if std is not None else torch.ones(1))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Representation in physical units, before standardisation."""
        if self.kind == "raw":
            return x
        if self.kind == "vcg":
            return torch.einsum("ij,bjt->bit", self.dower, x[:, self.lead_rows, :])
        if self.kind == "fft":
            return fft_torch(x, fs=self.fs)
        return stft_torch(x, fs=self.fs)

    @property
    def stat_dims(self) -> tuple[int, ...]:
        """Axes to pool when estimating standardisation statistics.

        Standardise over the axes along which the signal's statistics are
        homogeneous, and keep the axes along which they are not:

          raw/vcg  pool batch and time      -> one mean/std per lead
          fft      pool batch only          -> per (lead, frequency bin), since
                                               ECG spectral magnitude falls off
                                               steeply with frequency and a
                                               per-lead scalar would let the
                                               lowest bins dominate
          stft     pool batch and time      -> per (lead, frequency bin)
        """
        return {"raw": (0, 2), "vcg": (0, 2), "fft": (0,), "stft": (0, 3)}[self.kind]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return (z - self.mean) / self.std

    @torch.no_grad()
    def fit(self, loader, device, max_batches: int = 64) -> "Representation":
        """Estimate standardisation statistics from the training split.

        Raw/VCG are standardised per lead; STFT per (lead, frequency bin), since
        ECG power falls off steeply with frequency and a single scalar would let
        the low-frequency bins dominate.
        """
        dims = self.stat_dims
        total = sq = count = None
        for b, (x, _) in enumerate(loader):
            if b >= max_batches:
                break
            z = self.features(x.to(device, non_blocking=True)).double()
            s = z.sum(dim=dims)
            s2 = (z ** 2).sum(dim=dims)
            n = z.numel() / s.numel()
            total = s if total is None else total + s
            sq = s2 if sq is None else sq + s2
            count = n if count is None else count + n

        mean = total / count
        var = (sq / count) - mean ** 2
        std = var.clamp_min(1e-12).sqrt()
        # Broadcast back over the pooled axes: (1, C, 1) for raw/vcg/stft-freq,
        # (1, C, F) for fft where nothing beyond the batch axis was pooled.
        shape = (1, -1, 1) if mean.dim() == 1 else (1, *mean.shape)
        if self.kind == "stft":
            shape = (1, *mean.shape, 1)
        self.mean = mean.reshape(shape).float().to(device)
        self.std = std.reshape(shape).float().to(device)
        return self
