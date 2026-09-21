"""Training one ensemble member, and the dataloader that feeds it.

The five members differ only in initialisation and batch order.  They share
the frozen patient-level split written at preprocessing time, so
``Var_m p^(m)`` -- and the ensemble mean it produces -- reflect the fitting
procedure rather than which patients happened to land in training.

Normalisation statistics are computed from the training partition alone and
applied on the device by ``feed.BatchFeeder``, never baked into the stored
signals.  Computing them over the whole array would leak validation and test
information into training through the input scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Prepared:
    path: Path                  # directory holding signals.npy
    signals: np.ndarray         # memmap (N, 12, T) float16
    y: np.ndarray               # (N, 6) int8
    meta: dict
    manifest: dict


def load_prepared(path: Path) -> Prepared:
    path = Path(path)
    meta = dict(np.load(path / "meta.npz", allow_pickle=False))
    return Prepared(
        path=path,
        signals=np.load(path / "signals.npy", mmap_mode="r"),
        y=meta["y"],
        meta=meta,
        manifest=json.loads((path / "manifest.json").read_text(encoding="utf-8")))


def train_statistics(prep: Prepared, rows: np.ndarray,
                     sample: int = 20000, seed: int = 0) -> tuple[float, float]:
    """Per-dataset mean and SD over the training rows only.

    Estimated from a random subsample: the full training array is 30 GB, and
    the mean of 20,000 twelve-lead records is already stable to far more
    digits than a normalisation constant needs.
    """
    rng = np.random.default_rng(seed)
    take = rows if rows.size <= sample else rng.choice(rows, sample, replace=False)
    chunk = np.asarray(prep.signals[np.sort(take)], dtype=np.float32)
    return float(chunk.mean()), float(chunk.std() + 1e-6)


class ECGDataset(Dataset):
    """Reads rows from the memmap, normalising with train-partition stats.

    The memmap is opened lazily per worker rather than held on the instance:
    Windows spawns workers by pickling the dataset, and a 34 GB memmap cannot
    survive that. Storing the path and opening on first access keeps each
    worker mapping the file itself.
    """

    def __init__(self, prep: Prepared, rows: np.ndarray, mean: float,
                 std: float):
        self.sig_path = Path(prep.path) / "signals.npy"
        self.y = np.asarray(prep.y)
        self.rows = np.asarray(rows)
        self.mean, self.std = mean, std
        self._sig: np.ndarray | None = None

    def __len__(self) -> int:
        return self.rows.size

    @property
    def sig(self) -> np.ndarray:
        if self._sig is None:
            self._sig = np.load(self.sig_path, mmap_mode="r")
        return self._sig

    def __getitem__(self, i):
        """Accepts a single index or a whole batch of them.

        Per-record fetching dominated the epoch: raw memmap reads run at
        735-889 MB/s, enough for a 42 s epoch, but 311k Python calls each
        building a small tensor took the observed epoch to 150 s. Slicing a
        batch in one go and normalising it vectorised removes that overhead,
        so pass this a ``BatchSampler`` and leave ``batch_size=None``.
        """
        idx = np.atleast_1d(np.asarray(self.rows[i]))
        order = np.argsort(idx)                    # memmap likes ascending
        x = np.asarray(self.sig[idx[order]], dtype=np.float32)
        x = (x - self.mean) / self.std
        y = self.y[idx[order]].astype(np.float32)
        if np.isscalar(i) or np.ndim(i) == 0:
            return torch.from_numpy(x[0]), torch.from_numpy(y[0])
        return torch.from_numpy(x), torch.from_numpy(y)


def positive_weights(y: np.ndarray, cap: float = 50.0) -> torch.Tensor:
    """``neg/pos`` per class for BCE, capped.

    These labels sit near 2% prevalence, so unweighted BCE converges happily
    to predicting the negative class. The cap stops the rarest class from
    dominating the gradient outright.
    """
    y = np.asarray(y)
    pos = y.sum(axis=0).astype(np.float64)
    neg = y.shape[0] - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(np.clip(w, 1.0, cap), dtype=torch.float32)


@torch.no_grad()
def predict_probs(model: nn.Module, feeder, device: str,
                  amp: bool = True) -> np.ndarray:
    """Batches arrive already on the device, so nothing is moved here."""
    model.eval()
    out = []
    for x, _ in feeder:
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=amp and device == "cuda"):
            logits = model(x)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def train_member(model: nn.Module, train_feeder, val_feeder, device: str,
                 epochs: int, pos_weight: torch.Tensor, lr: float = 1e-3,
                 amp: bool = True, log=print) -> nn.Module:
    from .metrics import macro_auprc

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * len(train_feeder)))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    yv = val_feeder.labels()

    best_state, best = None, -1.0
    for ep in range(epochs):
        model.train()
        run, seen = 0.0, 0
        for x, y in train_feeder:
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=amp and device == "cuda"):
                loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]

        pv = predict_probs(model, val_feeder, device, amp)
        score = macro_auprc(yv, pv)
        star = ""
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            star = "  *"
        log(f"    epoch {ep:2d}  loss {run / max(seen, 1):.4f}  "
            f"val macro-AUPRC {score:.4f}{star}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
