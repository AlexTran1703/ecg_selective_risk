r"""Cohort, leave-one-source-out splits, and batch feeding for Study II.

The input representation is fixed once here and shared by every encoder:
10 s, 12 leads, decimated 500 -> 250 Hz. 250 Hz is not a neutral choice --
it is the highest rate that Study III found survivable on the STM32F411
once the input buffer is counted, so the benchmark trains models at a rate
they could actually run at.

Decimation is anti-aliased (scipy.decimate, FIR, zero-phase) rather than
plain strided slicing. Slicing would fold everything above 125 Hz back into
the band, which on an ECG means mains harmonics and EMG landing on top of
the QRS complex.

Records carrying no label inside the harmonised 13-class space are dropped
before any splitting. Such a record scores a loss of 1 whenever the model
predicts anything and no confidence score can rank it away, so keeping them
would impose a floor on selective risk that is a property of the label
vocabulary rather than of the model.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import decimate

HERE = Path(__file__).resolve().parents[2]      # 12_leads/
ROOT = HERE.parent
CACHE = ROOT / "data" / "cache"
META = ROOT / "data" / "meta"
PREP = HERE / "data"

# The model input rate is a deployment parameter, so it is settable
# from the environment: ECG_FS=100 selects the 100 Hz cohort. Each
# rate gets its own cache and its own results folder, so switching
# rate can never silently overwrite another rate's predictions.
FS_IN = 500
FS_OUT = int(os.environ.get("ECG_FS", "250"))
SECONDS = 10
N_LEADS = 12
LENGTH = FS_OUT * SECONDS
SOURCES = ("PTBXL", "Georgia", "Chapman", "Ningbo")


def classes() -> list[str]:
    return json.load(open(META / "classes.json"))["abbreviations"]


def prepare(chunk: int = 512, force: bool = False) -> None:
    """Decimate the retained cohort once to 250 Hz and cache it."""
    PREP.mkdir(parents=True, exist_ok=True)
    sig_out = PREP / f"signals_{FS_OUT}.npy"
    if sig_out.exists() and not force:
        return

    rec = pd.read_csv(META / "records.csv")
    keep = np.flatnonzero((rec.n_pos > 0).to_numpy())
    src = np.load(CACHE / "signals.npy", mmap_mode="r")
    n_cls = len(classes())

    y = np.zeros((keep.size, n_cls), dtype=np.int8)
    for i, lab in enumerate(rec.y.to_numpy()[keep]):
        for j in str(lab).split():
            y[i, int(j)] = 1

    out = np.lib.format.open_memmap(
        sig_out, mode="w+", dtype=np.float16,
        shape=(keep.size, N_LEADS, LENGTH))
    q = FS_IN // FS_OUT
    for s in range(0, keep.size, chunk):
        e = min(s + chunk, keep.size)
        x = np.asarray(src[keep[s:e]], dtype=np.float32)
        out[s:e] = decimate(x, q, ftype="fir", zero_phase=True,
                            axis=-1).astype(np.float16)
        if (s // chunk) % 20 == 0:
            print(f"  decimate {e}/{keep.size}", flush=True)
    out.flush()
    np.save(PREP / "labels.npy", y)
    rec.iloc[keep][["source", "record"]].to_csv(PREP / "index.csv",
                                                index=False)
    print(f"  wrote {sig_out.name}  {out.shape}  "
          f"{out.nbytes / 1e9:.1f} GB")


class Cohort:
    """The prepared cohort plus its leave-one-source-out partitions."""

    def __init__(self):
        prepare()
        self.sig_path = PREP / f"signals_{FS_OUT}.npy"
        self.y = np.load(PREP / "labels.npy")
        self.index = pd.read_csv(PREP / "index.csv")
        self.source = self.index["source"].to_numpy()
        self.classes = classes()

    def loso(self, held_out: str, val_frac: float = 0.1, seed: int = 0):
        """(train, val, test) row indices for one held-out source.

        Validation comes from the training sources, never from the held-out
        one: tuning on the target source would quietly convert this into a
        transductive setting and inflate every cross-source number.
        """
        test = np.flatnonzero(self.source == held_out)
        pool = np.flatnonzero(self.source != held_out)
        rng = np.random.default_rng(seed)
        pool = pool[rng.permutation(pool.size)]
        cut = int(round(val_frac * pool.size))
        return pool[cut:], pool[:cut], test

    def stats(self, rows: np.ndarray) -> tuple[float, float]:
        """Mean and SD from a subsample of the given rows only."""
        sig = np.load(self.sig_path, mmap_mode="r")
        rng = np.random.default_rng(0)
        take = rows if rows.size <= 4000 else rng.choice(rows, 4000, False)
        x = np.asarray(sig[np.sort(take)], dtype=np.float32)
        return float(x.mean()), float(x.std() + 1e-6)


class Feeder:
    """Block-shuffled batches straight off the memmap, normalised on device.

    Rows are sorted so each read is a contiguous run and only the block
    order is shuffled; fully random access over a multi-GB memmap costs far
    more than the stochasticity is worth.
    """

    def __init__(self, sig_path, y, rows, batch_size, mean, std, device,
                 shuffle=False, seed=0, prefetch=3):
        self.sig_path, self.y = Path(sig_path), np.asarray(y)
        self.bs, self.device, self.shuffle = int(batch_size), device, shuffle
        self.seed, self.prefetch, self.epoch = seed, max(1, prefetch), 0
        rows = np.asarray(rows)
        self.rows = np.sort(rows) if shuffle else rows
        self.drop_last = shuffle
        self._m = torch.tensor(float(mean), device=device)
        self._s = torch.tensor(float(std), device=device)

    def __len__(self):
        n = self.rows.size
        return n // self.bs if self.drop_last else -(-n // self.bs)

    def labels(self):
        n = self.rows.size
        if self.drop_last:
            n -= n % self.bs
        return self.y[self.rows[:n]]

    def _blocks(self):
        b = [self.rows[s:s + self.bs]
             for s in range(0, self.rows.size, self.bs)]
        if self.drop_last and b and b[-1].size < self.bs:
            b.pop()
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(b)
        return b

    def _produce(self, blocks, out):
        sig = np.load(self.sig_path, mmap_mode="r")
        try:
            for idx in blocks:
                out.put((torch.from_numpy(np.asarray(sig[idx], np.float32)),
                         torch.from_numpy(self.y[idx].astype(np.float32))))
        except Exception as exc:                        # noqa: BLE001
            out.put(exc)
        else:
            out.put(None)

    def __iter__(self):
        blocks = self._blocks()
        self.epoch += 1
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        t = threading.Thread(target=self._produce, args=(blocks, q),
                             daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            x, y = item
            x = x.to(self.device)
            x.sub_(self._m).div_(self._s)
            yield x, y.to(self.device)
        t.join()
