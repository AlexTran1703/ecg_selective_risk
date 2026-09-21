"""Batch feeding straight off the memmap, with no DataLoader machinery.

The hot path is deliberately short::

    one localised memmap gather -> one tensor -> one host-to-device copy
                                -> normalise on the GPU

What this removes, and why
--------------------------
Per-record ``__getitem__`` cost 311k Python calls and 311k small tensor
allocations per epoch, which took the observed step to ~123 ms against ~11 ms
of actual model time.  Batch fetching removed most of that; this removes the
rest -- default collation, worker processes, their queues, and pinned-memory
buffers.

Worker processes are avoided on purpose rather than tuned.  Each one maps the
same 34 GB file and carries its own prefetch buffers, and during the earlier
run free memory fell to 0.2 GB.  A single prefetch *thread* with a queue of
two or three batches gives the same overlap for ~75 MB, because the expensive
part -- the NumPy gather -- releases the GIL.

Locality without rewriting 34 GB
--------------------------------
Rows belonging to the training partition are scattered through the file, since
the split is by patient.  Fully random sampling therefore issues scattered
reads (measured 735 MB/s, against 889 MB/s contiguous).  Instead the row list
is sorted once and cut into contiguous blocks, and the *block order* is
shuffled each epoch.  Each read is then an ascending, locally clustered run,
and the stochasticity SGD needs comes from block order plus an optional
within-block shuffle.  That recovers most of the locality for free, where
physically reordering the memmap would cost a 34 GB rewrite for the remaining
~20%.

Normalisation happens on the device, so the CPU only moves bytes.  The
statistics still come from the training partition alone -- they are passed in,
never derived here.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
import torch

PREFETCH = 3


class BatchFeeder:
    """Yields ``(x, y)`` already on the device.

    ``shuffle=False`` preserves the order of ``rows`` exactly, which the
    evaluation path depends on: predictions are matched against labels
    indexed separately, so any reordering would pair them wrongly and corrupt
    every metric silently.
    """

    def __init__(self, sig_path: Path, y: np.ndarray, rows: np.ndarray,
                 batch_size: int, mean: float, std: float, device: str,
                 shuffle: bool = False, seed: int = 0,
                 drop_last: bool | None = None, prefetch: int = PREFETCH):
        self.sig_path = Path(sig_path)
        self.y = np.asarray(y)
        self.batch_size = int(batch_size)
        self.device = device
        self.shuffle = shuffle
        self.seed = seed
        self.prefetch = max(1, prefetch)
        self.drop_last = shuffle if drop_last is None else drop_last

        rows = np.asarray(rows)
        if shuffle:
            # Sorted so that each block is an ascending, clustered run.
            self.rows = np.sort(rows)
        else:
            if rows.size > 1 and np.any(np.diff(rows) <= 0):
                raise ValueError(
                    "unshuffled feeding requires ascending rows so output "
                    "order matches the caller's indexing")
            self.rows = rows

        self._mean = torch.tensor(float(mean), device=device)
        self._std = torch.tensor(float(std), device=device)
        self._epoch = 0

    def __len__(self) -> int:
        n = self.rows.size
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _blocks(self) -> list[np.ndarray]:
        n = self.rows.size
        edges = list(range(0, n, self.batch_size))
        blocks = [self.rows[s:s + self.batch_size] for s in edges]
        if self.drop_last and blocks and blocks[-1].size < self.batch_size:
            blocks.pop()
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(blocks)
        return blocks

    def _produce(self, blocks, out: queue.Queue) -> None:
        sig = np.load(self.sig_path, mmap_mode="r")
        try:
            for idx in blocks:
                x = np.asarray(sig[idx], dtype=np.float32)
                out.put((torch.from_numpy(x),
                         torch.from_numpy(self.y[idx].astype(np.float32))))
        except Exception as exc:                        # noqa: BLE001
            out.put(exc)
        else:
            out.put(None)

    def __iter__(self):
        blocks = self._blocks()
        self._epoch += 1
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
            x = x.to(self.device, non_blocking=False)
            x.sub_(self._mean).div_(self._std)          # normalise on device
            yield x, y.to(self.device, non_blocking=False)
        t.join()

    def labels(self) -> np.ndarray:
        """Labels in the order this feeder yields them, for unshuffled use."""
        if self.shuffle:
            raise ValueError("label order is only defined for shuffle=False")
        n = self.rows.size
        if self.drop_last:
            n -= n % self.batch_size
        return self.y[self.rows[:n]]


def train_statistics(sig_path: Path, rows: np.ndarray, sample: int = 20000,
                     seed: int = 0) -> tuple[float, float]:
    """Mean and SD over the training rows only, from a random subsample.

    20,000 twelve-lead records pin a scalar normalisation constant far more
    tightly than it needs; reading all 30 GB to refine it would be wasted.
    """
    sig = np.load(Path(sig_path), mmap_mode="r")
    rng = np.random.default_rng(seed)
    take = rows if rows.size <= sample else rng.choice(rows, sample,
                                                       replace=False)
    chunk = np.asarray(sig[np.sort(take)], dtype=np.float32)
    return float(chunk.mean()), float(chunk.std() + 1e-6)
