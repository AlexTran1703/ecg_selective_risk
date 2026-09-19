"""Throughput check for the cache-backed data loader (development aid)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgsr.datasets import ECGCacheDataset, load_index      # noqa: E402


def main() -> None:
    import psutil
    vm = psutil.virtual_memory()
    print(f"RAM total {vm.total / 2**30:.1f} GiB, available {vm.available / 2**30:.1f} GiB")

    index = load_index(ROOT / "data" / "meta")
    rows = np.random.default_rng(0).choice(len(index.y), 20_000, replace=False)
    cache = ROOT / "data" / "cache" / "signals.npy"

    for workers in (4, 8, 12):
        ds = ECGCacheDataset(cache, rows, index.y[rows])
        dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=workers,
                        pin_memory=True, persistent_workers=True)
        it = iter(dl)
        for _ in range(10):
            next(it)
        t0, seen = time.time(), 0
        for i, (x, _) in enumerate(it):
            seen += len(x)
            if i >= 150:
                break
        print(f"workers={workers:2d}  {seen / (time.time() - t0):7.0f} samples/s")
        del it, dl


if __name__ == "__main__":
    main()
