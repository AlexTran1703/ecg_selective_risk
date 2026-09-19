"""Measure the speed-up torch.compile gives on the 1-D backbone (development aid).

Eager and compiled are timed back-to-back under whatever load the GPU is already
under, so the *ratio* stays meaningful even if the absolute rates are depressed
by a concurrent training run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgsr.models import build_model            # noqa: E402


def rate(model, x, y, steps: int = 8) -> float:
    opt = torch.optim.AdamW(model.parameters())
    crit = nn.BCEWithLogitsLoss()
    for _ in range(4):                           # warm up / trigger compilation
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return steps * x.shape[0] / (time.time() - t0)


def main() -> None:
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    bs = 64
    x = (torch.randn(bs, 12, 5000, device=dev) * 0.5)
    y = torch.zeros(bs, 13, device=dev)

    eager = build_model("raw", 13).to(dev)
    r_eager = rate(eager, x, y)
    print(f"eager     {r_eager:7.1f} samples/s")
    del eager
    torch.cuda.empty_cache()

    try:
        comp = build_model("raw", 13).to(dev)
        comp = torch.compile(comp)
        t0 = time.time()
        r_comp = rate(comp, x, y)
        print(f"compiled  {r_comp:7.1f} samples/s   "
              f"(x{r_comp / r_eager:.2f}, first call cost {time.time() - t0:.0f}s)")
    except Exception as exc:                      # noqa: BLE001
        print(f"compiled  unavailable: {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    main()
