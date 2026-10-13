"""The 40 deployed configurations: width, depth and realised footprint."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from mcubench.footprint import (weight_bytes,            # noqa
                                float_fallback_frac,
                                peak_activation_bytes)

from mcubench.reference import (REFERENCE, PUBLISHED_DEPTH,  # noqa
                                build, fuse)
from mcubench.scaling import BUDGETS, fit                         # noqa
from mcubench import reference as R                               # noqa

BLOCKS = (R._BasicBlock, R._TemporalBlock, R._InvertedResidual,
          R._ShuffleBlock, R._GhostBottleneck, R._V3Bneck,
          R._FasterBlock, R._RepViTBlock, R._UIB)
import torch.nn as nn                                             # noqa


def depth(m):
    n = sum(isinstance(x, BLOCKS) for x in m.modules())
    return n or sum(isinstance(x, nn.Conv1d) for x in m.features)


for bname, cap in BUDGETS.items():
    print(f"=== {bname}  ({cap // 1024} KB INT8 weights) ===")
    print("{:<13} {:>6} {:>6} {:>5} {:>4} {:>8} {:>8} {:>7} {:>6}".format(
        "family", "w", "d", "blk", "pub", "bytes", "util%", "SRAM", "fp32%"))
    for meth in ("uniform", "joint"):
        print(f"-- {meth}")
        for name in REFERENCE:
            w, d = fit(name, cap, meth)
            m = fuse(build(name, w, d))
            b = weight_bytes(m)
            print("{:<13} {:>6.3f} {:>6.3f} {:>5} {:>4} {:>8,} {:>7.1f}% "
                  "{:>6.1f}K {:>5.0f}%".format(
                      name, w, d, depth(m), PUBLISHED_DEPTH[name], b,
                      100 * b / cap, peak_activation_bytes(m, 1000) / 1024,
                      100 * float_fallback_frac(m)))
    print()
