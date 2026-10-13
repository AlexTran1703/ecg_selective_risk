"""Fidelity check: every reference family builds, runs, and has the
published block count, and nothing blows the device's two hard limits."""
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from mcubench import reference as R          # noqa: E402
from mcubench.models import n_params, peak_activation_bytes  # noqa: E402

BLOCKS = (R._BasicBlock, R._TemporalBlock, R._InvertedResidual,
          R._ShuffleBlock, R._GhostBottleneck, R._V3Bneck,
          R._FasterBlock, R._RepViTBlock, R._UIB)

FLASH, SRAM = 512, 128          # KB, STM32F411VET6


def depth(m):
    n = sum(isinstance(x, BLOCKS) for x in m.modules())
    # FCN has no block class: count its conv-bn-relu triples instead
    return n or sum(isinstance(x, nn.Conv1d) for x in m.features)


print("{:<13} {:>6} {:>6} {:>10} {:>9} {:>9}".format(
    "family", "blk", "pub", "params", "INT8 KB", "SRAM KB"))
x = torch.randn(2, R.N_LEADS, 1000)
bad = 0
for name in R.REFERENCE:
    m = R.build(name).eval()
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, R.N_CLASSES), (name, y.shape)
    nb, pub = depth(m), R.PUBLISHED_DEPTH[name]
    p = n_params(m)
    kb, sram = p / 1024, peak_activation_bytes(m, 1000) / 1024
    flag = ""
    if nb != pub:
        flag, bad = " DEPTH", bad + 1
    if kb > FLASH:
        flag += " FLASH"
    if sram > SRAM:
        flag += " SRAM"
    print("{:<13} {:>6} {:>6} {:>10,} {:>9.1f} {:>9.1f}{}".format(
        name, nb, pub, p, kb, sram, flag))
print("\ndepth mismatches: {}".format(bad))
