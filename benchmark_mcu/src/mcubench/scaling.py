r"""Fit each reference family to an INT8 weight budget, two ways.

uniform (U)
    Published depth is kept and every channel count is multiplied by a
    single width multiplier, which is MobileNet's own scaling knob
    (Howard et al. 2017, section 3.1).

joint (J)
    Depth and width move together under fixed-resolution compound
    scaling (Tan and Le 2019): d = alpha^phi, w = beta^phi with
    alpha = 1.2 and beta = 1.1. The resolution term is dropped because
    the input length is fixed at ten seconds, and no stage is allowed to
    fall below one block, so a family keeps its stage structure however
    far it is scaled down.

Both the weight budget and the activation ceiling are enforced on
predicted *deployed* bytes rather than on parameter and tensor counts.
The two differ by up to 4x for any family using dilated convolutions,
which the int8 path refuses to quantise -- see `footprint`.
"""
from __future__ import annotations

from .footprint import peak_activation_bytes, weight_bytes
from .reference import build, fuse

ALPHA, BETA = 1.2, 1.1          # depth, width
BUDGETS = {"UC-128": 128 * 1024, "C-256": 256 * 1024}
ACT_CAP = 96 * 1024             # peak activation the device can host


def _wd(method: str, t: float) -> tuple[float, float]:
    """Map a scalar search coordinate to (width, depth) multipliers."""
    if method == "uniform":
        return t, 1.0
    return BETA ** t, ALPHA ** t


def fit(name: str, cap: int, method: str, act_cap: int = ACT_CAP,
        iters: int = 50) -> tuple[float, float]:
    """Largest model of this family that fits `cap` deployed weight bytes.

    Monotone in the search coordinate, so a bisection is exact to the
    precision of the bracket; the model is rebuilt at each step because
    channel rounding makes the cost a staircase rather than a curve.
    """
    lo, hi = ((0.01, 8.0) if method == "uniform" else (-40.0, 20.0))

    def ok(t):
        w, d = _wd(method, t)
        try:
            m = fuse(build(name, w, d))
        except Exception:                                     # noqa: BLE001
            return False
        return (weight_bytes(m) <= cap
                and peak_activation_bytes(m, 1000) <= act_cap)

    if not ok(lo):
        raise ValueError(f"{name}: does not fit {cap} B even at minimum")
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return _wd(method, lo)
