r"""Predict deployed INT8 weight bytes without invoking the toolchain.

The budget search needs a cost that can be evaluated thousands of times,
so it cannot call `stedgeai analyze`. Three facts, each measured rather
than assumed, make an analytic prediction possible:

  * batch-norm folds into the preceding convolution, so its parameters
    cost nothing and the convolution always ends up with a bias;
  * the folded bias is stored as int32 and each output channel also
    carries a float32 scale, so a layer pays 8 bytes per output channel
    on top of its weights;
  * **a dilated convolution is not quantised at all.** The INT8 path
    rejects any dilation above one and the layer stays float32 at four
    bytes per weight. A single 32x32 k=3 convolution measures 3,200 B at
    dilation 1 and 12,416 B at every dilation from 2 to 128 -- same
    parameter count, four times the flash.

That last point is why TCN overran its budget by 2.5x: fifteen of its
sixteen convolutions are dilated, so most of the network shipped as
float. It is a property of the deployment toolchain, not of the
architecture, but it is the toolchain this benchmark targets.
"""
from __future__ import annotations

import torch
import torch.nn as nn

SCALE_B = 4          # float32 per-channel scale
BIAS_B = 4           # int32 folded bias
FLOAT_B = 4          # bytes per weight when a layer cannot be quantised


def layer_bytes(m: nn.Module) -> int:
    """Deployed bytes for one weight-bearing layer, after BN folding."""
    if isinstance(m, nn.Conv1d):
        dil = m.dilation[0] if isinstance(m.dilation, tuple) else m.dilation
        per = FLOAT_B if dil > 1 else 1
        return (m.weight.numel() * per
                + m.out_channels * (SCALE_B + BIAS_B))
    if isinstance(m, nn.Linear):
        return m.weight.numel() + m.out_features * (SCALE_B + BIAS_B)
    return 0


def weight_bytes(model: nn.Module) -> int:
    """Predicted deployed weight bytes for a whole model."""
    return sum(layer_bytes(m) for m in model.modules())


def float_fallback_frac(model: nn.Module) -> float:
    """Share of weights that the INT8 path will refuse to quantise."""
    tot = fl = 0
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            n = m.weight.numel()
            tot += n
            dil = m.dilation[0] if isinstance(m.dilation, tuple) else m.dilation
            if dil > 1:
                fl += n
        elif isinstance(m, nn.Linear):
            tot += m.weight.numel()
    return fl / tot if tot else 0.0


@torch.no_grad()
def peak_activation_bytes(model: nn.Module, length: int = 1000) -> int:
    r"""Peak live tensor in bytes, charging float32 where int8 is refused.

    The earlier estimator assumed every activation is one byte, which is
    true only for a graph the int8 path fully accepts. A dilated
    convolution is not accepted: the toolchain brackets it with
    s8->f32 and f32->s8 conversions, so its input and its output are
    both float32 and cost four bytes an element.

    For TCN that is the difference between a 23 KB estimate and the
    165 KB the tool actually reports, which is the difference between a
    model that links on a 128 KB part and one that overruns RAM by
    72 KB. The estimate is still a proxy -- the allocator's true peak
    depends on liveness the graph alone does not give -- but it is now
    wrong in the right direction and for the right reason.
    """
    peak, hooks = 0, []

    def hook(mod, inp, out):
        nonlocal peak
        wide = (isinstance(mod, nn.Conv1d)
                and (mod.dilation[0] if isinstance(mod.dilation, tuple)
                     else mod.dilation) > 1)
        n = FLOAT_B if wide else 1
        if torch.is_tensor(out):
            peak = max(peak, out.numel() * n)
        if wide and inp and torch.is_tensor(inp[0]):
            peak = max(peak, inp[0].numel() * n)

    # Hook weight-bearing layers by type as well as leaves. A conv
    # carrying a weight-norm parametrisation has children, so a
    # leaves-only sweep skips it entirely -- which silently exempted
    # every dilated convolution in TCN from the very charge this
    # function exists to apply. Parametrisation submodules are excluded
    # because what they emit is a weight, not an activation.
    for name, mod in model.named_modules():
        if "parametrizations" in name:
            continue
        if isinstance(mod, (nn.Conv1d, nn.Linear)) or not list(mod.children()):
            hooks.append(mod.register_forward_hook(hook))
    was = model.training
    model.eval()
    model(torch.zeros(1, 12, length))
    model.train(was)
    for h in hooks:
        h.remove()
    return peak
