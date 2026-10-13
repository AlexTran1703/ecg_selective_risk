# Reconstruction: faithful architectures and measured budgets

*Written for the authors and co-authors of the BPEX submission — it is
the methods record for the rebuilt benchmark, and the place the
toolchain constraints are documented so they are not rediscovered.*

This document covers the design and the deployment constraints. Results
are added as the runs land; nothing here is provisional.

## 1. Why the benchmark was rebuilt

The earlier benchmark compared ten families at a fixed INT8 weight
budget, and the comparison was confounded in three ways.

**Depth was truncated inconsistently.** Nine of the ten families were
cut to between 15% and 100% of their published block count — RepViT kept
15%, GhostNet 19%, FasterNet 23%, MobileNetV2 24%, ShuffleNetV2 25%,
MobileNetV4 33%, ResNet 38% — while FCN and MobileNetV3 kept all of
theirs. A family was therefore being judged partly on how much of it had
survived, which is not a property of the design.

**TCN was not TCN.** It was non-causal, used batch normalisation in
place of weight normalisation, had no dropout, and inserted two
max-pools — which contradicted its own rationale, since pooling grows
the receptive field by stride where the design grows it by dilation.

**MobileNetV4's UIB put the stride on the wrong operator.** With no
middle depthwise, the stride fell to the final pointwise projection. A
strided 1×1 convolution subsamples without aggregating, which aliases.
Correcting it cut peak activation by 40% and MACs by 13% at identical
parameter count.

## 2. Design

Two tiers, because two different questions were being conflated.

| Tier | What it is | Answers |
|---|---|---|
| **REF** | 10 families, published topology, float32, **no device constraint** | What is the design worth as a design? (RQ1) |
| **Deployed** | 2 scaling methods × 2 budgets × 10 families = 40 configurations, INT8 | What does a design buy under a fixed budget? (RQ2, RQ3) |

Fifty configurations, four leave-one-source-out rotations each: 200 runs.

Every family is at its published block count in the reference tier —
FCN 3, ResNet 8, TCN 8, MobileNetV2 17, ShuffleNetV2 16, GhostNet 16,
MobileNetV3 11, FasterNet 13, RepViT 20, MobileNetV4 12 — and each keeps
its own published stem and stride schedule rather than a shared one.

### Scaling methods

**Uniform (U).** Published depth is kept; every channel count is
multiplied by a single width multiplier. This is MobileNet's own knob
(Howard et al. 2017, §3.1).

**Joint (J).** Depth and width move together under fixed-resolution
compound scaling (Tan and Le 2019): `d = 1.2^φ`, `w = 1.1^φ`. The
resolution term is dropped because the input length is fixed at ten
seconds. No stage may fall below one block, so a family keeps its stage
structure however far it is scaled down.

Channel rounding uses MobileNet's `_make_divisible` — divisor 8 with the
10% guard — not the earlier round-to-four, which could cost a layer more
than a tenth of its width to rounding alone.

### Budgets

| Name | INT8 weights | Peak activation |
|---|---|---|
| UC-128 | ≤ 128 KB | ≤ 96 KB |
| C-256 | ≤ 256 KB | ≤ 96 KB |

96 KB leaves room on a 128 KB part for stack, I/O buffers and firmware.
Both ceilings are enforced on **predicted deployed bytes**, not on
parameter counts — see §3, where the two are shown to differ by 4×.

## 3. Deployment constraints, measured

Three properties of the ST Edge AI 10.2.0 / X-CUBE-AI path were found by
probing the toolchain directly. All three are invisible until code
generation, which runs long after training; each was found before the
training run rather than after it.

### 3.1 Dilated convolutions are never quantised

Any `dilation > 1` is refused by the INT8 path and ships as float32 —
four bytes per weight, float32 activations, and `s8→f32`/`f32→s8`
conversions bracketing the layer.

One 32×32, k=3 convolution, parameter count identical throughout:

| Dilation | 1 | 2 | 4 | 8 | 32 | 128 |
|---|---|---|---|---|---|---|
| Weight bytes | 3,200 | 12,416 | 12,416 | 12,416 | 12,416 | 12,416 |

This is the whole explanation for the earlier TCN overrunning its weight
budget by 2.5×, and it is a property of the toolchain rather than of the
architecture. It is reported as such, but it is the toolchain this
benchmark targets, so TCN pays it.

### 3.2 A convolution fed by a slice fails code generation

`stedgeai generate` fails with `INTERNAL ERROR: 'onnxConvNNNN'` when a
Conv consumes a Python slice `x[:, :n]` or `tensor.chunk(...)`. The same
tensor taken with `torch.split(x, [a, b], dim=1)` generates and links.

| Construction | Code generation | Link |
|---|---|---|
| no subsetting (control) | ok | ok |
| `x[:, :n]` | **FAIL** | — |
| `tensor.chunk(2, 1)` | **FAIL** | — |
| `torch.split(x, [a, b], 1)` | ok | ok |

This affects partial convolution (FasterNet) and channel split
(ShuffleNetV2); a trailing slice after a concatenation (GhostNet) is the
same hazard, avoided by sizing the branch so no slice is needed. Slicing
along the **time** axis is unaffected — TCN's causal chomp generates
without complaint.

### 3.3 Untrained batch-norm produces a false codegen failure

This one is a trap rather than a constraint, and it cost a wrong
diagnosis before it was understood.

With untrained batch normalisation (weight 1, bias 0, mean 0, var 1)
every folded bias is the *identical* all-zero vector. The ONNX exporter
deduplicates them into one initializer aliased by `Identity` nodes, and
code generation fails on that with an error naming a Conv — which is
indistinguishable from an unsupported operator.

It only strikes blocks whose convolutions share an output width, which
is why ShuffleNetV2 and GhostNet appeared to fail while FCN, ResNet and
MobileNetV2 did not. Both generate cleanly once the norm parameters are
perturbed to look trained. **No architecture change was needed**, and
making one would have corrupted two faithful implementations.

Any probe run on untrained models must perturb batch-norm first.

### 3.4 Predicting deployed bytes

The budget search cannot call `stedgeai analyze` thousands of times, so
`mcubench.footprint` predicts deployed weight bytes analytically: one
byte per weight (four if the layer is dilated and therefore unquantised),
plus eight bytes per output channel for the int32 folded bias and the
float32 per-channel scale, with batch-norm folded away.

Validated against all twenty measurements from the earlier hardware run:

| | Max error | Mean error | Direction |
|---|---|---|---|
| Predicted vs measured weight bytes | +17.1% | +6.1% | **conservative in all 20** |

Always over-predicting is the safe direction: a configuration that
passes the predicted budget meets the real one.

The activation estimate is weaker, and knowingly so. For a graph the
INT8 path fully accepts it is accurate; for one it does not, it
under-predicts — TCN is predicted at 78 KB and measured at 110 KB, which
is the difference between linking on a 128 KB part and overrunning RAM
by 17 KB. Activation ceilings are therefore settled by **building**:
each configuration starts from the analytic fit and, while the firmware
fails to link, the ceiling is tightened and the family refitted until it
does. The verified result is frozen in `results/config_manifest.json`,
which every later stage reads instead of recomputing the search — so the
width a checkpoint was trained at cannot drift from the width that gets
deployed.

## 4. What binds, and for whom

Eight of the ten families are weight-bound: they sit at 89–99% of their
INT8 budget, which is the constraint the benchmark intends to apply.

Two are not.

**FCN-1D is activation-bound.** Faithful FCN does no downsampling
anywhere, so at 1000 samples it reaches the 96 KB activation ceiling at
w = 0.391 while using 33% of UC-128 and 17% of C-256. It is *the same
model* in all four deployed configurations: for this family the budget
contrast is empty.

**TCN is activation-bound as well**, for a different reason — the
float32 fallback of §3.1 applies to its activations, not only its
weights. Under uniform scaling it uses 36%/18% of the two budgets and is
the same model at both; 86% of its weights ship as float32, 52% under
joint scaling.

The 96 KB ceiling never bound in the earlier benchmark. It now binds for
exactly these two families, and only because both are now faithful.

## 5. Pipeline

| Stage | Script | Output |
|---|---|---|
| Freeze and verify configurations | `r0_manifest.py` | `results/config_manifest.json` |
| LOSO training, 200 folds | `r1_train.py` | `results/recon_100hz/` |
| INT8 quantisation, 160 runs | `r2_int8.py` | `results/recon_int8_100hz/` |
| Compile, flash, time, 40 configs | `r3_deploy.py` | `results/recon_deploy_100hz/` |
| Figures and tables | `r4_assets.py` | `figures/`, `tables/` |
| Consistency check | `r_verify.py` | — |

`r_verify.py` rebuilds every checkpoint from its stored width and depth,
re-runs the budget search, and reports any disagreement. It exists
because editing an architecture module mid-run once put twelve folds out
of step with their own stored widths: the live process kept the module
it had already imported and wrote pre-edit shapes into post-edit
records, and nothing surfaced until a `state_dict` load failed hours
later. Run it before quantisation and before deployment.

Results are written to new directories. The earlier benchmark's cached
predictions are not modified.
