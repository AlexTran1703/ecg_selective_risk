# Mobile-efficient is not MCU-deployable: ten 1-D encoder families
measured on an STM32F411

**Four clinical ECG sources, leave-one-source-out evaluation, two size
budgets, and every footprint and latency figure measured on the part.**

> Efficient-architecture research is evaluated on phones and GPUs. A
> microcontroller enforces a different constraint: not throughput, but
> 128 KB of SRAM shared with the rest of the firmware. Do the design
> innovations of 2016-2024 survive that, and can their benefit be
> predicted before anything is compiled?

The answer to the second question is no, and it is the more useful
finding. Peak activation derived from the graph -- the number a designer
can compute in seconds and the one most papers reason with -- carries
**no usable rank information** about what the device actually
allocates: Spearman rho between the two is **-0.14** at the compact
budget and **+0.13** at the standard budget, neither distinguishable
from zero. The proxy's cheapest model is the device's most expensive.

---

## Research questions

| | question | evidence |
|---|---|---|
| **RQ1** | At a fixed MCU budget, how much does the architecture decide external diagnostic performance? | Figure 2, Figure 3, Figure 4, Table 3 |
| **RQ2** | Do the efficiency innovations of 2016-2024 translate into measured Flash, SRAM and latency on a Cortex-M4? | Figure 5, Figure 6 |
| **RQ3** | What performance-resource trade-offs are observed, and can MCU feasibility be predicted from the graph rather than measured? | Figure 7 |

```
External discrimination  ->  Hardware characterisation  ->  Performance-resource relationships
```

## What is, and is not, claimed as new

Benchmarking 1-D CNNs on ECG is not new, and neither is observing that
MACs predicts latency poorly. Two things here are less common.

1. **Ten families are held to the same budget and the same cohort, then
   actually deployed.** Each is scaled to the largest version of itself
   that fits, rather than to a common width that would favour whichever
   design happens to be densest at that setting. All 20 model-budget
   combinations linked, flashed and returned correct output on a
   physical STM32F411 -- "deployed" is a measurement here, not a
   projection from a parameter count.

2. **The graph-to-device SRAM gap is quantified per family and shown to
   destroy the ranking**, not merely to shift it. This is the practical
   claim: a designer cannot screen architectures for an MCU using
   activation sizes read off the model definition.

What is *not* claimed: that these are the published networks. Each
family contributes its **defining block mechanism** -- ten distinct
mechanisms across ten families -- under a common stem, depth regime and
width-scaling rule. MCUNet and MicroNAS are cited as related work and
not benchmarked: both are architecture *search procedures*, paired in
MCUNet's case with a co-designed runtime, and reproducing a search is a
separate undertaking from benchmarking a block.

## Protocol

**Cohort.** Four public sources (PTB-XL, Georgia, Chapman, Ningbo),
77,333 records, 66,379 retained under a harmonised 13-label space. The
data layer is imported unchanged from the cross-source study, so the two
cannot drift apart on cohort definition.

**Partitions.** Leave-one-source-out: train on three, validate on a 10%
slice of those three, test on every record of the fourth. The held-out
source contributes nothing to training, validation, calibration or
threshold selection.

**Input.** 10 s, 12 leads, decimated to 100 Hz: 12 x 1000. Fixed before
anything was trained.

**Everything else pinned.** Same loss (BCE with positive weighting
capped at 50), AdamW at 2e-3, OneCycle, 12 epochs, batch 256, seed 0.
The only thing that varies is the architecture and its width.

**Width is set by the budget, not fixed.** For each family the largest
width multiplier satisfying both budgets is found by bisection:

| budget | int8 weights | peak activation (design target) |
|---|---|---|
| compact | <= 32 KB | <= 96 KB |
| standard | <= 128 KB | <= 96 KB |

96 KB leaves room on a 128 KB part for stack, I/O buffers and firmware.
It is a *design target* evaluated against a graph-derived proxy;
Figure 5 reports what the device actually required, and the two differ
substantially (Finding 9).

**Quantisation.** ONNX Runtime static PTQ, QDQ form, per-channel,
percentile-99.99 calibration on 256 records drawn only from each fold's
training sources.

**Deployment.** ST Edge AI Core (X-CUBE-AI 10.2.0) analyze and generate,
linked with arm-none-eabi 13.3.rel1 against the CM4 runtime, flashed
over SWD, timed with the DWT cycle counter at 100 MHz: 32 inferences
after a warm-up, median reported. The series is near-constant on this
part -- for a representative model, 7 distinct values across 32 runs
spanning 0.007% of the median -- so no p95 is reported separately; it
coincides with the median at the stated precision. Timing excludes
preprocessing and input transfer. Footprint and latency depend on the
architecture and budget, not on which source was held out, so one
fold's weights are deployed per combination.

---

## RQ1 - external discrimination

> **Finding 1. At a fixed MCU budget, architecture choice is nearly
> irrelevant to accuracy -- provided the family fits the budget at all.**
> Six of ten configurations sit within **0.010** equal-source
> macro-AUPRC of the best at each budget, with bootstrap intervals
> overlapping for almost every pair within that group. A 2016 ResNet and
> a 2024 RepViT are separated by **0.0035** (compact) and **0.0012**
> (standard).
>
> The full spread is nevertheless large -- **0.090** compact, **0.096**
> standard -- and that is almost entirely one family. Excluding
> MobileNetV3 the spread collapses to **0.038** and **0.025**. The
> honest statement is therefore two-part: among families that scale into
> the budget the choice hardly matters, and whether a family scales into
> the budget matters enormously. **Figure 2, Figure 3, Table 3.**

> **Finding 1b. The source ordering is not an artefact of class
> prevalence.** Macro-AUPRC averages 13 per-class average precisions,
> and each has a floor equal to that class's positive rate, so a source
> whose classes are commoner scores higher before any model is any
> good. That floor is worth computing rather than assuming, because the
> four sources differ sharply in composition: NSR is **88.3%** of
> PTB-XL and about **21%** of each of the other three, and PTB-XL's
> remaining twelve classes average 5.4% prevalence against 9.4-10.3%
> elsewhere (Table 1, Table S1).
>
> The macro-average of per-class prevalence -- what a random scorer
> attains -- is nevertheless almost flat across the four:
>
> | source | floor | achieved | lift | ratio |
> |---|---|---|---|---|
> | PTB-XL | **0.117** | 0.418 | 0.300 | 3.6x |
> | Georgia | 0.111 | 0.509 | 0.398 | 4.6x |
> | Chapman | 0.103 | 0.548 | 0.445 | 5.3x |
> | Ningbo | 0.105 | 0.571 | 0.467 | 5.5x |
>
> Macro-averaging washes the NSR difference out, because NSR is one
> class of thirteen. PTB-XL has the **highest** floor and the lowest
> achieved score, so prevalence works against the observed ordering
> rather than producing it, and the ordering is unchanged whether
> sources are ranked by the raw value, by lift over floor, or by ratio
> to it. Whatever makes PTB-XL hard, it is not that its classes are
> rare. **Table 1, Table S1.**

> **Finding 2. The newer the design, the less it helps.** The compact
> budget is led outright by **ResNet-1D (2016)**; the standard budget by
> **TCN (2018)**, with ResNet-1D and ShuffleNetV2 (2018) within 0.0003
> of each other behind it. The two 2024 entries place 4th and 6th
> (compact) and 4th and 8th (standard).
>
> Efficiency innovations developed for wide 2-D networks on mobile SoCs
> do not transfer to narrow 1-D models at MCU scale, where the channel
> counts are an order of magnitude smaller than the regime those blocks
> were tuned for.

> **Finding 3. MobileNetV3-Small does not scale into an MCU budget, and
> the reason is structural rather than a defect of the block.** It is
> last at both budgets by a wide margin (0.4311 and 0.4421, against
> 0.4827 and 0.5131 for the next-worst family).
>
> The cause is visible in the width column of Table 2. MobileNetV3-Small
> spends its parameter budget on *depth and gating*: eleven inverted
> residual blocks, squeeze-excitation on nine of them, and 5-wide kernels
> on eight. To bring that to 32 KB of int8 weights,
> bisection has to scale it to **w = 0.18** -- nearly four times
> narrower than the next-lowest family (ResNet-1D at 0.65). At that
> width the stem and three blocks are clamped at the 4-channel floor,
> and the first block's "expansion" is 4 channels wide, so the inverted
> residual degenerates to something with no expansion at all. The trunk
> carries 4 to 16 channels where its competitors carry dozens.
>
> This should be read as *"this architecture does not scale into this
> budget"*, not *"this is a weak architecture"*. The distinction matters
> for a designer: the failure is a width-scaling incompatibility between
> a deep, heavily gated topology and a hard weight ceiling, and it would
> not be fixed by training longer or tuning the block.

> **Finding 4. FasterNet's diagnosis is not reproduced at MCU scale.**
> Chen et al. argue that the efficient-architecture literature chased
> low FLOP *counts* while ignoring low FLOP *throughput*: depthwise
> convolutions have poor arithmetic intensity, so saved arithmetic does
> not become saved time. Partial convolution is the remedy -- convolve a
> contiguous quarter of the channels densely, pass the rest through.
>
> On this part the remedy does not win on its own axis. FasterNet
> reaches **66.4 ms per MMACC** at both budgets, placing **6th of ten**
> at each. At the standard budget it is beaten on arithmetic efficiency
> by all three of the designs it names: GhostNet 45.2, ShuffleNetV2
> 49.4, MobileNetV2 50.6. It is also 9th of ten on external AUPRC,
> because passing three quarters of the channels through untouched costs
> representational capacity at MCU width.
>
> This is a negative result against the family's own thesis in this
> regime, and it is reported as such. The thesis was argued for wide 2-D
> networks on GPUs and phones; at 12 x 1000 inputs and tens of channels
> the arithmetic-intensity argument does not carry over. See Finding 7
> for why it cannot: the ordering it predicts is not stable even within
> a single family across two budgets.

> **Finding 5. INT8 is free for every family, and now demonstrably so.**
> All **20 of 20** paired bootstrap intervals on the equal-source mean
> INT8 - FP32 difference lie entirely below zero, spanning **-0.0056 to
> -0.0017**; the worst single fold is -0.0133. Every one of those
> numbers is an order of magnitude below the gap between the best and
> worst family.
>
> The intervals are paired by construction -- one record index set per
> held-out source, both precisions scored on the same draw -- because
> FP32 and INT8 share most of their variance. An earlier version of
> Figure 4 drew the *range across the four sources* instead, which
> crossed zero for several families; that quantity describes how much
> the effect varies between datasets, not how precisely the mean is
> estimated, and it cannot shrink with more records. Which family
> degrades most is not stable across budgets (FCN-1D worst at compact,
> MobileNetV3 at standard), so no architectural story is
> attached to the ordering. **Figure 4.**

---

## RQ2 - hardware characterisation

> **Finding 6. MACs mis-ranks latency by a factor of 2.8.** Pooled
> Spearman rho between MACC and measured latency is **+0.88**, so the
> association is strong -- but within a budget, where the screening
> decision is actually made, Kendall tau falls to **+0.42** (compact)
> and **+0.56** (standard). Realised cost spans **40.2 to 111.5 ms per
> million MACC**.
>
> | compact | ms/MMACC | standard | ms/MMACC |
> |---|---|---|---|
> | ResNet-1D | **40.2** | GhostNet | **45.2** |
> | RepViT | 51.2 | ShuffleNetV2 | 49.4 |
> | FCN-1D | 53.3 | MobileNetV2 | 50.6 |
> | MobileNetV4-Conv | 59.8 | MobileNetV4-Conv | 55.4 |
> | MobileNetV2 | 61.8 | RepViT | 63.7 |
> | FasterNet | 66.4 | FasterNet | 66.4 |
> | ShuffleNetV2 | 77.0 | FCN-1D | 67.2 |
> | GhostNet | 86.7 | MobileNetV3 | 82.6 |
> | MobileNetV3 | 108.2 | ResNet-1D | 94.5 |
> | TCN | **111.5** | TCN | **98.6** |
>
> TCN is the least arithmetic-efficient design at both budgets: dilated
> convolutions break access locality, and no MAC count sees that.
> **Figure 5, Figure 6.**

> **Finding 7. Arithmetic efficiency is not a property of the block --
> it reverses with scale.** Read the two columns of Finding 6 side by
> side. **ResNet-1D is the most efficient design at the compact budget
> (40.2) and the second-least efficient at the standard budget (94.5)**,
> a 2.4x swing within one family. **GhostNet moves the opposite way**:
> 8th at compact (86.7), 1st at standard (45.2).
>
> The mechanism is working-set size. At the compact budget every
> intermediate tensor is small enough to stay in cache, and dense
> convolution vectorises best. At the standard budget ResNet-1D's wide
> dense kernels over a 1000-sample sequence no longer fit, and the
> design becomes memory-bound; GhostNet's cheap-operation half, which
> buys nothing when everything is already cached, starts paying.
>
> This is the strongest form of the screening warning in this report. It
> is not merely that MACs mispredicts latency by a fixed per-family
> factor that a designer could learn once and reuse -- the factor itself
> inverts between two budgets of the *same architecture on the same
> part*. **Figure 6, Figure S2.**

> **Finding 8. Every model runs far inside real time, so latency is not
> the binding constraint.** RTF spans **0.007 to 0.173** across both
> budgets; even the slowest standard-budget encoder (TCN, 1726 ms)
> finishes a 10 s recording in under two seconds. On this part, memory
> is what binds. **Figure 5.**

---

### A note on hardware-aware search

Direct latency measurement on the target follows the evaluation
philosophy MnasNet argued for: proxies such as FLOPs or MACs are poor
stand-ins for what a device does, so the device should be asked. This
study adopts that principle and nothing else from it. No architecture
search is run, no latency-aware reward is optimised, and MnasNet is not
benchmarked here. A prespecified set of families is measured, and the
Pareto analysis in RQ3 is a descriptive summary of the trade-offs
observed among them -- not a search result and not a claim of
optimality beyond the configurations tested.

---

## RQ3 - performance-resource relationships

> **Finding 9. No. The graph-derived SRAM proxy carries no usable rank
> information about what the device allocates.** Spearman rho between
> graph peak activation and measured peak SRAM is **-0.14** at the
> compact budget (p = 0.71) and **+0.13** at the standard budget
> (p = 0.71). Dropping MobileNetV3 as a possible outlier does not
> rescue it: +0.20 and +0.51, neither significant at n = 9.
>
> Standard budget, predicted against required:
>
> | model | graph proxy | measured SRAM | ratio |
> |---|---|---|---|
> | MobileNetV3 | **3.4 KB** | **77.7 KB** | **22.9x** |
> | GhostNet | 15.6 KB | 79.6 KB | 5.1x |
> | ResNet-1D | 7.8 KB | 30.8 KB | 3.9x |
> | TCN | 14.6 KB | 46.5 KB | 3.2x |
> | ShuffleNetV2 | 20.5 KB | 60.8 KB | 3.0x |
> | RepViT | 19.5 KB | 48.2 KB | 2.5x |
> | FasterNet | 21.5 KB | 50.5 KB | 2.4x |
> | FCN-1D | 16.6 KB | 32.9 KB | 2.0x |
> | MobileNetV4-Conv | 39.1 KB | 54.6 KB | **1.4x** |
> | MobileNetV2 | 39.1 KB | 54.5 KB | **1.4x** |
>
> Read the proxy column alone and MobileNetV3 is the obvious choice for
> a memory-constrained part at 3.4 KB, while the inverted-bottleneck
> families look like the memory hogs at 39.1 KB. On the device the order
> is roughly reversed: MobileNetV3 needs **77.7 KB**, leaving under
> 50 KB of a 128 KB part for everything else, while MobileNetV2 and V4
> sit mid-pack. At the compact budget the inversion is exact -- the
> proxy's cheapest model is the device's most expensive.

> **Finding 10. The gap decomposes into two terms the graph cannot
> see, and the larger one is architecture-dependent.** The toolchain's
> SRAM total is an activation arena plus the generated runtime's own
> RAM:
>
> - **The activation arena has a floor of 23.4 KB** -- the I/O buffering
>   for a 12 x 1000 input -- which no model can go below. For a family
>   whose internal tensors are genuinely small, the floor *is* the
>   arena, and the graph proxy misses it entirely.
> - **Runtime RAM spans 3.9 KB to 54.2 KB** and is *not* a constant
>   overhead that cancels in a comparison. FCN-1D's plain convolutions
>   need 3.9 KB of runtime state; MobileNetV3's eleven gated blocks need
>   **54.2 KB**, fourteen times more, and that single term is two thirds
>   of its measured footprint.
>
> This is why Finding 9 cannot be repaired by applying a correction
> factor. The error is a sum of a fixed floor the proxy omits and a
> per-architecture runtime cost that scales with operator count and
> operator variety rather than with tensor size -- neither of which is
> recoverable from the model definition.

> **Finding 11. The standard budget buys little for what it costs.**
> Quadrupling the weight allowance raises external macro-AUPRC by
> **+0.0183** on average (range +0.0096 to +0.0305 across families). The
> measured price is **2.5x the Flash** (median; 1.8x to 3.4x) and
> **3.6x the latency** (median; 2.1x to 9.2x).
>
> Note that Flash grows by 2.5x rather than the 4x the budget implies:
> the generated runtime is a fixed overhead that does not scale with the
> model, so small models are charged for it disproportionately. Latency
> scaling varies widely by family -- ResNet-1D pays 9.2x while GhostNet
> pays 2.1x -- which is Finding 7 seen from the other side, and another
> number that cannot be predicted from the budget alone. For a
> battery-powered device the compact budget is the defensible operating
> point. **Figure 5, Figure 7.**

> **Finding 12. Flash inflates too, and two families inflate
> differently.** The linked image is 1.2x to 1.5x the int8 weight size
> for eight of ten configurations -- the generated runtime and kernels.
> MobileNetV3 is 2.0x and **TCN is 3.2x** (113 KB of weights, 358 KB of
> Flash), the only family approaching the 512 KB limit, because its
> dilated kernels expand in the generated code. Weight count predicts
> Flash well enough to screen with, except where it does not, and which
> case you are in is not visible from the graph either.

---

## Headline

At a fixed microcontroller budget the architecture family barely affects
diagnostic accuracy among the families that fit it, the efficiency
innovations of 2016-2024 do not translate into lower measured latency,
and the memory figure a designer would use to screen candidates is
**rank-uncorrelated** with what the device allocates. The practical
recommendation is the uncomfortable one: **choose on measured footprint,
not on published efficiency claims or graph-derived estimates, and
measure on the part before committing.**

---

## Conclusions

1. **Architecture choice is budget-dependent.** ResNet-1D leads the
   compact budget, TCN the standard one, and no configuration leads
   both.

2. **Fitting the budget matters far more than the block.** Among the
   nine families that scale into the budget the spread is 0.038 AUPRC
   at the compact budget and 0.025 at the standard; adding
   MobileNetV3, which must shrink to w = 0.18 and starves its trunk to
   4-16 channels, widens them to 0.090 and 0.096.

3. **Rankings move across held-out sources.** The source term dominates
   the architecture term, and per-source orderings differ from the
   equal-source mean (Figure 3).

4. **MACC is useful but insufficient to predict measured latency.**
   Realised cost spans 40.2 to 111.5 ms per million MACC, a 2.8x range
   at equal arithmetic, and is not stable within a family across
   budgets -- ResNet-1D swings from most to second-least efficient.

5. **INT8 quantisation is free at this scale.** All 20 paired intervals
   lie below zero but span only -0.0056 to -0.0017, an order of
   magnitude below the gap between the best and worst family. Small
   aggregate changes may still hide per-class effects, which this study
   does not resolve.

6. **Several configurations are non-dominated.** Deployment choice
   therefore depends on whether latency, SRAM or discrimination binds
   in a given application, rather than on a single universal winner.

A seventh point follows from the measurements rather than the rankings,
and is the one with the clearest practical consequence: the memory
figure a designer would use to screen candidates has a rank correlation
with the measured figure of -0.14 to +0.13, so screening on
graph-derived activation sizes is no better than screening at random.

---

## Limitations

1. **Ten block mechanisms, no architecture search.** Neither MCUNet nor
   MicroNAS was reproduced: both are search procedures paired, in
   MCUNet's case, with a co-designed runtime. Nothing here speaks to
   what a search would find on this part.

2. **Block fidelity is to the mechanism, not the published topology.**
   Each family keeps its defining block -- verified against the papers,
   including squeeze-excitation placement for GhostNet (wider stages
   only, ratio 0.25), MobileNetV3 (Table 2 of Howard et al., verbatim)
   and RepViT (first block of each stage) -- but all are far shallower
   than published, share a stride-4 stem none of the originals has, and
   are width-scaled to budget. TCN is additionally non-causal with
   BatchNorm rather than weight normalisation. FasterNet is reported at
   its published partial ratio (n_div = 4) only. The published
   configurations do not run on this part at any width, so benchmarking
   the mechanisms under a common budget is the only available
   comparison, but "we benchmarked MobileNetV4" and "we benchmarked UIB
   blocks at MCU scale" are different claims and only the second is
   supported.

3. **Width scaling is one degree of freedom, and Finding 3 is partly a
   consequence of that choice.** Each family is scaled by channel width
   at fixed depth. MobileNetV3's collapse is a real incompatibility
   between a deep gated topology and a hard weight ceiling under
   *width* scaling; a depth-reduced MobileNetV3 might fare better and
   was not tried. These are results for "this family scaled this way",
   not for the family in general.

4. **One part, one toolchain.** STM32F411 at 100 MHz with X-CUBE-AI
   10.2.0 and CMSIS-NN kernels. The graph-to-device gap in Findings 9
   and 10 is partly a property of *this* allocator and these kernels;
   the direction should generalise, the factors need not. A second
   target would be the obvious strengthening.

5. **One seed per configuration.** Intervals cover sampling variation in
   the evaluation records, not variation in training. Differences of
   0.003 between families are not resolvable and are not treated as
   rankings anywhere in the text.

6. **Four source domains, and two of them share a labelling protocol.**
   Chapman and Ningbo come from the same group; the cross-source study
   documents the consequence. External performance here should be read
   as generalisation to these four held-out sources, not to clinical
   ECG at large.

7. **No energy measurement.** Latency is measured; joules per inference
   are not, and for a battery-powered device that is the number that
   ultimately matters.

---

## Files

```
benchmark_mcu/
  REPORT.md
  reference.bib                    verified entries
  src/mcubench/models.py           ten families
  scripts/b1_train.py              LOSO training, both budgets
  scripts/b2_int8.py               ONNX Runtime static PTQ
  scripts/b2b_int8_ci.py           paired bootstrap of INT8 - FP32
  scripts/b3_deploy.py             analyze, generate, build, flash, time
  scripts/b4_assets.py             tables and figures
  figures/  tables/                versioned
  results/                         measurement JSON versioned; the rest
                                   (checkpoints, ONNX, build trees) is
                                   regenerable and ignored
```

### Reproducing

```
python benchmark_mcu/scripts/b1_train.py      # 80 runs, ~45 min on one GPU
python benchmark_mcu/scripts/b2_int8.py       # 80 quantisations
python benchmark_mcu/scripts/b2b_int8_ci.py   # 400-replicate paired bootstrap
python benchmark_mcu/scripts/b3_deploy.py     # 20 builds, needs the board
python benchmark_mcu/scripts/b4_assets.py     # tables and figures
```

`ECG_FS=100` must be set; `b3_deploy.py` needs an STM32F411 attached and
the ST toolchain installed at the paths declared at the top of that file.

## Main figures

| | file | answers |
|---|---|---|
| 1 | `figures/figure1_design` | graphical overview in three columns: sources, benchmark, outputs. ECG traces are real records drawn from the cohort by index; the RQ1-RQ3 insets are the actual results in miniature; counts are read from the artefacts. Also exported as `.svg` |
| 2 | `figures/figure2_cross_source` | **RQ1** external AUPRC for every family on every held-out source, both budgets |
| 3 | `figures/figure3_distribution` | **RQ1** spread across sources per family, with bootstrap intervals -- the overlap is the point |
| 4 | `figures/figure4_int8_cost` | **RQ1** what quantisation costs: paired bootstrap intervals on the equal-source mean INT8 - FP32 difference, all twenty entirely below zero |
| 5 | `figures/figure5_hardware` | **RQ2** (A) Flash, (B) peak SRAM, (C) MACC, (D) measured latency, for every family at both budgets. Solid bars are the compact budget, hatched the standard. Replaces the deployment table outright, so the extremes a reader would have looked up are annotated and the part's capacities are stated in-panel; exact values for all twenty configurations are deposited as `results/deploy_*/deployment_measurements.csv` |
| 6 | `figures/figure6_complexity_latency` | **RQ2** (a) latency against MACC on log-log with Spearman and per-budget Kendall, (b) ms per million MACC paired across budgets -- the reversal in Finding 7 |
| 7 | `figures/figure7_tradeoffs` | **RQ3** equal-source external macro-AUPRC against measured latency, both budgets on one log axis. Marker shape is the budget, colour the family, area the measured peak SRAM. Outlined markers are non-dominated in the accuracy-latency plane within their budget; every tested configuration is drawn and **no connecting frontier is implied** -- these are twenty discrete measurements, not a continuum |
| S2 | `figures/figureS2_residual_latency` | OLS residual latency, retained for completeness; specification-dependent, which is why it is not in the main set |

## Main tables

| | file | answers |
|---|---|---|
| 1 | `tables/table1_protocol` | cohort composition: records, patients, age, sex, labels per ECG, the macro-AUPRC prevalence floor, and four diagnoses spanning the prevalence range |
| 2 | `tables/table2_architectures` | the ten families: identifier, year, defining block, width, size, MACs |
| 3 | `tables/table3_external` | external macro-AUPRC and AUROC per source, with bootstrap CIs |

There is no Table 4. The deployment measurements are Figure 5, and the
exact numbers are deposited as a machine-readable CSV
(`results/deploy_*/deployment_measurements.csv`) rather than as a
supplementary table: a reader comparing architectures wants the figure,
and a reader reproducing the work wants the file, and neither wants a
typeset grid of twenty rows.
| S1 | `tables/tableS1_prevalence` | prevalence of all 13 harmonised classes in each source, ordered by pooled prevalence |
