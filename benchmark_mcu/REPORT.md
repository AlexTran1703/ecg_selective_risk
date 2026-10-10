# Mobile-efficient is not MCU-deployable: ten 1-D encoder families
measured on an STM32F411

**Four clinical ECG sources, leave-one-source-out evaluation, two size
budgets, and every footprint and latency figure measured on the part.**

> Efficient-architecture research is evaluated on phones and GPUs. A
> microcontroller enforces a different constraint: not throughput, but
> 128 KB of SRAM shared with the rest of the firmware. Do the design
> innovations of 2018-2024 survive that, and can their benefit be
> predicted before anything is compiled?

The answer to the second question is no, and it is the more useful
finding. Peak activation derived from the graph -- the number a designer
can compute in seconds and the one most papers reason with -- is wrong
by 1.4x to 4.6x against what the allocator actually needs, and wrong by
a different factor for each family, so the ranking it induces is not the
ranking the device produces.

---

## Research questions

| | question | evidence |
|---|---|---|
| **RQ1** | At a fixed MCU budget, how much does the architecture family decide external diagnostic performance? | Figure 2, Figure 3, Table 3 |
| **RQ2** | Do the efficiency innovations of 2018-2024 translate into measured Flash, SRAM and latency on a Cortex-M4? | Figure 5, Figure S1, Table 4 |
| **RQ3** | Can MCU feasibility be predicted from the graph, or must it be measured? | Table 4 |

```
Does the family matter?  ->  Does its efficiency claim hold?  ->  Could we have known?
```

## What is, and is not, claimed as new

Benchmarking 1-D CNNs on ECG is not new, and neither is observing that
MACs predicts latency poorly. Two things here are less common.

1. **Ten families are held to the same budget and the same cohort, then
   actually deployed.** Each is scaled to the largest version of itself
   that fits, rather than to a common width that would favour whichever
   design happens to be densest at that setting. All 22 model-budget
   combinations linked, flashed and returned correct output on a
   physical STM32F411 -- "deployed" is a measurement here, not a
   projection from a parameter count.

2. **The graph-to-device gap is quantified per family and shown to
   invert the ranking.** This is the practical claim: a designer cannot
   screen architectures for an MCU using activation sizes read off the
   model definition.

What is *not* claimed: that these are the published networks. MCUNet and
MicroNAS are search outcomes, not blocks. `MCUNet-style` reproduces the
design discipline -- low expansion, early downsampling, small channels
where the sequence is long -- and is labelled as such everywhere.
MicroNAS is cited as related work and not benchmarked, because running
the search is a separate undertaking.

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
width multiplier satisfying both budgets is selected:

| budget | int8 weights | peak activation (design target) |
|---|---|---|
| compact | <= 32 KB | <= 96 KB |
| standard | <= 128 KB | <= 96 KB |

96 KB leaves room on a 128 KB part for stack, I/O buffers and firmware.
It is a *design target* evaluated against a graph-derived proxy; Table 4
reports what the device actually required, and the two differ
substantially.

**Quantisation.** ONNX Runtime static PTQ, QDQ form, per-channel,
percentile-99.99 calibration on 256 records drawn only from each fold's
training sources.

**Deployment.** ST Edge AI Core (X-CUBE-AI 10.2.0) analyze and generate,
linked with arm-none-eabi 13.3.rel1 against the CM4 runtime, flashed
over SWD, timed with the DWT cycle counter at 100 MHz: 32 inferences
after a warm-up, median and p95 reported. Footprint and latency depend
on the architecture and budget, not on which source was held out, so one
fold's weights are deployed per combination.

---

## RQ1 - does the family decide performance?

> **Finding 1. At a fixed MCU budget, architecture choice is nearly
> irrelevant to accuracy.** Eight of ten families sit within **0.010**
> macro-AUPRC of one another at the compact budget, and the bootstrap
> intervals overlap for almost every pair. A 2016 ResNet matches a 2024
> RepViT to within 0.003. The two exceptions are FCN-1D and FasterNet,
> which fall clearly behind.
>
> This echoes, in a different setting, the cross-source result that the
> deployment site explains far more variance than the encoder. Here the
> comparison is held at one cohort and one budget, and the families
> still do not separate. **Figure 2, Figure 3, Table 3.**

> **Finding 2. The newer the design, the less it helps.** At the
> standard budget the top three are TCN (2018), ShuffleNetV2 (2018) and
> ResNet-1D (2016). MobileNetV4-Conv (2024) ranks ninth of ten.
> Efficiency innovations developed for wide 2-D networks on mobile SoCs
> do not transfer to narrow 1-D models at MCU scale, where the channel
> counts are an order of magnitude smaller than the regime those blocks
> were tuned for.

> **Finding 3. The partial-convolution deficit is not an artefact of the
> partial ratio.** FasterNet is last at both budgets. The obvious
> explanation is starvation: at the published 1/4 ratio the block
> convolves only 5 of 20 channels at MCU widths, against the much wider
> networks the ratio was tuned on. So the ablation was run at
> `n_div=2`, which doubles the convolved fraction to 50% (10/20, 16/32,
> 26/52). The change is **+0.0044** (compact) and **-0.0001**
> (standard) -- no meaningful difference, and FasterNet stays last.
>
> One qualification, because the protocol makes this a budget question
> rather than a pure ratio question. Convolving half the channels costs
> more parameters per block, so `fit_to_budget` returns a narrower
> network for the ablation (width 0.79 against 0.85 at compact, 1.59
> against 1.79 at standard). The comparison is therefore "convolve more
> channels, at slightly reduced width, for the same Flash", which is the
> right contrast for a budget-constrained benchmark and the one every
> other family is also held to. It does not establish that the 1/4 ratio
> is optimal for PConv in general -- only that reallocating this budget
> towards a denser convolution does not recover the deficit.

> **Finding 4. int8 is free for every family.** Mean change in external
> macro-AUPRC spans **-0.0043 to -0.0010**, worst single fold -0.0080 --
> an order of magnitude below the gap between the best and worst family.
> RepViT degrades most at both budgets, plausibly because
> reparameterisation sums three branches into one kernel and widens the
> weight distribution the quantiser must cover. **Figure 4.**

---

## RQ2 - do the efficiency claims survive the device?

> **Finding 5. MACs mis-ranks latency by a factor of 2.7.** Measured
> milliseconds per million MACC, compact budget:
>
> | model | MACC | latency | ms / MMACC |
> |---|---|---|---|
> | ResNet-1D | 3.24 M | 132.4 ms | **40.9** |
> | FCN-1D | 2.80 M | 149.1 ms | 53.3 |
> | MCUNet-style | 1.35 M | **73.4 ms** | 54.4 |
> | MobileNetV4-Conv | 2.98 M | 178.0 ms | 59.7 |
> | GhostNet | 4.04 M | 259.3 ms | 64.2 |
> | ShuffleNetV2 | 3.57 M | 277.7 ms | 77.8 |
> | TCN | 4.26 M | 474.9 ms | **111.5** |
>
> TCN performs 1.3x the arithmetic of ResNet-1D and takes 3.6x as long:
> dilated convolutions break access locality. ShuffleNetV2's shuffle and
> GhostNet's concatenation both cost memory traffic that no MAC count
> can see. Plain convolutions vectorise, so the oldest design is the
> most efficient per operation performed. **Figure S1.**

> **Finding 6. Every model runs far inside real time, so latency is not
> the binding constraint.** RTF spans 0.007 to 0.173 across both
> budgets; even the slowest standard-budget encoder (TCN, 1726 ms)
> finishes a 10 s recording in under two seconds. On this part, memory
> is what binds. **Table 4.**

> **Finding 7. The standard budget buys little for what it costs.**
> Quadrupling the weight allowance raises external macro-AUPRC by
> **+0.0200** on average (range +0.0059 to +0.0373 across families). The
> measured price is **2.5x the Flash** (median; 2.3x to 3.4x) and
> **4.0x the latency** (median; 2.2x to 9.6x).
>
> Note that Flash grows by 2.5x rather than the 4x the budget implies:
> the generated runtime is a fixed overhead that does not scale with the
> model, so small models are charged for it disproportionately. Latency
> scaling varies widely by family -- ResNet-1D pays 9.6x while GhostNet
> pays 2.2x -- which is another consequence of Finding 5 and another
> number that cannot be predicted from the budget alone. For a
> battery-powered device the compact budget is the defensible operating
> point. **Figure 5.**

---

## RQ3 - could we have known without measuring?

> **Finding 8. No. The graph-derived SRAM proxy is wrong by up to 4.6x,
> and the error inverts the ranking.** Standard budget, predicted peak
> activation against what the device required:
>
> | model | graph proxy | measured SRAM | ratio |
> |---|---|---|---|
> | GhostNet | 20.5 KB | **94.9 KB** | **4.6x** |
> | ResNet-1D | 7.8 KB | 30.8 KB | 3.9x |
> | TCN | 14.6 KB | 46.5 KB | 3.2x |
> | ShuffleNetV2 | 20.5 KB | 60.8 KB | 3.0x |
> | MCUNet-style | 13.2 KB | 30.9 KB | 2.3x |
> | RepViT | 21.5 KB | 46.9 KB | 2.2x |
> | MobileNetV4-Conv | 39.1 KB | 54.6 KB | **1.4x** |
> | MobileNetV2 | 39.1 KB | 54.5 KB | 1.4x |
>
> Read the proxy column alone and the inverted-bottleneck families look
> like the memory hogs at 39.1 KB while GhostNet looks cheap at 20.5 KB.
> On the device the order reverses: GhostNet needs **94.9 KB**, leaving
> barely 33 KB of a 128 KB part for everything else, while MobileNetV2
> and V4 sit mid-pack. The error is not a constant factor that cancels
> in a comparison -- it is architecture-dependent, so screening designs
> on activation sizes read off the model definition selects the wrong
> ones.

> **Finding 9. Flash inflates too, and one family inflates differently.**
> The linked image is 1.2x to 1.4x the int8 weight size for nine of
> eleven configurations -- the generated runtime and kernels. TCN is
> **3.2x** (113 KB of weights, 358 KB of Flash), the only family
> approaching the 512 KB limit, because its dilated kernels expand in
> the generated code. Weight count predicts Flash well enough to screen
> with, except where it does not, and which case you are in is not
> visible from the graph either.

---

## Headline

At a fixed microcontroller budget the architecture family barely affects
diagnostic accuracy, the efficiency innovations of 2018-2024 do not
translate into lower measured latency, and the memory figure a designer
would use to screen candidates is wrong by up to 4.6x in an
architecture-dependent way. The practical recommendation is the
uncomfortable one: **choose on measured footprint, not on published
efficiency claims or graph-derived estimates, and measure on the part
before committing.**

---

## Limitations

1. **Ten families, one hand-specified stand-in.** `MCUNet-style` is not
   the searched MCUNet topology and MicroNAS was not run. Both are
   search procedures; reproducing them is separate work. Results
   attributed to "MCUNet-style" are properties of the design discipline,
   not of the published network.

2. **One part, one toolchain.** STM32F411 at 100 MHz with X-CUBE-AI
   10.2.0 and CMSIS-NN kernels. The graph-to-device gap in Finding 8 is
   partly a property of *this* allocator and these kernels; the direction
   should generalise, the factors need not. A second target would be the
   obvious strengthening.

3. **One seed per configuration.** Intervals cover sampling variation in
   the evaluation records, not variation in training. Differences of
   0.003 between families are not resolvable and are not treated as
   rankings anywhere in the text.

4. **Width scaling is one degree of freedom.** Each family is scaled by
   channel width at fixed depth. A family might do better at a different
   depth-to-width ratio, so these are results for "this family scaled
   this way", not for the family in general.

5. **Four source domains, and two of them share a labelling protocol.**
   Chapman and Ningbo come from the same group; the cross-source study
   documents the consequence. External performance here should be read
   as generalisation to these four held-out sources, not to clinical
   ECG at large.

6. **No energy measurement.** Latency is measured; joules per inference
   are not, and for a battery-powered device that is the number that
   ultimately matters.

---

## Files

```
benchmark_mcu/
  REPORT.md
  reference.bib                    12 entries, 2023-25 verified
  src/mcubench/models.py           ten families + one ablation
  scripts/b1_train.py              LOSO training, both budgets
  scripts/b2_int8.py               ONNX Runtime static PTQ
  scripts/b3_deploy.py             analyze, generate, build, flash, time
  scripts/b4_assets.py             tables and figures
  figures/  tables/                versioned
  results/                         regenerable, not versioned
```

### Reproducing

```
python benchmark_mcu/scripts/b1_train.py      # 88 runs, ~45 min on one GPU
python benchmark_mcu/scripts/b2_int8.py       # 88 quantisations
python benchmark_mcu/scripts/b3_deploy.py     # 22 builds, needs the board
python benchmark_mcu/scripts/b4_assets.py     # tables and figures
```

`ECG_FS=100` must be set; `b3_deploy.py` needs an STM32F411 attached and
the ST toolchain installed at the paths declared at the top of that file.

## Main figures

| | file | answers |
|---|---|---|
| 1 | `figures/figure1_design` | the control structure: one cohort, leave-one-source-out, ten families scaled to two budgets, and what each RQ consumes. Counts are read from the artefacts, not typed |
| 2 | `figures/figure2_cross_source` | **RQ1** external AUPRC for every family on every held-out source, both budgets |
| 3 | `figures/figure3_distribution` | **RQ1** spread across sources per family, with bootstrap intervals -- the overlap is the point |
| 4 | `figures/figure4_int8_cost` | **RQ1** what quantisation costs, against the gap between families |
| 5 | `figures/figure5_accuracy_latency` | **RQ2** external AUPRC against measured latency, bubble area = measured peak SRAM |
| S1 | `figures/figureS1_macs_vs_latency` | **RQ2** MACs against measured latency: the proxy, tested |

## Main tables

| | file | answers |
|---|---|---|
| 1 | `tables/table1_protocol` | the cohort and the LOSO rotations |
| 2 | `tables/table2_architectures` | the ten families: year, defining block, width, size, MACs |
| 3 | `tables/table3_external` | external macro-AUPRC and AUROC per source, with bootstrap CIs |
| 4 | `tables/table4_deployment` | **measured** Flash, peak SRAM, latency, p95, cycles, ms/MMACC, deployed yes/no |
