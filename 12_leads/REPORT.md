# Cross-source robustness and resource-constrained deployment of
lightweight 12-lead ECG classifiers

**Four clinical sources, leave-one-source-out evaluation, and measured
STM32F411 deployment at 100 Hz.**

> Can compact ECG classifiers stay reliable under real-world source and
> signal-quality shift, and does that reliability survive deployment on
> a microcontroller?

Everything here answers that one question in three stages. External
generalisation is the spine: the quality analysis explains failures in
it, selective prediction asks whether those failures can be recognised,
and deployment asks whether the externally validated behaviour survives
implementation. Every downstream analysis uses the same leave-one-
source-out predictions.

| | question | evidence |
|---|---|---|
| **RQ1** | How well do compact ECG encoders generalise to previously unseen clinical sources? | Figure 2, Table III |
| **RQ2** | How are source-specific signal characteristics and record-level impairments associated with external performance, and can selective prediction identify unreliable cases? | Figure 3, Table III, Figure 4 |
| **RQ3** | Which externally evaluated models retain performance after int8 deployment, and what accuracy-resource operating points do they offer? | Figure 5 |

```
Generalises?  ->  Can we trust it?  ->  Can we deploy it?
```

Four public sources, 77,333 records. Six encoder families trained
leave-one-source-out at 100 Hz. **Every deployment number was measured
on the part**: each quantised model was compiled into a bare-metal
firmware, flashed to an STM32F411 over SWD, and timed with the DWT
cycle counter at 100 MHz.

## What is, and is not, claimed as new

Cross-dataset ECG evaluation is not new, and a benchmark of six familiar
1D CNNs is not a contribution by itself. Neither is claimed here.

The contribution is the chain, on one cohort and one frozen input
specification: compact encoders transfer unevenly across clinical
sources (RQ1); specific measured impairments track that unevenness, and
confidence degrades alongside discrimination rather than compensating
for it (RQ2); and the same externally evaluated models then run on a
Cortex-M4 with their accuracy essentially intact (RQ3).

The deployment constraint is fixed first and carried unrelaxed through
every analysis, so the accuracy numbers belong to models that
demonstrably run on the part -- not to a model zoo deployed as an
afterthought.

```
12_leads/
  REPORT.md
  figures/  figures/supplementary/     versioned
  tables/   tables/supplementary/      versioned
  firmware/                            bare-metal timing firmware
  scripts/  src/ecgmcu/                versioned
  data/  results/  archive_250hz/      regenerable, not versioned
```

### Reproducing

All model work is at `ECG_FS=100`; the quality audit deliberately runs on the
native 500 Hz signal and ignores that variable. Each stage caches its
output, so a later stage never re-runs an earlier one.

```
s1_signal_quality.py      RQ2a   quality audit, native 500 Hz
s2_train.py               RQ1    24 LOSO runs
s2d_repredict_fp32.py     RQ1    re-score checkpoints in true float32
s9_reliability.py         RQ2b   excess-risk curves, E-AURC
s6_quality_robustness.py  RQ2b   prevalence-matched dAUPRC
s10_quality_reliability.py RQ2c  quality-stratified E-AURC
s3_analyze.py             RQ3    float32 envelope
s3b_int8.py               RQ3    int8 PTQ + footprint
s8_latency.py             RQ3    build, flash, time on the F411
s7_manuscript.py          assembles figures 1-5 and tables I-V
```

`s8_latency.py` needs the board attached and the STM32CubeIDE toolchain;
every other stage runs from the cached cohort alone.

---

## Stage 0 - the input specification, fixed before anything is trained

A 10 s 12-lead record at 100 Hz is 12,000 samples: **11.7 KB as int8**,
against 46.9 KB at float32 and 117 KB at 250 Hz float32. The input buffer
alone decides feasibility before any architecture does, so the
specification -- **10 s, 12 leads, 100 Hz, int8** -- is settled first and
then frozen.

**Quality is measured before decimation, at 500 Hz.** A 100 Hz signal has
a 50 Hz Nyquist limit, so mains interference and high-frequency content
are not observable in the model input at all; measuring them there would
be meaningless. The audit therefore runs on the native signal and the models
run on an anti-aliased (FIR, zero-phase) resampling of it. Figure 1 shows
the two branches explicitly.

---

## RQ1 - external-source generalisation

66,379 records after dropping those with no label in the harmonised
13-class space. Everything except the encoder is held constant.
Validation is always drawn from the training sources, never the held-out
one. 24 runs. **Figure 2**.

### Discrimination across held-out sources

Macro AUPRC by held-out source, **float32** -- the primary analysis. E-AURC sits alongside it in Table III, and what int8 costs is kept separate so deployment does not contaminate the modelling result.

| model | PTB-XL | Georgia | Chapman | Ningbo | **macro** |
|---|---|---|---|---|---|
| TinyCNN | 0.425 | 0.499 | 0.527 | 0.558 | 0.502 |
| DS-CNN | 0.430 | 0.524 | 0.547 | 0.576 | 0.520 |
| MobileNet1D | 0.399 | 0.511 | 0.551 | 0.574 | 0.509 |
| **MBConv1D** | 0.434 | 0.537 | 0.580 | 0.609 | **0.540** |
| TCN-Lite | 0.425 | 0.524 | 0.569 | 0.589 | 0.526 |
| **ResNet1D-Lite** | 0.435 | 0.544 | 0.599 | 0.618 | **0.549** |

> **Finding 2. PTB-XL is the hardest held-out source for every encoder**
> (0.396-0.432 against 0.557-0.614 for Ningbo), even though RQ2a shows
> it is the *cleanest* source on every indicator. The cross-source gap is
> therefore not signal quality. It must be acquisition, population or
> labelling -- which is what makes the quality audit explanatory rather
> than decorative, and also what this design cannot further separate.

> **Finding 3. 100 Hz costs nothing, and helps the smallest models.** The
> rate was chosen for the memory budget, not for accuracy, so a penalty
> was expected. There is none: against the same pipeline at 250 Hz,
> TinyCNN gains **+0.056** macro AUPRC and DS-CNN **+0.037**, while the
> largest encoders are unchanged within noise. Anti-aliased decimation
> removes content these models were not exploiting, and shortens the
> sequence a small encoder has to summarise. The 250 Hz runs supporting
> this comparison are in `archive_250hz/`.

## RQ2 - reliability under source and quality shift

RQ1 shows that external performance depends on which source is held
out. This asks why, and whether the model can tell. Three steps: what
differs between the sources, whether those differences track the
performance gap, and whether confidence recognises the records it gets
wrong. **Figure 3**, **Table III**, **Figure 4**.

### RQ2a - what shifts between sources

All 77,333 records, six indicators reported separately. No composite
index, which would bury arbitrary weights under a scientific-looking
decimal. **Figure 2**, one panel per indicator.

> **Finding 1. The four sources differ on one axis, not six.** The three
> spectral indicators -- HF noise, baseline wander, mains -- are nearly
> superimposed across sources. The separation is structural: a flat or
> clipped lead occurs in **4.24% of Ningbo records against 0.005% of
> PTB-XL**, a factor of 920. QRS-detectability failure (0.86-1.91%) and
> implausible RR intervals differ far less. A "noise robustness" framing
> would have looked in the wrong place entirely.

---

### RQ2b - does the shift matter diagnostically?

For each indicator, strata are
formed **within each held-out source** -- a threshold applied to the
pooled cohort would sort records by source and measure source identity
instead of quality. Each replicate then draws from the impaired and the
clean stratum so that both match a common distribution of labels per
record, because AUPRC moves with prevalence; the difference is taken
inside the replicate. **Table III**. Negative means the
impairment costs accuracy.

Pooled across the six encoders with 95% record-level bootstrap intervals, 200 replicates. The pooled effect is formed **inside** each replicate, on the same resampled records for every architecture: the six encoders scored the same ECGs, so averaging six separately drawn intervals would treat them as independent observations and understate the uncertainty.

| impairment | dAUPRC | direction |
|---|---|---|
| **flat/clipped lead** | **-0.154 (-0.190, -0.116)** | lower performance |
| **RR implausibility** | **-0.080 (-0.114, -0.037)** | lower performance |
| **QRS failure** | **-0.047 (-0.089, -0.003)** | lower performance |
| mains | +0.014 (+0.002, +0.030) | higher performance |
| baseline wander | +0.022 (+0.012, +0.034) | higher performance |
| HF noise | +0.022 (+0.005, +0.036) | higher performance |

> **Finding 4. Structural signal failure is associated with lower performance; spectral variation is not.**
> After prevalence matching, the three *spectral* indicators -- HF noise,
> baseline wander, mains -- show slightly **higher** AUPRC in the more
> impaired stratum, and the three *structural* indicators show clearly
> lower AUPRC. These are associations, not causal effects: the strata
> differ in whatever else travels with the indicator, and only label
> count is matched, so the positive columns should not be read as noise
> improving anything. The defensible claim is the contrast between the
> two families, and that the largest negative association is on exactly
> the indicator where the sources differ by a factor of 920.

> **Finding 5. Vulnerability does not follow accuracy.** On flat or
> clipped leads TinyCNN is the *most* damaged (-0.180) despite being the
> least accurate, and ResNet1D-Lite the least damaged (-0.140) despite
> being the most accurate. The intuition that capacity is paid for with
> fragility does not hold here. Supplementary Table S4 reports the same
> contrast relative to each encoder's own baseline, which matters because
> a weaker model has less accuracy available to lose.

---


### RQ2c - does confidence degrade with the signal?

RQ2b shows impaired records are harder. Figure 4(b) shows confidence
ranks errors to some degree. Neither says whether the two interact, and
the interaction is what a referral deployment depends on:

    dE-AURC = E-AURC(impaired) - E-AURC(clean)

Positive means degradation produces errors the model is *also* worse at
recognising. Restricted to the three structural indicators, since those
are the ones RQ2b found consistently associated with lower
discrimination. **Figure 4(c)**, 1000 replicates, seed 0:

| indicator | dE-AURC (95% CI) | sources |
|---|---|---|
| flat/clipped lead | **+0.038** (+0.028, +0.049) | 1 |
| RR implausibility | **+0.034** (+0.024, +0.044) | 3 |
| QRS failure | **+0.031** (+0.021, +0.042) | 3 |

> **Finding. Degradation is compounding: discrimination falls and the
> confidence ordering falls with it.** All three structural indicators
> show higher E-AURC in the impaired stratum, with intervals clear of
> zero. The practical consequence is the unwelcome one -- abstaining on
> low-confidence records is *least* dependable exactly on the records
> where discrimination has already dropped, so selective prediction
> cannot be relied on to absorb poor signal quality. A deployment that
> wants to defer on bad recordings should gate on the measured quality
> indicator directly, not on model confidence.

---
## RQ3 - deployment-preserved performance

Target confirmed over SWD: **STM32F411xC/E, Cortex-M4, 128 KB SRAM,
512 KB Flash**, ST-Link V2. Toolchain: ST Edge AI Core v2.2.0, GNU Arm
13.3.rel1 bundled with STM32CubeIDE 1.18.1, STM32CubeProgrammer 2.20.0.

### RQ3a - does quantisation preserve the benchmark conclusions?

ST Edge AI imports quantised models; it does not quantise. The int8 graphs
are built with ONNX Runtime static PTQ in QDQ form, calibrated on 256
records drawn **only from each fold's training sources**. Both precisions
are scored on the same records.

### RQ3b - what does it cost on the part?

Each model was compiled into a bare-metal firmware (`firmware/`), flashed
over SWD, and timed with the DWT cycle counter at 100 MHz: 32 runs after
one warm-up, median reported, results read back from a fixed SRAM address.
RTF is latency over the 10 s acquisition window.

**Supplementary Table S7 carries the full numbers; every row ran on the board.**

| model | Flash KB | left | RAM KB | MACC M | **latency ms** | RTF | int8 AUPRC | dAUPRC |
|---|---|---|---|---|---|---|---|---|
| TinyCNN | 57.8 | 454.2 | 27.3 | 4.28 | **222.4** | 0.022 | 0.4999 | -0.0025 |
| DS-CNN | 44.8 | 467.2 | 28.6 | 3.33 | **282.9** | 0.028 | 0.5164 | -0.0032 |
| MobileNet1D | 48.2 | 463.8 | 29.9 | 2.67 | **222.3** | 0.022 | 0.5076 | -0.0013 |
| MBConv1D | 133.5 | 378.5 | 45.7 | 9.99 | **640.1** | 0.064 | 0.5371 | -0.0029 |
| TCN-Lite | 179.8 | 332.2 | 53.5 | 10.25 | **1067.6** | 0.107 | 0.5249 | -0.0012 |
| ResNet1D-Lite | 508.8 | **3.2** | 52.4 | 54.47 | **4247.1** | 0.425 | 0.5467 | -0.0023 |

Flash and RAM are totals including the generated ST Edge AI runtime
(about 26 KB Flash and 4.7 KB RAM), not weights alone.

> **Finding 6. All six run, and all six are faster than real time.** RTF
> spans 0.022 to 0.425, so even the heaviest encoder finishes a 10 s
> record in 4.2 s. On this part latency is not the binding constraint --
> Flash is.

> **Finding 7. int8 is free.** The largest accuracy change across the six
> encoders is -0.0032, one to two orders of magnitude below the gaps
> between architectures. Quantisation is not a trade-off here; it is a
> precondition that costs nothing.

> **Finding 8. MACC mis-ranks latency, and by a factor of two.**
> MobileNet1D has **36% fewer MACC than TinyCNN and identical latency**
> (222.3 against 222.4 ms). Measured cost spans 52 ms per MMACC for
> TinyCNN to 104 ms per MMACC for TCN-Lite. Plain convolutions vectorise;
> depthwise convolutions are memory-bound; dilated convolutions break
> access locality. A benchmark reporting MACC or parameter count as a
> latency proxy would order these encoders wrongly -- which is the whole
> argument for measuring on the part.

> **Finding 9. Physical fit and practical fit are different questions.**
> ResNet1D-Lite occupies 508.8 of 512 KB and did run on the board, so it
> fits physically. Against a reserved budget of 448 KB Flash and 112 KB
> SRAM -- leaving 64 KB and 16 KB for application firmware, drivers,
> acquisition buffers and a bootloader -- it is **Flash-limited**, and
> the other five pass. So int8 brings five of six reference
> architectures inside a realistic F411 envelope. ResNet is also 6.6x
> slower than MBConv1D for **+0.009 AUPRC**, which makes **MBConv1D**
> the practical choice: 98% of the best accuracy, 15% of the latency,
> 26% of the Flash.

---

## Limitations

1. **"Source" is not a single variable.** Holding out a dataset varies
   acquisition hardware, population and labelling protocol at once.
   Finding 2 shows the gap is not signal quality; it cannot say which of
   the remaining three it is.
2. **Prevalence matching is on label count, not label identity.** A
   stratum may still contain a different *mix* of diagnoses rather than a
   different *number*. The residual gap is reported in Supplementary
   Table S4 rather than left for a reviewer to find.
3. **Quality indicators are heuristic**, with stated but not externally
   validated thresholds and no expert-adjudicated quality reference.
4. **One seed per configuration.** Intervals cover sampling variation in
   the evaluation set, not variation in training.
5. **Latency is single-board, single-clock**: 100 MHz from HSI via PLL,
   3 wait states, caches enabled, a fixed synthetic input. No energy is
   reported, because no calibrated current instrumentation was available
   and a bad energy number is worse than none.
6. **Preprocessing is not timed.** The reported latency is inference
   only.
7. **Post-training quantisation only.** QAT was not attempted.
8. **Labels are the dataset authors'**, with the inconsistency that
   implies across four independently annotated sources.

---

## What is measured, and what is not

| | status |
|---|---|
| RQ2a quality audit, 77,333 records at 500 Hz | measured |
| RQ1, 24 LOSO runs at 100 Hz | measured |
| RQ2b, per-diagnosis matched, 1000 replicates | measured |
| RQ2c, quality-stratified E-AURC, 1000 replicates | measured |
| int8 accuracy, 24 folds | measured |
| Flash and RAM including generated runtime | measured, ST Edge AI with ARM gcc |
| **On-device latency, all six encoders** | **measured on the STM32F411** |
| Energy per inference | **not done** - no calibrated instrumentation |
| Preprocessing latency | **not done** - out of scope |
| QAT | **not attempted** |

---

## Main figures

| | file | answers |
|---|---|---|
| 1 | `figures/figure1_design` | the experimental logic, three RQs |
| 2 | `figures/figure2_generalisation` | **RQ1** do the encoders generalise? AUPRC by held-out source, and the mean with bootstrap CI |
| 3 | `figures/figure3_signal_quality` | **RQ2a** how do the sources differ? six indicators, native 500 Hz |
| 4 | `figures/figure4_reliability` | **RQ2c** can confidence identify unreliable predictions? risk-coverage per source, and E-AURC |
| 5 | `figures/figure5_operating_points` | **RQ3** which reliable models are practical on-device? int8 AUPRC against Flash, SRAM and measured latency |

## Main tables

| | file | answers |
|---|---|---|
| I | `tables/table1_datasets` | what data |
| II | `tables/table2_architectures` | what models |
| III | `tables/table3_quality_association` | **RQ2b** do the source differences matter diagnostically? |

Tables I and II define the experimental objects. Table III is the one result
that does not compress into a figure: model x indicator estimates with intervals.
Everything spatial or comparative is a figure, so no result appears twice.

Two tables were deliberately removed rather than demoted. The leave-one-
source-out discrimination table is gone entirely: Figure 2 carries both its
halves, the per-source values in the heatmap and the mean with its interval
in the dot plot, so reprinting 24 numbers underneath would be a second container
for evidence the reader has just seen. The deployment table moved to the
supplement: Figure 5 answers what performance a resource budget buys, which is
the scientific question, and the kilobyte-level detail is reproducibility material.

The one thing the deployment table carried that Figure 5 does not is that int8
costs almost nothing. That belongs in prose: across the six encoders the change
in mean external macro AUPRC spans **-0.0032 to -0.0012**, one to two orders of
magnitude below the gaps between architectures.

## Supplementary

| | file | shows |
|---|---|---|
| S1 | `tables/supplementary/tableS1_per_diagnosis` | AUPRC for each of the 13 diagnoses |
| S2 | `figures/supplementary/figureS2_risk_coverage` | risk-coverage per source with each model's oracle drawn in |
| S3 | `figures/supplementary/figureS3_deployment_detail` | float32 against int8 Flash and SRAM, and the quantisation cost |
| S4 | `figures/supplementary/figureS4_eaurc_by_source` and `tables/supplementary/tableS4_quality_robustness_full` | E-AURC by held-out source; full quality-association output |
| S5 | `figures/supplementary/figureS5_quality_reliability` | **RQ2c extension** whether confidence degrades along with the signal |
| S6 | `tables/supplementary/tableS6_auroc` | macro AUROC by held-out source |
| S7 | `tables/supplementary/tableS7_stm32f411_deployment` | the numerical deployment detail behind Figure 5 |
