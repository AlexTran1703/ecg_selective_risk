# Protocol: MCU-first cross-source 12-lead ECG architecture benchmark

**Version:** 2.0 (10 October 2026)  
**Status:** prospective analysis and reporting specification; no results are claimed here.  
**Detailed manuscript display specification:** [`figures_tables_reporting.md`](figures_tables_reporting.md) (exact table templates, panel layouts, captions, data schemas, QC).  
**Target hardware:** STM32F411VET6, Cortex-M4F, 512 KiB Flash, 128 KiB SRAM (confirm usable memories, clock, board, and reserved firmware regions on the actual hardware).  
**Purpose:** Compare *externally validated* multi-label 12-lead ECG accuracy and *measured MCU feasibility* of lightweight time-series architectures, including architectures adapted from mobile vision and sensing. **Do not imply the original papers validated the 1D ECG adaptations.**

## 1. Research questions and endpoints

- **RQ1 — External generalization:** Which fixed architecture achieves the strongest cross-source macro-AUPRC and macro-AUROC under identical leave-one-source-out (LOSO) training?
- **RQ2 — Quantization:** How does full-integer INT8 export affect predictions on each unseen source?
- **RQ3 — Deployment:** Which models execute correctly on a physical STM32F411VET6 within Flash and peak SRAM budgets, and at what median/p95 latency and cycle count?
- **RQ4 — Performance–resource trade-off:** Which *feasible* models lie on the external-performance–latency–SRAM Pareto frontier?

**Primary predictive endpoint:** unweighted average of *source-wise* macro-AUPRC, calculated on the same preregistered common class set. **Co-primary hardware condition:** correct full INT8 inference on the F411 within available device memory. Secondary: macro-AUROC, worst-source macro-AUPRC, per-class AUPRC/AUROC, quantization delta, median/p95 latency, cycles, Flash, SRAM. AUPRC is prevalence-sensitive; show prevalence and never interpret between-source differences as pure model effects.

## 2. Prespecified model roster

### 2.1 Eight fixed families (main benchmark)

| ID | Architecture (1D ECG adaptation) | Key original source | Defining operator/block; preserve in adaptation | Track |
|---|---|---|---|---|
| M1 | FCN-1D | Wang, Yan & Oates (2017), *Time Series Classification from Scratch with Deep Neural Networks* | Plain convolution–normalization–activation blocks; global pooling | Fixed |
| M2 | ResNet-1D | He et al. (2016), *Deep Residual Learning*; Ribeiro et al. (2020), ECG application | Residual Conv1D blocks | Fixed |
| M3 | TCN | Bai, Kolter & Koltun (2018), *Empirical Evaluation of Generic Convolutional and Recurrent Networks* | Causal/dilated residual convolutions (state precise padding convention) | Fixed |
| M4 | MobileNetV2-1D | Sandler et al. (2018), *MobileNetV2* | Inverted residual, linear bottleneck, depthwise Conv1D | Fixed |
| M5 | ShuffleNetV2-1D | Ma et al. (2018), *ShuffleNet V2* | Channel split, shuffle, pointwise/depthwise convolution | Fixed |
| M6 | GhostNet-1D | Han et al. (2020), *GhostNet* | Ghost modules using cheap operations | Fixed |
| M7 | FasterNet-1D | Chen et al. (2023), *Run, Don't Walk* | Partial channel convolution, feature mixing | Fixed |
| M8 | MobileNetV4-Conv-1D | Qin et al. (2024), *MobileNetV4* | Universal inverted bottleneck; **convolution-only** (no Mobile MQA) | Fixed |

**Not in primary roster:** TinyHAR, DeepConvLSTM, MLP-HAR, PatchTST, Mamba, large Transformers, ECG foundation models, RepViT. RepViT is an optional *supplementary* sensitivity model only, after implementation/operator support audit.

**Citation rule:** Give original paper DOI/venue and source-code commit in Supplementary Table S1. No paper is evidence of STM32F411 performance until independently deployed here. For modified 1D blocks, provide code and diagrams.

### 2.2 Separate hardware-aware design comparator

- **H1 — MCUNet-style co-design (2020):** Include only if running an actual documented architecture/engine co-design method; simply using a small MobileNet is **not** MCUNet. Label this comparator **MCUNet-inspired search/co-design** unless the original method is faithfully reproduced.
- **H2 — MicroNAS (2025):** Run a reproducible memory/latency-aware architecture search if available; report exact search space, constraint predictor, device measurements, search GPU-hours, trials/seeds and final architecture. Not directly paired with fixed families in an architecture-only inference claim.

**Core study remains complete with M1–M8.** H1/H2 may be reported separately, never allowed to alter the primary roster retrospectively.

### 2.3 Fixed size tiers and architecture freeze

Use two target tiers per fixed family, calibrated using development-only information:

| Tier | Maximum stored INT8 model weights | Other requirements |
|---|---:|---|
| Compact (C) | 32 KiB | Full compiled firmware and runtime peak SRAM must fit hardware |
| Standard (S) | 128 KiB | Full compiled firmware and runtime peak SRAM must fit hardware |

Both are **caps**, not equal parameter counts or guaranteed deployment. Freeze for each family: stage count, width multipliers, stem stride, kernels, normalization, skip paths, head, chosen quantization operator mapping, and preprocessing **before viewing external results**. Include exact configurations in `configs/architectures/*.yaml`. Use activation RAM checks on real firmware or valid allocation trace; do not silently resize a model after inspecting target-source metrics. If a tier cannot fit despite documented size-search on *development data and hardware only*, report infeasible.

Primary run matrix: **8 families × 2 tiers × 4 LOSO folds × 3 seeds = 192 fitted models**. One exported checkpoint per run, with separate deployment/quantization status.

## 3. Data, harmonization and leakage prevention

- **Data sources:** PTB-XL, Chapman–Shaoxing, Georgia, Ningbo. Version and actual count of included/excluded records to be measured, not copied from publication headline counts.
- **Signal:** 12 leads, common lead ordering, 10-second crop/padding policy, 100 Hz after documented anti-aliased resampling; shape `[12,1000]`. Record baseline correction/filtering and whether MCU inference includes any of these operations.
- **Targets:** up to 13 harmonized diagnoses. Publish per-source SNOMED/SCP/raw-code mapping, positive counts, negative counts, and *unknown* masks. Do not mark unknown/unobserved diagnoses as negatives.
- **Primary common classes:** freeze the subset valid and estimable across all four sources *before model results*. Report the inclusion rule explicitly, e.g. complete label observability and adequate positive/negative counts. Secondary analysis may use all evaluable classes with denominators per source/class.
- **Splits:** four LOSO folds. In each fold, the held-out source is external test **only**. Within the remaining three sources, form patient-grouped train/development validation using fixed IDs; no target-source examples for early stopping, normalizer fitting, tuning, calibration, quantizer representative data, architecture scaling, or threshold selection.
- **Seeds:** 0, 1, 2. Use an identical split per fold/seed across models. Save `patient_id`, `record_id`, `source`, class mask and prediction array in every export.
- **Preprocessing:** any learnt statistics come solely from development train; robust fixed physical amplitude scaling permitted if precisely specified.

## 4. Training, quantization and MCU measurement

### 4.1 Shared training specification

Use multi-label masked BCEWithLogits; optimizer AdamW (initial LR 1e-3, weight decay 1e-4); cosine schedule; maximum 50 epochs; patience 8 on development source-balanced macro-AUPRC; effective batch size 64; identical permitted augmentation; same input and output semantics. State whether positive weighting is used; if yes, compute weights only from development training and cap with a prespecified bound. Report epoch, best checkpoint and seed. Do not tune held-out test thresholds. For thresholded clinical metrics, use thresholds selected on development validation and describe separately.

### 4.2 Full-integer quantization

Use a clearly named deployment stack/version (e.g. TFLite Micro/LiteRT Micro with CMSIS-NN); full-int8 weights **and activations** where supported. Calibrate PTQ on a deterministic class-aware or random sample from development **training** only. Verify operator list, quantization scales and zero points, arena allocation, and input/output parity. If PTQ unsupported, show exact failure; optional QAT is secondary and labelled separately, never silently substituted for PTQ.

**Output parity:** compare device raw/logit/probability outputs with the same fully quantized host graph on a fixed source-independent ECG test-vector suite, reporting max absolute difference, mean absolute difference and disagreement under frozen thresholds. A deployment counts as **successful** only after executing the network with numerically acceptable parity under a prespecified tolerance.

### 4.3 Physical-board measurements

Fix clock (nominal maximum 100 MHz if safely configured), board ID, power source, compiler flags, HAL, TFLM/CMSIS-NN versions and debug instrumentation. Record: total linked or flashed firmware bytes; weight bytes; peak tensor arena + stack + other SRAM occupancy **without double counting**; median and p95 inference times across ≥1000 warmed inferences with interrupts policy specified; DWT cycle counter results (handle overflow); optional *measured* energy using voltage/current integration. Log preprocessing or data-transfer time separately; label MCU latency as inference-only unless acquisition and preprocessing also run there. For all **12 LOSO×seed checkpoints per configuration**, report feasibility rates and range/median hardware metrics; fixed model structure alone does not guarantee identical linked weight/runtime metadata.

**Deployment status codes:** `PASS`, `CONVERT_FAIL`, `UNSUPPORTED_OP`, `FLASH_OOM`, `SRAM_OOM`, `RUNTIME_FAIL`, `PARITY_FAIL`. Missing timing must be `NA`, never zero. Record the phase and diagnostic error.

## 5. Publication structure: where every table and figure belongs

### 5.1 Main manuscript layout

1. **Introduction:** limits of within-source accuracy and the importance of physically measured MCU deployment; hypotheses RQ1–RQ4.
2. **Methods**
   - 2.1 Cohorts, harmonized labels and exclusions → **Table 1**
   - 2.2 LOSO design and leakage protection → **Figure 1**
   - 2.3 Model definitions and size tiers → **Table 2**
   - 2.4 Training, inference, INT8 conversion and MCU methods
   - 2.5 Endpoints, inference and multiplicity strategy
3. **Results**
   - 3.1 Dataset coverage and audit → **Table 1** (refer back; supplementary S2)
   - 3.2 Cross-source discrimination → **Figure 2 + Table 3**
   - 3.3 Source robustness and model ranking → **Figure 3**
   - 3.4 FP32 versus INT8 predictive changes → **Figure 4 + Table 5**
   - 3.5 Device feasibility and runtime efficiency → **Table 4 + Figure 6**
   - 3.6 Externally validated performance per device resource → **Figure 5**
   - 3.7 Search-based H1/H2 comparator (if completed) → **Supplement S5**, short text summary
4. **Discussion:** interpretation by architecture/operators, domain shift, MCU operator support, limitations (source count, code/label mapping, adaptation vs reproduction), reproducibility.

### 5.2 Table specifications (main text)

#### Table 1 — Dataset selection and label observability

**One row per source**, not a mixed roster of papers. Columns: `Source | Version | Original ECG n | Excluded n (reason in footnote) | Included ECG n | Unique patients n / unknown | Recording length | Sampling rate (original→100 Hz) | Common classes evaluable (k/K) | Positive labels per class (reference S2)`. Under table: `n` = ECG recordings; multi-label totals can exceed ECG count. All full 13-class counts and absent/unverified label status go in **S2**.

#### Table 2 — Frozen model and operator manifest

**One row per model×tier (16 rows)**. Columns: `ID | Architecture + original citation | Tier | 1D block summary | Stages / channels | Parameters | MACs @12×1000 | INT8 weight KiB | Peak activation estimate KiB (predeployment) | Operator support`. All measured/calculated using the *same counting tool*. Footnote source vs adapted architecture and code commit. Hardware feasibility is **not** concluded from this table.

#### Table 3 — LOSO external discrimination (primary results)

**One row per model×tier (16 rows)**, grouped by family. Columns: `ID | Tier | PTB-XL macro-AUPRC | Chapman macro-AUPRC | Georgia macro-AUPRC | Ningbo macro-AUPRC | Equal-source mean macro-AUPRC | Equal-source mean macro-AUROC | Worst-source macro-AUPRC`. Each source cell: **three-seed mean** with `95% CI` in compact footnote/supplement (do not imply CI over three seeds alone). Show all precision to 3 decimals; bold best *eligible* model only per prespecified primary score. Add independent row(s) for H1/H2 in a **visually separated** block only if search costs/conditioning are detailed elsewhere. CIs should be derived from patient-clustered paired bootstraps of held-out predictions, accounting for seeds as repeated fits and using the same bootstrap resamples when comparing models. Include per-class breakdown S3.

**Printed miniature template:**

| Model | Tier | PTB-XL | Chapman | Georgia | Ningbo | Mean AUPRC | Mean AUROC | Worst AUPRC |
|---|---|---|---|---|---|---|---|---|
| M4 MobileNetV2 | C | — | — | — | — | — | — | — |
| M4 MobileNetV2 | S | — | — | — | — | — | — | — |
| M8 MobileNetV4-Conv | C | — | — | — | — | — | — | — |
| M8 MobileNetV4-Conv | S | — | — | — | — | — | — | — |

`—` means not yet measured in this protocol, **not** a zero value.

#### Table 4 — Actual STM32F411VET6 deployment

**One row per model×tier (16 rows)** in main text, summarizing 12 checkpoints; per-checkpoint logs in S4. Columns: `ID | Tier | PASS/attempts (of 12) | Median linked Flash KiB [min,max] | Peak SRAM KiB [min,max] | Median latency ms [IQR] | P95 latency ms | Median cycles | Status / first failure reason`. If 0/12 pass, Flash/latency/success-only fields show `NA`; where a compile-time memory estimate exists it can appear explicitly as **estimate** (not measurement). Report *actual reserved capacity*, not a free-standing 512/128 annotation without runtime/stack caveat. For hardware metrics, only PASS runs contribute to numerical summaries; also report failures.

#### Table 5 — Predictive quantization loss and parity

**One row per model×tier**, separate by source in S3/S4. Columns: `ID | Tier | FP32 equal-source macro-AUPRC | INT8 equal-source macro-AUPRC | Δ AUPRC (INT8−FP32; 95% paired CI) | Δ AUROC (95% CI) | Host↔MCU maximum absolute output difference | Fully quantized exports/attempts`. Crucially: present **INT8 external scores even for models that cannot run on MCU** when the host quantized graph exists; label them `host INT8` rather than on-device accuracy. MCU parity is NA when not executable.

### 5.3 Figure specifications (main text)

**Figure 1 — Study design / no-leakage workflow (Methods; vector SVG/PDF).**
- Panel A: four cohorts, exclusion and common-label audit, input 12×1000.
- Panel B: four LOSO runs shown as a compact 4-row train/validation/held-out schematic, *not* all 192 runs as icons.
- Panel C: FP32 training → development-only PTQ → deployment attempt → MCU measurement / status.
- Annotate `8×2×4×3 = 192` fixed-run total, hardware pass criterion, and **held-out source never used for adaptation**.

**Figure 2 — Cross-source heatmaps (Results 3.2; primary performance figure).**
- Panel A macro-AUPRC; panel B macro-AUROC.
- Y = same 16 model×tier rows in Table 3, paired tiers adjacent; X = PTB-XL, Chapman, Georgia, Ningbo, **same column order always**.
- Each cell = average metric over three fitted seeds on that external source; annotate to `0.000`; missing or undefined = hatched `NA`.
- Continuous colorbars separately labelled and consistently scaled *within each metric across all models*; do not use color to encode rank while annotating raw score.
- In caption: exact common class set, unweighted macro and 3-seed averaging rule. Main text call out best and worst sources without post-hoc claims of significance.

**Figure 3 — Robustness/ranking across external sources (Results 3.3).**
- One horizontal dot-and-line plot, Y = 16 configurations ordered by prespecified mean AUPRC (rank can be displayed but state ranking is descriptive).
- X = external macro-AUPRC; **four small source-colored dots per row**, large neutral diamond = equal-source mean. Connect source minimum↔maximum with thin grey line labelled `source range`, *not CI*.
- Optional right narrow column = worst-source AUPRC; if included, do not duplicate whole Table 3. **Avoid calling four sources a probability distribution or adding fake cross-hospital CI.**

**Figure 4 — Quantization effect (Results 3.4).**
- X = `Δ macro-AUPRC = host INT8 − FP32`; Y = 16 model×tier rows.
- Point = mean paired delta over sources/seeds; whisker = patient-clustered bootstrap 95% CI conditional on these four sources; grey vertical zero line; negative is worse.
- Color by architecture family; shape by tier. Indicate unsupported full-int8 export as `NA` with reason in Table 5 / S4. Optional supplementary panel per source.

**Figure 5 — Externally validated MCU Pareto frontier (Results 3.6; synthesis figure).**
- X = physical STM32 **median inference latency (ms, lower better)**; Y = equal-source external **INT8** macro-AUPRC (higher better), not FP32.
- Bubble **area** proportional to peak SRAM KiB (caption specifies exact scale); color = model family; shape = tier. Label nondominated designs directly using ID and tier.
- **Eligibility:** a configuration must have a successful correct deployment for all twelve LOSO×seed checkpoints (strict primary panel). Supplement may show partially feasible models, explicitly flagged. Use a single preregistered aggregation rule (e.g. latency median over 12 successful checkpoints).
- Draw Pareto frontier only over eligible points using 2-dimensional dominance on (AUPRC↑, latency↓); SRAM is a displayed constraint. Clearly distinguish 3D Pareto if instead computed over three variables; don't say 3D Pareto when frontier ignores SRAM. No invented values or drawn frontier before results.

**Figure 6 — Resource utilization and feasibility (Results 3.5).**
- Panel A: horizontal bars, total linked Flash KiB per configuration; limit = measured usable Flash, not just silicon headline capacity.
- Panel B: peak SRAM KiB, including arena/stack/other occupancy without double-counting; limit = measured usable SRAM.
- Panel C: median inference latency with p95 marker/range, **only PASS** models; failure rows carry a status glyph and `NA`, not zero-height bars.
- Same model order as Fig. 2/Table 4; units on all axes; tiers visually paired; captions enumerate failure counts and exact board/compiler/runtime configuration.

### 5.4 Supplementary items (required for transparency)

- **S1** model block diagrams, original citations, exact code commits, architecture YAMLs, MAC-count procedure, op compatibility matrix.
- **S2** complete label mapping, unknown masks, per-class positives/negatives, class prevalence and missingness across datasets.
- **S3** model×tier×seed×source×class AUROC/AUPRC, 95% CIs, thresholded metrics (threshold fitted on development only), FP32 vs host INT8.
- **S4** all 192 per-checkpoint conversion/firmware executions, failure messages, Flash, SRAM, runtime, parity and firmware git/hash.
- **S5** H1/H2 search space, resource estimates, measured feasibility, compute/search budget; replication protocol. Only if conducted.
- **S6** sensitivity analyses: source-balanced vs pooled development loss, alternative sample rate if undertaken; must be labelled exploratory unless preregistered.

## 6. Reproducibility: required machine-readable files

Keep a single long-form results table and derive every graphic from it; **never hand-transcribe figure numbers**.

- `manifest/records.csv`: `record_id,patient_id,source,signal_path,quality_flag,split_fold,labels,unknown_mask`.
- `results/predictions.parquet`: `model_id,tier,fold,seed,record_id,class_id,y_true,label_observed,p_fp32,p_int8_host` (one row per class/test record).
- `results/models.csv`: `model_id,tier,fold,seed,params,macs,weights_kib,checkpoint_hash,model_config_hash,conversion_status`.
- `results/mcu_runs.csv`: `model_id,tier,fold,seed,firmware_hash,device_id,clock_mhz,toolchain,flash_kib,peak_sram_kib,latency_ms_median,latency_ms_p95,cycles_median,parity_max_abs,status,failure_reason`.
- `results/source_metrics.csv`: `model_id,tier,fold,seed,source,quantization,metric,class_set,estimate,ci_lo,ci_hi,n_pos,n_neg`.
- `results/figures_manifest.json`: plot script version, input hashes, paper table/figure output filenames.

All reporting scripts must apply one class-set inclusion rule, one patient-resampling scheme and one quantization naming convention. Include figure-generation scripts and a `README` explaining how to reproduce Tables 1–5 and Figures 1–6.

## 7. Statistical inference and selection

1. Source-wise per-class AUPRC/AUROC; macro across *fixed common evaluable classes*; `NA` for undefined class metrics, with counts. Average the four source macro metrics with equal source weights; average the three seeds in each source first.
2. For **within-source** confidence intervals, resample patients with replacement as clusters (or records if patient IDs cannot be recovered, explicitly indicating this limitation), preserving model pairing and recomputing both compared models on identical draws. Use 2,000 bootstrap draws with fixed RNG seed and report percentile CIs; summarize CIs across four *fixed* sources conditional on the source set. Do not infer to unseen hospitals from patient-level bootstrap alone.
3. Report paired differences for models when making superiority claims, with CIs and a stated multiple-comparison strategy or characterize comparisons as descriptive/exploratory. Do not declare a winner based on overlapping individual model CIs.
4. **Primary winner definition:** among models with `PASS` on all 12 exported checkpoints, maximize equal-source **INT8 macro-AUPRC**; if results essentially tied under a prespecified tolerance, choose lower median MCU latency, then lower peak SRAM. Also report the external-performance/latency frontier so the paper does not hinge on one model.
5. Pre-register the handling of a model with inconsistent deployability across folds/seeds. Strict primary Fig. 5 excludes it; Table 4 and S4 retain all attempts. Separately show a sensitivity analysis for partial success if useful.

## 8. Execution sequence and QC gates

1. Freeze dataset versions, patient IDs, common label set and four LOSO splits; produce Table 1 draft and S2.
2. Implement M1–M8, two tiers; archive YAMLs and operator lists; produce Table 2 draft.
3. Compile **one representative model per family** on F411 to audit operator compatibility and memory before launching all 192 fits. Failures remain in the roster.
4. Run all 192 FP32 fits, export external predictions, and verify record/class masks; calculate Fig. 2/3 and Table 3.
5. Perform source-safe PTQ, validate parity, evaluate host INT8; calculate Fig. 4 and Table 5.
6. Flash all eligible exports, measure physical device resources for each; calculate Table 4 and Fig. 6; log every failure.
7. Construct only the **measured, feasible** Pareto Fig. 5 and check statistical consistency between figures/tables.
8. Run H1/H2 *separately* if time/resources allow, with preregistered search budgets, then add S5.

## 9. Results-writing template (do not fill before data exist)

- **3.2:** “Across four held-out sources, [configuration] attained the highest equal-source macro-AUPRC of [value], compared with [comparator] ([paired delta and CI]). Performance ranged from [min] in [source] to [max] in [source] (Fig. 2; Table 3).”
- **3.4:** “INT8 conversion changed macro-AUPRC by [delta, 95% CI] for [model], with [number] full-integer conversion failures (Fig. 4; Table 5).”
- **3.5:** “[x]/16 configurations deployed successfully across all 12 checkpoints. The leading causes of failure were [reasons]. Among fully feasible configurations, median inference latency ranged from [x] to [y] ms (Table 4; Fig. 6).”
- **3.6:** “[models] occupied the measured external-AUPRC/latency nondominated frontier (Fig. 5). No claim about clinical readiness is made solely from offline ECG classification or inference latency.”

**Interpretation safeguard:** This is an *architecture benchmarking study using ECG as the clinical task*, not evidence that original mobile-vision networks are clinically validated, and not proof of prospective clinical suitability.
