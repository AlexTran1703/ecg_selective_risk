# Manuscript-ready figures and tables reporting guide

**Study:** MCU-first, four-source, 12-lead ECG benchmarking on STM32F411VET6  
**Companion:** `protocol.md` (v2.0)  
**Status:** Prespecified display and analysis plan. Every empty cell below must be populated from experiment outputs; no example values are measured results.

## 0. Reporting contract (freeze before analyzing results)

- **Sources:** PTB-XL, Chapman-Shaoxing, Georgia, Ningbo. Four leave-one-source-out (LOSO) experiments; each source is test-only once.
- **Models:** M1 FCN-1D; M2 ResNet-1D; M3 TCN; M4 MobileNetV2-1D; M5 ShuffleNetV2-1D; M6 GhostNet-1D; M7 FasterNet-1D; M8 MobileNetV4-Conv-1D.
- **Configurations:** Compact and Standard tiers × three seeds (0, 1, 2). The complete fixed-model matrix is 8 × 2 × 4 × 3 = **192 trained checkpoints**. Don't silently average unlike tiers.
- **Outputs:** Multilabel sigmoid outputs for the prespecified common eligible class set, with per-source class coverage, prevalence and counts documented. Undocumented or unknown labels are **not** counted as negatives.
- **Primary score:** Macro-AUPRC averaged **equally across the four source-wise macro-AUPRC values**, using the same eligible classes for each comparison. **Secondary:** macro-AUROC; per-class scores; worst-source macro-AUPRC; FP32→INT8 delta; hardware success; Flash, SRAM, cycles, inference latency.
- **Uncertainty:** Paired patient-cluster bootstrap within each held-out source (record-cluster bootstrap only if patient IDs cannot be recovered; disclose this); 2,000 replicates, 95% percentile CI. If reporting a four-source aggregate interval, resample patients within each source in every replicate, then recompute the equal-source mean; **do not claim that this captures uncertainty across future hospitals**. State how variation across 3 training seeds is summarized separately.
- **Tier/model comparison:** Show Standard and Compact as separate rows/marks. Select deployable winners using frozen prespecified rules only. For each held-out fold, final checkpoints and PTQ calibration can use development sources only.
- **No manufactured data:** Use `NR` (not reported), `NE` (not estimable), `CF` (conversion failed), `OP` (unsupported operator), `FM` (Flash overflow), `RM` (SRAM overflow), `VP` (verification/parity failure), or `—` (not measured) consistently with a table footnote; **never use 0 for failed latency**.
- **Units:** Parameters count, MACs for one 12×1000 inference, Flash/SRAM in KiB (1 KiB=1024 bytes), latency in ms, cycles in millions, energy in µJ only if measured; show device clock and compiler settings.

## 1. Exact placement in the manuscript

| Manuscript section | Main displays | Claim supported |
|---|---|---|
| 2.1 Data and label harmonization | **Table 1** | Comparable, well-defined cohorts and labels |
| 2.2 Study design | **Figure 1** | Dataset-held-out evaluation without source leakage |
| 2.3 Architectures | **Table 2** | Traceable model family, size tier, and complexity |
| 3.1 External predictive performance | **Table 3 + Figure 2** | Model × unseen-source discrimination |
| 3.2 Source robustness | **Figure 3** | Robustness, variation, and worst-source outcome |
| 3.3 Quantization | **Table 5 + Figure 4** | Impact of real integer conversion on prediction |
| 3.4 Hardware feasibility | **Table 4 + Figure 6** | Actual Flash/SRAM/latency and failures |
| 3.5 Accuracy–resource frontier | **Figure 5** | Which fully verified INT8 models are nondominated |
| Supplement | **S1–S7 / FS1–FS3** | Full per-class, seed, checkpoint, operator and power detail |

**Recommended narrative:** describe Table 1 and Figure 1 in Methods; then read Table 3 → Figures 2–3 → Table 5/Figure 4 → Table 4/Figure 6 → Figure 5 in Results. The figure numbering remains fixed even when the text references Figure 6 before Figure 5.

## 2. Main tables: exact templates

### Table 1. Cohort composition and label observability

**Layout:** one row per data source plus an optional total row; total must not imply independent patients across institutions without deduplication. Main columns:

| Source | Raw ECGs, n | Eligible ECGs, n (%) | Known patients, n | Used labels, n | AF positives, n (%) | Unmapped/excluded, n (%) |
|---|---:|---:|---:|---:|---:|---:|
| PTB-XL | TBD | TBD | TBD/NR | TBD | TBD | TBD |
| Chapman-Shaoxing | TBD | TBD | TBD/NR | TBD | TBD | TBD |
| Georgia | TBD | TBD | TBD/NR | TBD | TBD | TBD |
| Ningbo | TBD | TBD | TBD/NR | TBD | TBD | TBD |

**Supplement S2:** full *source × each harmonized class* matrix: n positive, n explicit negative, n unknown/unmapped. If 13 harmonized labels are used, give all 13, not just AF. Avoid class counts from overlapping multi-label classes being interpreted as a partition of total ECGs. **Table note:** raw dataset versions, resampling, duration, licensing, label crosswalk, patient identifier availability.

### Table 2. Frozen architectures (before outcome inspection)

**Layout:** 16 rows = 8 families × 2 tiers, in M1–M8 order; group tiers under each family.

| ID | Model | Tier | Source architecture / adaptation | Params (k) | MACs (M) | INT8 weights (KiB) | Peak activation (KiB, host estimate) |
|---|---|---|---|---:|---:|---:|---:|
| M1 | FCN-1D | Compact | Plain Conv1D + GAP | TBD | TBD | TBD | TBD |
| M1 | FCN-1D | Standard | Plain Conv1D + GAP | TBD | TBD | TBD | TBD |
| M2–M8 | Each family | Compact + Standard | See frozen configs | TBD | TBD | TBD | TBD |

**Additional source information:** year, DOI, implementation commit, 1D adaptation, block counts, widths, kernel sizes, expansion ratio, downsampling and supported inference operators go in **S1**. Host activation estimate is *not* actual MCU SRAM and must never replace Table 4. MAC accounting method and whether normalization is folded into convolution belong in footnotes.

### Table 3. Unseen-source discrimination (primary predictive results)

**Layout:** 16 rows (model × tier) and paired four-source columns. Prefer **macro-AUPRC main table**, with AUROC in companion panel or **S3** to avoid a wide 11-column mess.

| Model | Tier | PTB-XL | Chapman | Georgia | Ningbo | Mean across sources | Worst source |
|---|---|---|---|---|---|---|---|
| FCN-1D | Compact | TBD | TBD | TBD | TBD | TBD | TBD |
| FCN-1D | Standard | TBD | TBD | TBD | TBD | TBD | TBD |
| … | … | … | … | … | … | … | … |
| MobileNetV4-Conv-1D | Standard | TBD | TBD | TBD | TBD | TBD | TBD |

**Each source cell:** `0.xxx [0.xxx, 0.xxx]` (95% CI for the prespecified seed aggregation procedure); headings explicitly state *macro-AUPRC*. **Mean:** four-source equal-weight mean, never pool source records for this summary. **Worst source:** minimum of the four, with the source name in parentheses. Summarize seed variability in **S4**; do not interpret a patient bootstrap CI as a three-seed robustness interval. Add macro-AUROC as a separate Table S3; if journal page budget allows, Table 3A (AUPRC) and 3B (AUROC).

### Table 4. Physical STM32F411VET6 results

**Layout:** one row per **selected deployed checkpoint** with model, tier, held-out target source and seed made explicit. Main text can show a prespecified representative checkpoint per configuration (such as development-selected seed 0 in a nominated LOSO fold), but **S5 must contain every attempted checkpoint**. Never pool 4 folds' weights into a fictitious single firmware image.

| Model | Tier | Target-excluded fold | Seed | Full Flash (KiB) | Peak SRAM (KiB) | Median latency (ms) | p95 (ms) | Outcome |
|---|---|---|---:|---:|---:|---:|---:|---|
| FCN-1D | Compact | PTB-XL held out | 0 | TBD | TBD | TBD | TBD | PASS / failure code |
| … | … | … | … | … | … | … | … | … |

**Add in S5:** actual model weights, operator/version manifest, build ID, cycles, clock rate, debugger/UART interference, measured stack, tensor arena, input/output buffers, board voltage, flash mapping, model accuracy parity. **Pass requires:** successful firmware build, within *usable* Flash and peak SRAM, accurate output parity, and repeated physical-board inference. If energy is measured, add energy median and method; no proxy energy from a guessed current draw.

### Table 5. Quantization and host/device parity

**Layout:** 16 configurations × 4 held-out sources (64 rows, ideally compact appendix with main text containing equal-source aggregate only). Distinguish FP32, host INT8, device integer outputs.

| Model | Tier | Source | FP32 macro-AUPRC | Host INT8 macro-AUPRC | Δ AUPRC [95% CI] | Device parity | Status |
|---|---|---|---|---|---|---|---|
| FCN-1D | Compact | PTB-XL | TBD | TBD | TBD | Pass/Fail | TBD |
| … | … | … | … | … | … | … | … |

Define `Δ = INT8 − FP32`; **negative means degradation**. Quantization calibration samples originate only from development data in that LOSO fold. Define the parity statistic and tolerance before execution (e.g., max absolute output-score deviation on an auditable fixed test-vector set). Only label a model device-verified if this parity passes. A single device test vector cannot support claims about complete held-out-set accuracy; full held-out scores require all samples to be run through equivalent integer inference or an explicitly validated inference-equivalence method.

## 3. Main figures: exact panels, encodings and captions

### Figure 1. Dataset, train/test separation, and embedded pipeline

**Format:** one landscape workflow, 3 panels (A–C).

- **A, sources:** four dataset boxes with raw → retained counts and 12-lead, 10-s, 100-Hz harmonization.
- **B, LOSO:** four columns, one held-out source shaded for each fold; arrows from the other three sources into *train → internal validation → frozen checkpoint*, with no arrows originating at test to calibration or selection.
- **C, hardware:** FP32 checkpoint → train-source-only INT8 calibration → operator compatibility/build → physical F411 measurement → predictive/resource results.

**Caption must state:** dataset held out never contributes to weight updates, threshold tuning, architecture selection or PTQ calibration. Boxes are methodology, not measured outcomes.

### Figure 2. External performance heatmaps

**Format:** two panels aligned horizontally: **A macro-AUPRC; B macro-AUROC**.

- **Y:** 16 configurations, ordered M1 Compact, M1 Standard, ... M8 Compact, M8 Standard.
- **X:** PTB-XL, Chapman, Georgia, Ningbo.
- **Cell value:** three-seed mean, displayed to 3 decimal places, with same eligible class-set rule.
- **Color:** perceptually uniform sequential palette *within each metric*, shared colorbar across sources; do **not** use identical scale between AUPRC and AUROC automatically. Report both chosen limits.
- **Special cells:** blank/crosshatch for not estimable; never color undefined values as zero.
- **CI:** keep individual 95% CIs in Table 3/S3; annotate heatmap with point estimates only to prevent unreadable cells.

**Caption:** all columns correspond to entirely unseen source; macro-AUPRC is prevalence-sensitive. Give seed aggregation and class eligibility definitions.

### Figure 3. Robustness to unseen sources

**Format:** paired panels: **A model-wise source range; B worst-source vs mean**.

- **A:** horizontal range plot: y = 16 model-tier configurations; x = macro-AUPRC. For each row, four colored shape-coded dots (sources), plus a black diamond for the equal-source mean. Thin connector from minimum to maximum is a descriptive range, **not a confidence interval**.
- **B:** scatter: x = equal-source mean macro-AUPRC, y = worst-source macro-AUPRC; each point is a configuration. Draw y=x dashed for context. Mark compact vs standard using filled/open symbols. No fabricated statistical inference over four sources.

**Caption:** descriptive between-source robustness; label imbalance and source prevalence may contribute to differences. Include source color/marker key.

### Figure 4. Quantization effect

**Format:** forest plot with 16 rows, or 8 rows × 2 tier panels.

- **X:** `Δ macro-AUPRC = INT8 − FP32` (signed scale, units = absolute AUPRC points).
- **Y:** architecture; group Compact and Standard separately.
- **Mark:** filled dot = paired estimate; horizontal bar = 95% patient-cluster bootstrap CI; vertical reference at zero.
- **Colors:** consistent architecture palette; CIs same color as point, not thick gray connectors.
- **Averaging:** equal-weight across four target sources; source-specific estimates in **FS1**.
- **Device failures:** append a status glyph/text; do not invent INT8 effect when conversion failed.

**Caption:** include calibration protocol, source aggregation, paired resampling unit and number of bootstraps.

### Figure 5. External utility versus deployed resource Pareto frontier (primary synthesis)

**Format:** A = performance–latency bubble plot; B = performance–memory bubble plot (optional if space).

- **A X:** median **physical MCU** latency (ms, log scale only if justified); **Y:** equal-source INT8 macro-AUPRC of the **same defined group of checkpoints**; bubble **area** proportional to measured peak SRAM (KiB), not radius; color = architecture; shape = tier.
- **B X:** full firmware Flash occupancy (KiB); Y = same performance; bubble area = peak SRAM.
- **Eligible points:** only verified, Flash/SRAM-feasible MCU checkpoints with correctly matched predictive results. Indicate if the main plot uses seed 0 for each fold or summarized fold measurements; never use a best-of-seeds accuracy with a different untracked firmware build.
- **Pareto criterion:** no other eligible point with higher/equal macro-AUPRC and lower/equal latency, with at least one strict improvement; show nondominated points and connect **only** those sorted by latency. If memory enters Pareto definition, explicitly use the 3D dominance test (performance, latency, SRAM) and state it in legend.
- **Failures:** list counts by reason below panel or refer to Table 4; don't plot them at zero latency.

**Caption:** all performance estimates are external to each checkpoint's training sources; each latency is measured on the F411. State device clock and measurement protocol.

### Figure 6. Board feasibility and runtime breakdown

**Format:** 3 aligned panels, same 16-row y-axis if possible; **A firmware Flash; B peak SRAM; C median MCU latency**.

- **A:** horizontal bars in KiB; vertical dashed line at *measured usable Flash*, not blindly 512 KiB if firmware reserves space.
- **B:** horizontal bars in KiB; vertical dashed line at measured usable SRAM, accounting for stack/runtime allocation.
- **C:** latency dots (median) with whiskers for p95 latency **only if clearly labeled as percentiles, not statistical CIs**. Prefer dot and a thin line to p95.
- **Ordering:** M1→M8, Compact then Standard; same architecture colors as Figures 3–5.
- **Failure annotations:** `CF`, `OP`, `FM`, `RM`, `VP` adjacent to the relevant row. Do not draw bars for unavailable measurements.

**Caption:** full firmware rather than only model storage; report clock rate, toolchain/compiler flags, inference runtime version, warmup, number of iterations, and measurement instrumentation.

## 4. Supplementary displays (required)

| Item | Content / grain |
|---|---|
| **S1** | Original architecture references and 1D implementation manifest (16 configurations + commits, operator lists) |
| **S2** | Harmonized label code crosswalk and source × class positive/negative/unknown prevalence matrix |
| **S3** | Per-source macro-AUROC with 95% CIs; per-class AUPRC/AUROC across held-out sources |
| **S4** | Per-seed results for every model × tier × source; training/validation learning curves; selection rationale |
| **S5** | All MCU build/flash/parity trials: checkpoint × fold × seed × quantizer and compiler/runtime info |
| **S6** | FP32 → INT8 class-wise deltas and source-specific paired effect CIs |
| **S7** | Bootstrapping / multiplicity policy, dataset-level sample counts, code hashes and sensitivity analyses |
| **FS1** | Quantization effects faceted by held-out source (four panels) |
| **FS2** | MACs versus *measured* latency, log axes if necessary, hardware-verified points only |
| **FS3** | Peak SRAM versus maximum intermediate activation size / operator breakdown (when available) |

## 5. Data schemas: one authoritative source per display

Produce tables/figures from exported artifacts, **never from transcribed values**. All identifiers below are mandatory join keys.

```text
cohorts.csv
  source, dataset_version, n_raw, n_retained, n_patients_known, label, n_pos, n_neg, n_unknown

model_manifest.csv
  model_id, family, tier, architecture_commit, config_hash, parameters, macs,
  int8_weight_bytes, input_shape, operator_manifest_hash

predictions.parquet
  heldout_source, model_id, tier, seed, checkpoint_hash, patient_id, record_id,
  class_id, truth, truth_known, fp32_score, host_int8_score

performance.csv
  heldout_source, model_id, tier, seed, inference_mode, class_set_hash,
  n_records, macro_auprc, macro_auroc, ci_low, ci_high, bootstrap_unit

quant_effects.csv
  heldout_source, model_id, tier, seed, delta_auprc, ci_low, ci_high,
  calibration_hash, host_device_parity_status

mcu_runs.csv
  heldout_source, model_id, tier, seed, checkpoint_hash, firmware_build_hash,
  quantizer_version, runtime_version, cpu_mhz, clock_config, flash_bytes,
  peak_sram_bytes, arena_bytes, stack_bytes, median_ms, p95_ms, median_cycles,
  number_of_inferences, parity_metric, parity_pass, deployment_status,
  failure_code
```

**Plot input mapping:** Table 1 ← `cohorts.csv`; Table 2 ← `model_manifest.csv`; Table 3 and Figures 2–3 ← `performance.csv` backed by `predictions.parquet`; Table 5 and Figure 4 ← `quant_effects.csv`; Table 4 and Figure 6 ← `mcu_runs.csv`; Figure 5 ← explicit checkpoint-level join of `performance.csv` + `mcu_runs.csv`. Reject missing/duplicate key matches, not just mismatched model names.

## 6. Visual style and quality-control checklist

- Use consistent architecture colors in all performance and hardware plots; use shape or line style for Compact vs Standard; use a separate marker encoding for source.
- Prefer vector `PDF/SVG` for plots, 300 dpi or better for raster-only panels. Use readable labels at journal column widths (target ≥7–8 pt after reduction).
- Use concise titles with panel letters **A/B/C** inside upper-left of each panel; avoid repeating the caption inside the plot.
- Always state the metric, direction of benefit (↑/↓), unit and observation level. Prevalence-sensitive metrics require source prevalence nearby (Table 1/S2).
- Show CIs as `estimate [lower, upper]`; distinguish CI from seed spread and from min–max source range.
- Don't mark significance with stars without named comparison, correction method and paired analysis protocol. Prefer effect estimates and intervals.
- For absent ECG diagnoses, report `NE`; don't assume their AUROC/AUPRC is zero. Only calculate prespecified common-class macros when those classes are estimable in all required sources.
- Every plotted deployed point has a traceable checkpoint hash, build hash, fold, seed, quantization calibration hash, and correctness/parity status.
- Add exact figure and table captions explaining denominator, pooling, aggregation, deployment device and whether values reflect FP32, host INT8, or device inference.
- Do not declare a model fastest from parameter count or MACs. The MCU measurements decide.

## 7. Recommended manuscript Results outline

```markdown
## 3. Results
### 3.1 Cohort retention and class observability
(Refer to Table 1; avoid repeating each count in prose.)
### 3.2 External classification performance across four unseen sources
(Table 3 and Figure 2; report which model/tier is highest on the preregistered primary endpoint.)
### 3.3 Robustness and worst-source classification performance
(Figure 3; describe mean–worst trade-offs; avoid causal explanations without evidence.)
### 3.4 Effect of integer quantization on external predictions
(Table 5 and Figure 4; highlight magnitude and CI of paired FP32→INT8 change.)
### 3.5 On-device execution, memory feasibility and latency
(Table 4 and Figure 6; report successes, failures and their exact reasons.)
### 3.6 Joint external performance–resource efficiency
(Figure 5; nondominated configurations; clearly disclose selection rule.)
```

**Suggested manuscript conclusion shape:** state (i) the externally strongest architecture/tier, (ii) the most resource-efficient correctly deployed architecture/tier, and (iii) whether those are the same. No winner, numerical result, or clinical claim should be written until runs are complete.
