# Risk-Controlled Selective Classification of 12-Lead ECG

Code for a study of how much coverage a strong 12-lead ECG classifier can retain
on an **unseen clinical source** while its prediction risk stays certified below
a target level, and how that depends on the signal representation.

The classification machinery is deliberately conventional. The contribution is
the selective-risk layer: a frozen classifier, a record-level confidence score,
and an abstention threshold calibrated with a finite-sample guarantee on a small
labelled sample from the target source.

## Setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verified on Windows 11, Python 3.13, torch 2.11.0+cu128, RTX 4070 (12 GB).

## Data

Four databases, 77,333 records, expected under `dataset/`:

| Source | Records | Layout |
| --- | ---: | --- |
| PTB-XL 1.0.1 | 21,837 | `dataset/ptbxl/records500/…` + `ptbxl_database.csv` |
| Georgia (G12EC) | 10,344 | `dataset/georgia/*.hea,*.mat` |
| Chapman-Shaoxing | 10,247 | `dataset/WFDB_ChapmanShaoxing/` |
| Ningbo | 34,905 | `dataset/WFDB_Ningbo/` |

### Physical units and amplitude QC

All signals are converted to physical millivolts and must pass **source-wise
physiological amplitude quality control before any model is allowed to train**.
`02_cache_signals.py` exits non-zero if a source's median limb-lead QRS
peak-to-peak amplitude falls outside 0.5-2.5 mV, and `03_train.py` refuses to
start without a passing `amplitude_qc.json`.

This is a gate rather than a warning because a wrong ADC gain yields a perfectly
clean cache that is silently off by a constant factor, surfacing only as
inexplicably poor cross-source transfer. It also cannot be normalised away
later: low QRS voltage (LQRSV) is one of the 13 labels, so per-record amplitude
normalisation would delete the very information that label depends on.

Georgia required a gain override (see `GAIN_OVERRIDE` in `src/ecgsr/ingest.py`).
Its headers declare 4880 ADC units/mV against 1000 for the other three sources;
applied literally this put 94% of Georgia records below the clinical
low-QRS-voltage threshold while only 3.6% carry the LQRSV label, and the model
then predicted LQRSV on 97% of that source. Treating the stored samples as
scaled for gain 1000 aligns all four sources:

```
source     median      q05      q95  P(<0.5mV)
Chapman     1.225    0.715    2.459      0.005
Georgia     1.299    0.695    2.620      0.008   (gain override 1000)
Ningbo      1.251    0.729    2.683      0.004
PTBXL       1.261    0.749    2.516      0.003
```

This is a scaling inconsistency in the files as distributed to this pipeline.
Absent confirmation from the original G12EC documentation, it should be
described as such rather than as an error in the published headers.

### Labels

Georgia, Chapman and Ningbo already carry PhysioNet/CinC 2021 SNOMED-CT `#Dx`
labels. PTB-XL 1.0.1 does not, so its SCP-ECG statements are converted in
`src/ecgsr/labels.py`. That conversion is not taken on trust: every rule is
checked against the per-code record counts published in the Challenge's
`dx_mapping_{scored,unscored}.csv`, and `01_build_index.py` refuses to run if any
count disagrees. Three unscored umbrella codes (MI, VEB, NSSTTA) could not be
reproduced from the published SCP statements and are left unmapped rather than
guessed at; none is a scored diagnosis, so none can enter the label space.

## Pipeline

```powershell
.\.venv\Scripts\python.exe scripts\00_validate_risk_control.py   # check the guarantee
.\.venv\Scripts\python.exe scripts\01_build_index.py             # labels + common classes
.\.venv\Scripts\python.exe scripts\02_cache_signals.py           # filtered signal cache
.\.venv\Scripts\python.exe scripts\05_run_all.py                 # 12 pipelines, resumable
.\.venv\Scripts\python.exe scripts\06_tables_figures.py          # tables + figures
.\.venv\Scripts\python.exe scripts\08_selective_sweep.py         # offline, seconds
```

### Logits are cached, so selective-risk work is offline

`04_calibrate_eval.py` performs the only GPU work after training: one inference
pass over dev/cal/test, written to `predictions.npz` (logits, labels, record
IDs). Everything downstream — confidence scores, decision thresholds,
risk-coverage curves, AURC, the oracle bound, finite-sample calibration, every
risk level — is a function of those arrays alone and lives in
`src/ecgsr/analysis.py`.

`08_selective_sweep.py` therefore answers a new question in **seconds**, with no
ECG decoding and no forward pass. Changing the confidence score does not require
retraining or re-inferring anything. Risk-coverage curves are one argsort plus
one cumulative sum, O(N log N), rather than a loop over candidate thresholds.

`05_run_all.py` is resumable: a pipeline with an existing `eval.json` is skipped.

### Label space

Pre-registered rule, applied before any training: keep a *scored* SNOMED
diagnosis only if it has **≥100 positives in every one of the four sources**,
after merging the Challenge's equivalence pairs (CRBBB≡RBBB, CLBBB≡LBBB,
PAC≡SVPB, PVC≡VPB). That yields **13 classes** — NSR, SB, TAb, STach, LAD, TInv,
CRBBB, IAVB, PAC, QAb, NSIVCB, LQRSV, CLBBB — and avoids testing on a hospital
for a diagnosis the training hospitals never labelled.

### Representations

One fixed configuration each, never tuned. All standardised to 10 s @ 500 Hz,
0.5–50 Hz band-pass plus a site-appropriate mains notch, amplitude kept in
millivolts (QRS voltage is itself diagnostic).

- **Raw** — 12 × 5000
- **VCG** — 3 × 5000, inverse Dower applied to physical amplitudes *before*
  normalisation
- **STFT** — `n_fft=256`, Hann, `hop=64`, 0.5–50 Hz retained, `log(1+|STFT|²)`

### Backbones

| representation | network | GMAC/sample | params |
| --- | --- | ---: | ---: |
| Raw, VCG | `resnet1d18` | 1.41 | 6.3 M |
| STFT | `resnet2d18` | 1.21 | 11.2 M |

`resnet1d18` is the 1-D counterpart of the ResNet-18 used for the STFT, so all
three representations share one depth/width family and the comparison isolates
the representation rather than model capacity. Deeper 1-D options
(`resnet1d34`, `xresnet1d50`, `xresnet1d101`) remain available via
`--backbone`; `xresnet1d101` is 6.4× the MACs and measured 8.9× slower on an
RTX 4070 for no gain in dev AUROC at matched epochs.

A single signal cache serves all three; VCG and STFT are derived on the GPU, so
the representations see byte-identical underlying signals.

### Protocol

Four leave-one-source-out experiments. Network weights see only the three
non-target sources. The target source is split 20% calibration / 80% test; the
test split is read exactly once.

Per-class decision thresholds `t_k` are fitted on a development split of the
*training* sources (grouped by patient) and then frozen. They are never refitted
on the target.

### Selective risk

Confidence is the **class-normalised mean decision margin**,

    d_k(x) = logit p_k(x) − logit t_k
    s(x)   = mean_k |d_k(x)| / MAD_k

with `MAD_k` a robust per-class scale estimated on the development split of the
training sources only, so the score introduces no target leakage. Normalising
stops a naturally high-margin diagnosis from dominating a naturally hard one.

`min_k` is **not** used, and that is a finding rather than a detail. It asks
whether *every* one of K binary decisions is far from flipping, and the chance
that at least one output sits near its boundary grows quickly with K. At K = 13
it saturates and stops ranking records — measurably worse than the mean. It is
retained in `src/ecgsr/scores.py` only so the failure can be reported.
`max_k p_k` is worse still: in a multi-label problem it says nothing about the
classes the model called negative.

Risk is record-wise Jaccard loss, `L = 1 − |Y ∩ Ŷ| / |Y ∪ Ŷ|`, with both-empty
counted as a perfect prediction. Note this is a demanding loss: on a record with
`Y = {AF, RBBB}`, predicting only `{AF}` already costs `L = 0.5`.

### Metrics

**AURC is the primary metric.** It summarises the whole risk-coverage ranking
and does not depend on an arbitrary risk level. Fixed-risk operating points
`C@R<=alpha` are reported as interpretable secondary numbers at levels declared
in `ALPHA_LEVELS` (0.10, 0.20, 0.30) before any test split is read.

Every table also carries the **oracle bound** — the best risk-coverage curve any
confidence score could achieve, obtained by ranking records by their true loss.
It separates two very different failures:

- oracle `C@R<=alpha` also near zero → the classifier never produces enough
  near-perfect records, and no confidence score can rescue that operating point;
- oracle high but achieved coverage low → the predictions are good and the
  *ranking* is what is failing.

### Calibration

The goal is `max_τ C(τ)` subject to `R(τ) ≤ α` (α = 0.10, δ = 0.05). Two details
in `src/ecgsr/ltt.py` are what make this a guarantee rather than a hopeful cutoff:

1. Selective risk is a *ratio*, `E[L·g]/E[g]`, so a bounded-mean bound does not
   apply directly. Instead of bounding numerator and denominator separately
   (valid but very lossy) we test the equivalent linear form
   `E[(L−α)·g] ≤ 0` via the shifted variable `v = (L−α)·g + α ∈ [0,1]`, which
   uses the whole calibration set with no union bound.
2. Threshold selection over a grid is multiple testing, handled by Bonferroni.
   Fixed-sequence testing is *not* usable here — at the conservative end of the
   grid coverage tends to zero, the constraint holds with equality, nothing can
   be rejected, and the sequence aborts before reaching a useful threshold.

`00_validate_risk_control.py` verifies the procedure by Monte Carlo before any
ECG result is produced. Representative output:

```
procedure             mean cov  mean risk   P(test risk > alpha)
LTT (calibrated)         0.431     0.0445                  0.000
naive (empirical)        0.696     0.0997                  0.487
```

Picking the threshold where the *empirical* calibration risk first drops below α
violates the target risk about half the time. That gap is what the guarantee
buys, and the coverage difference is its price.

## Layout

```
configs/    official Challenge dx_mapping tables
src/ecgsr/  labels, ingest, representations, datasets, models, metrics,
            train, scores, selective, ltt, analysis
scripts/    00 validate · 01 index · 02 cache · 03 train · 04 calibrate+eval
            05 run-all · 06 tables+figures
artifacts/  runs/ (checkpoints, predictions.npz, eval.json) · tables/ · figures/
```

## Numerical notes

- Autocast uses **bfloat16**, not float16. float16 overflowed the deeper
  `xresnet1d101` activations late in training: the loss was still falling and
  then the forward pass produced NaN. bf16 has fp32's exponent range, so it
  cannot overflow, and Ada-class GPUs run it at the same speed. `GradScaler` is
  enabled only for fp16, where it is actually needed.
- Metrics sanitise non-finite scores to a neutral 0.5 rather than raising. A
  monitoring call should never be able to destroy hours of training; a genuine
  numerical problem surfaces as a degraded score plus the per-epoch
  `nonfinite_batches` counter.
- `torch.compile` was measured at ×1.06 on this backbone (with
  `triton-windows`) against a 71 s compile cost, so it is not used.

## Notes

- Reported numbers use the **calibrated** threshold. `oracle_coverage` in
  `eval.json` is the label-peeking upper reference and is not a headline result.
- `04_calibrate_eval.py` sweeps a range of α and stores all of them, so
  operating-point sensitivity is available without retraining. α = 0.10 is the
  pre-registered headline.
- The guarantee assumes calibration and deployment records are exchangeable
  within the target source. It says nothing about a further, unobserved shift.
- ECGFounder is not included here. If added, it belongs as an external SOTA
  reference rather than the primary backbone: it was pretrained on Harvard-Emory
  data and the Georgia database originates from Emory, so overlap cannot be
  excluded.
