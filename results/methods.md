# Supplementary Methods

Cardiologist disagreement and predictive uncertainty in automated 12-lead ECG
classification.

All quantities reported in the Letter and in Figure 1 are produced by the
pipeline described below. Every model choice was fixed on the development
dataset before the test set was read.

---

## 1. Data

**Development set — CODE-15%.** 345,779 12-lead ECG exams from 233,770
patients, a public 15% sample of the CODE cohort (Telehealth Network of Minas
Gerais, Brazil). Six diagnostic labels derived from the automatic report and
expert review: first-degree AV block (1dAVb), right bundle branch block
(RBBB), left bundle branch block (LBBB), sinus bradycardia (SB), atrial
fibrillation (AF) and sinus tachycardia (ST). Used for training, model
selection and threshold fitting only.

**Test set — CODE-test.** 827 ECGs from distinct patients, annotated
independently by two cardiologists across all tracings. The gold standard is
the two cardiologists' common diagnosis where they agreed; where they
disagreed, a third senior specialist, aware of both prior annotations,
decided. The dataset also contains annotations from two 4th-year cardiology
residents, two 3rd-year emergency residents and two 5th-year medical students
(each pair splitting the dataset), and predictions from the original
publication's neural network. None of these were used here.

CODE-test was held out entirely: it contributed to no training step, no
architecture or hyperparameter choice, and no threshold.

---

## 2. Preprocessing and data partition

The train/validation split was **frozen before any signal was written to
disk**, at the patient level (a patient's exams cannot straddle the split),
90/10, seed 0: 311,153 training and 34,626 validation exams.

Signals were written once to a float16 memory-mapped array of shape
(n, 12, 4096): 12 leads, 400 Hz, 4096 samples per lead, zero-padded where the
original recording was shorter. Lead order I, II, III, aVR, aVL, aVF, V1–V6.

No training-derived transformation was baked into the stored signals. A single
scalar mean and standard deviation were estimated from a 20,000-exam random
subsample of the **training partition only** (mean −0.0664, SD 1.4592) and
applied at load time. Storing normalised signals would have made the stored
array a function of the split.

---

## 3. Model

A deliberately conventional lightweight 1-D convolutional network,
**169,030 parameters**:

| stage | operation |
|---|---|
| 1 | Conv1d(12 → 32, kernel 15, stride 2), BatchNorm, ReLU |
| 2 | Conv1d(32 → 64, kernel 11, stride 2), BatchNorm, ReLU |
| 3 | Conv1d(64 → 128, kernel 7, stride 2), BatchNorm, ReLU |
| 4 | Conv1d(128 → 128, kernel 5, stride 2), BatchNorm, ReLU |
| head | global average pooling → Linear(128 → 6) → sigmoid |

Convolutions are bias-free (BatchNorm supplies the offset). Kernels narrow
with depth while the receptive field grows through stride: wide early filters
span a QRS complex, later filters combine them.

Architecture is not the contribution. The claim concerns the relationship
between reader disagreement and model error, and is stronger if it holds for a
conventional classifier than for a bespoke one.

---

## 4. Training

Five members were trained, differing **only** in random seed (0–4), which sets
both weight initialisation and batch order. All five shared the same frozen
split, so the spread across members reflects initialisation and optimisation
variance rather than a mixture of that and data-split variance.

| setting | value |
|---|---|
| optimiser | AdamW, weight decay 1e-4 |
| learning rate | 1e-3, OneCycle schedule over all steps |
| loss | BCEWithLogits, per-class positive weighting |
| positive weights | neg/pos per class, capped at 50 |
| | 1dAVb 50, RBBB 35, LBBB 50, SB 50, AF 48, ST 45 |
| batch size | 256 |
| epochs | 15 |
| precision | bfloat16 autocast on GPU |
| checkpoint | epoch with best validation macro-AUPRC |

Positive weighting is necessary because these labels sit near 2% prevalence,
at which unweighted BCE converges to predicting the negative class; the cap
prevents the rarest class from dominating the gradient.

Validation macro-AUPRC by member: 0.6321, 0.6262, 0.6322, 0.6295, 0.6289.
The ensemble reached 0.6381, exceeding every individual member.

---

## 5. Ensemble and decision thresholds

The ensemble probability is the unweighted mean of member probabilities,
p̄(i,c) = (1/5) Σ_m p^(m)(i,c).

Per-class decision thresholds t_c were chosen to maximise F1 **on the
validation partition only**, over a 199-point grid on (0, 1), then frozen
before the test set was read:

| | 1dAVb | RBBB | LBBB | SB | AF | ST |
|---|---|---|---|---|---|---|
| t_c | 0.915 | 0.895 | 0.955 | 0.905 | 0.950 | 0.925 |

Thresholds are high because of the positive weighting, which inflates
predicted probabilities. Fitting them on data later used for evaluation would
have made every downstream confidence value optimistic, since confidence is
defined relative to t_c.

Predictions are ŷ(i,c) = 1[p̄(i,c) ≥ t_c] and the per-label loss is
ℓ(i,c) = 1[ŷ(i,c) ≠ y\*(i,c)] against the adjudicated gold standard.

**Test-set discrimination:** macro AUROC 0.989, macro AUPRC 0.801.

---

## 6. Uncertainty score

One score is used throughout — for the referral ranking and for the
uncertainty comparison alike, so that the supporting and primary analyses
cannot come from two differently behaved quantities:

    C(i,c) = |logit(p̄(i,c)) − logit(t_c)|        (decision confidence)
    U(i,c) = −C(i,c)                              (decision uncertainty)

This is the distance of the prediction from its own operating point on the
logit scale. No bounded transformation is applied. Ensemble variance and
predictive entropy appear only as prespecified sensitivity analyses
(Section 10).

---

## 7. Reader disagreement

    D(i,c) = 1[y1(i,c) ≠ y2(i,c)]

computed from the two cardiologists' independent annotations **after**
prediction. D never enters training, threshold selection, the uncertainty
score, or the referral decision. A deployed model cannot know whether two
readers will disagree about a record it is about to see; using D to select
what to retain would measure an oracle rather than a referral policy.

Observed: 33 disputed labels among 4,962 diagnosis-record pairs (0.67%),
concentrated in 1dAVb (14 of 33). All 33 fall in 33 distinct ECGs, so no
record carries more than one disputed label.

---

## 8. Selective risk and referral

At coverage q the most confident fraction q of predictions is retained
**per diagnosis**, so every class retains the same proportion; a shared
threshold would retain very different fractions across classes whose
confidence distributions differ. Selective risk R_c(q) is the error rate among
retained predictions for class c, and the reported risk is the macro average
over the six diagnoses. Macro AURC is the area under that curve.

Risk was computed over the full range q ∈ (0, 1] at 0.01 resolution; q = 0
retains nothing and is undefined. Figure 1A displays q ∈ [0.50, 1.00], where
1 − q is the referral fraction.

**Results:** macro selective risk 0.0125 (95% CI 0.0095–0.0157) at full
coverage, 0.0013 (0.0004–0.0034) at 90%, 0.0003 (0.0000–0.0010) at 80%, and
0.0000 at 50%. Macro AURC 0.00056 (0.00033–0.00087). Residual errors fall from
62 at full coverage to 18 at 95% and 6 at 90%.

---

## 9. Bootstrap

All confidence intervals are 95% percentile bootstrap intervals over
**1,000 replicates, resampling whole ECGs**. An ECG is the unit of
resampling, so all six of its labels are drawn together and the within-record
correlation among labels is preserved. Replicates in which a statistic is
undefined — a resample containing no disputed label, or none with a residual
error — are dropped rather than counted as zero, and the interval is taken
over the remainder.

Ratios are formed **inside each replicate** and the percentiles taken over the
resulting distribution of ratios. Dividing the endpoints of two separately
bootstrapped intervals would ignore the correlation between numerator and
denominator; for the enrichment statistic that approach gives 8.97–34.70
against the correct 10.06–34.82.

---

## 10. Reported comparisons

**Uncertainty by reader agreement (text only).** Median U was −7.234 on
consensus labels and −1.189 on disputed labels, a difference of 6.045
(95% CI 5.442–6.409).

**Disagreement prevalence (Figure 1B, left).** Disputed labels were 0.67% of
all evaluated pairs (33/4,962) but 14.5% of model errors (9/62), a 22-fold
overrepresentation (95% CI 10–35).

**Error rate by reader agreement (Figure 1B, right).** Model error was 1.08%
(95% CI 0.79–1.38) among consensus labels and 27.3% (12.1–42.9) among disputed
labels, a 25-fold higher error rate (95% CI 11–45).

**Sensitivity to the uncertainty score.** The primary score was fixed before
unblinding and is not revised. Under the two alternatives, macro AURC was
0.0040 (ensemble variance) and 0.0038 (predictive entropy) against 0.00056 for
the decision margin.

---

## 11. Limitations of the reference standard

Where the two cardiologists agreed, the gold standard **is** their shared
answer by construction. The consensus stratum is therefore defined by the same
two readers whose agreement defines it, and part of the contrast in
Figure 1B (right) is close to definitional: consensus labels are those on
which two experts independently converged, which selects for unambiguous
tracings. The composition result (Figure 1B, left) is less exposed, since its
denominator is all evaluated labels rather than the consensus stratum.
Adjudication of disputed cases was not blinded — the third specialist saw both
prior annotations.

With 33 disputed labels in total, error counts fall quickly under referral (62
at full coverage, 6 at 90%), so enrichment among *residual* errors after
aggressive referral could not be estimated with useful precision. Both
reported comparisons are therefore made at full coverage.

---

## 12. Reproducibility

Predictions from all five members are cached, so every post-hoc choice —
coverage grid, bootstrap replicate count, uncertainty definition — is
recomputed from stored probabilities and never triggers retraining. Random
seeds are fixed for the split (0), the members (0–4) and every bootstrap.

    scripts/01_prepare.py        build memmaps, freeze the split
    scripts/02_train_ensemble.py train 5 members, fit thresholds, cache
    scripts/03_analyse.py        all statistics and Figure 1
