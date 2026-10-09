# Source heterogeneity dominates architecture in external 12-lead ECG
classification, and selective reliability inherits it

**Four clinical sources, six compact encoders, leave-one-source-out
evaluation at 100 Hz.**

> When a 12-lead ECG classifier is applied to a clinical source it has
> never seen, how much of what happens is about the model, how much is
> about the source, and can the model tell when it is wrong?

Everything here answers that one question in three stages. External
generalisation is the spine: the quality analysis explains failures in
it, selective prediction asks whether those failures can be recognised,
and deployment asks whether the externally validated behaviour survives
implementation. Every downstream analysis uses the same leave-one-
source-out predictions.

| | question | evidence |
|---|---|---|
| **RQ1** | How much does external performance vary across unseen sources, relative to variation across architectures? | Figure 2 |
| **RQ2** | Which diagnostic-distribution, annotation and signal-quality differences are associated with that variation? | Figure 3, Table III |
| **RQ3** | Does model confidence identify unreliable predictions consistently across unseen sources? | Figure 4 |

```
How much does the source matter?  ->  Why?  ->  Can the model tell?
```

Each question is the previous answer's leftover. RQ1 finds that the
held-out source explains 92% of the variance in external AUPRC and the
architecture 6%, so RQ2 asks what it is about a source that matters --
and finds that it is not signal quality, which runs the wrong way.
RQ3 then asks whether a model can at least recognise when it is in
trouble, and finds that confidence does rank errors in every source --
but that selective reliability varies across sources more than across
architectures, exactly as discrimination does.

Four public sources, 77,333 records, 13 harmonised labels. Six encoder
families trained leave-one-source-out at 100 Hz, so every reported number
comes from a source the model never saw in training, under one frozen
input specification.

## What is, and is not, claimed as new

Cross-dataset ECG evaluation is not new, and a benchmark of six familiar
1D CNNs is not a contribution by itself. Neither is claimed here.

Three claims are made, on one cohort and one frozen input
specification, and two of them are negative.

1. **The held-out source, not the architecture, decides external
   performance.** The held-out source accounts for 91.9% of the variance
   in external macro AUPRC and the encoder for 6.4%, with an interaction
   of 1.7% (RQ1). Benchmarks that rank architectures on a single
   held-out split are measuring the split.

2. **Signal quality does not explain which sources are hard, and runs
   the wrong way.** The cleanest corpus in this cohort is the worst to be
   tested on and the dirtiest is the best (RQ2a). What tracks external
   performance instead is how far the site's label distribution sits
   from the training pool, and part of the residual is the sources
   disagreeing about what a diagnosis means -- separable from
   generalisation failure because AUROC survives where AUPRC collapses.
   Poor signal quality does cost discrimination *within* a source
   (RQ2b); it is not what distinguishes the sources from each other.

3. **Confidence remains informative under transfer, but inherits the
   same source dependence.** Selective risk falls with coverage for every
   encoder on every held-out source, so abstention is not merely tracking
   source difficulty. Yet E-AURC spans 0.130-0.161 across architectures
   and 0.104-0.202 across the 24 model-by-source folds, and degradation
   compounds: where a measured quality defect costs discrimination, the
   confidence ordering degrades with it (RQ3). An abstention budget
   calibrated on one source is not a reliability guarantee on another.

What is *not* claimed: a causal account of the
cross-source spread. Four sources give four points, and Findings 1c and
1d establish sign and consistency, not effect size.

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
s7_manuscript.py          assembles figures 1-5 and tables I-V
```


---

## Stage 0 - the input specification, fixed before anything is trained

The input specification -- **10 s, 12 leads, 100 Hz** -- is settled
first and then frozen, so that no later result is the product of having
tuned the representation to it. 100 Hz was chosen because it costs
nothing against 250 Hz and helps the smallest encoders (Finding 3); the
250 Hz runs are archived rather than discarded.

**Quality is measured before decimation, at 500 Hz.** A 100 Hz signal has
a 50 Hz Nyquist limit, so mains interference and high-frequency content
are not observable in the model input at all; measuring them there would
be meaningless. The audit therefore runs on the native signal and the models
run on an anti-aliased (FIR, zero-phase) resampling of it. Figure 1 shows
the two branches explicitly.

---

## RQ1 - is it the model or the site?

66,379 records after dropping those with no label in the harmonised
13-class space. Everything except the encoder is held constant.
Validation is always drawn from the training sources, never the held-out
one. 24 runs. **Figure 2**.

### Discrimination across held-out sources

Macro AUPRC by held-out source, **float32**. AUROC is reported beside it per diagnosis in Figure 3(b), because average precision has a prevalence floor and the two metrics do not agree on which source is hardest.

| model | PTB-XL | Georgia | Chapman | Ningbo | **macro** |
|---|---|---|---|---|---|
| TinyCNN | 0.425 | 0.499 | 0.527 | 0.558 | 0.502 |
| DS-CNN | 0.430 | 0.524 | 0.547 | 0.576 | 0.520 |
| MobileNet1D | 0.399 | 0.511 | 0.551 | 0.574 | 0.509 |
| **MBConv1D** | 0.434 | 0.537 | 0.580 | 0.609 | **0.540** |
| TCN-Lite | 0.425 | 0.524 | 0.569 | 0.589 | 0.526 |
| **ResNet1D-Lite** | 0.435 | 0.544 | 0.599 | 0.618 | **0.549** |

> **Finding 1b. The held-out source decides external performance; the
> architecture barely participates.** Decomposing the 6 x 4 table of
> external macro AUPRC into a source term, an architecture term and
> their interaction, over the 13 diagnoses scorable in all four sources
> (400 record-level bootstrap replicates, seed 0):
>
> | component | share of variance | 95% CI |
> |---|---|---|
> | held-out source | **91.9%** | 90.8%, 92.9% |
> | architecture | 6.4% | 5.6%, 7.2% |
> | interaction | **1.7%** | 1.3%, 2.3% |
>
> The source effect spans 0.163 AUPRC against 0.047 for the architecture
> effect, a ratio of **3.5x** (3.3, 3.8). The interaction share is the
> one that carries a design consequence: at 1.7% there is almost no
> architecture-by-site matching to exploit, so the choice of encoder is
> close to separable from where the device will be used. For a
> deployment constrained in cost, latency or power, that is permission
> to take the cheapest encoder that suffices rather than the most
> accurate one.
> **Figure 2(c).**

> **Finding 1g. Most of the spread is not a transfer penalty. It is how
> hard the source is.** Finding 1b says the held-out source decides
> external performance, but not whether that is because the model never
> saw the source or because its diagnoses are harder there. Those are
> different claims and only one of them is about generalisation.
>
> Each source was split patient-disjointly in half. The external arm is
> the LOSO model reported everywhere else, scored on one half. The
> reference arm is the same architecture trained with the *other* half
> swapped into the pool -- and an equal number of records removed from
> the other three sources, so both arms train on the same record count,
> epochs, optimiser, schedule and seed. Only representation differs.
>
> | held-out source | external AP | reference AP | transfer gap | CI excludes 0 |
> |---|---|---|---|---|
> | PTB-XL | 0.429 | 0.468 | **+0.039** | 6/6 |
> | Georgia | 0.522 | 0.545 | +0.023 | 6/6 |
> | Chapman | 0.566 | 0.591 | +0.026 | 6/6 |
> | Ningbo | 0.586 | 0.608 | +0.022 | 6/6 |
>
> There is a real penalty -- all 24 model-by-source intervals exclude
> zero -- and PTB-XL's is roughly 1.7x the others, which is the first
> result here that connects RQ1's heterogeneity to RQ2's annotation
> findings rather than leaving them adjacent.
>
> But the penalty is small against the thing it was invoked to explain.
> The spread across sources is **0.157** externally and **0.140** when
> every source is represented in training: **89% of the cross-source
> spread survives**, and only about **11%** is attributable to transfer.
> Training on a source barely closes the gap to it.
>
> This is the result that makes RQ2 interpretable. The search for a
> mechanism behind the spread kept coming up empty because most of the
> spread is not a transfer effect to have a mechanism for -- it is
> intrinsic difficulty of the source, its case mix and its label
> definitions. **Table IV.**

> **Finding 2. PTB-XL is the hardest held-out source for every encoder**
> (0.399-0.434 against 0.558-0.618 for Ningbo), even though RQ2a shows
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

## RQ2 - examining explanations for source heterogeneity

RQ1 shows that external performance depends on which source is held
out. This asks why, and whether the model can tell. Three steps: what
differs between the sources, whether those differences track the
performance gap, and whether confidence recognises the records it gets
wrong. **Figure 3**, **Table III**, **Figure 4**.

### RQ2a - it is not signal quality

RQ1 leaves a spread between sources to account for -- though Finding 1g
has already removed about 89% of it from the brief, since that part is
present even when the source is in the training set and is therefore not
a transfer effect at all. What follows asks what distinguishes the
sources, transfer or not. Signal quality is the obvious candidate and
the one this study was originally built around. It does not survive
contact with the data.

All 77,333 records, six indicators reported separately, measured on the
native 500 Hz signal before decimation. No composite index, which would
bury arbitrary weights under a scientific-looking decimal.
One indicator per panel, all 77,333 records.

> **Finding 1. The four sources differ on one axis, not six.** The three
> spectral indicators -- HF noise, baseline wander, mains -- are nearly
> superimposed across sources. The separation is structural: a flat or
> clipped lead occurs in **4.24% of Ningbo records against 0.005% of
> PTB-XL**, a factor of 920. QRS-detectability failure (0.86-1.91%) and
> implausible RR intervals differ far less. A "noise robustness" framing
> would have looked in the wrong place entirely.

> **Finding 1c. Signal quality does not explain the cross-source spread.
> It anti-correlates with it.** Taking each source's mean indicator value
> against its external macro AUPRC, five of seven indicators correlate
> *positively* -- worse measured quality, better external performance:
>
> | source | HF noise | flat leads | clipping | external mAP |
> |---|---|---|---|---|
> | PTB-XL | 0.0030 | 0.0000 | 0.0006 | **0.434** |
> | Georgia | 0.0038 | 0.0009 | 0.0007 | 0.537 |
> | Chapman | 0.0047 | 0.0037 | 0.0008 | 0.580 |
> | Ningbo | 0.0052 | 0.2011 | 0.0173 | **0.609** |
>
> PTB-XL is the cleanest corpus on nearly every indicator and the worst
> source to be tested on; Ningbo has flat leads in a fifth of its records
> and is the best. The correlations are +0.94 for baseline wander and
> +0.97 for HF noise. With four sources these are four points and
> establish sign, not effect size -- but the sign is the opposite of the
> one a quality-driven account requires. **Figure 3(a).**

> **Finding 1d. A source-level association with label-distribution
> shift does not survive moving to the diagnosis level.** Across the
> four sources, the L1 distance between a held-out source's prevalence
> vector and its training pool correlates -0.79 with external AUPRC, and
> -0.74 to -0.81 for each architecture separately. That is four points,
> and it does not hold up when the unit of analysis is the 13 x 4
> diagnosis-by-source cells, where the dependence can be modelled:
>
> | test | estimate | 95% CI | p |
> |---|---|---|---|
> | mixed model, random intercept per diagnosis | -0.011 | -0.046, +0.024 | 0.54 |
> | sign test over 13 within-diagnosis slopes | 9/13 negative | -- | 0.27 |
>
> AUROC is the outcome rather than AP, so that a prevalence difference is
> not partly regressed on itself. **The honest reading is that this study
> does not demonstrate a mechanism for the cross-source spread.** The
> source-level correlation is reported because it is what the aggregate
> data shows, and the diagnosis-level null is reported beside it because
> it is the stronger test. Finding 1b stands on its own: the spread is
> real and large whatever produces it. **Figure 3(c).**

> **Finding 1f. Average precision and AUROC disagree about which source
> is hardest, and that disagreement is informative.** Averaging over the
> 13 diagnoses and six encoders:
>
> | source | mean AP | mean AUROC | AP rank | AUROC rank |
> |---|---|---|---|---|
> | PTB-XL | 0.425 | 0.864 | 1 (worst) | 2 |
> | Georgia | 0.523 | 0.846 | 2 | **1 (worst)** |
> | Chapman | 0.562 | 0.911 | 3 | 3 |
> | Ningbo | 0.587 | 0.928 | 4 | 4 |
>
> PTB-XL is the worst source by average precision and the second-worst by
> rank-based discrimination; by AUROC it is Georgia. PTB-XL's AP deficit
> is therefore substantially a precision problem rather than a ranking
> problem -- the model is ordering its records roughly as well as it
> orders Georgia's, and losing precision against a different labelling
> convention (Finding 1e). This is the quantitative form of the SB case.
>
> A prevalence-standardised sensitivity check rules out the mechanical
> explanation: replacing AP with (AP - pi) / (1 - pi) per diagnosis ranks
> the four sources **identically**, so the AP ordering is not an artefact
> of the prevalence floor. **Figure 3(a-b).**

> **Finding 1e. Some of the gap is the sources disagreeing about what a
> label means.** Comparing each source's labelling propensity for a given
> diagnosis -- how often it applies the label relative to how often the
> model, trained on the other sources, finds the pattern -- against the
> median of its peer sources isolates convention from case mix, because
> the model reads the same signal everywhere. Eleven of 52
> source-by-diagnosis cells deviate by more than a doubling, and **seven
> of the eleven are PTB-XL**.
>
> The clearest is sinus bradycardia. In PTB-XL the model reaches
> **AUROC 0.930** -- it ranks the bradycardic records correctly -- while
> AUPRC collapses to 0.196 against 0.985 elsewhere, because PTB-XL
> labels SB at roughly **one eighth** the rate its peers do given the
> same evidence. That is an annotation convention, not a generalisation
> failure, and AUROC/AUPRC divergence is what separates the two.
>
> It does **not** account for the aggregate spread: removing every
> flagged diagnosis leaves the cross-source range at 0.195, slightly
> *wider* than the 0.163 over all 13. Label-definition drift is
> demonstrably present and concentrated in one corpus; it is not the
> mechanism behind Finding 1b. **Figure 3(c).**

---

### RQ2b - does the shift matter diagnostically?

Scope, stated before the numbers: everything in RQ2b and RQ2c is a
**within-source** contrast. Records are split into impaired and clean
strata inside each held-out source and matched per diagnosis, so these
results say what a bad recording costs relative to a good one *at the
same site*. They are not an account of the between-source spread, and
Finding 1c is the reason that distinction is drawn so sharply.

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
> differ in whatever else travels with the indicator even once every
> diagnosis prevalence is matched, so the positive columns should not be
> read as noise improving anything. The defensible claim is the contrast between the
> two families, and that the largest negative association is on exactly
> the indicator where the sources differ by a factor of 920.

> **Finding 5. Vulnerability tends to follow accuracy rather than
> resist it.** On flat or clipped leads, per encoder, with the contrast
> also expressed relative to that encoder's own clean-stratum baseline
> because a weaker model has less accuracy available to lose:
>
> | model | mean AUPRC | dAUPRC | relative |
> |---|---|---|---|
> | TinyCNN | 0.502 | -0.133 | -20.0% |
> | DS-CNN | 0.520 | **-0.122** | **-17.4%** |
> | MobileNet1D | 0.509 | -0.164 | -22.9% |
> | MBConv1D | 0.540 | -0.155 | -21.6% |
> | TCN-Lite | 0.526 | -0.153 | -21.2% |
> | ResNet1D-Lite | **0.549** | **-0.172** | **-23.2%** |
>
> The most accurate encoder is the most damaged in both absolute and
> relative terms, and correlation across the six is -0.55 (absolute) and
> -0.39 (relative). With six models and a contrast resting on Ningbo
> alone this is a tendency, not a law -- DS-CNN breaks the ordering --
> but it points the opposite way to the comforting reading that capacity
> buys robustness. Nothing here licenses choosing a smaller encoder for
> resilience.

---


## RQ3 - does confidence stay informative under transfer?

RQ1 establishes that the held-out source dominates external
discrimination. RQ2 fails to find a mechanism for it. Neither says
whether a deployed model can *tell* when it is in trouble, which is the
property an abstention or referral workflow would rest on.

Selective prediction is evaluated on exactly the same leave-one-source-out
predictions, so the discrimination and reliability results describe the
same models on the same records. Records are ranked by confidence
`min_c |p_c - 0.5|`, retained from most to least confident, and selective
risk is the mean record-wise Jaccard loss over the retained fraction.
E-AURC is the area between that curve and the model's own oracle, so it
measures ranking quality and not the difficulty of the source.

> **Threshold protocol.** No operating threshold is fitted on a held-out
> source. The curves and E-AURC are label-free in their ordering -- the
> confidence score uses only model outputs -- and every reported
> coverage is a property of that ordering, not a tuned cut. Any
> deployment threshold would have to be chosen on training sources and
> would not be guaranteed to land on the same coverage, which is stated
> here because it is the point at which these numbers stop being
> directly actionable.

**Figure 4(a-d)**, risk-coverage per held-out source; **Figure 4(e)**,
E-AURC with the per-source values shown beside the mean.

> **Finding 6. Confidence does rank errors, in every source.** Selective
> risk falls monotonically as coverage is reduced for all six encoders on
> all four held-out sources, and every model's E-AURC is clear of its own
> oracle. Withholding the least-confident tenth lowers record-wise risk
> by 0.009 to 0.014, and the least-confident fifth by 0.019 to 0.031
> Confidence is informative
> under transfer; it is not merely tracking source difficulty.

> **Finding 7. But reliability is source-dependent in the same way
> discrimination is, and more strongly than it is architecture-dependent.**
> E-AURC spans **0.130 to 0.161** across the six architectures and
> **0.104 to 0.202** across the twenty-four model-by-source folds. The
> ordering of sources is not the ordering of discrimination either:
> ResNet1D-Lite ranks errors best on Ningbo (0.104) and worst on Chapman
> (0.177), while DS-CNN is worst overall yet reaches 0.124 on Ningbo.
>
> The practical consequence is that an abstention budget calibrated on
> one source does not transfer as a reliability guarantee to another,
> for the same reason an accuracy estimate does not. **Figure 4(e).**

> **Finding 9. An abstention threshold fixed before deployment lands
> close to its target on an unseen source.** The risk-coverage curves and
> E-AURC are statements about *ordering*. A deployment cannot use an
> ordering; it has to fix a number in advance. So the confidence quantile
> retaining 90% of the training-side validation split was computed, then
> applied unchanged to the held-out source -- no labels, predictions or
> recalibration from the target.
>
> Achieved coverage spans **0.866 to 0.916** against the 0.90 target:
> mean absolute error **0.012**, worst case 0.034. At a 0.80 target,
> 0.752 to 0.820, mean error 0.018, worst 0.048. By source the mean
> error is 0.010 for PTB-XL, Chapman and Ningbo, and 0.030 for Georgia.
>
> So the operational form of selective prediction survives the source
> change, which does not follow from the curves alone and is the
> positive result of this section. The caveat is that the error is not
> negligible at a clinical scale: a service planning for a 10% referral
> rate should expect roughly 8.4% to 11.6%, and should monitor the
> realised rate rather than assume it. **Figure 4(f).**

> **Finding 8. Degradation is compounding: where discrimination falls,
> the confidence ordering falls with it.** Restricted to the three
> structural quality indicators, E-AURC is higher in the impaired
> stratum than the clean one for all three, with intervals clear of
> zero: flat/clipped lead **+0.038** (+0.028, +0.049), RR implausibility
> **+0.034** (+0.024, +0.044), QRS failure **+0.031** (+0.021, +0.042);
> 1000 replicates, seed 0.
>
> This is the unwelcome direction. Abstaining on low-confidence records
> is *least* dependable exactly on the records where discrimination has
> already dropped, so selective prediction cannot be relied on to absorb
> poor signal quality. A deployment that wants to defer on bad
> recordings should gate on the measured quality indicator directly
> rather than trusting the model to be suitably unconfident.
> The flat/clipped contrast rests on Ningbo alone, so that row in
> particular should be read as indicative.

---

## Limitations

0. **The percentile bootstrap is mildly biased for E-AURC in principle,
   though not visibly here.** Resampling records with replacement
   introduces duplicates, which changes the tie structure a confidence
   ranking depends on, and E-AURC is a non-linear functional of that
   ranking. In an earlier analysis comparing precisions this bias was
   large enough to put a point estimate outside its own 1000-replicate
   interval. In the float32 results reported here it is not detectable:
   all six E-AURC point estimates fall inside their own intervals
   (Figure 4(e)). The caution is recorded because the mechanism is real
   and would matter to anyone bootstrapping E-AURC on coarser scores;
   intervals are drawn as absolute endpoints so that a point falling
   outside its interval would be shown rather than hidden.

0. **Four databases, but about three annotation domains.** Chapman and
   Ningbo come from the same group (Zheng et al., sharing first and
   senior authors) and their diagnosis prevalence vectors are nearly
   identical: L1 distance **0.196**, against 0.658-0.676 for every other
   pair involving them and ~1.6 for every PTB-XL pair. SB is 0.450 and
   0.438, NSR 0.211 and 0.218, STach 0.182 and 0.196. Different
   hospitals, but one labelling convention.

   Three consequences, none of which invalidate the decomposition but
   all of which narrow what it licenses. First, "four held-out sources"
   overstates the domain diversity actually sampled, so the source-level
   inference in Findings 1c and 1d rests on roughly three independent
   points rather than four. Second, when Chapman or Ningbo is held out
   its training pool contains a near-twin, and when PTB-XL is held out
   it does not -- so the label-shift correlation in Finding 1d partly
   measures whether a source has a twin in training, which is a real
   mechanism but a narrower claim than "distribution shift costs
   performance". Third, the peer-median baseline in Finding 1e is a
   median over three sources of which two share a convention, so the
   consensus it compares against is weighted toward the Zheng
   convention; that is one reason seven of eleven drift flags land on
   PTB-XL, and the flags should be read as deviation from that
   consensus rather than from a neutral standard.

   The observation is worth reporting in its own right: a four-database
   public-ECG benchmark of this shape samples fewer independent
   annotation domains than its source count suggests.

0. **Four source domains, not a sample of them.** The intervals throughout
   are record-level bootstraps: they quantify sampling variation of records
   *within* these four sources. They say nothing about variation across the
   population of clinical sources, because four domains cannot estimate that.
   Read every result as generalisation to *these four* held-out sources, and
   treat the spread across the columns of Figure 2 -- not the interval on the
   mean -- as the honest measure of source dependence.

1. **"Source" is not a single variable.** Holding out a dataset varies
   acquisition hardware, population and labelling protocol at once.
   Finding 2 shows the gap is not signal quality; it cannot say which of
   the remaining three it is.
2. **Prevalence matching equalises each diagnosis, not the joint
   label distribution.** For every class the two strata are sampled to a
   common n+ and n-, so the per-diagnosis confound is removed. What
   remains free is co-occurrence: a record carrying three diagnoses and
   one carrying one are not distinguished, so the strata can still
   differ in multi-label structure even with every marginal matched.
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
   implies across four independently annotated sources.

---

## What is measured, and what is not

| | status |
|---|---|
| RQ2a quality audit, 77,333 records at 500 Hz | measured |
| RQ1, 24 LOSO runs at 100 Hz | measured |
| RQ2b, per-diagnosis matched, 1000 replicates | measured |
| RQ2c, quality-stratified E-AURC, 1000 replicates | measured |
| Energy per inference | **not done** - no calibrated instrumentation |
| Preprocessing latency | **not done** - out of scope |
| QAT | **not attempted** |

---

## Main figures

**Figure 1 caption.** Overview of the study design and research questions.
Four harmonised clinical ECG sources were used. Signal-quality indicators
were measured on the native 500 Hz recordings before decimation, while all
models operated on anti-aliased 100 Hz inputs. RQ1 evaluates cross-source
diagnostic generalisation using leave-one-source-out testing of six compact
encoders. RQ2 uses those held-out predictions together with the source-specific
quality indicators to study reliability under source and signal-quality shift,
through descriptive characterisation, prevalence-matched quality-performance
association, and selective reliability. RQ3 evaluates whether the externally
assessed models rank their own errors informatively on sources they have
never seen. In short: how much does the source matter, what about it, and
can the model tell?

| | file | answers |
|---|---|---|
| 1 | `figures/figure1_design` | the experimental logic, three RQs |
| 2 | `figures/figure2_generalisation` | **RQ1** AUPRC by held-out source (a), mean with bootstrap CI (b), and the source/architecture/interaction variance decomposition (c) |
| 3 | `figures/figure3_diagnosis_transfer` | **RQ2** transfer at the diagnosis level: average precision (a) and AUROC (b) for all 13 x 4 cells, which disagree on the hardest source, and the dependence-aware test of prevalence shift against discrimination (c), which is null |
| 4 | `figures/figure4_reliability` | **RQ3** does confidence identify unreliable predictions? Risk-coverage per held-out source (a-d); E-AURC (e) with per-source values beside the mean, showing reliability varies more across sources than architectures; and (f) the coverage a 0.90 abstention threshold actually achieves when fixed on training-side validation and carried to the unseen source |

## Main tables

| | file | answers |
|---|---|---|
| I | `tables/table1_datasets` | what data, from which institutions, under whose annotation protocol -- including that Chapman and Ningbo share one |
| II | `tables/table2_architectures` | what models |
| III | `tables/table3_quality_association` | **RQ2b** do the source differences matter diagnostically? |
| IV | `tables/table4_transfer_gap` | **RQ1** is there a measured transfer penalty, or only heterogeneity? External versus target-represented models on shared test records, matched training budget |

Tables I and II define the experimental objects. Table III is the one result
that does not compress into a figure: model x indicator estimates with intervals.
Everything spatial or comparative is a figure, so no result appears twice.

Two tables were deliberately removed rather than demoted. The leave-one-
source-out discrimination table is gone entirely: Figure 2 carries both its
halves, the per-source values in the heatmap and the mean with its interval
in the dot plot, so reprinting 24 numbers underneath would be a second container
for evidence the reader has just seen.

