# Reference uncertainty and selective reliability in AI-ECG

One question:

> Does uncertainty in the **human reference** change how we should interpret
> model uncertainty and selective reliability?

AI-ECG models are scored against a reference treated as definitive, but ECG
interpretation carries genuine diagnostic uncertainty. This study asks whether
the error that survives confidence-based referral is concentrated on exactly
the labels humans were unsure about.

Two datasets supply two different kinds of reference uncertainty, and the same
model family is trained independently on each — PTB-XL is an independent
replication of the phenomenon, **not** a transfer target. Training once and
transferring would vary domain shift, label mapping and reference uncertainty
simultaneously, leaving no result attributable to any of them.

```
CODE-15%  ->  5 x ResNet1D-Lite  ->  CODE-test   ->  D = reader disagreement
PTB-XL    ->  5 x ResNet1D-Lite  ->  fold 10     ->  L = statement likelihood
```

## Design

**Model uncertainty** `U = Var_m p^(m)` — ensemble disagreement over 5
independently initialised networks.

**Decision confidence** `C = |logit(pbar) - logit(t_c)|` — distance from the
operating point, with `t_c` fitted on validation only and then frozen.

These are not interchangeable. An ensemble can be far from the threshold while
its members disagree about the probability; `U` is compared against human
uncertainty, `C` is what a deployed system would rank by.

**Selection uses confidence only.** Reference uncertainty is applied after the
retained set is fixed, because a deployed model cannot know whether two
cardiologists will disagree about a record it is about to see. Using
disagreement to choose what to keep would measure an oracle, not a referral
policy.

Risk is computed per diagnosis and macro-averaged. With six labels at 2–3%
prevalence, pooling all (record, diagnosis) pairs would be dominated by true
negatives.

## What the data actually supports

**CODE-test** — 827 ECGs, two independent cardiologists plus adjudication.
Disagreement is 0.67% of label-record pairs overall but concentrated where it
matters: 14 disputes on 1dAVb against only 28 gold positives.

**PTB-XL** — likelihood is attached to *diagnostic* statements, not *rhythm*
statements, and a stored 0 means "not specified" rather than "certain":

```
label    positives   with a specified likelihood
1dAVb          797                          797
RBBB           542                          542
LBBB           536                          536
SB             637                            0
AF            1514                           48
ST             826                            4
```

So only **1dAVb, RBBB, LBBB** can enter the PTB-XL likelihood analysis. The
zeros for SB/AF/ST are structurally missing metadata; reading them as low
certainty would manufacture a result.

## Layout

```
src/ecguq/      model, uncertainty, selective, bootstrap, data
tests/          the analysis core
scripts/        preparation, training, analysis
results/        tables/ and figures/ -- the only outputs that matter
data/           preprocessed signal cache and record index
dataset/        raw CODE and PTB-XL
archive/        the superseded beat-relation study, intact
```

## Status

Done: package skeleton, uncertainty and selective-risk core with 24 passing
tests, clustered ECG-level bootstrap, data loaders validated against both
datasets.

Not yet: ensemble training, and the analysis producing 1 figure (3 panels) and
2 tables.

## Archive

`archive/beat_relation_study/` holds the previous direction complete with
trained runs. Its findings that still constrain this study: encoder capacity
degraded cross-source confidence ranking while barely moving accuracy, which
is why the backbone here is small and fixed; and out-of-vocabulary labels
impose an irreducible selective-risk floor, which is why label spaces here are
matched explicitly rather than harmonised by projection.
