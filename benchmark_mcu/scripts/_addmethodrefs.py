"""References that ground the two choices Efron and Tibshirani do not.

Two gaps were left open: how many bootstrap replicates, and why each
held-out source carries equal weight. Both have literature.

  replicate count    Hall (1986) is the reference for choosing B;
                     Carpenter and Bithell (2000) give the applied
                     clinical recommendation for percentile intervals.
  clustered draws    Field and Welsh (2007) is the cluster bootstrap,
                     which is what resampling patients rather than
                     records actually is.
  per-source summary Debray et al. (2015) and Riley et al. (2016)
                     establish that in multi-dataset external
                     validation the cluster is the unit of analysis and
                     that how clusters are combined changes the
                     estimand; Riley et al. (2010) covers the weighting
                     question directly.

What these still do not do is prescribe *equal* weight. They establish
that the choice is consequential and must be stated, which is weaker
and is what Methods should claim.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(os.path.dirname(HERE), "reference.bib")

NEW = r"""

% ======================================================================
% RESAMPLING DESIGN AND MULTI-SOURCE AGGREGATION.
%
% These replace two bare assertions. Previously the replicate count and
% the equal-source weighting rested on nothing; now the first is
% grounded and the second is at least framed.
%
% VERIFICATION: volumes and page ranges are as commonly cited. A DOI is
% recorded only where it is certain; the three marked CHECK-DOI need
% confirming against the publisher record before submission.
% ======================================================================

% ------------------------------------- how many bootstrap replicates
% The reference for choosing B. Percentile intervals need far more
% replicates than a standard error does, which is why 2,000 rather
% than the 50-200 sufficient for an SE.
@article{Hall1986BootstrapN,
  author  = {Hall, Peter},
  title   = {On the Number of Bootstrap Simulations Required to Construct
             a Confidence Interval},
  journal = {The Annals of Statistics},
  year    = {1986},
  volume  = {14},
  number  = {4},
  pages   = {1453--1462},
  note    = {CHECK-DOI}
}

% The applied recommendation, in a clinical-statistics venue: at least
% 1,000-2,000 replicates for percentile intervals.
@article{Carpenter2000BootstrapCI,
  author  = {Carpenter, James and Bithell, John},
  title   = {Bootstrap Confidence Intervals: When, Which, What? A
             Practical Guide for Medical Statisticians},
  journal = {Statistics in Medicine},
  year    = {2000},
  volume  = {19},
  number  = {9},
  pages   = {1141--1164},
  note    = {CHECK-DOI}
}

% ------------------------------------------- resampling the cluster
% Resampling patients rather than records is the cluster bootstrap.
% This matters for PTB-XL, where 20,487 records come from 17,928
% patients; treating correlated records as independent narrows the
% interval.
@article{Field2007ClusterBootstrap,
  author  = {Field, Christopher A. and Welsh, A. H.},
  title   = {Bootstrapping Clustered Data},
  journal = {Journal of the Royal Statistical Society: Series B},
  year    = {2007},
  volume  = {69},
  number  = {3},
  pages   = {369--390},
  note    = {CHECK-DOI}
}

% ------------------------------ combining several validation datasets
@article{Debray2015Framework,
  author  = {Debray, Thomas P. A. and Vergouwe, Yvonne and Koffijberg,
             Hendrik and Nieboer, Daan and Steyerberg, Ewout W. and
             Moons, Karel G. M.},
  title   = {A New Framework to Enhance the Interpretation of External
             Validation Studies of Clinical Prediction Models},
  journal = {Journal of Clinical Epidemiology},
  year    = {2015},
  volume  = {68},
  number  = {3},
  pages   = {279--289},
  doi     = {10.1016/j.jclinepi.2014.06.018}
}

@article{Riley2016ExternalValidation,
  author  = {Riley, Richard D. and Ensor, Joie and Snell, Kym I. E. and
             Debray, Thomas P. A. and Altman, Douglas G. and Moons,
             Karel G. M. and Collins, Gary S.},
  title   = {External Validation of Clinical Prediction Models Using Big
             Datasets from e-Health Records or {IPD} Meta-Analysis:
             Opportunities and Challenges},
  journal = {BMJ},
  year    = {2016},
  volume  = {353},
  pages   = {i3140},
  doi     = {10.1136/bmj.i3140}
}

% Weighting across datasets: the question equal-source averaging
% answers by choice rather than by derivation.
@article{Riley2010IPDMeta,
  author  = {Riley, Richard D. and Lambert, Paul C. and Abo-Zaid, Ghada},
  title   = {Meta-Analysis of Individual Participant Data: Rationale,
             Conduct, and Reporting},
  journal = {BMJ},
  year    = {2010},
  volume  = {340},
  pages   = {c221},
  doi     = {10.1136/bmj.c221}
}
"""

s = io.open(p, encoding="utf-8").read()
for key in ("Hall1986BootstrapN", "Riley2010IPDMeta"):
    if key in s:
        raise SystemExit("already present: " + key)
io.open(p, "w", encoding="utf-8").write(s.rstrip() + NEW)
out = io.open(p, encoding="utf-8").read()
print("appended 6 entries")
print("total         : %d" % out.lower().count("\n@"))
print("braces        : %d / %d" % (out.count("{"), out.count("}")))
print("FIXME         : %d" % out.count("FIXME"))
print("CHECK-DOI     : %d" % out.count("CHECK-DOI"))
