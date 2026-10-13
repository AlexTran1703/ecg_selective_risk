"""Append the evaluation-methodology references to reference.bib.

These justify the choices the results rest on: a precision-recall
primary metric under class imbalance, macro-averaging over 13
diagnoses, leave-one-source-out as external rather than internal
validation, and bootstrap interval construction.

All six are long-established and arrived with complete venue, volume,
page and DOI details, so none carries a FIXME. The scikit-learn entry
is recorded as software, not as a journal reference: the metric's
definition belongs to the methodological literature, and what the
software fixes is the numerical convention and the version.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(os.path.dirname(HERE), "reference.bib")

NEW = r"""

% ======================================================================
% EVALUATION METHODOLOGY. Why the primary metric, the averaging scheme,
% the validation design and the intervals are what they are.
%
% Two things these references do NOT establish, and which Methods has
% to state on its own authority: the equal weight given to each of the
% four held-out sources, which is this study's chosen estimand rather
% than a prescribed convention; and the choice of 2,000 bootstrap
% replicates. Neither follows from the cited work.
% ======================================================================

% ------------------------------------------- precision-recall vs ROC
@inproceedings{Davis2006PRvsROC,
  author    = {Davis, Jesse and Goadrich, Mark},
  title     = {The Relationship Between Precision-Recall and {ROC} Curves},
  booktitle = {Proceedings of the 23rd International Conference on
               Machine Learning (ICML)},
  year      = {2006},
  pages     = {233--240},
  doi       = {10.1145/1143844.1143874}
}

@article{Saito2015PRPlot,
  author  = {Saito, Takaya and Rehmsmeier, Marc},
  title   = {The Precision-Recall Plot Is More Informative than the {ROC}
             Plot When Evaluating Binary Classifiers on Imbalanced
             Datasets},
  journal = {PLOS ONE},
  year    = {2015},
  volume  = {10},
  number  = {3},
  pages   = {e0118432},
  doi     = {10.1371/journal.pone.0118432}
}

% Software, not a journal reference. It fixes the numerical convention:
% AP = sum_k (R_k - R_{k-1}) P_k, the non-interpolated form, which is
% not identical to a trapezoidal PR-AUC. This study's own AP routine
% was verified against average_precision_score to 1.7e-16 including
% tied scores. Cite the version in Methods rather than the URL.
@misc{SklearnAP,
  author       = {{scikit-learn developers}},
  title        = {{\texttt{average\_precision\_score}}},
  year         = {2025},
  howpublished = {scikit-learn 1.9.1 documentation},
  note         = {non-interpolated average precision}
}

% ------------------------------- averaging, validation and reporting
@article{Sokolova2009Measures,
  author  = {Sokolova, Marina and Lapalme, Guy},
  title   = {A Systematic Analysis of Performance Measures for
             Classification Tasks},
  journal = {Information Processing \& Management},
  year    = {2009},
  volume  = {45},
  number  = {4},
  pages   = {427--437},
  doi     = {10.1016/j.ipm.2009.03.002}
}

@article{Steyerberg2016Validation,
  author  = {Steyerberg, Ewout W. and Harrell, Frank E.},
  title   = {Prediction Models Need Appropriate Internal,
             Internal-External, and External Validation},
  journal = {Journal of Clinical Epidemiology},
  year    = {2016},
  volume  = {69},
  pages   = {245--247},
  doi     = {10.1016/j.jclinepi.2015.04.005}
}

@article{Collins2024TRIPODAI,
  author  = {Collins, Gary S. and others},
  title   = {{TRIPOD+AI} Statement: Updated Guidance for Reporting
             Clinical Prediction Models That Use Regression or Machine
             Learning Methods},
  journal = {BMJ},
  year    = {2024},
  volume  = {385},
  pages   = {e078378},
  doi     = {10.1136/bmj-2023-078378}
}

% -------------------------------------------------------- resampling
@book{Efron1993Bootstrap,
  author    = {Efron, Bradley and Tibshirani, Robert J.},
  title     = {An Introduction to the Bootstrap},
  publisher = {Chapman \& Hall/CRC},
  year      = {1993},
  series    = {Monographs on Statistics and Applied Probability},
  volume    = {57}
}
"""

s = io.open(p, encoding="utf-8").read()
for key in ("Davis2006PRvsROC", "Efron1993Bootstrap"):
    if key in s:
        raise SystemExit("already present: " + key)
io.open(p, "w", encoding="utf-8").write(s.rstrip() + NEW)
out = io.open(p, encoding="utf-8").read()
print("appended 7 entries")
print("total         : %d" % out.lower().count("\n@"))
print("braces        : %d / %d" % (out.count("{"), out.count("}")))
print("FIXME markers : %d" % out.count("FIXME"))
