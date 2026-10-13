"""Append the four source-dataset references to reference.bib.

These are the papers the cohort itself comes from, so they belong with
the data description rather than the related work. Unlike the
related-work batch, all four arrived with complete author lists and
DOIs, so none carries a FIXME.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(os.path.dirname(HERE), "reference.bib")

NEW = r"""

% ======================================================================
% SOURCE DATASETS. The four corpora the cohort is drawn from.
%
% Table 1 reports post-harmonisation counts, which are smaller than the
% released totals these papers describe: records carrying no label
% inside the 13-class space are dropped before any splitting. Cite the
% paper for the dataset and Table 1 for what this study actually used.
% ======================================================================

@article{Wagner2020PTBXL,
  author  = {Wagner, Patrick and Strodthoff, Nils and Bousseljot,
             Ralf-Dieter and Kreiseler, Dieter and Lunze, Fatima I. and
             Samek, Wojciech and Schaeffter, Tobias},
  title   = {{PTB-XL}, a large publicly available electrocardiography
             dataset},
  journal = {Scientific Data},
  year    = {2020},
  volume  = {7},
  pages   = {154},
  doi     = {10.1038/s41597-020-0495-6}
}

% The Challenge paper is the primary citable description of the Georgia
% 12-lead set; there is no separate dataset-description publication.
@article{PerezAlday2020Challenge,
  author  = {Perez Alday, Erick A. and others},
  title   = {Classification of 12-lead {ECGs}: The {PhysioNet/Computing
             in Cardiology} Challenge 2020},
  journal = {Physiological Measurement},
  year    = {2020},
  volume  = {41},
  number  = {12},
  pages   = {124003},
  doi     = {10.1088/1361-6579/abc960}
}

@article{Zheng2020Chapman,
  author  = {Zheng, Jianwei and Zhang, Jianming and Danioko, Sidy and
             Yao, Hai and Guo, Hangyuan and Rakovski, Cyril},
  title   = {A 12-lead electrocardiogram database for arrhythmia research
             covering more than 10,000 patients},
  journal = {Scientific Data},
  year    = {2020},
  volume  = {7},
  pages   = {48},
  doi     = {10.1038/s41597-020-0386-x}
}

% The Challenge 2021 release is where the Ningbo First Hospital database
% was published. Cite the PhysioNet record (10.13026/jz9p-0m02) in the
% data availability statement alongside this paper.
@inproceedings{Reyna2021Challenge,
  author    = {Reyna, Matthew A. and others},
  title     = {Will Two Do? Varying Dimensions in Electrocardiography:
               The {PhysioNet/Computing in Cardiology} Challenge 2021},
  booktitle = {Computing in Cardiology},
  year      = {2021},
  volume    = {48},
  note      = {Dataset: PhysioNet, doi:10.13026/jz9p-0m02}
}
"""

s = io.open(p, encoding="utf-8").read()
for key in ("Wagner2020PTBXL", "Zheng2020Chapman"):
    if key in s:
        raise SystemExit("already present: " + key)
io.open(p, "w", encoding="utf-8").write(s.rstrip() + NEW)
out = io.open(p, encoding="utf-8").read()
print("appended 4 dataset entries")
print("total entries : %d" % out.count("\n@"))
print("braces        : %d open / %d close" % (out.count("{"), out.count("}")))
print("FIXME markers : %d" % out.count("FIXME"))
