"""Append the related-work references to reference.bib.

Entries are recorded as supplied. Where a field was not supplied it is
left out and flagged rather than guessed: an invented author list or a
plausible-looking DOI is worse than a visible gap, because the gap gets
fixed and the invention gets cited.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(os.path.dirname(HERE), "reference.bib")

NEW = r"""

% ======================================================================
% RELATED WORK. Added 2026-10-13 from a supplied reading list.
%
% VERIFICATION STATUS: these were NOT checked against Crossref or the
% publisher record. Venue, volume and page details are as supplied.
% Three entries arrived without an author list and carry a FIXME; they
% cannot be submitted in this state. No DOI is recorded for an entry
% whose DOI string was not supplied -- a fabricated identifier resolves
% to the wrong paper, which is worse than none.
% ======================================================================

% ------------------------------------- MCU benchmarking and inference
@inproceedings{Banbury2021MLPerfTiny,
  author    = {Banbury, Colby and Reddi, Vijay Janapa and Torelli, Peter
               and Holleman, Jeremy and Jeffries, Nat and Kiraly, Csaba
               and Montino, Pietro and Kanter, David and Ahmed, Sebastian
               and Pau, Danilo and Thakker, Urmish and Torrini, Antonio
               and Warden, Peter and Cordaro, Jay and Di Guglielmo,
               Giuseppe and Duarte, Javier and Gibellini, Stephen and
               Parekh, Videet and Tran, Honson and Tran, Nhan and
               Wenxu, Niu and Xuesong, Xu},
  title     = {{MLPerf} Tiny Benchmark},
  booktitle = {Advances in Neural Information Processing Systems,
               Datasets and Benchmarks Track},
  year      = {2021},
  note      = {arXiv:2106.07597}
}

@inproceedings{Lin2021MCUNetV2,
  author    = {Lin, Ji and Chen, Wei-Ming and Cai, Han and Gan, Chuang
               and Han, Song},
  title     = {{MCUNetV2}: Memory-Efficient Patch-based Inference for Tiny
               Deep Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2021},
  note      = {arXiv:2110.15352}
}

% ------------------------------------ PTB-XL benchmarking and reuse
@article{Bickmann2024Benchmarking,
  author  = {Bickmann, Lucas and Plagwitz, Lucas and Varghese, Julian},
  title   = {Benchmarking Approaches: Time Series Versus Feature-Based
             Machine Learning in {ECG} Analysis on the {PTB-XL} Dataset},
  journal = {Studies in Health Technology and Informatics},
  year    = {2024},
  volume  = {316},
  pages   = {589--593}
}

% A preprint, not peer reviewed. Cite as such or replace with the
% published version if one has since appeared.
@misc{Gitau2025ReproducePTBXL,
  author       = {Gitau, A. and others},
  title        = {[Re] Deep Learning for {ECG} Analysis: Benchmarks and
                  Insights from {PTB-XL}},
  year         = {2025},
  howpublished = {medRxiv preprint},
  note         = {FIXME: complete the author list before submission}
}

% ----------------------------------------- ECG TinyML and edge devices
@article{Kim2023TinyMLECG,
  author  = {Kim, Eunchan and others},
  title   = {{TinyML}-Based Classification in an {ECG} Monitoring Embedded
             System},
  journal = {Computers, Materials \& Continua},
  year    = {2023},
  volume  = {75},
  number  = {1},
  pages   = {1751--1764},
  note    = {FIXME: complete the author list before submission}
}

@inproceedings{ATC2024TinyMLECG,
  title     = {{TinyML}-Based System for Cost-Effective {ECG} Rhythm
               Classification on Embedded Device},
  booktitle = {International Conference on Advanced Technologies for
               Communications (ATC)},
  year      = {2024},
  note      = {FIXME: author list not supplied. Conference held October
               2024; IEEE Xplore record added March 2025}
}

@article{Akbari2025AFEdge,
  author  = {Akbari, H. and others},
  title   = {Atrial Fibrillation Detection on the Embedded Edge:
             Energy-Efficient Inference on a Low-Power Microcontroller},
  journal = {Sensors},
  year    = {2025},
  volume  = {25},
  number  = {21},
  pages   = {6601},
  note    = {FIXME: complete the author list before submission}
}

@article{Sensors2025ECGAnomaly,
  title   = {Reliable {ECG} Anomaly Detection on Edge Devices for Internet
             of Medical Things Applications},
  journal = {Sensors},
  year    = {2025},
  volume  = {25},
  number  = {8},
  pages   = {2496},
  note    = {FIXME: author list not supplied}
}

@article{TinyMLCardio2025Survey,
  title   = {{TinyML} and Edge Intelligence Applications in Cardiovascular
             Disease: A Survey},
  journal = {Computers in Biology and Medicine},
  year    = {2025},
  volume  = {186},
  pages   = {109653},
  note    = {FIXME: author list not supplied. A review, not an original
             deployment study}
}
"""

s = io.open(p, encoding="utf-8").read()
for key in ("Banbury2021MLPerfTiny", "Lin2021MCUNetV2"):
    if key in s:
        raise SystemExit("already present: " + key)
io.open(p, "w", encoding="utf-8").write(s.rstrip() + NEW)
print("appended 9 entries")
print("total entries: %d" % io.open(p, encoding="utf-8").read().count("\n@"))
