# IOP Publishing LaTeX template

Official template for Biomedical Physics & Engineering Express, and for
every other IOP journal: IOP ships one house class rather than a
per-journal one.

    iopjournal.cls              class, v2024/01/31, LPPL 1.3c, (c) IOP 2025
    iopjournal-template.tex     skeleton manuscript
    iopjournal-guidelines.pdf   IOP's own usage notes
    figure1.pdf, orcid.pdf      assets the skeleton references

Source: https://publishingsupport.iopscience.iop.org/questions/latex-template/
(bundle `ioplatextemplate.zip`, retrieved 2026-10-06)

Verified: compiles unmodified with the local MiKTeX 25.12 / pdfTeX 4.23,
two pages, no missing packages.

## Notes that matter for this manuscript

`\documentclass[anonymous]{iopjournal}` strips author names, affiliations
and acknowledgements for double-anonymous review. Everything except the
supplementary-data section is removed by that option, so do not hide
anything load-bearing in the back matter.

Using this class is **not** required -- IOP accepts any common TeX variant
and redoes the page design in production, so there is no point chasing
the published look. What they do require: figures and tables embedded at
the point of discussion rather than collected at the end, at least 12 pt
with generous leading for the review copy, standard font families only,
and either Vancouver numerical or Harvard alphabetical references.
