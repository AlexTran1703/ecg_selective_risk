r"""Build the submission bundle: cover letter, manuscript, and the merge.

    .\.venv\Scripts\python.exe scripts\05_build_submission.py

Runs pdflatex twice on each source -- the manuscript's manual bibliography
needs the second pass to resolve \cite against the .aux -- then concatenates
cover_letter.pdf ahead of letter.pdf into jacc_letter.pdf.

The journal wants the cover letter and the manuscript as separate uploads, so
the two component PDFs are kept; jacc_letter.pdf is the single file to read or
send when one document is more convenient.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pypdf import PdfWriter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AUX = (".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".synctex.gz", ".blg")


def build(stem: str) -> Path:
    for _ in range(2):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"],
            cwd=RESULTS, capture_output=True, text=True)
    pdf = RESULTS / f"{stem}.pdf"
    log = (RESULTS / f"{stem}.log").read_text(encoding="utf-8",
                                              errors="ignore")
    errs = [ln for ln in log.splitlines() if ln.startswith("! ")]
    undef = [ln for ln in log.splitlines() if "undefined" in ln.lower()]
    if errs or not pdf.exists():
        print(f"  {stem}: FAILED")
        print("\n".join(errs[:5]) or r.stdout[-800:])
        sys.exit(1)
    print(f"  {stem}.pdf  ok  ({len(undef)} undefined references)")
    return pdf


def main() -> None:
    print("building:")
    cover = build("cover_letter")
    letter = build("letter")

    out = RESULTS / "jacc_letter.pdf"
    w = PdfWriter()
    for pdf in (cover, letter):
        w.append(str(pdf))
    with out.open("wb") as fh:
        w.write(fh)
    w.close()

    for stem in ("cover_letter", "letter"):
        for ext in AUX:
            (RESULTS / f"{stem}{ext}").unlink(missing_ok=True)

    from pypdf import PdfReader
    n = [len(PdfReader(str(p)).pages) for p in (cover, letter, out)]
    print(f"\nmerged: {n[0]} + {n[1]} = {n[2]} pages -> "
          f"{out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
