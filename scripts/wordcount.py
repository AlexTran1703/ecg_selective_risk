r"""Recount the JACC word budget and write it into the title page.

    .\.venv\Scripts\python.exe scripts\wordcount.py

Counts everything after the title page -- body text, references and figure
legend -- with LaTeX markup stripped, which is the basis the journal states.
Kept as a script rather than done by hand so the declared count cannot drift
away from the manuscript, and broken out by section so it is obvious where
the budget is going when it needs trimming.
"""
import pathlib
import re

LIMIT = 1000
p = pathlib.Path(__file__).resolve().parents[1] / "results" / "letter.tex"
s = p.read_text(encoding="utf-8")


def wc(text: str) -> int:
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\cite\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", text)
    return len(re.sub(r"[{}$\\]", " ", text).split())


body = s[s.index(r"\doublespacing"):s.index(r"\end{document}")]
i_ref = body.index(r"\begin{thebibliography}")
j_ref = body.index(r"\end{thebibliography}")
i_leg = body.index("FIGURE LEGEND")

parts = [("body text", body[:i_ref] + body[j_ref:i_leg]),
         ("references", body[i_ref:j_ref]),
         ("figure legend", body[i_leg:])]
for name, chunk in parts:
    print(f"  {name:<14}{wc(chunk):>5}")
total = wc(body)
print(f"  {'-' * 19}")
print(f"  {'TOTAL':<14}{total:>5}  / {LIMIT}"
      f"   ({LIMIT - total:+d} headroom)")

# The title page may or may not declare a word count; update it if present
# rather than failing, so this stays a read-only check when it is not.
s, k = re.subn(r"(\\textbf\{Word count:\} )\d+", r"\g<1>%d" % total, s)
if k:
    p.write_text(s, encoding="utf-8")
else:
    print("  (no word-count field on the title page; nothing written)")

# The cover letter has no stated limit, but it is the other thing being
# submitted and its length is worth knowing before it goes.
cov = p.with_name("cover_letter.tex")
if cov.exists():
    t = cov.read_text(encoding="utf-8")
    t = t[t.index(r"\begin{document}"):t.index(r"\end{document}")]
    print(f"\n  {'cover letter':<14}{wc(t):>5}  (no stated limit)")
