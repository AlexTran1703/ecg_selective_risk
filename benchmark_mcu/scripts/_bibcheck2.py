"""Does every architecture cite the right paper, and is anything orphaned?

The project refers to papers in prose ("Sandler et al. 2018"), not by
bibtex key, so a key-string search finds nothing and a search of the
scripts that *wrote* the bib finds everything. Both are useless. This
matches on author surname plus year instead, and excludes the bib
maintenance scripts.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SKIP = ("_addrefs.py", "_adddatarefs.py", "_bibcheck.py", "_bibcheck2.py")

bib = io.open(os.path.join(ROOT, "reference.bib"), encoding="utf-8").read()
entries = re.findall(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", bib, re.S | re.I)


def field(body, name):
    m = re.search(name + r"\s*=\s*[{\"](.*?)[}\"]\s*,?\s*\n", body, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


# gather project prose
text = ""
for d, _, fs in os.walk(ROOT):
    if any(p in d for p in (".git", "results", "__pycache__", "st_ai")):
        continue
    for f in fs:
        if f.endswith((".md", ".py")) and f not in SKIP:
            try:
                text += io.open(os.path.join(d, f), encoding="utf-8",
                                errors="ignore").read()
            except OSError:
                pass

print("=== orphan check (surname + year in project prose) ===")
orphans = []
for _, key, body in entries:
    key = key.strip()
    auth = field(body, "author")
    year = field(body, "year")
    surname = auth.split(",")[0].strip() if auth else ""
    surname = surname.split(" and ")[0].strip()
    hit = bool(surname) and re.search(
        re.escape(surname) + r"[^.\n]{0,40}" + re.escape(year), text)
    if not hit and surname:
        hit = bool(re.search(re.escape(surname) + r"\s+et al", text))
    if not hit:
        orphans.append((key, surname or "(no author)", year))
    print("  %-28s %-14s %s" % (key, surname[:14], "cited" if hit else "--"))

print("\nnot referenced in any project text: %d" % len(orphans))
for k, s, y in orphans:
    print("   %-28s %s %s" % (k, s, y))

# --- architecture to citation map, read from reference.py docstrings
print("\n=== architecture citations ===")
src = io.open(os.path.join(ROOT, "src", "mcubench", "reference.py"),
              encoding="utf-8").read()
EXPECT = {"FCN1D": "Wang", "ResNet1D": "He", "TCN": "Bai",
          "MobileNetV2": "Sandler", "ShuffleNetV2": "Ma",
          "GhostNet": "Han", "MobileNetV3": "Howard",
          "FasterNet": "Chen", "RepViT": "Wang", "MobileNetV4": "Qin"}
for cls, want in EXPECT.items():
    m = re.search(r"class %s\(nn\.Module\):\s*\n\s*(?:r?\"\"\")(.*?)\"\"\""
                  % cls, src, re.S)
    doc = re.sub(r"\s+", " ", m.group(1))[:60] if m else "(no docstring)"
    ok = want.lower() in doc.lower()
    print("  %-14s %-8s %s  %s" % (cls, want, "OK " if ok else "?? ", doc))
