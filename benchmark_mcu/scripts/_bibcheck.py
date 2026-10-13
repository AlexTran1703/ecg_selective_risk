"""Audit reference.bib: duplicate keys, near-duplicate titles, orphans.

An orphan is an entry nothing in the project cites. It is not an error
-- the manuscript cites things the code does not -- but an entry that
no text anywhere mentions is usually one that was added speculatively
and will be submitted uncited.
"""
import io
import os
import re
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
bib = io.open(os.path.join(ROOT, "reference.bib"), encoding="utf-8").read()

entries = re.findall(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", bib,
                     re.S | re.I)
print("entries: %d" % len(entries))

# --- duplicate keys
keys = [k.strip() for _, k, _ in entries]
dup = {k for k in keys if keys.count(k) > 1}
print("\nduplicate keys: %s" % (sorted(dup) if dup else "none"))


def title_of(body):
    m = re.search(r"title\s*=\s*\{(.*?)\}\s*,?\s*\n", body, re.S)
    if not m:
        return ""
    t = re.sub(r"[{}\\]", "", m.group(1))
    return re.sub(r"\s+", " ", t).strip().lower()


def year_of(body):
    m = re.search(r"year\s*=\s*\{?(\d{4})", body)
    return m.group(1) if m else "----"


titles = [(k.strip(), title_of(b), year_of(b)) for _, k, b in entries]

# --- near-duplicate titles
print("\nnear-duplicate titles (ratio > 0.70):")
hits = 0
for i in range(len(titles)):
    for j in range(i + 1, len(titles)):
        a, b = titles[i][1], titles[j][1]
        if not a or not b:
            continue
        r = SequenceMatcher(None, a, b).ratio()
        if r > 0.70:
            hits += 1
            print("  %.2f  %s (%s)" % (r, titles[i][0], titles[i][2]))
            print("        %s (%s)" % (titles[j][0], titles[j][2]))
            print("        A: %s" % titles[i][1][:78])
            print("        B: %s" % titles[j][1][:78])
if not hits:
    print("  none")

# --- which keys are mentioned anywhere in the project
text = ""
for d, _, fs in os.walk(ROOT):
    if any(p in d for p in (".git", "results", "__pycache__", "st_ai_output")):
        continue
    for f in fs:
        if f.endswith((".md", ".py", ".tex")) and f != "reference.bib":
            try:
                text += io.open(os.path.join(d, f), encoding="utf-8",
                                errors="ignore").read()
            except OSError:
                pass

cited, orphan = [], []
for k, t, y in titles:
    (cited if k in text else orphan).append((k, y))
print("\ncited somewhere in the project: %d" % len(cited))
for k, y in cited:
    print("   %-28s %s" % (k, y))
print("\nnot cited anywhere yet: %d" % len(orphan))
for k, y in orphan:
    print("   %-28s %s" % (k, y))

# --- entries still carrying a FIXME
print("\nentries with FIXME:")
for _, k, b in entries:
    if "FIXME" in b:
        print("   %s" % k.strip())
