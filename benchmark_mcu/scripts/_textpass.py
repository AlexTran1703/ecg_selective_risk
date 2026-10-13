"""Figures 2-7: larger text, black text, and no descriptive footnotes.

Three passes over the asset builder.

1. Every fontsize and labelsize is scaled by 1.15. These figures are
   printed at journal column width, where 5.8 pt is at the edge of
   legibility.
2. Secondary ink (grey) is promoted to primary ink for anything that is
   text: labels, ticks and legend entries. Grey was being used to make
   text recede, which at this size makes it hard to read rather than
   quiet. Colour that carries meaning -- the red device limits, the
   blue/orange method cues -- is left alone.
3. The explanatory footnotes under Figures 3 and 4 are removed. They
   describe the method, which belongs in the caption; a figure should
   not have to carry its own prose.

Figure 1 is untouched: it lives in r_fig1.py and its text was sized
against a dense three-column layout.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "r4_assets.py")
s = io.open(p, encoding="utf-8").read()

# --- 1. remove the two in-figure footnotes first, before sizes move
F3 = '''    fig.text(0.5, 0.018, "coloured points are source-specific estimates, "
             "not source-specific intervals;  black bar = patient-cluster "
             f"paired bootstrap 95% CI ({REPS} replicates)",
             fontsize=6.0, color=INK2, ha="center")
'''
F4 = '''    fig.text(0.5, 0.020,
             "paired patient-cluster bootstrap, "
             f"{REPS} replicates, equal weight per held-out source;   "
             "* weight budget never binds (activation-bound)",
             fontsize=6.1, color=INK2, ha="center")
'''
F4B = '''    fig.text(0.5, 0.020,
             "paired patient-cluster bootstrap, "
             f"{REPS} replicates, equal weight per held-out source;   "
             "* activation memory, not the weight budget, "
             "determines the configuration",
             fontsize=6.1, color=INK2, ha="center")
'''
removed = 0
for block in (F3, F4, F4B):
    if block in s:
        s = s.replace(block, "", 1)
        removed += 1
print("footnotes removed: %d" % removed)

# the figures reserved vertical space for those footnotes
s = s.replace("fig.subplots_adjust(left=0.215, right=0.985, top=1 - 0.30 / h,\n"
              "                        bottom=0.95 / h)",
              "fig.subplots_adjust(left=0.225, right=0.985, top=1 - 0.30 / h,\n"
              "                        bottom=0.72 / h)")
s = s.replace("fig.subplots_adjust(left=0.175, right=0.985, top=1 - 0.42 / h,\n"
              "                        bottom=0.92 / h, wspace=0.07)",
              "fig.subplots_adjust(left=0.185, right=0.985, top=1 - 0.42 / h,\n"
              "                        bottom=0.62 / h, wspace=0.07)")

# --- 2. grey text becomes black
before = s.count("INK2")
s = s.replace("labelcolor=INK2", "labelcolor=INK")
s = s.replace("colors=INK2", "colors=INK")
s = s.replace("color=INK2", "color=INK")
s = s.replace("colour=INK2", "colour=INK")
print("INK2 references: %d -> %d" % (before, s.count("INK2")))


# --- 3. scale every type size by 1.15
def bump(mo):
    key, val = mo.group(1), float(mo.group(2))
    return "%s=%.1f" % (key, round(val * 1.15, 1))


s, n = re.subn(r"\b(fontsize|labelsize)=(\d+(?:\.\d+)?)", bump, s)
print("type sizes scaled: %d" % n)

io.open(p, "w", encoding="utf-8").write(s)
print("written")
