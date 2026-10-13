"""Figure 1, panel C: put each research-question bullet on one line.

The bullets were hard-wrapped mid-phrase, so the first line of each
stopped well short of the box edge and the second carried two or three
words. Shortened so every bullet is a single line that fills the
column, which also lets the boxes close up.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "r_fig1.py")
s = io.open(p, encoding="utf-8").read()
NL = chr(92) + "n"
n = 0


def rep(a, b):
    global s, n
    assert a in s, "MISSING: " + repr(a[:70])
    s = s.replace(a, b, 1)
    n += 1


rep('["external discrimination across' + NL + 'four held-out sources",',
    '["external discrimination, four held-out sources",')
rep('"class-specific AP and paired' + NL + 'architecture comparisons"],',
    '"class-specific AP and paired contrasts"],')
rep('["compare scaling strategies under' + NL + 'deployment budgets",',
    '["scaling strategy at matched budgets",')
rep('"assess prediction, quantisation' + NL + 'effects, Flash, SRAM, latency"],',
    '"quantisation, Flash, SRAM and latency"],')
rep('["identify favorable deployed' + NL + 'configurations",',
    '["which deployed configurations are favourable",')
rep('"trade-offs among external' + NL + 'macro-AP, latency and memory"],',
    '"macro-AP against latency and memory"],')

# one line each now, so the entries sit closer and the boxes shrink
rep("        ey = 0.586 - k * 0.278", "        ey = 0.622 - k * 0.268")
rep("        _rr(fig, cx_ + 0.016, ey, 0.244, 0.244, fc, ec=ec, lw=1.0, r=0.009,",
    "        _rr(fig, cx_ + 0.016, ey, 0.244, 0.216, fc, ec=ec, lw=1.0, r=0.009,")
rep('        _t(fig, cx_ + 0.030, ey + 0.198, f"{tag} {title}", size=8.0,',
    '        _t(fig, cx_ + 0.030, ey + 0.170, f"{tag} {title}", size=8.0,')
rep("            by = ey + 0.132 - j * 0.064", "            by = ey + 0.104 - j * 0.052")

io.open(p, "w", encoding="utf-8").write(s)
print("applied %d edits" % n)
