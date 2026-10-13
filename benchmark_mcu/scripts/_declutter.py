"""Figure 7: drop the frontier line. Figure 6: drop the point labels.

Both are the same decision -- an annotation that competed with the data
it sat on top of. In Figure 7 the black outlines already identify the
non-dominated set, so the connecting line added no information and cut
across the cloud. In Figure 6 two floating identifiers sat in the middle
of the plot while every other point relied on the legend, which is an
inconsistency as much as a clutter problem.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "r4_assets.py")
s = io.open(p, encoding="utf-8").read()
n = 0


def rep(a, b):
    global s, n
    assert a in s, "MISSING: " + repr(a[:80])
    s = s.replace(a, b, 1)
    n += 1


# --- figure 6: the two annotated extremes
rep('''        slowest = max(pts, key=lambda t: t[0]["ms_median"] / t[0]["macc"])
        biggest = max(pts, key=lambda t: t[0]["macc"])
        for r, m, cfg in {id(slowest): slowest, id(biggest): biggest}.values():
            ax.annotate(D.LABEL[m] + " " + m,
                        (r["macc"] / 1e6, r["ms_median"]),
                        textcoords="offset points", xytext=(9, 7),
                        fontsize=7.8, color=INK, zorder=7)

''', '')

# --- figure 6 docstring no longer promises annotations
rep('''    Colour is the architecture family, as in Figure 7, so a family can
    be carried between the two figures; the per-point identifiers are
    gone and the legend carries identity instead.
    """''',
    '''    Colour is the architecture family, as in Figure 7, so a family can
    be carried between the two figures; identity comes from the legend
    rather than from labels on the points, which at forty points would
    sit on top of the data they describe.
    """''')

# --- figure 7: the connecting frontier
rep('''        # The pooled frontier, identical in both panels. Connecting
        # only this panel's members would draw a line that is neither
        # the pooled frontier nor a method-specific one.
        if len(fr_all) > 1:
            ax.plot([p["ms"] for p in fr_all], [p["ap"] for p in fr_all],
                    lw=1.0, color="#8a8a84", ls=(0, (4, 2)), zorder=2)
''', '')

rep('''    fr_all = sorted([p for p in pts if (p["m"], p["cfg"]) in front],
                    key=lambda p: p["ms"])
''', '')

# --- figure 7 docstring: the outlines now carry the whole message
rep('''    One panel per scaling method so forty bubbles are not piled onto one
    pair of axes, but the Pareto front is computed **once over the
    combined set**. Fronts computed per method would be two different
    questions wearing one name: a configuration that another method
    dominates outright would still appear non-dominated within its own
    panel, and the figure would recommend it.''',
    '''    One panel per scaling method so forty bubbles are not piled onto one
    pair of axes, but the non-dominated set is computed **once over the
    combined set**. Sets computed per method would be two different
    questions wearing one name: a configuration that another method
    dominates outright would still appear non-dominated within its own
    panel, and the figure would recommend it.

    Membership is shown by outline alone. A line joining the members ran
    across the cloud without adding anything the outlines did not
    already say.''')

io.open(p, "w", encoding="utf-8").write(s)
print("applied %d edits" % n)
