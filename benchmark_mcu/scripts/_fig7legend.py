"""Figure 7: family-colour legend as in Figure 6, no per-point labels.

Also corrects the frontier. Each panel previously connected only its own
method's members of the pooled non-dominated set, which draws a polyline
that is neither the pooled frontier nor a method-specific one. Both
panels now show the same pooled frontier, and the black outline marks
which of that panel's points lie on it -- so the dashed line is the
pooled frontier projected onto each panel, exactly as the caption says.
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


# --- the pooled frontier, drawn identically in both panels
rep('''        fr = sorted([p for p in sel if (p["m"], p["cfg"]) in front],
                    key=lambda p: p["ms"])
        if len(fr) > 1:
            ax.plot([p["ms"] for p in fr], [p["ap"] for p in fr], lw=1.0,
                    color="#8a8a84", ls=(0, (4, 2)), zorder=2)''',
    '''        # The pooled frontier, identical in both panels. Connecting
        # only this panel's members would draw a line that is neither
        # the pooled frontier nor a method-specific one.
        if len(fr_all) > 1:
            ax.plot([p["ms"] for p in fr_all], [p["ap"] for p in fr_all],
                    lw=1.0, color="#8a8a84", ls=(0, (4, 2)), zorder=2)''')

rep('''    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), facecolor=SURFACE,
                             sharey=True)''',
    '''    fr_all = sorted([p for p in pts if (p["m"], p["cfg"]) in front],
                    key=lambda p: p["ms"])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), facecolor=SURFACE,
                             sharey=True)''')

# --- drop the per-point identifiers; the legend carries identity
rep('''            ax.annotate(D.LABEL[p["m"]], (p["ms"], p["ap"]),
                        textcoords="offset points", xytext=(0, 0),
                        ha="center", va="center", fontsize=4.8,
                        color=INK, zorder=7)
''', '')

# --- legend in the Figure 6 idiom: families, then the keys
rep('''    hs = [plt.Line2D([], [], lw=0, marker=BSHAPE[b], markersize=5,
                     markerfacecolor="#8a8a84", markeredgecolor="none",
                     label=b) for b in ("Tight", "Relaxed")]
    hs.append(plt.Line2D([], [], lw=0, marker="o", markersize=7,
                         markerfacecolor="none", markeredgecolor=INK,
                         label="non-dominated (both methods pooled)"))
    for s_kb in (np.percentile([p["sram"] for p in pts], 20),
                 np.percentile([p["sram"] for p in pts], 80)):
        hs.append(plt.Line2D(
            [], [], lw=0, marker="o",
            markersize=np.sqrt(28 + 300 * (s_kb / smax)) * 0.62,
            markerfacecolor="#cfcfc8", markeredgecolor="none",
            label=f"{s_kb:.0f} KB SRAM"))
    fig.legend(handles=hs, fontsize=6.3, frameon=False, labelcolor=INK2,
               ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               columnspacing=1.5, handletextpad=0.5)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.925, bottom=0.185,
                        wspace=0.06)''',
    '''    fam = [plt.Line2D([], [], lw=0, marker="o", markersize=5.5,
                      markerfacecolor=B.MODEL_COLOR[D.ICON[m]],
                      markeredgecolor="none", label=D.LABEL[m] + " " + m)
           for m in D.ORDER if any(p["m"] == m for p in pts)]
    # Black, not grey: the keys share a legend grid with the families,
    # and one of those families is already grey.
    keys = [plt.Line2D([], [], lw=0, marker=BSHAPE[b], markersize=5.5,
                       markerfacecolor="#1a1a18", markeredgecolor="none",
                       label="budget " + b) for b in ("Tight", "Relaxed")]
    keys.append(plt.Line2D([], [], lw=0, marker="o", markersize=7.5,
                           markerfacecolor="none", markeredgecolor=INK,
                           label="non-dominated (pooled)"))
    for s_kb in (np.percentile([p["sram"] for p in pts], 20),
                 np.percentile([p["sram"] for p in pts], 80)):
        keys.append(plt.Line2D(
            [], [], lw=0, marker="o",
            markersize=np.sqrt(28 + 300 * (s_kb / smax)) * 0.62,
            markerfacecolor="#cfcfc8", markeredgecolor="none",
            label="{:.0f} KB SRAM".format(s_kb)))
    fig.legend(handles=fam + keys, fontsize=6.6, frameon=False,
               labelcolor=INK2, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.008), columnspacing=1.5,
               handletextpad=0.5)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.935, bottom=0.265,
                        wspace=0.06)''')

# --- Figure 4 footnote: name the constraint that actually binds
rep('"* weight budget never binds (activation-bound)"',
    '"* activation memory, not the weight budget, "\n'
    '             "determines the configuration"')

io.open(p, "w", encoding="utf-8").write(s)
print("applied %d edits" % n)
