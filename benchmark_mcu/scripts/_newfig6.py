"""Replace figure6 with a two-panel, vertically stacked version."""
import io
import os

F6 = '''def figure6(runs=None):
    """RQ2: is arithmetic cost a usable proxy for latency on this part?

    One panel per scaling method, stacked so the two share an x axis and
    a reader can drop a vertical line through both. If MAC count were a
    good proxy every point would lie on its panel's trend and a designer
    could stop at a FLOP count.

    Rank discordance is reported beside the correlation, per panel,
    because what a designer does with a proxy is *order* candidates, and
    a proxy can correlate strongly while still mis-ordering the pairs
    that decide a choice. Splitting by method also asks a sharper
    question than pooling would: whether the proxy fails for everyone,
    or only for the networks one method produces.

    Colour is the architecture family, as in Figure 7, so a family can
    be carried between the two figures; the per-point identifiers are
    gone and the legend carries identity instead.
    """
    dep, synth = deploy_table()
    if not dep:
        print("  figure6 skipped: no deployment records")
        return
    all_pts = [(r, m, cfg) for (m, cfg), r in dep.items()
               if r.get("deployed", True) and r.get("ms_median")]
    if len(all_pts) < 6:
        print("  figure6 skipped: too few deployed")
        return
    from scipy.stats import spearmanr

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.4), facecolor=SURFACE,
                             sharex=True, sharey=True)
    for k, (meth, ax) in enumerate(zip(("uniform", "joint"), axes)):
        pts = [p for p in all_pts if D.METHOD_OF[p[2]] == meth]
        if not pts:
            continue
        x = np.array([r["macc"] / 1e6 for r, _, _ in pts])
        y = np.array([r["ms_median"] for r, _, _ in pts])

        # Least squares in log-log, where constant throughput would be
        # a straight line of slope one.
        a, b = np.polyfit(np.log10(x), np.log10(y), 1)
        xs = np.linspace(np.log10(x.min()), np.log10(x.max()), 50)
        ax.plot(10 ** xs, 10 ** (a * xs + b), lw=1.1, color="#8a8a84",
                ls=(0, (4, 2)), zorder=2)

        for r, m, cfg in pts:
            ax.scatter(r["macc"] / 1e6, r["ms_median"], s=48,
                       marker=BSHAPE[D.BUDGET_OF[cfg]],
                       color=B.MODEL_COLOR[D.ICON[m]], edgecolor=SURFACE,
                       lw=0.5, zorder=5, alpha=0.92)

        rho = spearmanr(x, y).statistic
        disc = sum(1 for i in range(len(x)) for j in range(i + 1, len(x))
                   if (x[i] - x[j]) * (y[i] - y[j]) < 0)
        tot = len(x) * (len(x) - 1) // 2
        ax.text(0.035, 0.955,
                "Spearman rho = {:.2f}".format(rho) + chr(10)
                + "mis-ordered pairs: {}/{} ({:.0f}%)".format(
                    disc, tot, 100 * disc / tot),
                transform=ax.transAxes, fontsize=7.2, color=INK, va="top",
                linespacing=1.6)

        slowest = max(pts, key=lambda t: t[0]["ms_median"] / t[0]["macc"])
        biggest = max(pts, key=lambda t: t[0]["macc"])
        for r, m, cfg in {id(slowest): slowest, id(biggest): biggest}.values():
            ax.annotate(D.LABEL[m] + " " + m,
                        (r["macc"] / 1e6, r["ms_median"]),
                        textcoords="offset points", xytext=(9, 7),
                        fontsize=6.8, color=INK, zorder=7)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("(" + chr(65 + k) + ") " + meth + " scaling",
                     fontsize=8.4, color=INK, pad=5)
        ax.set_ylabel("measured latency (ms, log)", fontsize=7.6,
                      color=INK)
        ax.tick_params(labelsize=6.8, colors=INK2)
        B._frame(ax)
        ax.grid(color=GRID, lw=0.4, which="major")
        ax.set_axisbelow(True)

    axes[-1].set_xticks([2, 5, 10, 20, 50])
    axes[-1].set_yticks([200, 500, 1000, 2000, 4000])
    for axis in (axes[-1].xaxis, axes[-1].yaxis, axes[0].yaxis):
        axis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[-1].set_xlabel("arithmetic cost (million MACC, log)", fontsize=7.8,
                        color=INK)

    fam = [plt.Line2D([], [], lw=0, marker="o", markersize=5.5,
                      markerfacecolor=B.MODEL_COLOR[D.ICON[m]],
                      markeredgecolor="none", label=D.LABEL[m] + " " + m)
           for m in D.ORDER if any(p[1] == m for p in all_pts)]
    # Black, not grey: the shape key shares a legend grid with the
    # families, and one of those families is already grey.
    shp = [plt.Line2D([], [], lw=0, marker=BSHAPE[bn], markersize=5.5,
                      markerfacecolor="#1a1a18", markeredgecolor="none",
                      label="budget " + bn)
           for bn in ("UC-128", "C-256")]
    fig.legend(handles=fam + shp, fontsize=6.8, frameon=False,
               labelcolor=INK2, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.004), columnspacing=1.4,
               handletextpad=0.4)
    fig.subplots_adjust(left=0.110, right=0.985, top=0.965, bottom=0.165,
                        hspace=0.16)
    if synth:
        _stamp(fig)
    B.save(fig, "figure6_complexity_latency", dest=D.FIG, svg=True)


'''

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "r4_assets.py")
s = io.open(p, encoding="utf-8").read()
i6 = s.index("def figure6(")
i7 = s.index("def figure7(")
assert i6 < i7, "unexpected function order"
io.open(p, "w", encoding="utf-8").write(s[:i6] + F6 + s[i7:])
print("figure6 replaced with stacked two-panel version")
