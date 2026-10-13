"""Replace figure5 and figure6 in r4_assets with the revised designs.

Figure 5 drops to the two memory panels; Figure 6 becomes a single
scatter coloured by family. Written as a splice rather than by hand so
the surrounding file is untouched.
"""
import io
import os

F5 = '''def figure5(runs=None):
    """RQ2: does a configuration fit, and can it run?

    Two panels, because these are the two questions the part asks
    before any other matters: flash decides whether the model can be
    stored, SRAM whether it can be executed at all. Arithmetic cost and
    latency are not memory constraints and have a figure of their own.

    Four points per row -- the two scaling methods at the two budgets --
    so a family's whole footprint reads at a glance, and the dashed
    lines are the device's real limits rather than the budgets the
    search targeted. A point left of the line is feasible on that
    resource.
    """
    dep, synth = deploy_table()
    if not dep:
        print("  figure5 skipped: no deployment records")
        return
    ms = [m for m in D.ORDER if any(k[0] == m for k in dep)]
    panels = [("(A) firmware flash",
               lambda r: (r.get("total_flash_b") or 0) / 1024,
               D.FLASH_KB, "flash limit"),
              ("(B) peak SRAM",
               lambda r: (r.get("total_ram_b") or 0) / 1024,
               D.SRAM_KB, "SRAM limit")]
    off = {"U-UC128": 0.255, "U-C256": 0.085,
           "J-UC128": -0.085, "J-C256": -0.255}

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.5), facecolor=SURFACE,
                             sharey=True)
    for ax, (title, val, limit, lname) in zip(axes, panels):
        for y_, m in enumerate(ms):
            for cfg in D.DEPLOYED:
                r = dep.get((m, cfg))
                if not r:
                    continue
                colour, shape = _cfg_style(cfg)
                ok = r.get("deployed", True)
                ax.scatter(val(r), y_ + off[cfg], s=26, marker=shape,
                           color=colour if ok else "#ffffff",
                           edgecolor=SURFACE if ok else "#b03a4a",
                           lw=0.4 if ok else 0.9, zorder=5)
        ax.axvline(limit, color="#b03a4a", lw=1.0, ls=(0, (3, 2)), zorder=4)
        ax.text(limit, -0.54, "  " + lname + " " + str(limit) + " KB",
                fontsize=6.0, color="#b03a4a", ha="left", va="center")
        ax.set_xlim(0, limit * 1.14)
        ax.set_title(title, fontsize=8.2, color=INK, pad=5)
        ax.set_xlabel("KB", fontsize=7.0, color=INK2)
        ax.tick_params(axis="x", labelsize=6.4, colors=INK2)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(ms) - 0.4, -0.75)
        B._frame(ax)
        ax.grid(axis="x", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(ms)))
    axes[0].set_yticklabels([D.LABEL[m] + "  " + m for m in ms],
                            fontsize=7.0, color=INK)
    _deploy_legend(fig, y=0.018)
    fig.subplots_adjust(left=0.185, right=0.985, top=0.925, bottom=0.165,
                        wspace=0.08)
    if synth:
        _stamp(fig)
    B.save(fig, "figure5_hardware", dest=D.FIG, svg=True)


'''

F6 = '''def figure6(runs=None):
    """RQ2: is arithmetic cost a usable proxy for latency on this part?

    One panel, one question. If MAC count were a good proxy every point
    would lie on the trend and a designer could stop at a FLOP count.
    Rank discordance is reported next to the correlation because what a
    designer does with a proxy is *order* candidates, and a proxy can
    correlate strongly while still mis-ordering the pairs that decide a
    choice.

    Colour is the architecture family, as in Figure 7, so a family can
    be carried between the two figures. The per-point identifiers are
    gone -- forty of them turned the middle of the plot into a smear --
    and the legend carries identity instead. Only the two extremes are
    annotated, because those are the points a reader asks about.
    """
    dep, synth = deploy_table()
    if not dep:
        print("  figure6 skipped: no deployment records")
        return
    pts = [(r, m, cfg) for (m, cfg), r in dep.items()
           if r.get("deployed", True) and r.get("ms_median")]
    if len(pts) < 3:
        print("  figure6 skipped: too few deployed")
        return

    fig, ax = plt.subplots(figsize=(7.4, 5.2), facecolor=SURFACE)
    x = np.array([r["macc"] / 1e6 for r, _, _ in pts])
    y = np.array([r["ms_median"] for r, _, _ in pts])

    # Least squares in log-log, which is where constant throughput
    # would be a straight line of slope one.
    a, b = np.polyfit(np.log10(x), np.log10(y), 1)
    xs = np.linspace(np.log10(x.min()), np.log10(x.max()), 50)
    ax.plot(10 ** xs, 10 ** (a * xs + b), lw=1.1, color="#8a8a84",
            ls=(0, (4, 2)), zorder=2)

    for r, m, cfg in pts:
        ax.scatter(r["macc"] / 1e6, r["ms_median"], s=48,
                   marker=BSHAPE[D.BUDGET_OF[cfg]],
                   color=B.MODEL_COLOR[D.ICON[m]], edgecolor=SURFACE,
                   lw=0.5, zorder=5, alpha=0.92)

    from scipy.stats import spearmanr
    rho = spearmanr(x, y).statistic
    disc = sum(1 for i in range(len(x)) for j in range(i + 1, len(x))
               if (x[i] - x[j]) * (y[i] - y[j]) < 0)
    tot = len(x) * (len(x) - 1) // 2
    ax.text(0.035, 0.965,
            "Spearman rho = {:.2f}".format(rho) + chr(10)
            + "mis-ordered pairs: {}/{} ({:.0f}%)".format(
                disc, tot, 100 * disc / tot),
            transform=ax.transAxes, fontsize=7.4, color=INK, va="top",
            linespacing=1.6)

    slowest = max(pts, key=lambda t: t[0]["ms_median"] / t[0]["macc"])
    biggest = max(pts, key=lambda t: t[0]["macc"])
    for r, m, cfg in (slowest, biggest):
        ax.annotate(D.LABEL[m] + " " + m, (r["macc"] / 1e6, r["ms_median"]),
                    textcoords="offset points", xytext=(9, 7),
                    fontsize=6.8, color=INK, zorder=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_yticks([200, 500, 1000, 2000, 4000])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("arithmetic cost (million MACC, log)", fontsize=7.8,
                  color=INK)
    ax.set_ylabel("measured latency on STM32F411 (ms, log)", fontsize=7.8,
                  color=INK)
    ax.tick_params(labelsize=6.8, colors=INK2)
    B._frame(ax)
    ax.grid(color=GRID, lw=0.4, which="major")
    ax.set_axisbelow(True)

    fam = [plt.Line2D([], [], lw=0, marker="o", markersize=5.5,
                      markerfacecolor=B.MODEL_COLOR[D.ICON[m]],
                      markeredgecolor="none",
                      label=D.LABEL[m] + " " + m)
           for m in D.ORDER if any(p[1] == m for p in pts)]
    shp = [plt.Line2D([], [], lw=0, marker=BSHAPE[bn], markersize=5.5,
                      markerfacecolor="#8a8a84", markeredgecolor="none",
                      label=bn) for bn in ("UC-128", "C-256")]
    fig.legend(handles=fam + shp, fontsize=6.8, frameon=False,
               labelcolor=INK2, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.004), columnspacing=1.4,
               handletextpad=0.4)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.245)
    if synth:
        _stamp(fig)
    B.save(fig, "figure6_complexity_latency", dest=D.FIG, svg=True)


'''

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "r4_assets.py")
s = io.open(p, encoding="utf-8").read()
i5 = s.index("def figure5(")
i6 = s.index("def figure6(")
i7 = s.index("def figure7(")
assert i5 < i6 < i7, "unexpected function order"
io.open(p, "w", encoding="utf-8").write(s[:i5] + F5 + F6 + s[i7:])
print("figure5 and figure6 replaced")
