r"""Figure 1: benchmark design and evaluation flow.

Three columns -- the data, the models, the outputs -- with the flow
running left to right between them. Panel B carries the one structural
claim the design has to make visible: a single set of reference models
branches into two scaling methods, each at two weight budgets, and only
then reaches the device.

Every box gets a title and at most two short lines. Anything needing a
sentence belongs in the text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "src"))

import b4_assets as B                                           # noqa: E402
import r_data as D                                              # noqa: E402
from ecgmcu.data import SOURCES                                 # noqa: E402

INK, INK2 = "#1a1a18", "#55554f"
COL_A, COL_B, COL_C = "#1f5fa8", "#1d7a44", "#b05a18"
TINT_A, TINT_B, TINT_C = "#eaf2fb", "#eaf6ee", "#fdf3e8"
EDGE_A, EDGE_B, EDGE_C = "#7aa9d8", "#79bb94", "#e3ab6e"
SRC_COL = {"PTBXL": "#1f5fa8", "Georgia": "#b05a18",
           "Chapman": "#1d7a44", "Ningbo": "#6a4fb0"}
UNI_C, JNT_C = "#2f6fb0", "#3f8f5f"
GRID_PINK = "#f0d8d8"


def _rr(fig, x, y, w, h, fc, ec="none", lw=1.0, r=0.010, z=2, ls="solid"):
    fig.add_artist(matplotlib.patches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        transform=fig.transFigure, facecolor=fc, edgecolor=ec, linewidth=lw,
        linestyle=ls, zorder=z,
        mutation_aspect=fig.get_figwidth() / fig.get_figheight()))


def _t(fig, x, y, s, size=7.0, colour=INK2, ha="center", weight="normal",
       z=8, ls=1.35):
    fig.text(x, y, s, fontsize=size, color=colour, ha=ha, va="center",
             fontweight=weight, zorder=z, linespacing=ls)


def _fat_arrow(fig, x, y, w=0.022, h=0.030):
    """The chunky grey connector between columns."""
    fig.add_artist(matplotlib.patches.FancyArrow(
        x, y, w, 0, transform=fig.transFigure, width=h * 0.42,
        head_width=h, head_length=w * 0.52, length_includes_head=True,
        facecolor="#9a9a93", edgecolor="none", zorder=9))


def _ecg_card(fig, x, y, w, h, name, sig, rec, count):
    """A source card: title, and a real trace on ECG graph paper."""
    _rr(fig, x, y, w, h, "#ffffff", ec=EDGE_A, lw=0.9, r=0.008, z=4)
    _t(fig, x + w / 2, y + h - 0.022, name, size=8.0, colour=INK,
       weight="bold", z=9)
    _t(fig, x + w / 2, y + 0.018, count, size=6.3, colour=INK2, z=9)
    ax = fig.add_axes([x + 0.012, y + 0.034, w - 0.024, h - 0.074], zorder=6)
    ax.set_facecolor("#fffafa")
    for s in ax.spines.values():
        s.set_color("#d8c4c4")
        s.set_linewidth(0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    if rec is not None:
        v = np.asarray(sig[rec, 1], dtype=np.float32)[:420]
        for gx in np.linspace(0, len(v) - 1, 13):
            ax.axvline(gx, color=GRID_PINK, lw=0.4, zorder=1)
        lo, hi = float(v.min()), float(v.max())
        pad = 0.18 * (hi - lo or 1.0)
        for gy in np.linspace(lo - pad, hi + pad, 7):
            ax.axhline(gy, color=GRID_PINK, lw=0.4, zorder=1)
        ax.plot(v, lw=0.75, color="#2b2b28", zorder=3,
                solid_joinstyle="round")
        ax.set_xlim(0, len(v) - 1)
        ax.set_ylim(lo - pad, hi + pad)


def _stack(fig, x, y, w, h, colour, caption, drop):
    """Published network shrinking: thinner, and for joint also shorter."""
    ax = fig.add_axes([x, y, w, h], zorder=6)
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    def blocks(cx, bw, n, top=9.3, bot=0.7):
        step = (top - bot) / max(n, 1)
        bh = step * 0.56
        for i in range(n):
            yy = top - step * (i + 0.5)
            ax.add_patch(matplotlib.patches.FancyBboxPatch(
                (cx - bw / 2, yy - bh / 2), bw, bh,
                boxstyle="round,pad=0,rounding_size=0.18",
                facecolor=colour, alpha=0.85, edgecolor=colour,
                linewidth=0.7))
            if i < n - 1:
                ax.annotate("", xy=(cx, yy - step * 0.92 + bh / 2),
                            xytext=(cx, yy - bh / 2),
                            arrowprops=dict(arrowstyle="-|>", color="#55554f",
                                            lw=0.7, shrinkA=0, shrinkB=0))

    blocks(2.0, 2.6, 4)
    ax.annotate("", xy=(7.0, 5.0), xytext=(4.6, 5.0),
                arrowprops=dict(arrowstyle="-|>", color="#2b2b28", lw=1.0))
    ax.text(5.8, 6.5, caption, fontsize=6.0, color=INK, ha="center",
            va="center", linespacing=1.3)
    if drop:
        blocks(8.8, 1.8, 3, top=7.9, bot=2.1)
    else:
        blocks(8.8, 1.1, 4)


def figure1(runs=None):
    c = B.cohort()
    ix = c["index"]
    sig = np.load(c["signals"], mmap_mode="r")

    # Wide and short: this goes at the top of a paper, where column
    # height is the scarce dimension.
    fig = plt.figure(figsize=(12.2, 4.9), facecolor="#fdfcf8")

    ax_, bx_, cx_ = 0.012, 0.344, 0.712
    aw, bw_, cw = 0.300, 0.340, 0.276
    top, bot = 0.918, 0.020
    _rr(fig, ax_, bot, aw, top - bot, TINT_A, ec=EDGE_A, lw=1.3, r=0.012)
    _rr(fig, bx_, bot, bw_, top - bot, TINT_B, ec=EDGE_B, lw=1.3, r=0.012)
    _rr(fig, cx_, bot, cw, top - bot, TINT_C, ec=EDGE_C, lw=1.3, r=0.012)
    _fat_arrow(fig, 0.315, 0.470)
    _fat_arrow(fig, 0.685, 0.470)

    # =============================================== A. clinical sources
    _t(fig, ax_ + 0.018, 0.872, "A.  Clinical ECG sources", size=11.0,
       colour=COL_A, ha="left", weight="bold")
    for i, s in enumerate(SOURCES):
        px = ax_ + 0.016 + (i % 2) * 0.144
        py = 0.640 - (i // 2) * 0.186
        k = (ix["source"] == s).values
        rec = B._clean_record(sig, np.flatnonzero(k))
        _ecg_card(fig, px, py, 0.136, 0.176, B.sname(s), sig, rec,
                  f"{int(k.sum()):,} ECGs")

    _rr(fig, ax_ + 0.016, 0.352, 0.268, 0.090, "#ffffff", ec=EDGE_A, lw=0.9,
        r=0.008, z=4)
    _t(fig, ax_ + 0.150, 0.416, "Harmonised label space", size=8.4,
       colour=COL_A, weight="bold", z=9)
    _t(fig, ax_ + 0.150, 0.380,
       "13 diagnostic classes retained across all sources", size=6.4, z=9)

    _rr(fig, ax_ + 0.016, 0.038, 0.268, 0.280, "#ffffff", ec=EDGE_A, lw=0.9,
        r=0.008, z=4)
    _t(fig, ax_ + 0.150, 0.288, "Leave-one-source-out evaluation", size=8.4,
       colour=COL_A, weight="bold", z=9)
    _rr(fig, ax_ + 0.028, 0.058, 0.244, 0.198, "none", ec="#9ebfe0", lw=0.9,
        r=0.008, z=5, ls=(0, (3, 2)))
    for i, s in enumerate(SOURCES):
        fx = ax_ + 0.036 + i * 0.0585
        _rr(fig, fx, 0.118, 0.054, 0.124, "#fbfdff", ec=SRC_COL[s], lw=0.9,
            r=0.006, z=6)
        _t(fig, fx + 0.027, 0.218, f"Fold {i + 1}", size=7.2, colour=INK,
           weight="bold", z=9)
        _t(fig, fx + 0.027, 0.183, "test:", size=6.2, z=9)
        _t(fig, fx + 0.027, 0.148, B.sname(s), size=6.4, colour=SRC_COL[s],
           weight="bold", z=9)
    _t(fig, ax_ + 0.150, 0.082,
       "train on three sources, test on the fourth", size=6.4, z=9)

    # ================================ B. reference models and scaling
    _t(fig, bx_ + 0.018, 0.878, "B.  Reference models and", size=11.0,
       colour=COL_B, ha="left", weight="bold")
    _t(fig, bx_ + 0.018, 0.838, "deployment-oriented scaling", size=11.0,
       colour=COL_B, ha="left", weight="bold")

    _rr(fig, bx_ + 0.016, 0.714, 0.308, 0.082, "#f1edfb", ec="#a88fd0",
        lw=1.0, r=0.008, z=4)
    _t(fig, bx_ + 0.170, 0.772, "10 reference models (unconstrained)",
       size=8.6, colour=INK, weight="bold", z=9)
    _t(fig, bx_ + 0.170, 0.738,
       "published topology, float32, no device budget", size=6.3, z=9)

    _rr(fig, bx_ + 0.016, 0.392, 0.308, 0.302, "#f4fbf6", ec=EDGE_B, lw=0.9,
        r=0.008, z=4)
    _t(fig, bx_ + 0.170, 0.670, "Two scaling strategies for MCU deployment",
       size=8.6, colour=INK, weight="bold", z=9)
    for k, (nm, cap, col, drop, blurb) in enumerate([
            ("Uniform width scaling", "width only",
             UNI_C, False, "thinner, same depth"),
            ("Joint width-depth scaling", "width + depth",
             JNT_C, True, "thinner and shallower")]):
        px = bx_ + 0.026 + k * 0.152
        _rr(fig, px, 0.408, 0.142, 0.242, "#ffffff", ec=col, lw=1.0,
            r=0.008, z=5)
        _t(fig, px + 0.071, 0.626, nm, size=7.4, colour=col, weight="bold",
           z=9)
        _stack(fig, px + 0.008, 0.470, 0.126, 0.144, col, cap, drop)
        _t(fig, px + 0.071, 0.442, blurb, size=7.2, colour=INK,
           weight="bold", z=9)

    _rr(fig, bx_ + 0.016, 0.046, 0.308, 0.302, "#f4fbf6", ec=EDGE_B, lw=0.9,
        r=0.008, z=4)
    _t(fig, bx_ + 0.170, 0.318, "Training, quantisation, MCU deployment",
       size=8.6, colour=INK, weight="bold", z=9)
    for k, (nm, sub, fc, ec) in enumerate([
            ("Train", "(FP32)", "#eaf2fb", "#7aa9d8"),
            ("Quantise", "(INT8)", "#f1edfb", "#a88fd0")]):
        px = bx_ + 0.028 + k * 0.102
        _rr(fig, px, 0.160, 0.088, 0.118, fc, ec=ec, lw=1.0, r=0.008, z=5)
        _t(fig, px + 0.044, 0.242, nm, size=8.4, colour=INK, weight="bold",
           z=9)
        _t(fig, px + 0.044, 0.198, sub, size=7.4, colour=INK2, z=9)
        _fat_arrow(fig, px + 0.090, 0.219, w=0.010, h=0.022)
    px = bx_ + 0.232
    _rr(fig, px, 0.112, 0.086, 0.176, "#eaf6ee", ec=EDGE_B, lw=1.0, r=0.008,
        z=5)
    _t(fig, px + 0.043, 0.270, "Deploy on MCU", size=7.6, colour=INK,
       weight="bold", z=9)
    B.board_icon(fig, px + 0.012, 0.128, 0.062, 0.116)
    _t(fig, bx_ + 0.170, 0.090, "measured on STM32F411VET6:", size=6.3, z=9)
    _t(fig, bx_ + 0.170, 0.066, "Flash, peak SRAM, latency, macro-AP",
       size=6.3, z=9)

    # ============================================= C. evaluation outputs
    _t(fig, cx_ + 0.018, 0.872, "C.  Evaluation outputs", size=11.0,
       colour=COL_C, ha="left", weight="bold")
    for k, (tag, title, bullets, col, fc, ec) in enumerate([
            ("RQ1.", "Cross-source generalization",
             ["external discrimination across\nfour held-out sources",
              "class-specific AP and paired\narchitecture comparisons"],
             "#c0392b", "#fdeeee", "#e8a8a0"),
            ("RQ2.", "Scaling and MCU efficiency",
             ["compare scaling strategies under\ndeployment budgets",
              "assess prediction, quantisation\neffects, Flash, SRAM, latency"],
             "#1f5fa8", "#eaf2fb", "#7aa9d8"),
            ("RQ3.", "Performance-resource trade-offs",
             ["identify favorable deployed\nconfigurations",
              "trade-offs among external\nmacro-AP, latency and memory"],
             "#1d7a44", "#eaf6ee", "#79bb94")]):
        ey = 0.586 - k * 0.278
        _rr(fig, cx_ + 0.016, ey, 0.244, 0.244, fc, ec=ec, lw=1.0, r=0.009,
            z=4)
        _t(fig, cx_ + 0.030, ey + 0.198, f"{tag} {title}", size=8.0,
           colour=col, ha="left", weight="bold", z=9)
        for j, b in enumerate(bullets):
            by = ey + 0.132 - j * 0.064
            fig.add_artist(matplotlib.lines.Line2D(
                [cx_ + 0.036], [by], transform=fig.transFigure, lw=0,
                marker="o", markersize=3.0, markerfacecolor=col,
                markeredgecolor="none", zorder=9))
            _t(fig, cx_ + 0.048, by, b, size=6.6, colour=INK, ha="left",
               z=9)

    B.save(fig, "figure1_design", dest=D.FIG, svg=True)
