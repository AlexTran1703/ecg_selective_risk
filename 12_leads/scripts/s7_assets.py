r"""The 100 Hz manuscript set: 5 figures, 5 tables, plus supplementary.

    .\.venv\Scripts\python.exe 12_leads\scripts\s7_assets.py

    Fig 1  study design, with the two-branch signal path
    Fig 2  six signal-quality characteristics across four sources
    Fig 3  which impairments cost accuracy, and for which encoder
    Fig 4  int8 deployment: SRAM, Flash, compute, accuracy cost
    Fig 5  accuracy against parameters and against measured SRAM

Study I is computed on the native 500 Hz signal and the models run on an
anti-aliased 100 Hz resampling of it. That split is deliberate: a 100 Hz
signal has a 50 Hz Nyquist limit, so mains and high-frequency content are
not observable after decimation and would have to be either dropped or
measured dishonestly.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

from ecgmcu.data import SOURCES, FS_OUT, SECONDS, N_LEADS      # noqa: E402
from ecgmcu.models import ZOO, build, n_params                 # noqa: E402
from ecgmcu.specs import SPECS                                 # noqa: E402

RATE_TAG = "" if FS_OUT == 250 else f"_{FS_OUT}hz"
S1D = HERE / "results" / "study1"
S2D = HERE / "results" / ("study2" if FS_OUT == 250 else f"study2{RATE_TAG}")
S3D = HERE / "results" / f"study3{RATE_TAG}"
S16 = HERE / "results" / f"study16{RATE_TAG}"
S3B = HERE / "results" / f"study3b{RATE_TAG}"
FIG, TAB = HERE / "figures", HERE / "tables"
SFIG, STAB = FIG / "supplementary", TAB / "supplementary"

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"
OKC, BADC = "#2a78d6", "#eb6834"
SRAM_KB, FLASH_KB = 128, 512
RESERVE_FLASH, RESERVE_SRAM = 64, 16
ORDER = ["tiny", "dscnn", "mobilenet", "mbconv", "tcn", "resnet"]
NICE = {k: v["name"] for k, v in SPECS.items()}
SRC_NICE = {"PTBXL": "PTB-XL"}
ALPHA_RE = re.compile(r"a[0-9.]+__")
QRS_LO, QRS_HI = 5, 30
# Must match the labels emitted by s6_quality_robustness.py: a mismatch
# here silently drops columns from Figure 3 rather than raising.
FACTOR_ORDER = ["HF noise", "baseline wander", "mains",
                "flat/clipped lead", "QRS failure", "RR implausibility"]


def sname(s):
    return SRC_NICE.get(s, s)


def emit(df, stem, caption, folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.txt").write_text(
        caption + "\n" + "-" * 78 + "\n" + df.to_string() + "\n",
        encoding="utf-8")
    df.to_csv(folder / f"{stem}.csv")
    (folder / f"{stem}.tex").write_text(
        df.to_latex(caption=caption, escape=True), encoding="utf-8")
    print(f"  {folder.name}/{stem}")


def save(fig, stem, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{stem}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"  {folder.name}/{stem}")


def _frame(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7, length=2.5, pad=2)
    ax.set_axisbelow(True)


def ap_fast(y, s):
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    tp = np.cumsum(y)
    n_pos = tp[-1]
    if not n_pos:
        return np.nan
    end = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    tp_e = tp[end]
    return float((np.diff(np.r_[0.0, tp_e / n_pos]) * (tp_e / (end + 1.0))
                  ).sum())


def macro_ap(y, p):
    v = [ap_fast(y[:, c], p[:, c]) for c in range(y.shape[1])
         if 0 < y[:, c].sum() < y.shape[0]]
    return float(np.mean(v)) if v else np.nan


def macro_auroc(y, p):
    from sklearn.metrics import roc_auc_score
    v = [roc_auc_score(y[:, c], p[:, c]) for c in range(y.shape[1])
         if 0 < y[:, c].sum() < y.shape[0]]
    return float(np.mean(v)) if v else np.nan


def aurc_eaurc(loss, conf):
    n = loss.size
    k = np.arange(1, n + 1)
    risk = np.cumsum(loss[np.argsort(-conf, kind="stable")]) / k
    oracle = np.cumsum(np.sort(loss)) / k
    return float(risk.mean()), float(risk.mean() - oracle.mean())


def jaccard_loss(y, p, thr=0.5):
    yh = (p >= thr).astype(np.int8)
    inter = (yh & y).sum(1)
    union = (yh | y).sum(1)
    return 1.0 - np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def load_runs():
    runs = {}
    for f in sorted(S2D.glob("*.npz")):
        if "__" not in f.stem or ALPHA_RE.search(f.stem):
            continue
        d = np.load(f, allow_pickle=True)
        runs[(str(d["model"]), str(d["held_out"]))] = {
            "p": d["test_prob"].astype(np.float64),
            "y": d["test_y"].astype(np.int8), "rows": d["test_rows"],
            "params": int(d["params"])}
    return runs


S3C = HERE / "results" / f"study3c{RATE_TAG}"


def load_latency() -> dict:
    """Measured on-device results, keyed by model.

    This file is the authority for Flash and RAM as well as latency: it is
    produced with the ARM compiler available, so the analyser could size
    the generated runtime, which the earlier weights-only numbers could
    not include.
    """
    f = S3C / "latency.json"
    if not f.exists():
        return {}
    return {r["model"]: r for r in json.loads(f.read_text())}


def quality():
    q = pd.read_parquet(S1D / "quality.parquet")
    q["qrs_fail"] = ((q.n_qrs < QRS_LO) | (q.n_qrs > QRS_HI)).astype(int)
    return q


# ------------------------------------------------------------- figure 1
def figure1():
    """Study overview: a central spine with one side branch.

    An overview figure should show where the data go, what each question
    consumes, and what flows into the next stage. It should not try to be
    the methods section, so every box carries a title and a single line,
    the three RQ blocks are the dominant objects, and the detail that used
    to sit inside them lives in the caption instead.

    RQ1, RQ2 and RQ3 are the same width and centred, so the eye reads one
    vertical story. The quality branch leaves the native 500 Hz signal and
    enters RQ2 horizontally -- a real input, not an annotation floating in
    the margin.
    """
    FIG_H, YLO, YHI = 5.0, 12.0, 99.0
    YSPAN = YHI - YLO
    fig, ax = plt.subplots(figsize=(7.0, FIG_H), facecolor=SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(YLO, YHI)
    ax.axis("off")
    NL = chr(10)

    FILL = {"data": "#eaf2fc", "qual": "#fdf0e6", "hw": "#e9f7f1"}
    EDGE = {"data": "#3f7cbf", "qual": "#d9792b", "hw": "#2f9a6d"}
    ARR, TXT = "#5a5a55", "#2d2d2b"
    SPINE_L, SPINE_R = 20, 80          # the RQ column
    MID = (SPINE_L + SPINE_R) / 2

    def box(x, y, w, h, kind, title, line, fs_t=8.4, fs_d=7.2):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.9", linewidth=1.4,
            facecolor=FILL[kind], edgecolor=EDGE[kind], zorder=2))
        pt2y = YSPAN / (FIG_H * 72.0)
        nl = line.count(NL) + 1
        th = fs_t * 1.3 * pt2y
        dh = nl * fs_d * 1.5 * pt2y
        top = y + h / 2 + (th + 0.8 + dh) / 2
        ax.text(x + w / 2, top - th / 2, title, ha="center", va="center",
                fontsize=fs_t, color=INK, fontweight="bold", zorder=4)
        ax.text(x + w / 2, top - th - 0.8 - dh / 2, line, ha="center",
                va="center", fontsize=fs_d, color=TXT, zorder=4,
                linespacing=1.5)

    def down(x, y1, y2):
        ax.add_patch(FancyArrowPatch(
            (x, y1), (x, y2), arrowstyle="-|>", mutation_scale=11,
            color=ARR, lw=1.2, zorder=1, shrinkA=0, shrinkB=0))

    def elbow(pts, head):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, lw=1.2, color=ARR, zorder=1,
                solid_joinstyle="round")
        ax.add_patch(FancyArrowPatch(
            pts[-1], head, arrowstyle="-|>", mutation_scale=11,
            color=ARR, lw=1.2, zorder=1, shrinkA=0, shrinkB=0))

    def label(x, y, t, ha="left"):
        ax.text(x, y, t, fontsize=6.6, color=INK2, ha=ha, va="center",
                style="italic", zorder=4)

    # ---- the data -----------------------------------------------------
    box(6, 86, 88, 11, "data", "Four clinical ECG sources",
        "PTB-XL | Georgia | Chapman | Ningbo" + NL +
        "77,333 records  ->  66,379 retained, 13 harmonised labels",
        fs_t=9.0, fs_d=7.4)

    # Split into the two preparation paths.
    ax.plot([50, 50], [86, 83], lw=1.2, color=ARR, zorder=1)
    ax.plot([26, 74], [83, 83], lw=1.2, color=ARR, zorder=1)
    down(26, 83, 79.4)
    down(74, 83, 79.4)

    box(6, 69, 40, 10, "qual", "Native 500 Hz ECG",
        "quality indicators measured" + NL + "before decimation")
    box(54, 69, 40, 10, "data", "Anti-aliased resampling to 100 Hz",
        "model input: 12 x 1000")

    # ---- the spine ----------------------------------------------------
    elbow([(74, 69), (74, 65.5), (MID, 65.5)], (MID, 63.4))

    # All three RQ boxes are the same height and carry a one-line
    # subtitle. No period after the label: "RQ1. External" reads as a
    # numbered sentence, "RQ1  External" reads as a tag.
    box(SPINE_L, 51, SPINE_R - SPINE_L, 12, "data",
        "RQ1   Generalisation heterogeneity",
        "leave-one-source-out: train on three sources, test on the"
        + NL + "fourth, four rotations. Source, architecture and"
        + NL + "interaction effects on external AUPRC")

    down(MID, 51, 42.4)
    label(MID + 2, 46.5, "held-out predictions")

    box(SPINE_L, 32, SPINE_R - SPINE_L, 10, "qual",
        "RQ2   Examining explanations for source heterogeneity",
        "diagnostic distribution, annotation heterogeneity,"
        + NL + "and signal-quality associations")

    # The quality branch runs down the outside and enters RQ2 on its left
    # edge. The label sits along the run rather than floating beside the
    # bend, which is where it used to collide with RQ1.
    elbow([(26, 69), (26, 66.5), (12.5, 66.5), (12.5, 37)], (19.4, 37))
    ax.text(10.8, 52, "quality descriptors", rotation=90, ha="center",
            va="center", fontsize=6.6, color=INK2, style="italic",
            zorder=4)

    down(MID, 32, 24.4)
    label(MID + 2, 28, "externally evaluated models")

    box(SPINE_L, 14, SPINE_R - SPINE_L, 10, "hw",
        "RQ3   Reliability under transfer",
        "does confidence rank errors consistently"
        + NL + "across the same unseen sources?")

    fig.tight_layout(pad=0.2)
    save(fig, "figure1_design", FIG)



# ------------------------------------------------------------- figure 2
def figure2(q, dest=None, name="figure3_signal_quality"):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3), facecolor=SURFACE)
    cmap = plt.get_cmap("tab10")
    srcs = [s for s in SOURCES if s in set(q.source)]

    def ecdf(ax, col, title, logx=True):
        for i, s in enumerate(srcs):
            v = np.sort(q.loc[q.source == s, col].to_numpy(dtype=float))
            v = v[np.isfinite(v)]
            if logx:
                v = np.maximum(v, 1e-6)
            ax.plot(v, np.linspace(0, 1, v.size), lw=1.3, color=cmap(i),
                    label=sname(s))
        if logx:
            ax.set_xscale("log")
        ax.set_ylabel("cumulative fraction", fontsize=7, color=INK)
        ax.set_title(title, fontsize=7.6, color=INK, pad=3)
        _frame(ax)
        ax.grid(color=GRID, lw=0.5)

    def bars(ax, vals, title, ylab):
        ax.bar(range(len(srcs)), vals, 0.62,
               color=[cmap(i) for i in range(len(srcs))],
               edgecolor=SURFACE, lw=0.6)
        for i, v in enumerate(vals):
            # 0.005% would print as "0.00" and read as a true zero.
            lab_ = f"{v:.2f}" if v >= 0.1 else f"{v:.3f}"
            ax.text(i, v, lab_, ha="center", va="bottom",
                    fontsize=6.3, color=INK)
        ax.set_xticks(range(len(srcs)))
        ax.set_xticklabels([sname(s) for s in srcs], fontsize=6.6,
                           rotation=18, ha="right")
        ax.set_ylabel(ylab, fontsize=7, color=INK)
        ax.set_title(title, fontsize=7.6, color=INK, pad=3)
        ax.margins(y=0.20)
        _frame(ax)
        ax.grid(axis="y", color=GRID, lw=0.5)

    ecdf(axes[0][0], "hf_rel", "(a) high-frequency noise >40 Hz")
    ecdf(axes[0][1], "bw_rel", "(b) baseline wander <0.5 Hz")
    ecdf(axes[0][2], "pli_rel", "(c) mains 50/60 Hz")
    bars(axes[1][0], [100 * (q.loc[q.source == s, "flat_leads"] > 0).mean()
                      for s in srcs],
         "(d) flat or clipped lead", "% of records")
    bars(axes[1][1], [100 * q.loc[q.source == s, "qrs_fail"].mean()
                      for s in srcs],
         "(e) QRS detectability failure", "% of records")
    # RR implausibility is zero in ~98% of records, so an ECDF is a
    # vertical line at zero and says nothing. The useful quantity is how
    # often any implausible interval occurs at all.
    bars(axes[1][2], [100 * (q.loc[q.source == s, "rr_implaus"] > 0).mean()
                      for s in srcs],
         "(f) any implausible RR interval", "% of records")
    axes[0][0].legend(fontsize=6.2, frameon=False, labelcolor=INK2,
                      loc="lower right")
    fig.tight_layout(pad=0.5, w_pad=1.2, h_pad=1.4)
    save(fig, name, dest or FIG)


# ------------------------------------------------------------- figure 3
def figure4(dep, acc, f32):
    if not dep:
        print("  (no int8 deployment)")
        return
    ms = [m for m in ORDER if m in dep]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), facecolor=SURFACE)
    cmap = plt.get_cmap("tab10")
    x = np.arange(len(ms))

    def mem(ax, key32, key8, lim, res, ttl, lab):
        a = [f32.get(m, {}).get(key32, np.nan) for m in ms]
        b = [dep[m][key8] for m in ms]
        ax.bar(x - 0.2, a, 0.38, color=BADC, label="float32",
               edgecolor=SURFACE, lw=0.5)
        ax.bar(x + 0.2, b, 0.38, color=OKC, label="int8",
               edgecolor=SURFACE, lw=0.5)
        ax.axhspan(lim - res, lim, color=INK, alpha=0.09, lw=0, zorder=0)
        ax.axhline(lim, color=INK, lw=1.0, ls="--")
        ax.text(len(ms) - 0.45, lim * 1.07, f"F411 {lim} KB", fontsize=6.2,
                color=INK, ha="right", va="bottom")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([NICE[m] for m in ms], fontsize=6.3, rotation=26,
                           ha="right")
        ax.set_ylabel(lab, fontsize=7, color=INK)
        ax.set_title(ttl, fontsize=7.4, color=INK, pad=3)
        _frame(ax)
        ax.grid(axis="y", color=GRID, lw=0.5)

    mem(axes[0][0], "total_sram_kb", "sram_kb", SRAM_KB, RESERVE_SRAM,
        "(a) peak SRAM", "KB")
    mem(axes[0][1], "flash_kb", "flash_kb", FLASH_KB, RESERVE_FLASH,
        "(b) Flash", "KB")
    axes[0][0].legend(fontsize=6.2, frameon=False, labelcolor=INK2, ncol=2,
                      loc="upper left")

    # (c) measured on-device latency, STM32F411 at 100 MHz.
    ax = axes[1][0]
    lat = load_latency()
    vals = [lat.get(m, {}).get("ms_median", np.nan) for m in ms]
    ax.bar(x, vals, 0.62, color=[cmap(i) for i in range(len(ms))],
           edgecolor=SURFACE, lw=0.5)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=6.0, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[m] for m in ms], fontsize=6.3, rotation=26,
                       ha="right")
    ax.set_ylabel("inference latency (ms)", fontsize=7, color=INK)
    ax.set_title("(c) measured inference latency, F411 at 100 MHz",
                 fontsize=7.4, color=INK, pad=3)
    ax.margins(y=0.18)
    _frame(ax)
    ax.grid(axis="y", color=GRID, lw=0.5)

    # (d) quantisation cost
    ax = axes[1][1]
    d = [acc[acc.model == m].delta.mean() if not acc.empty else np.nan
         for m in ms]
    ax.bar(x, d, 0.62, color=[BADC if v < 0 else OKC for v in d],
           edgecolor=SURFACE, lw=0.5)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[m] for m in ms], fontsize=6.3, rotation=26,
                       ha="right")
    ax.set_ylabel("int8 minus float32 AUPRC", fontsize=7, color=INK)
    ax.set_title("(d) int8 PTQ discrimination cost", fontsize=7.4,
                 color=INK, pad=3)
    ax.annotate("mean over four held-out sources", xy=(0.5, -0.46),
                xycoords="axes fraction", ha="center", fontsize=6.0,
                color=INK2)
    _frame(ax)
    ax.grid(axis="y", color=GRID, lw=0.5)
    fig.tight_layout(pad=0.5, w_pad=1.6, h_pad=1.6)
    save(fig, "figureS3_deployment_detail", SFIG)


# ------------------------------------------------------------- figure 5
def figure5(runs, dep):
    """RQ3: does deployment preserve what RQ1 and RQ2 established, and
    what does that preservation cost?

    Two questions in the right order. The top row asks whether int8
    changes the discrimination from RQ1 or the selective reliability from
    RQ2 -- if quantisation scrambled the confidence ordering, every
    selective conclusion would be void on the device that actually runs.
    The bottom row asks what resources the preserved behaviour needs.

    Plotted on int8 accuracy throughout: these are deployment operating
    points, so the axes have to describe the model on the part.
    """
    from matplotlib.ticker import MaxNLocator, NullFormatter
    lat = load_latency()
    pres = {}
    fp = S2D / "int8_preservation.json"
    if fp.exists():
        pres = {r["model"]: r for r in json.loads(fp.read_text())}
    acc = {m: pres[m]["auprc_int8"] for m in ORDER if m in pres}
    if not acc:
        f8 = S3B / "int8.json"
        if f8.exists():
            a8 = pd.DataFrame(json.loads(f8.read_text()))
            acc = {m: float(a8[a8.model == m].auprc_int8.mean())
                   for m in ORDER if len(a8[a8.model == m])}
    if not acc:
        return
    models = [m for m in ORDER if m in acc]
    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(7.6, 4.9), facecolor=SURFACE)
    # The rows answer different questions -- what quantisation cost, then
    # what it costs to run -- so they are grouped rather than evenly
    # spaced. The gap is only as wide as the row heading needs.
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], hspace=0.60,
                          wspace=1.5, top=0.90, bottom=0.14, left=0.08,
                          right=0.98)
    for y_, t in ((0.965, "quantisation effect"),
                  (0.505, "embedded operating points")):
        fig.text(0.008, y_, t, fontsize=7.0, color=INK2, style="italic",
                 ha="left", va="center")

    def preserve(ax, key, lo, hi, title, xlabel, worse_positive):
        if not pres:
            return
        yy = np.arange(len(models))[::-1]
        # Intervals are drawn as absolute endpoints rather than as xerr
        # offsets. A full-data point estimate can fall outside its own
        # bootstrap percentile range, and when it does that is something
        # to show, not a reason for the figure to fail to render.
        for y_, m in zip(yy, models):
            v, a, b = pres[m][key], pres[m][lo], pres[m][hi]
            ax.plot([a, b], [y_, y_], lw=1.1, color=INK2, zorder=2,
                    solid_capstyle="butt")
            for e in (a, b):
                ax.plot([e, e], [y_ - 0.17, y_ + 0.17], lw=1.1,
                        color=INK2, zorder=2)
            ax.scatter(v, y_, s=34, color=cmap(ORDER.index(m)),
                       edgecolor=SURFACE, lw=0.8, zorder=3)
        ax.axvline(0, color=INK, lw=0.9, ls="--")
        ax.set_yticks(yy)
        ax.set_yticklabels([NICE[m] for m in models], fontsize=6.6)
        ax.set_xlabel(xlabel, fontsize=6.9, color=INK)
        ax.set_title(title, fontsize=7.2, color=INK, pad=4)
        _frame(ax)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="x", labelsize=6.2)
        ax.grid(axis="x", color=GRID, lw=0.5)

    preserve(fig.add_subplot(gs[0, 0:3]), "d_auprc", "d_auprc_lo",
             "d_auprc_hi", "(a) change in macro AUPRC after int8",
             "dAUPRC, int8 minus float32", worse_positive=False)
    # (b) was the int8 E-AURC change, which is not comparable across
    # precisions: it is a functional of the confidence ranking, and the
    # two rankings have ~16,500 and ~24 distinct levels. The lattice
    # that makes it incomparable is the finding, so it is plotted here.
    lf = S2D / "confidence_lattice.json"
    if lf.exists():
        ax = fig.add_subplot(gs[0, 3:6])
        lat_d = pd.DataFrame(json.loads(lf.read_text()))
        i8 = lat_d[lat_d.precision == "int8"]
        yy = np.arange(len(models))[::-1]
        for y_, m in zip(yy, models):
            g = i8[i8.model == m]
            if g.empty:
                continue
            near = np.array([t["0.90"]["nearest_cov"] for t in g.targets])
            ax.plot([near.min(), near.max()], [y_, y_], lw=1.1,
                    color=INK2, zorder=2, solid_capstyle="butt")
            for v in near:
                ax.scatter(v, y_, s=30,
                           color=cmap(ORDER.index(m)),
                           marker="X" if v > 0.999 else "o",
                           edgecolor=SURFACE, lw=0.8, zorder=3)
        ax.axvline(0.90, color=INK, lw=1.0, ls="--", zorder=1)
        ax.annotate("target", xy=(0.90, len(models) - 0.45),
                    fontsize=5.8, color=INK, ha="center", va="bottom")
        # Annotated in the empty bottom-right rather than as a legend
        # box, which landed on top of the ResNet row.
        ax.annotate("X = target not reachable,"
                    + chr(10) + "nearest option is no abstention",
                    xy=(1.018, -0.46), ha="right", va="bottom",
                    fontsize=5.6, color=INK2, style="italic",
                    linespacing=1.3)
        ax.set_yticks(yy)
        ax.set_yticklabels([NICE[m] for m in models], fontsize=7.0)
        ax.set_xlabel("nearest achievable coverage to a 0.90 target, "
                      "one point per held-out source",
                      fontsize=6.6, color=INK)
        ax.set_title("(b) int8 confidence lattice", fontsize=7.2,
                     color=INK, pad=4)
        ax.set_xlim(0.78, 1.025)
        _frame(ax)
        ax.grid(axis="x", color=GRID, lw=0.5)

    def cost(ax, getter, lim, xlabel, title, logx=True, scale=1024.0):
        for i_, m in enumerate(models):
            v = getter(m)
            if v:
                ax.scatter(v / scale, acc[m], s=44,
                           color=cmap(ORDER.index(m)), edgecolor=SURFACE,
                           lw=1.0, zorder=3,
                           label=NICE[m] if ax is first_cost else None)
        if lim:
            ax.axvline(lim, color=INK, lw=1.0, ls="--")
            ax.text(lim, ax.get_ylim()[0], f"F411 {lim:.0f} KB  ",
                    fontsize=5.8, color=INK, rotation=90, va="bottom",
                    ha="right")
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xlabel, fontsize=6.9, color=INK)
        ax.set_title(title, fontsize=7.2, color=INK, pad=4)
        _frame(ax)
        ax.grid(color=GRID, lw=0.5)

    first_cost = fig.add_subplot(gs[1, 0:2])
    first_cost.set_ylabel("int8 macro AUPRC", fontsize=6.9, color=INK)
    cost(first_cost, lambda m: lat.get(m, {}).get("total_flash_b"),
         FLASH_KB, "total Flash (KB)", "(c) int8 AUPRC vs Flash")
    cost(fig.add_subplot(gs[1, 2:4]),
         lambda m: lat.get(m, {}).get("total_ram_b"), SRAM_KB,
         "total RAM (KB)", "(d) int8 AUPRC vs SRAM", logx=False)
    ax_lat = fig.add_subplot(gs[1, 4:6])
    cost(ax_lat, lambda m: lat.get(m, {}).get("ms_median"), None,
         "measured latency (ms)", "(e) int8 AUPRC vs latency", scale=1.0)
    # The real-time factors belong in the caption. In-panel commentary
    # reads as a slide note and competes with the data it sits on; the
    # range is printed by supplementary_deployment and quoted in the text.
    rtf = [lat[m]["rtf"] for m in models if lat.get(m, {}).get("rtf")]
    if rtf:
        print(f"    RTF {min(rtf):.3f}-{max(rtf):.3f} (for the caption)")

    handles, labels = first_cost.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.3, frameon=False,
               labelcolor=INK2, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.005))
    save(fig, "figure5_operating_points", FIG)


def figure3_diagnosis_transfer():
    """RQ2 at the diagnosis level, where the units are not four.

    The source-level version of this analysis had four points and could
    not carry an explanatory claim. This one works on the 13 x 4
    diagnosis-by-source cells and reports the dependence honestly.

    (a) and (b) are the same cells under two metrics. Average precision
    has a prevalence floor; AUROC is rank-only and has none. Printing
    both lets a reader separate "this diagnosis is rare in this source"
    from "the model cannot discriminate it here" -- and they disagree
    about which source is hardest, which is the point.

    (c) tests whether prevalence shift tracks discrimination, with AUROC
    as the outcome so that a prevalence difference is not partly
    regressed on itself. The 52 cells are not 52 independent
    observations, so the pooled slope is a mixed model with a random
    intercept per diagnosis, shown beside a sign test over the 13
    within-diagnosis slopes that assumes nothing.
    """
    f = S2D / "diagnosis_level.json"
    if not f.exists():
        print("  (no diagnosis-level analysis)")
        return
    d = json.loads(f.read_text())
    t = pd.DataFrame(d["cells"])
    srcs = [s for s in d["sources"]]
    classes = [c for c in d["classes"] if c in set(t["class"])]
    order = (t.groupby("class").ap.mean().reindex(classes)
             .sort_values(ascending=False).index.tolist())

    fig = plt.figure(figsize=(7.9, 3.5), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.12],
                          wspace=0.48, left=0.085, right=0.975,
                          top=0.87, bottom=0.17)

    def heat(ax, col, title, cmap_name, lo, hi):
        M = np.full((len(order), len(srcs)), np.nan)
        for _, r in t.iterrows():
            M[order.index(r["class"]), srcs.index(r["source"])] = r[col]
        im = ax.imshow(M, cmap=cmap_name, vmin=lo, vmax=hi, aspect="auto")
        ax.set_xticks(range(len(srcs)))
        ax.set_xticklabels([sname(x) for x in srcs], fontsize=6.2,
                           rotation=20, ha="right")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=5.9)
        ax.set_title(title, fontsize=7.3, color=INK, pad=4)
        ax.tick_params(length=0, colors=INK2)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.ax.tick_params(labelsize=5.6, colors=INK2, length=2)
        return M

    heat(fig.add_subplot(gs[0, 0]), "ap",
         "(a) average precision", "YlGnBu", 0.0, 1.0)
    ax = fig.add_subplot(gs[0, 1])
    heat(ax, "auroc", "(b) AUROC, no prevalence floor", "YlGnBu", 0.5, 1.0)
    ax.set_yticklabels([])

    # (c) the dependence-aware test, and it is null.
    ax = fig.add_subplot(gs[0, 2])
    cmap = plt.get_cmap("Dark2")
    for i, src in enumerate(srcs):
        g = t[t.source == src]
        ax.scatter(g.abs_log2_shift, g.auroc, s=22, color=cmap(i),
                   edgecolor=SURFACE, lw=0.6, zorder=3, label=sname(src))
    xg = np.linspace(0, t.abs_log2_shift.max() * 1.03, 20)
    b0 = t.auroc.mean() - d["mixed_slope"] * t.abs_log2_shift.mean()
    ax.plot(xg, b0 + d["mixed_slope"] * xg, lw=1.2, ls="--", color=INK,
            zorder=2)
    ax.fill_between(xg, b0 + d["mixed_lo"] * xg, b0 + d["mixed_hi"] * xg,
                    color=INK2, alpha=0.13, lw=0, zorder=1)
    ax.set_xlabel("|log2 prevalence shift| from training pool",
                  fontsize=6.8, color=INK)
    ax.set_ylabel("AUROC", fontsize=6.8, color=INK)
    ax.set_title("(c) no consistent association\ndetected",
                 fontsize=7.3, color=INK, pad=4)
    ax.annotate("unit: diagnosis x source cell (n = 52)" + chr(10)
                + "mixed model, random intercept per diagnosis" + chr(10)
                + f"slope {d['mixed_slope']:+.3f}" + chr(10)
                + f"95% CI ({d['mixed_lo']:+.3f}, {d['mixed_hi']:+.3f}),"
                + f" p = {d['mixed_p']:.2f}" + chr(10)
                + f"sign test over {d['sign_n']} diagnoses: "
                + f"{d['sign_neg']} negative, "
                + f"p = {d['sign_p']:.2f}",
                xy=(0.97, 0.04), xycoords="axes fraction", ha="right",
                va="bottom", fontsize=5.7, color=INK2, linespacing=1.45)
    ax.legend(fontsize=5.8, frameon=False, labelcolor=INK2, ncol=2,
              loc="upper left", bbox_to_anchor=(-0.01, 1.02),
              handletextpad=0.3, columnspacing=0.8)
    ax.set_ylim(0.45, 1.17)
    _frame(ax)
    ax.grid(color=GRID, lw=0.5)
    save(fig, "figure3_diagnosis_transfer", FIG)


def figure3_explaining_spread():
    """RQ2: what accounts for the spread Figure 2 found between sources?

    The obvious candidate is signal quality, and it is wrong. Panel (a)
    correlates each source-level quality indicator with that source's
    external AUPRC and almost every one comes out positive: the cleanest
    corpus is the worst to be tested on and the dirtiest is the best. A
    quality-driven account of cross-source generalisation would need
    these bars to be negative.

    Panel (b) is the candidate that does track performance -- how far the
    held-out source's label distribution sits from the pool the model was
    trained on. Panel (c) is the sharper, narrower version of the same
    idea: individual diagnoses where a source applies the label at a
    different rate than its peers do, given the same signal evidence.

    Four sources is four points. These panels establish sign and
    consistency, not effect size, and the caption says so.
    """
    f = S2D / "label_drift.json"
    if not f.exists():
        print("  (no label drift)")
        return
    d = json.loads(f.read_text())
    runs = load_runs()
    q = quality()
    srcs = [s for s in SOURCES if ("tiny", s) in runs]
    if not srcs:
        return

    mAP = {}
    for s in srcs:
        mAP[s] = float(np.mean([macro_ap(runs[(m, s)]["y"], runs[(m, s)]["p"])
                                for m in ORDER if (m, s) in runs]))
    Y = {s: runs[("tiny", s)]["y"] for s in srcs}
    P = {s: np.array([Y[s][:, c].mean() for c in range(Y[s].shape[1])])
         for s in srcs}
    L1 = {}
    for s in srcs:
        tr = [t for t in srcs if t != s]
        w = np.array([Y[t].shape[0] for t in tr], dtype=float)
        L1[s] = float(np.abs(P[s] - np.average([P[t] for t in tr], axis=0,
                                               weights=w)).sum())

    IND = [("hf_rel", "HF noise"), ("bw_rel", "baseline wander"),
           ("pli_rel", "mains"), ("rr_implaus", "RR implausibility"),
           ("qrs_fail", "QRS failure"), ("flat_leads", "flat leads"),
           ("clip_frac", "clipping")]
    gq = q.groupby("source")[[k for k, _ in IND]].mean()
    y = np.array([mAP[s] for s in srcs])
    rs = [float(np.corrcoef(np.array([gq.loc[s, k] for s in srcs]), y)[0, 1])
          for k, _ in IND]

    fig = plt.figure(figsize=(7.6, 2.9), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.85, 1.15],
                          wspace=0.52, left=0.14, right=0.985,
                          top=0.86, bottom=0.21)
    WRONG, RIGHT = "#d9792b", "#3f7cbf"

    # (a) every quality indicator, so nothing is cherry-picked.
    ax = fig.add_subplot(gs[0, 0])
    yy = np.arange(len(IND))[::-1]
    ax.barh(yy, rs, height=0.62, zorder=2,
            color=[WRONG if v > 0 else RIGHT for v in rs])
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(yy)
    ax.set_yticklabels([n for _, n in IND], fontsize=6.6)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("corr(indicator, source AUPRC)" + chr(10)
                  + "right of zero = worse quality, better AUPRC",
                  fontsize=6.4, color=INK, linespacing=1.4)
    ax.set_title("(a) signal quality does not\nexplain the spread",
                 fontsize=7.3, color=INK, pad=4)
    _frame(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)

    # (b) the candidate that does track it.
    ax = fig.add_subplot(gs[0, 1])
    cmap = plt.get_cmap("Dark2")
    for i, s in enumerate(srcs):
        ax.scatter(L1[s], mAP[s], s=52, color=cmap(i), edgecolor=SURFACE,
                   lw=1.0, zorder=3)
        ax.annotate(sname(s), (L1[s], mAP[s]), textcoords="offset points",
                    xytext=(0, -11), ha="center", fontsize=6.2, color=INK2)
    xs = np.array([L1[s] for s in srcs])
    b = np.polyfit(xs, y, 1)
    xg = np.linspace(xs.min() * 0.92, xs.max() * 1.04, 20)
    ax.plot(xg, np.polyval(b, xg), lw=1.0, ls="--", color=INK2, zorder=1)
    r = float(np.corrcoef(xs, y)[0, 1])
    ax.annotate(f"r = {r:+.2f}", xy=(0.95, 0.93), xycoords="axes fraction",
                ha="right", va="top", fontsize=6.6, color=INK)
    ax.set_xlabel("label-distribution shift\nfrom training pool (L1)",
                  fontsize=6.9, color=INK)
    ax.set_ylabel("external macro AUPRC", fontsize=6.9, color=INK)
    ax.set_title("(b) label-distribution\nshift does", fontsize=7.3, color=INK,
                 pad=4)
    ax.margins(x=0.20, y=0.24)
    _frame(ax)
    ax.grid(color=GRID, lw=0.5)

    # (c) per-diagnosis labelling propensity against the peer sources.
    ax = fig.add_subplot(gs[0, 2])
    per = d["per_class"]
    classes = sorted({r["class"] for r in per})
    M = np.full((len(classes), len(srcs)), np.nan)
    for r in per:
        if r["source"] in srcs:
            M[classes.index(r["class"]), srcs.index(r["source"])] = r["drift"]
    v = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    for r in per:
        if r["source"] in srcs and r.get("flagged"):
            ax.scatter(srcs.index(r["source"]), classes.index(r["class"]),
                       s=9, marker="o", facecolor="none", edgecolor=INK,
                       lw=0.8, zorder=3)
    ax.set_xticks(range(len(srcs)))
    ax.set_xticklabels([sname(s) for s in srcs], fontsize=6.3, rotation=20,
                       ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=5.9)
    ax.set_title("(c) label-definition drift\nper diagnosis", fontsize=7.3,
                 color=INK, pad=4)
    ax.tick_params(length=0, colors=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=5.6, colors=INK2, length=2)
    cb.set_label("log2 labelling propensity\nvs peer sources", fontsize=5.6,
                 color=INK2, linespacing=1.3)
    save(fig, "figureS12_source_level", SFIG)


def figure3_reliability():
    """Selective reliability under held-out clinical-source shift.

    Two questions, one panel each: does confidence-ranking quality depend
    on the source, and what is each architecture's overall level. The
    excess-risk curves that E-AURC integrates are informative but ask the
    reader to subtract an oracle, accept a forced convergence to zero at
    full coverage, and then integrate a hump by eye; they are better
    placed in the supplement, where anyone who wants to inspect ranking
    behaviour can find them.
    """
    f = S2D / "reliability_curves.json"
    if not f.exists():
        print("  (no reliability curves)")
        return
    d = json.loads(f.read_text())
    models = [m for m in ORDER if m in d["models"]]
    srcs = [s for s in SOURCES
            if s in d["models"][models[0]]["eaurc_by_source"]]
    fig, axes = plt.subplots(
        1, 2, figsize=(7.4, 3.0), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.0, 0.95]})

    # (a) is the source-dependence, which is the point of the figure.
    ax = axes[0]
    mat = np.array([[d["models"][m]["eaurc_by_source"][s] for s in srcs]
                    for m in models])
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(srcs)))
    ax.set_xticklabels([sname(s) for s in srcs], fontsize=6.8, rotation=20,
                       ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([NICE[m] for m in models], fontsize=7.0)
    lim = mat.max()
    for a_ in range(mat.shape[0]):
        for b_ in range(mat.shape[1]):
            ax.text(b_, a_, f"{mat[a_, b_]:.3f}", ha="center", va="center",
                    fontsize=6.4,
                    color="#ffffff" if mat[a_, b_] > 0.75 * lim else INK)
    ax.set_title("(a) E-AURC by held-out source", fontsize=7.4, color=INK,
                 pad=4)
    ax.tick_params(length=0, colors=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=6, colors=INK2, length=2)
    cb.set_label("lower E-AURC = better error ranking", fontsize=6.0,
                 color=INK2)

    # (b) overall level, ordered, with record-level bootstrap intervals.
    ax = axes[1]
    order = sorted(models, key=lambda m: d["models"][m]["eaurc"])
    yy = np.arange(len(order))[::-1]
    pt = [d["models"][m]["eaurc"] for m in order]
    lo = [d["models"][m]["eaurc"] - d["models"][m]["eaurc_lo"]
          for m in order]
    hi = [d["models"][m]["eaurc_hi"] - d["models"][m]["eaurc"]
          for m in order]
    ax.errorbar(pt, yy, xerr=[lo, hi], fmt="none", ecolor=INK2,
                elinewidth=1.1, capsize=2.6, zorder=2)
    cmap = plt.get_cmap("tab10")
    for y_, m in zip(yy, order):
        ax.scatter(d["models"][m]["eaurc"], y_, s=42,
                   color=cmap(ORDER.index(m)), edgecolor=SURFACE, lw=0.9,
                   zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels([NICE[m] for m in order], fontsize=7.0)
    ax.set_xlabel("mean E-AURC over four held-out sources", fontsize=7.4,
                  color=INK)
    ax.set_title("(b) mean E-AURC, record-level bootstrap 95% CI",
                 fontsize=7.2, color=INK, pad=4)
    ax.annotate("lower = better error ranking", xy=(0.5, -0.30),
                xycoords="axes fraction", ha="center", fontsize=6.2,
                color=INK2)
    _frame(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout(pad=0.5, w_pad=1.8)
    save(fig, "figureS4_eaurc_by_source", SFIG)


def figureS2_risk_coverage():
    """Risk-coverage curves per held-out source, with the oracle shown.

    This is the detail Figure 3 summarises: ordinary selective risk, not a
    subtracted quantity, with each model's own oracle as a dashed floor so
    the gap E-AURC integrates is visible directly.
    """
    f = S2D / "reliability_curves.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    grid = np.array(d["grid"])
    models = [m for m in ORDER if m in d["models"]]
    srcs = [s for s in SOURCES
            if s in d["models"][models[0]].get("risk_coverage", {})]
    if not srcs:
        return
    fig, axes = plt.subplots(1, len(srcs), figsize=(2.1 * len(srcs), 3.1),
                             facecolor=SURFACE, sharey=True)
    cmap = plt.get_cmap("tab10")
    for k, s in enumerate(srcs):
        ax = axes[k]
        for i, m in enumerate(models):
            rc = d["models"][m]["risk_coverage"][s]
            ax.plot(grid, rc["risk"], lw=1.2, color=cmap(ORDER.index(m)),
                    label=NICE[m] if k == 0 else None)
            ax.plot(grid, rc["oracle"], lw=0.7, ls="--",
                    color=cmap(ORDER.index(m)), alpha=0.45)
        ax.set_title(sname(s), fontsize=7.4, color=INK, pad=3)
        ax.set_xlabel("coverage", fontsize=7.0, color=INK)
        if k == 0:
            ax.set_ylabel("selective risk", fontsize=7.0, color=INK)
        _frame(ax)
        ax.grid(color=GRID, lw=0.5)
    # The legend and the key sit below the panels. tight_layout only
    # reserves space for artists inside the axes, so the margin is set
    # explicitly and both are placed in figure coordinates.
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.30,
                        wspace=0.16)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=5.8,
               frameon=False, labelcolor=INK2, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, 0.10))
    fig.text(0.5, 0.035, "solid = selective risk under the model's own "
             "confidence ranking;  dashed = that model's oracle",
             ha="center", fontsize=6.0, color=INK2, style="italic")
    save(fig, "figureS2_risk_coverage", SFIG)


def figure2_generalisation(boot):
    """RQ1: how well the encoders transfer to an unseen clinical source.

    The primary diagnostic result had been living in a table while the
    secondary analyses got the figures, which made the supporting work
    look like the main event. Same visual language as the reliability
    panels so the two read as one study.
    """
    runs = load_runs()
    if not runs or not boot:
        print("  (no runs for figure 2)")
        return
    models = [m for m in ORDER if any(k[0] == m for k in runs)]
    srcs = [s for s in SOURCES if (models[0], s) in runs]
    mat = np.array([[macro_ap(runs[(m, s)]["y"], runs[(m, s)]["p"])
                     for s in srcs] for m in models])

    vd = S2D / "variance_decomposition.json"
    dec = json.loads(vd.read_text()) if vd.exists() else None
    fig, axes = plt.subplots(
        1, 3 if dec else 2,
        figsize=(8.4 if dec else 7.4, 3.0), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.0, 0.95, 0.72] if dec
                     else [1.0, 0.95]})
    ax = axes[0]
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(srcs)))
    ax.set_xticklabels([sname(s) for s in srcs], fontsize=6.8, rotation=20,
                       ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([NICE[m] for m in models], fontsize=7.0)
    lim = mat.max()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=6.4,
                    color="#ffffff" if mat[i, j] > 0.80 * lim else INK)
    ax.set_title("(a) macro AUPRC by held-out source", fontsize=7.4,
                 color=INK, pad=4)
    ax.tick_params(length=0, colors=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=6, colors=INK2, length=2)
    cb.set_label("higher AUPRC = better discrimination", fontsize=6.0,
                 color=INK2)

    ax = axes[1]
    order = sorted(models, key=lambda m: -boot[m][0])
    yy = np.arange(len(order))[::-1]
    pt = [boot[m][0] for m in order]
    lo = [boot[m][0] - boot[m][1] for m in order]
    hi = [boot[m][2] - boot[m][0] for m in order]
    ax.errorbar(pt, yy, xerr=[lo, hi], fmt="none", ecolor=INK2,
                elinewidth=1.1, capsize=2.6, zorder=2)
    cmap = plt.get_cmap("tab10")
    for y_, m in zip(yy, order):
        ax.scatter(boot[m][0], y_, s=42, color=cmap(ORDER.index(m)),
                   edgecolor=SURFACE, lw=0.9, zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels([NICE[m] for m in order], fontsize=7.0)
    ax.set_xlabel("mean macro AUPRC over four held-out sources",
                  fontsize=7.4, color=INK)
    ax.set_title("(b) mean AUPRC, record-level bootstrap 95% CI",
                 fontsize=7.2, color=INK, pad=4)
    ax.annotate("higher = better", xy=(0.5, -0.30),
                xycoords="axes fraction", ha="center", fontsize=6.2,
                color=INK2)
    _frame(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)

    # (c) The heatmap is read row-wise as an architecture ranking. Read
    # column-wise the columns differ far more, and this is that
    # comparison made explicit rather than left to the reader's eye.
    if dec:
        ax = axes[2]
        # Three bars rather than one stacked bar: the stack needed a
        # legend, the legend sat on the bar, and stacking hid the
        # intervals, which are the reason this is a decomposition and
        # not an assertion.
        parts = [("held-out\nsource", "share_source", "#3f7cbf"),
                 ("architecture", "share_model", "#d9792b"),
                 ("interaction", "share_interaction", "#8a8a85")]
        yy = np.arange(len(parts))[::-1]
        for y_, (name, key, col) in zip(yy, parts):
            v = dec[key] * 100
            lo, hi = dec[key + "_lo"] * 100, dec[key + "_hi"] * 100
            ax.barh([y_], [v], height=0.52, color=col, zorder=2)
            ax.plot([lo, hi], [y_, y_], lw=1.1, color=INK, zorder=3,
                    solid_capstyle="butt")
            ax.text(v + 4, y_, f"{v:.0f}%", va="center", ha="left",
                    fontsize=7.0, color=INK, fontweight="bold", zorder=4)
        ax.set_yticks(yy)
        ax.set_yticklabels([n for n, _, _ in parts], fontsize=6.8,
                           linespacing=1.2)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("share of variance in the table at (a)",
                      fontsize=6.9, color=INK)
        ax.set_title("(c) decomposition of observed" + chr(10)
                     + "AUPRC variation",
                     fontsize=7.2, color=INK, pad=4)
        ax.annotate(f"variance of the observed 6x4 matrix; source effect" + chr(10)
                    + f" is {dec['ratio']:.1f}x the "
                    f"architecture effect, bars are 95% CI",
                    xy=(0.5, -0.30), xycoords="axes fraction",
                    ha="center", fontsize=6.0, color=INK2,
                    linespacing=1.3)
        _frame(ax)
        ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout(pad=0.5, w_pad=1.8)
    save(fig, "figure2_generalisation", FIG)


def figure4_reliability(qr):
    """RQ2c: can confidence identify unreliable external predictions?

    Two things only: the selective behaviour, then its summary. Panels
    (a-d) are the risk-coverage curves, one per held-out source; panel (e)
    is the integral, E-AURC, with record-level bootstrap intervals.

    The risk axis is exactly the quantity E-AURC integrates: records are
    ranked by confidence, retained from most to least confident, and
    selective risk is the mean loss over the retained fraction. Using a
    different selective metric here because it plots more prettily would
    make the figure and the summary describe different things.

    Selective risk at fixed coverage -- what withholding the least
    confident tenth actually buys -- is the operational reading of the
    same curves, and it is a supplementary figure rather than a sixth
    panel here. Two summaries of one quantity side by side crowd each
    other and neither is read.
    """
    f = S2D / "reliability_curves.json"
    if not f.exists():
        print("  (no reliability curves)")
        return
    d = json.loads(f.read_text())
    grid = np.array(d["grid"])
    models = [m for m in ORDER if m in d["models"]]
    srcs = [s for s in SOURCES
            if s in d["models"][models[0]].get("risk_coverage", {})]
    if not srcs:
        print("  (no per-source curves)")
        return

    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(7.6, 4.7), facecolor=SURFACE)
    # One panel row, one summary row, and a gap between them sized for the
    # shared legend and nothing else.
    gs = fig.add_gridspec(2, len(srcs), height_ratios=[1.0, 0.82],
                          hspace=0.95, wspace=0.42,
                          top=0.93, bottom=0.11, left=0.09, right=0.98)

    ax0 = None
    for k, s in enumerate(srcs):
        ax = fig.add_subplot(gs[0, k], sharey=ax0) if ax0 else             fig.add_subplot(gs[0, k])
        ax0 = ax0 or ax
        for m in models:
            rc = d["models"][m]["risk_coverage"][s]
            ax.plot(grid, rc["risk"], lw=1.3, color=cmap(ORDER.index(m)),
                    label=NICE[m] if k == 0 else None)
        ax.set_title(f"({chr(97 + k)}) {sname(s)}", fontsize=7.6,
                     color=INK, pad=3)
        ax.set_xlabel("coverage", fontsize=7.2, color=INK)
        if k == 0:
            ax.set_ylabel("selective risk", fontsize=7.2, color=INK)
        else:
            ax.tick_params(labelleft=False)
        _frame(ax)
        ax.grid(color=GRID, lw=0.5)

    # (e) spans the full width: the intervals are the point of the panel,
    # so they get the resolution rather than sharing the row.
    ax = fig.add_subplot(gs[1, 0:2])
    order = sorted(models, key=lambda m: d["models"][m]["eaurc"])
    yy = np.arange(len(order))[::-1]
    pt = [d["models"][m]["eaurc"] for m in order]
    lo = [d["models"][m]["eaurc"] - d["models"][m]["eaurc_lo"]
          for m in order]
    hi = [d["models"][m]["eaurc_hi"] - d["models"][m]["eaurc"]
          for m in order]
    ax.errorbar(pt, yy, xerr=[lo, hi], fmt="none", ecolor=INK2,
                elinewidth=1.1, capsize=2.6, zorder=2)
    # The per-source values are plotted behind the mean. The question is
    # not only which encoder ranks errors best but whether any of them
    # ranks them consistently, and a mean alone cannot show that: the
    # spread across sources dwarfs the spread across architectures here
    # exactly as it does for discrimination in Figure 2.
    smap = plt.get_cmap("Dark2")
    for y_, m in zip(yy, order):
        by = d["models"][m].get("eaurc_by_source", {})
        for k, src in enumerate(srcs):
            if src in by:
                ax.scatter(by[src], y_ + 0.26, s=13, color=smap(k),
                           edgecolor=SURFACE, lw=0.4, zorder=3,
                           label=sname(src) if m == order[0] else None)
    for y_, m in zip(yy, order):
        ax.scatter(d["models"][m]["eaurc"], y_, s=42,
                   color=cmap(ORDER.index(m)), edgecolor=SURFACE, lw=0.9,
                   zorder=4)
    ax.set_yticks(yy)
    ax.set_yticklabels([NICE[m] for m in order], fontsize=7.2)
    ax.set_xlabel("E-AURC, lower = better.  Large = mean with 95% CI",
                  fontsize=6.6, color=INK)
    ax.set_title(f"({chr(97 + len(srcs))}) reliability varies more "
                 f"across sources than architectures",
                 fontsize=7.3, color=INK, pad=3)
    src_handles = ax.collections[:len(srcs)]
    _frame(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)

    # (f) An ordering is not an operating point. This is the coverage a
    # threshold actually delivers when it is fixed on the training-side
    # validation split and carried to the held-out source unchanged --
    # the only way a deployment could set it.
    tf = S2D / "threshold_transfer.json"
    if tf.exists():
        th = pd.DataFrame(json.loads(tf.read_text()))
        ax = fig.add_subplot(gs[1, 2:])
        yy = np.arange(len(order))[::-1]
        for y_, m in zip(yy, order):
            for k, src in enumerate(srcs):
                g = th[(th.model == m) & (th.source == src)
                       & (th.target == 0.90)]
                if not g.empty:
                    ax.scatter(float(g.coverage.iloc[0]), y_, s=24,
                               color=smap(k), edgecolor=SURFACE, lw=0.5,
                               zorder=3)
        ax.axvline(0.90, color=INK, lw=1.0, ls="--", zorder=1)
        ax.set_yticks(yy)
        ax.set_yticklabels([NICE[m] for m in order], fontsize=7.2)
        ax.set_xlabel("coverage achieved on the held-out source",
                      fontsize=6.6, color=INK)
        ax.set_title("(f) a 0.90 target, fixed on training-side"
                     + chr(10) + "validation, transfers within a few points",
                     fontsize=7.3, color=INK, pad=3)
        err = (th[th.target == 0.90].coverage - 0.90).abs()
        ax.annotate(f"mean |error| {err.mean():.3f}, worst {err.max():.3f}",
                    xy=(0.03, 0.04), xycoords="axes fraction", ha="left",
                    va="bottom", fontsize=5.9, color=INK2)
        _frame(ax)
        ax.grid(axis="x", color=GRID, lw=0.5)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.6, frameon=False,
               labelcolor=INK2, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 0.545))
    smap2 = plt.get_cmap("Dark2")
    fig.legend([plt.Line2D([], [], marker="o", ls="", ms=4,
                           color=smap2(k)) for k in range(len(srcs))],
               [sname(x) for x in srcs], fontsize=6.0, frameon=False,
               labelcolor=INK2, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.495),
               title="held-out source, panels (e) and (f)",
               title_fontsize=6.0)
    save(fig, "figure4_reliability", FIG)


def figureS8_abstention():
    """Supplementary: what abstention buys, in units a reader can act on.

    E-AURC is the rigorous summary and it is what Figure 4(e) ranks, but
    it is an integral and says nothing directly about a deployment. This
    is the same curves read at three operating points: selective risk with
    everything retained, and after withholding the least-confident tenth
    and fifth.

    It was a sixth panel of Figure 4. Standing alone it is readable, and
    Figure 4 is better without a second summary competing with (e).
    """
    fp = S2D / "int8_preservation.json"
    f = S2D / "reliability_curves.json"
    if not (fp.exists() and f.exists()):
        return
    pres = {r["model"]: r for r in json.loads(fp.read_text())}
    d = json.loads(f.read_text())
    order = sorted([m for m in ORDER if m in d["models"] and m in pres],
                   key=lambda m: d["models"][m]["eaurc"])
    if not order:
        return

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(5.4, 2.9), facecolor=SURFACE)
    yy = np.arange(len(order))[::-1]
    for y_, m in zip(yy, order):
        r = pres[m]["risk_f32"]
        full, c90, c80 = r["1.0"], r["0.9"], r["0.8"]
        ax.plot([c80, full], [y_, y_], lw=1.6,
                color=cmap(ORDER.index(m)), alpha=0.45, zorder=2,
                solid_capstyle="round")
        ax.scatter(full, y_, s=40, color=cmap(ORDER.index(m)),
                   edgecolor=SURFACE, lw=0.9, zorder=3,
                   label="100% (no abstention)" if m == order[0] else None)
        ax.scatter(c90, y_, s=26, marker="s", color=cmap(ORDER.index(m)),
                   edgecolor=SURFACE, lw=0.8, zorder=3,
                   label="90%" if m == order[0] else None)
        ax.scatter(c80, y_, s=26, marker="D", facecolor="white",
                   edgecolor=cmap(ORDER.index(m)), lw=1.1, zorder=3,
                   label="80%" if m == order[0] else None)
    ax.set_yticks(yy)
    ax.set_yticklabels([NICE[m] for m in order], fontsize=7.4)
    ax.set_xlabel("selective risk at retained coverage, lower = better",
                  fontsize=7.4, color=INK)
    _frame(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.legend(fontsize=6.4, frameon=False, labelcolor=INK2, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.24),
              title="retained coverage", title_fontsize=6.4)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.95, bottom=0.30)
    save(fig, "figureS8_abstention", SFIG)


def figureS5_quality_reliability():
    """Supplementary: does confidence degrade along with the signal?

    A second-order interaction -- quality against confidence, given that
    quality already costs discrimination. Kept because the answer is the
    unwelcome one and it bears on how a deployment should gate, but out of
    the main sequence, where it would pull RQ2c back towards being a
    signal-quality study rather than a reliability one.
    """
    f = S2D / "quality_reliability.json"
    if not f.exists():
        return
    q2 = pd.DataFrame(json.loads(f.read_text()))
    pq = q2[q2.model == "__pooled__"].set_index("label")
    labs = [c for c in FACTOR_ORDER if c in pq.index]
    if not labs:
        return
    fig, ax = plt.subplots(figsize=(4.4, 2.3), facecolor=SURFACE)
    yy = np.arange(len(labs))[::-1]
    pt = [pq.loc[c, "delta_eaurc"] for c in labs]
    lo = [pq.loc[c, "delta_eaurc"] - pq.loc[c, "lo"] for c in labs]
    hi = [pq.loc[c, "hi"] - pq.loc[c, "delta_eaurc"] for c in labs]
    ax.errorbar(pt, yy, xerr=[lo, hi], fmt="none", ecolor=INK2,
                elinewidth=1.1, capsize=2.4, zorder=2)
    for y_, v in zip(yy, pt):
        ax.scatter(v, y_, s=38, color=BADC if v > 0 else OKC,
                   edgecolor=SURFACE, lw=0.9, zorder=3)
    ax.axvline(0, color=INK, lw=0.9, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels(labs, fontsize=7.0)
    ax.set_xlabel("dE-AURC, impaired minus clean", fontsize=7.2, color=INK)
    ax.annotate("positive = errors the model is also worse at ranking",
                xy=(0.5, -0.34), xycoords="axes fraction", ha="center",
                fontsize=6.2, color=INK2)
    _frame(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout(pad=0.4)
    save(fig, "figureS5_quality_reliability", SFIG)
