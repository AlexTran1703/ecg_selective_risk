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
    """Study design as a map of the five research questions.

    Each box carries a small icon of what it does, a bold title and two
    lines of detail, with the RQ number on a tab sitting across the top
    edge. Three functional colours: amber for signal quality, blue for
    data and modelling, green for hardware, and neutral grey for the
    synthesis, which belongs to both worlds.

    The arrow into RQ5 comes from RQ2 and RQ4, not from RQ3: the
    accuracy-resource trade-off combines discrimination with deployment
    cost, while the quality association answers its own question and feeds
    nothing downstream.
    """
    FIG_H, YLO, YHI = 5.2, 13.0, 101.0
    YSPAN = YHI - YLO
    fig, ax = plt.subplots(figsize=(7.6, FIG_H), facecolor=SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(YLO, YHI)
    ax.axis("off")
    NL = chr(10)

    FILL = {"data": "#e8f1fb", "qual": "#fdefe2", "hw": "#e7f6ef",
            "syn": "#f2f2f0"}
    EDGE = {"data": "#4a86c8", "qual": "#e07f33", "hw": "#36a173",
            "syn": "#8a8a88"}

    def box(x, y, w, h, kind, title, detail, fs_t=8.0, fs_d=6.8,
            tx=None):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.9", linewidth=1.3,
            facecolor=FILL[kind], edgecolor=EDGE[kind], zorder=2))
        cx = tx if tx is not None else x + w / 2
        # Points to data units: the axis spans YSPAN units over
        # FIG_H inches, and there are 72 points to the inch. Guessing this
        # factor is how the title ends up sitting on the detail.
        pt2y = YSPAN / (FIG_H * 72.0)
        nt = title.count(NL) + 1
        nd = detail.count(NL) + 1 if detail else 0
        th = nt * fs_t * 1.30 * pt2y
        dh = nd * fs_d * 1.55 * pt2y
        gap = 0.9 if nd else 0.0
        top = y + h / 2 + (th + gap + dh) / 2
        ax.text(cx, top - th / 2, title, ha="center", va="center",
                fontsize=fs_t, color=INK, fontweight="bold", zorder=4,
                linespacing=1.30)
        if detail:
            ax.text(cx, top - th - gap - dh / 2, detail, ha="center",
                    va="center", fontsize=fs_d, color="#3c3c3a", zorder=4,
                    linespacing=1.55)

    def tab(x, y, label, kind):
        """RQ number on a tab straddling the top edge of its box."""
        ax.add_patch(FancyBboxPatch(
            (x, y - 1.8), 8.6, 3.6, boxstyle="round,pad=0.5",
            linewidth=0, facecolor=EDGE[kind], zorder=5))
        ax.text(x + 4.3, y, label, ha="center", va="center",
                fontsize=6.6, color="white", fontweight="bold", zorder=6)

    ARR = "#54544f"

    def arrow(x1, y1, x2, y2, cs=None):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
            color="#54544f", lw=1.1, zorder=1, shrinkA=1, shrinkB=1,
            connectionstyle=cs or "arc3,rad=0"))

    def note(x, y, t, ha="center"):
        ax.text(x, y, t, fontsize=6.3, color=INK2, ha=ha, va="center",
                style="italic", zorder=4, linespacing=1.3)

    # ------------------------------------------------------------ icons
    def ecg_icon(cx, cy, w, h, c):
        """One PQRST complex, drawn rather than approximated by a zigzag."""
        t = np.linspace(0, 1, 260)
        y = (0.10 * np.exp(-((t - 0.17) / 0.035) ** 2)          # P
             - 0.13 * np.exp(-((t - 0.40) / 0.012) ** 2)        # Q
             + 1.00 * np.exp(-((t - 0.45) / 0.011) ** 2)        # R
             - 0.30 * np.exp(-((t - 0.51) / 0.014) ** 2)        # S
             + 0.22 * np.exp(-((t - 0.72) / 0.045) ** 2))       # T
        ax.plot(cx + (t - 0.5) * w, cy + y * h * 0.42, lw=1.3, color=c,
                zorder=4, solid_capstyle="round")

    def resample_icon(cx, cy, w, h, c):
        t = np.linspace(0, 1, 200)
        wave = np.sin(2 * np.pi * 3 * t) * np.exp(-1.1 * t)
        ax.plot(cx - w * 0.33 + (t - 0.5) * w * 0.34,
                cy + wave * h * 0.26, lw=1.1, color=c, zorder=4)
        k = np.linspace(0, 1, 13)
        ax.plot(cx + w * 0.30 + (k - 0.5) * w * 0.34,
                cy + (np.sin(2 * np.pi * 3 * k) * np.exp(-1.1 * k))
                * h * 0.26, lw=1.1, color=c, zorder=4, marker="o", ms=1.9)
        ax.annotate("", xy=(cx + w * 0.10, cy), xytext=(cx - w * 0.10, cy),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1),
                    zorder=4)

    def bars_icon(cx, cy, w, h, c):
        hs = [0.42, 0.78, 0.55, 1.0, 0.33]
        bw = w / 7.5
        for k, v in enumerate(hs):
            ax.add_patch(plt.Rectangle(
                (cx - w * 0.36 + k * bw * 1.4, cy - h * 0.30),
                bw, v * h * 0.60, facecolor=c, edgecolor="none", zorder=4))

    def nn_icon(cx, cy, w, h, c):
        layers = [(-0.30, 2), (0.0, 3), (0.30, 2)]
        pos = []
        for dx, n in layers:
            ys = np.linspace(-0.26, 0.26, n)
            pos.append([(cx + dx * w, cy + y * h) for y in ys])
        for a, b in zip(pos[:-1], pos[1:]):
            for p1 in a:
                for p2 in b:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], lw=0.5,
                            color=c, alpha=0.55, zorder=3)
        for layer in pos:
            for (px, py) in layer:
                ax.add_patch(plt.Circle((px, py), w * 0.052,
                                        facecolor="white", edgecolor=c,
                                        lw=1.0, zorder=4))

    def scatter_icon(cx, cy, w, h, c):
        ax.plot([cx - w * 0.34, cx - w * 0.34, cx + w * 0.36],
                [cy + h * 0.30, cy - h * 0.28, cy - h * 0.28],
                lw=0.9, color=c, zorder=4)
        rng = np.random.default_rng(3)
        xs = np.linspace(-0.24, 0.28, 9)
        ys = xs * 0.9 + rng.normal(0, 0.055, xs.size)
        ax.scatter(cx + xs * w, cy + ys * h * 0.8, s=5.5, color=c,
                   zorder=5)
        ax.plot(cx + xs * w, cy + xs * 0.9 * h * 0.8, lw=0.9, ls="--",
                color=c, alpha=0.75, zorder=4)

    def chip_icon(cx, cy, w, h, c):
        s = w * 0.30
        ax.add_patch(FancyBboxPatch(
            (cx - s, cy - s), 2 * s, 2 * s, boxstyle="round,pad=0.25",
            facecolor="white", edgecolor=c, lw=1.2, zorder=4))
        ax.add_patch(plt.Rectangle((cx - s * 0.42, cy - s * 0.42),
                                   s * 0.84, s * 0.84, facecolor=c,
                                   edgecolor="none", zorder=5))
        for k in (-0.5, 0.0, 0.5):
            ax.plot([cx + k * 2 * s * 0.6] * 2,
                    [cy + s, cy + s * 1.45], lw=1.0, color=c, zorder=4)
            ax.plot([cx + k * 2 * s * 0.6] * 2,
                    [cy - s, cy - s * 1.45], lw=1.0, color=c, zorder=4)
            ax.plot([cx - s, cx - s * 1.45],
                    [cy + k * 2 * s * 0.6] * 2, lw=1.0, color=c, zorder=4)
            ax.plot([cx + s, cx + s * 1.45],
                    [cy + k * 2 * s * 0.6] * 2, lw=1.0, color=c, zorder=4)

    def pareto_icon(cx, cy, w, h, c):
        ax.plot([cx - w * 0.34, cx - w * 0.34, cx + w * 0.36],
                [cy + h * 0.32, cy - h * 0.30, cy - h * 0.30],
                lw=0.9, color=c, zorder=4)
        xs = np.linspace(-0.26, 0.30, 8)
        ys = 0.30 * np.exp(-6.0 * (xs + 0.26)) - 0.12
        ax.scatter(cx + xs * w, cy + ys * h * 1.5, s=6.0, color=c,
                   zorder=5)
        ax.text(cx - w * 0.40, cy + h * 0.34, "AUPRC", fontsize=5.0,
                color=INK2, ha="center", va="bottom", rotation=90)
        ax.text(cx + w * 0.02, cy - h * 0.40, "resource cost",
                fontsize=5.0, color=INK2, ha="center", va="top")

    # ------------------------------------------------- level 1: the data
    # Columns 3-46 and 54-97 leave an 8-unit centre gap, which is what the
    # arrows need in order to run orthogonally instead of diagonally.
    box(3, 90, 94, 9, "data",
        "Four clinical ECG sources",
        "PTB-XL | Georgia | Chapman | Ningbo" + NL +
        "77,333 records  ->  66,379 retained, 13 harmonised labels",
        fs_t=8.4, fs_d=6.8, tx=56)
    ecg_icon(16, 94.5, 10, 5, EDGE["data"])

    ax.plot([50, 50], [90, 87.5], lw=1.0, color=ARR, zorder=1)
    ax.plot([24, 76], [87.5, 87.5], lw=1.0, color=ARR, zorder=1)
    note(51.5, 88.6, "10 s ECG extraction", ha="left")
    arrow(24, 87.5, 24, 84.5)
    arrow(76, 87.5, 76, 84.5)

    box(3, 76, 43, 8, "qual", "Native 500 Hz ECG",
        "signal-quality characterisation", fs_t=7.4, fs_d=6.4, tx=30)
    ecg_icon(10, 80, 8, 4.5, EDGE["qual"])
    box(54, 76, 43, 8, "data", "Anti-aliased resampling to 100 Hz",
        "FIR zero phase | input 12 x 1000", fs_t=7.4, fs_d=6.4, tx=81)
    resample_icon(61, 80, 9, 4.5, EDGE["data"])

    arrow(24, 76, 24, 72.5)
    arrow(76, 76, 76, 72.5)

    # ------------------------------------------- level 2: RQ1 to RQ4
    box(3, 60, 43, 12, "qual", "Source signal characteristics",
        "HF | baseline | mains" + NL + "flat/clipped | QRS | RR",
        fs_t=7.4, fs_d=6.4, tx=31)
    tab(6, 72, "RQ1", "qual")
    bars_icon(10, 65, 6.5, 5.5, EDGE["qual"])

    box(54, 60, 43, 12, "data",
        "External-source discrimination" + NL + "and selective reliability",
        "leave-one-source-out | 6 encoders x 4 folds" + NL +
        "AUPRC higher | E-AURC lower", fs_t=7.4, fs_d=6.4, tx=83)
    tab(57, 72, "RQ2", "data")
    nn_icon(61, 65, 7, 5.5, EDGE["data"])

    box(3, 38, 43, 12, "qual",
        "Quality-performance" + NL + "association",
        "within-source strata | prevalence matching" + NL +
        "dAUPRC with 95% CI", fs_t=7.4, fs_d=6.4, tx=31)
    tab(6, 50, "RQ3", "qual")
    scatter_icon(10, 43, 7, 5.5, EDGE["qual"])

    box(54, 38, 43, 12, "hw",
        "Embedded feasibility of the" + NL + "externally evaluated models",
        "STM32F411 100 MHz | int8 PTQ" + NL +
        "Flash | SRAM | latency | dAUPRC",
        fs_t=7.4, fs_d=6.4, tx=83)
    tab(57, 50, "RQ4", "hw")
    chip_icon(61, 43, 7, 5.5, EDGE["hw"])

    arrow(24, 60, 24, 50.5)
    note(25.5, 55.5, "quality indicators", ha="left")
    arrow(76, 60, 76, 50.5)
    note(77.5, 55, "trained encoders", ha="left")

    # One trunk out of RQ2, branching: left into RQ3, onward into RQ5.
    # Drawn as a T rather than two lines that cross each other.
    ax.plot([58, 50, 50], [60, 56, 28], lw=1.0, color=ARR, zorder=1,
            solid_joinstyle="round")
    arrow(50, 44, 46.6, 44)
    note(51.3, 56, "LOSO predictions", ha="left")

    # ------------------------------------------- level 3: RQ5 synthesis
    box(3, 15, 94, 12, "syn", "Accuracy-resource operating points",
        "diagnostic value  <->  deployment cost" + NL +
        "macro AUPRC | Flash / SRAM / measured latency" + NL +
        "which encoders are worth running on the part",
        fs_t=8.4, fs_d=6.8, tx=57)
    tab(6, 27, "RQ5", "syn")
    pareto_icon(26, 20, 9, 5.5, "#6f6f6d")

    arrow(50, 28, 50, 27)
    note(51.5, 31, "discrimination", ha="left")
    arrow(76, 38, 76, 27)
    note(77.5, 32, "hardware metrics", ha="left")

    fig.tight_layout(pad=0.2)
    save(fig, "figure1_design", FIG)



# ------------------------------------------------------------- figure 2
def figure2(q):
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
    save(fig, "figure2_signal_quality", FIG)


# ------------------------------------------------------------- figure 3
def figure3(qr):
    if qr is None or qr.empty:
        print("  (no quality-robustness results)")
        return
    qm = qr[qr.model != "__pooled__"]
    qp = qr[qr.model == "__pooled__"]
    piv = qm.pivot(index="model", columns="label", values="delta")
    piv = piv.reindex(index=[m for m in ORDER if m in piv.index],
                      columns=[c for c in FACTOR_ORDER if c in piv.columns])
    fig, axes = plt.subplots(
        1, 2, figsize=(7.4, 3.0), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.45, 1.0]})

    # (a) heatmap
    ax = axes[0]
    lim = float(np.nanmax(np.abs(piv.to_numpy())))
    im = ax.imshow(piv.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim,
                   aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, fontsize=6.4, rotation=28, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([NICE[m] for m in piv.index], fontsize=6.8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        fontsize=5.9,
                        color="#ffffff" if abs(v) > 0.6 * lim else INK)
    ax.set_title("(a) prevalence-matched dAUPRC: higher minus lower\nimpairment stratum", fontsize=7.2, color=INK, pad=4)
    # Two families of indicator, measured on different signals: the
    # first three come from the native 500 Hz recording and describe
    # acquisition conditions, the last three are structural and survive
    # decimation. A divider stops the heatmap reading as six
    # perturbations applied to the 100 Hz model input.
    n_spec = sum(1 for c in piv.columns
                 if c in ("HF noise", "baseline wander", "mains"))
    if 0 < n_spec < piv.shape[1]:
        ax.axvline(n_spec - 0.5, color=INK, lw=1.8)
        ax.text((n_spec - 1) / 2.0, -0.92, "native spectral (500 Hz)",
                ha="center", va="bottom", fontsize=6.0, color=INK2,
                style="italic")
        ax.text((n_spec + piv.shape[1] - 1) / 2.0, -0.92,
                "structural / rhythm", ha="center", va="bottom",
                fontsize=6.0, color=INK2, style="italic")
    ax.tick_params(length=0, colors=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=6, colors=INK2, length=2)

    # (b) pooled effect, model-averaged, with CI
    ax = axes[1]
    # Pooled effect and interval are produced inside the record-level
    # bootstrap, where every architecture sees the same resampled
    # records. Averaging six separately drawn intervals would treat
    # the encoders as independent observations, which they are not.
    pooled = (qp.set_index("label")[["delta", "lo", "hi"]]
              .rename(columns={"delta": "d"})
              .reindex([c for c in FACTOR_ORDER if c in set(qp.label)]))
    yy = np.arange(len(pooled))[::-1]
    ax.errorbar(pooled["d"], yy,
                xerr=[pooled["d"] - pooled["lo"], pooled["hi"] - pooled["d"]],
                fmt="o", ms=4.6, lw=1.2, capsize=2.4,
                color=INK, ecolor=INK2, zorder=3)
    for y_, v in zip(yy, pooled["d"]):
        ax.scatter(v, y_, s=34, zorder=4,
                   color=BADC if v < 0 else OKC, edgecolor=SURFACE, lw=0.9)
    ax.axvline(0, color=INK, lw=0.9, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels(pooled.index, fontsize=6.8)
    ax.set_xlabel("dAUPRC, model-averaged", fontsize=7.4, color=INK)
    ax.set_title("(b) pooled effect, record-level bootstrap 95% CI",
                 fontsize=7.2, color=INK, pad=4)
    _frame(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout(pad=0.5, w_pad=1.6)
    save(fig, "figure4_quality_association", FIG)
    return pooled


# ------------------------------------------------------------- figure 4
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
    save(fig, "figure4_deployment", FIG)


# ------------------------------------------------------------- figure 5
def figure5(runs, dep):
    acc = {}
    for m in ORDER:
        v = [macro_ap(runs[(m, s)]["y"], runs[(m, s)]["p"])
             for s in SOURCES if (m, s) in runs]
        if v:
            acc[m] = float(np.mean(v))
    if not acc:
        return
    lat = load_latency()
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2), facecolor=SURFACE)
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(acc):
        L0 = lat.get(m, {})
        if L0.get("total_flash_b"):
            axes[0].scatter(L0["total_flash_b"] / 1024, acc[m], s=48,
                            color=cmap(i), edgecolor=SURFACE, lw=1.0,
                            zorder=3, label=NICE[m])
        L = lat.get(m, {})
        if L.get("total_ram_b"):
            axes[1].scatter(L["total_ram_b"] / 1024, acc[m], s=48,
                            color=cmap(i), edgecolor=SURFACE, lw=1.0,
                            zorder=3)
        if L.get("ms_median"):
            axes[2].scatter(L["ms_median"], acc[m], s=48, color=cmap(i),
                            edgecolor=SURFACE, lw=1.0, zorder=3)
    # matplotlib labels log minor ticks by default at narrow decade spans,
    # which collides into an unreadable smear on a panel this wide.
    from matplotlib.ticker import NullFormatter
    axes[0].set_xscale("log")
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].axvline(FLASH_KB, color=INK, lw=1.0, ls="--")
    axes[0].text(FLASH_KB, axes[0].get_ylim()[0], "F411 512 KB  ",
                 fontsize=6.0, color=INK, rotation=90, va="bottom",
                 ha="right")
    axes[0].set_xlabel("measured total Flash (KB)", fontsize=7.4,
                       color=INK)
    axes[0].set_ylabel("cross-source macro AUPRC", fontsize=7.4, color=INK)
    axes[0].set_title("(a) accuracy vs Flash", fontsize=7.2,
                      color=INK)

    axes[1].axvline(SRAM_KB, color=INK, lw=1.0, ls="--")
    axes[1].text(SRAM_KB, axes[1].get_ylim()[0], "F411 128 KB  ",
                 fontsize=6.0, color=INK, rotation=90, va="bottom")
    axes[1].set_xlabel("measured total RAM (KB)", fontsize=7.4, color=INK)
    axes[1].set_title("(b) accuracy vs peak SRAM", fontsize=7.2,
                      color=INK)

    axes[2].set_xscale("log")
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].set_xlabel("measured latency on F411 (ms)", fontsize=7.4,
                       color=INK)
    axes[2].set_title("(c) accuracy vs inference latency", fontsize=7.2,
                      color=INK)
    for ax in axes:
        _frame(ax)
        ax.grid(color=GRID, lw=0.5)
    axes[0].legend(fontsize=6.2, frameon=False, ncol=6,
                   labelcolor=INK2, loc="upper center",
                   bbox_to_anchor=(1.75, -0.22))
    fig.tight_layout(pad=0.5, w_pad=1.4)
    save(fig, "figure5_accuracy_resource", FIG)


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
    save(fig, "figure3_selective_reliability", FIG)


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
    fig, axes = plt.subplots(1, len(srcs), figsize=(2.1 * len(srcs), 2.6),
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
    axes[0].legend(fontsize=5.8, frameon=False, labelcolor=INK2, ncol=3,
                   loc="upper center", bbox_to_anchor=(2.2, -0.26))
    fig.text(0.5, -0.10, "solid = selective risk under the model's own "
             "confidence ranking;  dashed = that model's oracle",
             ha="center", fontsize=6.0, color=INK2, style="italic")
    fig.tight_layout(pad=0.4, w_pad=1.0)
    save(fig, "figureS2_risk_coverage", SFIG)


