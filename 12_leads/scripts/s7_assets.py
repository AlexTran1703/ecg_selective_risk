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
    """Study design as one chain of three research questions.

    External generalisation is the spine. The quality work explains
    failures in it, selective prediction asks whether those failures can
    be recognised, and deployment asks whether the externally validated
    behaviour survives implementation -- so the three blocks are stacked,
    not arranged side by side as independent studies.

    The quality branch leaves the native 500 Hz signal and rejoins at RQ2,
    because those indicators are measured before decimation while the
    classifiers see the 100 Hz input.
    """
    FIG_H, YLO, YHI = 6.0, 1.0, 101.0
    YSPAN = YHI - YLO
    fig, ax = plt.subplots(figsize=(7.4, FIG_H), facecolor=SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(YLO, YHI)
    ax.axis("off")
    NL = chr(10)

    FILL = {"data": "#e8f1fb", "qual": "#fdefe2", "hw": "#e7f6ef"}
    EDGE = {"data": "#4a86c8", "qual": "#e07f33", "hw": "#36a173"}
    ARR = "#54544f"

    def box(x, y, w, h, kind, title, detail, fs_t=8.0, fs_d=6.8, tx=None,
            lw=1.3):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.9", linewidth=lw,
            facecolor=FILL[kind], edgecolor=EDGE[kind], zorder=2))
        cx = tx if tx is not None else x + w / 2
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
        ax.add_patch(FancyBboxPatch(
            (x, y - 1.8), 8.6, 3.6, boxstyle="round,pad=0.5", linewidth=0,
            facecolor=EDGE[kind], zorder=5))
        ax.text(x + 4.3, y, label, ha="center", va="center", fontsize=6.6,
                color="white", fontweight="bold", zorder=6)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
            color=ARR, lw=1.1, zorder=1, shrinkA=1, shrinkB=1))

    def note(x, y, t, ha="center"):
        ax.text(x, y, t, fontsize=6.3, color=INK2, ha=ha, va="center",
                style="italic", zorder=4)

    def ecg_icon(cx, cy, w, h, c):
        t = np.linspace(0, 1, 240)
        y = (0.10 * np.exp(-((t - 0.17) / 0.035) ** 2)
             - 0.13 * np.exp(-((t - 0.40) / 0.012) ** 2)
             + 1.00 * np.exp(-((t - 0.45) / 0.011) ** 2)
             - 0.30 * np.exp(-((t - 0.51) / 0.014) ** 2)
             + 0.22 * np.exp(-((t - 0.72) / 0.045) ** 2))
        ax.plot(cx + (t - 0.5) * w, cy + y * h * 0.42, lw=1.2, color=c,
                zorder=4, solid_capstyle="round")

    # ---------------------------------------------------------- the data
    box(5, 90, 90, 9, "data", "Four clinical ECG sources",
        "PTB-XL | Georgia | Chapman | Ningbo" + NL +
        "77,333 records  ->  66,379 retained, 13 harmonised labels",
        fs_t=8.6, fs_d=7.0, tx=56)
    ecg_icon(17, 94.5, 11, 5.5, EDGE["data"])

    ax.plot([50, 50], [90, 87], lw=1.1, color=ARR, zorder=1)
    ax.plot([22, 72], [87, 87], lw=1.1, color=ARR, zorder=1)
    note(51.5, 88.2, "10 s extraction", ha="left")
    arrow(22, 87, 22, 84)
    arrow(72, 87, 72, 84)

    box(5, 76, 34, 8, "qual", "Native 500 Hz signal",
        "quality indicators measured" + NL + "before decimation",
        fs_t=7.4, fs_d=6.4)
    box(50, 76, 45, 8, "data", "Anti-aliased resampling to 100 Hz",
        "model input: 12 x 1000", fs_t=7.4, fs_d=6.4)

    # --------------------------------------------------- RQ1, the spine
    arrow(72, 76, 72, 71)
    box(40, 58, 55, 13, "data", "External generalisation",
        "leave-one-source-out, six compact encoders" + NL +
        "macro AUPRC across unseen clinical sources", fs_t=8.2)
    tab(43, 71, "RQ1", "data")

    arrow(67, 58, 67, 53)
    note(68.5, 55.5, "held-out predictions", ha="left")

    # ------------------------------------ RQ2, three steps in one block
    # Drawn as a plain panel: box() centres its title, which here would
    # land on step (b) rather than heading the block.
    ax.add_patch(FancyBboxPatch(
        (14, 27), 81, 25, boxstyle="round,pad=0.9", linewidth=1.3,
        facecolor=FILL["qual"], edgecolor=EDGE["qual"], zorder=2))
    ax.text(55, 49.3, "Reliability under source and quality shift",
            ha="center", va="center", fontsize=8.2, color=INK,
            fontweight="bold", zorder=4)
    tab(17, 52, "RQ2", "qual")
    steps = [
        (43.5, "a", "what shifts between the sources",
         "six indicators, four sources"),
        (36.0, "b", "do those shifts matter diagnostically?",
         "within-source strata, prevalence-matched dAUPRC"),
        (28.5, "c", "can the model tell when it is unreliable?",
         "E-AURC, and whether it degrades with the signal"),
    ]
    for y, letter, head, sub in steps:
        ax.text(23, y, letter, fontsize=7.0, color=EDGE["qual"],
                fontweight="bold", ha="center", va="center", zorder=4)
        ax.text(27, y + 1.1, head, fontsize=7.2, color=INK, ha="left",
                va="center", zorder=4)
        ax.text(27, y - 1.6, sub, fontsize=6.3, color="#55554f", ha="left",
                va="center", zorder=4)
    for y in (39.8, 32.3):
        ax.add_patch(FancyArrowPatch(
            (23, y + 1.2), (23, y - 1.2), arrowstyle="-|>",
            mutation_scale=7, color=EDGE["qual"], lw=0.9, zorder=4))

    # The quality branch rejoins here, not at RQ1.
    ax.plot([10, 10, 13], [76, 43.5, 43.5], lw=1.1, color=ARR, zorder=1,
            solid_joinstyle="round")
    ax.add_patch(FancyArrowPatch(
        (12, 43.5), (14.6, 43.5), arrowstyle="-|>", mutation_scale=10,
        color=ARR, lw=1.1, zorder=1))
    note(10.6, 60, "quality" + NL + "indicators", ha="left")

    arrow(67, 27, 67, 22)
    note(68.5, 24.5, "externally evaluated encoders", ha="left")

    # ------------------------------------------------- RQ3, deployment
    box(14, 8, 81, 13, "hw", "Deployment-preserved performance",
        "int8 post-training quantisation | STM32F411 at 100 MHz" + NL +
        "Flash | SRAM | measured latency | accuracy-resource trade-off",
        fs_t=8.2)
    tab(17, 21, "RQ3", "hw")

    ax.text(50, 3.4,
            "Generalises?   ->   Can we trust it?   ->   Can we deploy it?",
            ha="center", va="center", fontsize=7.4, color=INK2,
            style="italic")
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
    save(fig, "figure3_signal_quality", FIG)


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
    """RQ3: what diagnostic performance each resource budget buys.

    Plotted on int8 accuracy, not float32: these are deployment operating
    points, so the y-axis has to describe the model that actually runs on
    the part. The frontier is left for the reader to read off the scatter
    rather than ringed: with six points per panel the markers were doing
    the work already, and the rings only added ink.
    """
    acc = {}
    f8 = S3B / "int8.json"
    if f8.exists():
        a8 = pd.DataFrame(json.loads(f8.read_text()))
        for m in ORDER:
            sub = a8[a8.model == m]
            if len(sub):
                acc[m] = float(sub.auprc_int8.mean())
    if not acc:                      # fall back to float32 if int8 absent
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
    axes[0].set_ylabel("int8 cross-source macro AUPRC", fontsize=7.4,
                       color=INK)
    axes[0].set_title("(a) int8 AUPRC vs Flash", fontsize=7.2,
                      color=INK)

    axes[1].axvline(SRAM_KB, color=INK, lw=1.0, ls="--")
    axes[1].text(SRAM_KB, axes[1].get_ylim()[0], "F411 128 KB  ",
                 fontsize=6.0, color=INK, rotation=90, va="bottom")
    axes[1].set_xlabel("measured total RAM (KB)", fontsize=7.4, color=INK)
    axes[1].set_title("(b) int8 AUPRC vs peak SRAM", fontsize=7.2,
                      color=INK)

    axes[2].set_xscale("log")
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].set_xlabel("measured latency on F411 (ms)", fontsize=7.4,
                       color=INK)
    axes[2].set_title("(c) int8 AUPRC vs measured latency", fontsize=7.2,
                      color=INK)
    for ax in axes:
        _frame(ax)
        ax.grid(color=GRID, lw=0.5)
    axes[0].legend(fontsize=6.2, frameon=False, ncol=6,
                   labelcolor=INK2, loc="upper center",
                   bbox_to_anchor=(1.75, -0.22))
    fig.tight_layout(pad=0.5, w_pad=1.4)
    save(fig, "figure5_operating_points", FIG)


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

    fig, axes = plt.subplots(
        1, 2, figsize=(7.4, 3.0), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.0, 0.95]})
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
    fig.tight_layout(pad=0.5, w_pad=1.8)
    save(fig, "figure2_generalisation", FIG)


def figure4_reliability(qr):
    """RQ2c: can confidence identify unreliable external predictions?

    Panels (a-d) are the risk-coverage curves themselves, one per held-out
    source, so the reader sees the selective behaviour rather than only
    its integrated scalar. Panel (e) is the integral, E-AURC, with
    record-level bootstrap intervals.

    The risk axis is exactly the quantity E-AURC integrates: records are
    ranked by confidence, retained from most to least confident, and
    selective risk is the mean loss over the retained fraction. Using a
    different selective metric here because it plots more prettily would
    make the figure and the summary describe different things.
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
    fig = plt.figure(figsize=(7.6, 4.3), facecolor=SURFACE)
    # The legend sits in the gap between the rows, so the gap has to be
    # big enough for it and no bigger.
    gs = fig.add_gridspec(2, len(srcs), height_ratios=[1.0, 0.80],
                          hspace=0.62, wspace=0.18,
                          top=0.93, bottom=0.10, left=0.09, right=0.98)

    ax0 = None
    for k, s in enumerate(srcs):
        ax = fig.add_subplot(gs[0, k], sharey=ax0) if ax0 else \
            fig.add_subplot(gs[0, k])
        ax0 = ax0 or ax
        for m in models:
            rc = d["models"][m]["risk_coverage"][s]
            ax.plot(grid, rc["risk"], lw=1.3, color=cmap(ORDER.index(m)),
                    label=NICE[m] if k == 0 else None)
        ax.set_title(f"({chr(97 + k)}) {sname(s)}", fontsize=7.4,
                     color=INK, pad=3)
        ax.set_xlabel("coverage", fontsize=7.0, color=INK)
        if k == 0:
            ax.set_ylabel("selective risk", fontsize=7.0, color=INK)
        else:
            ax.tick_params(labelleft=False)
        _frame(ax)
        ax.grid(color=GRID, lw=0.5)

    ax = fig.add_subplot(gs[1, :])
    order = sorted(models, key=lambda m: d["models"][m]["eaurc"])
    yy = np.arange(len(order))[::-1]
    pt = [d["models"][m]["eaurc"] for m in order]
    lo = [d["models"][m]["eaurc"] - d["models"][m]["eaurc_lo"]
          for m in order]
    hi = [d["models"][m]["eaurc_hi"] - d["models"][m]["eaurc"]
          for m in order]
    ax.errorbar(pt, yy, xerr=[lo, hi], fmt="none", ecolor=INK2,
                elinewidth=1.1, capsize=2.6, zorder=2)
    for y_, m in zip(yy, order):
        ax.scatter(d["models"][m]["eaurc"], y_, s=42,
                   color=cmap(ORDER.index(m)), edgecolor=SURFACE, lw=0.9,
                   zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels([NICE[m] for m in order], fontsize=7.0)
    ax.set_xlabel("mean E-AURC over four held-out sources, "
                  "lower = better error ranking", fontsize=7.2, color=INK)
    ax.set_title(f"({chr(97 + len(srcs))}) selective reliability, "
                 f"record-level bootstrap 95% CI", fontsize=7.4,
                 color=INK, pad=3)
    _frame(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.4, frameon=False,
               labelcolor=INK2, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 0.515))
    save(fig, "figure4_reliability", FIG)


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


