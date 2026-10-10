r"""Tables and figures for the MCU architecture benchmark.

    .\.venv\Scripts\python.exe benchmark_mcu\scripts\b4_assets.py

Four tables and five figures, built only from measured artefacts:
cached held-out predictions, the int8 comparison, and what the board
reported. Nothing here is estimated from a parameter count.

    Table 1   cohort and leave-one-source-out protocol
    Table 2   the ten families: year, defining block, size, MACs
    Table 3   external macro-AUPRC and AUROC per held-out source, CIs
    Table 4   measured deployment: Flash, SRAM, latency, pass/fail

    Figure 2  cross-source performance heatmaps, both budgets
    Figure 3  external performance distribution per family
    Figure 4  what int8 costs
    Figure 5  accuracy against measured latency, SRAM as bubble area
    Figure S1 MACs against measured latency -- the proxy under test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "benchmark_mcu"
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(ROOT / "12_leads" / "src"))

from ecgmcu.data import SOURCES, FS_OUT                          # noqa: E402
from mcubench.models import ZOO, NICE, YEAR, BLOCK               # noqa: E402

import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

TRAIN = HERE / "results" / f"train_{FS_OUT}hz"
INT8 = HERE / "results" / f"int8_{FS_OUT}hz"
DEPLOY = HERE / "results" / f"deploy_{FS_OUT}hz"
FIG, TAB = HERE / "figures", HERE / "tables"
BUDGETS = ["compact", "standard"]
NL = chr(10)
FLASH_KB, SRAM_KB = 512, 128
REPS = 400

SURFACE, INK, INK2, GRID = "#fdfcf8", "#2d2d2b", "#6b6b66", "#e3e1da"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 8, "axes.grid": False, "axes.spines.top": False,
    "axes.spines.right": False})


def sname(s):
    return {"PTBXL": "PTB-XL"}.get(s, s)


def _frame(ax):
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
        ax.spines[sp].set_linewidth(0.8)


def save(fig, name, dest=FIG):
    dest.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(dest / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}")


def emit(df, name, caption, dest=TAB):
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / f"{name}.csv")
    (dest / f"{name}.tex").write_text(df.to_latex(escape=False), "utf-8")
    (dest / f"{name}.txt").write_text(
        caption + "\n" + "-" * 78 + "\n" + df.to_string(), "utf-8")
    print(f"  tables/{name}")


def ap_fast(y, s):
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    tp = np.cumsum(y)
    if not tp[-1]:
        return np.nan
    end = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    te = tp[end]
    return float((np.diff(np.r_[0.0, te / tp[-1]]) * (te / (end + 1.0))).sum())


def macro_ap(y, p):
    v = [ap_fast(y[:, c], p[:, c]) for c in range(y.shape[1])
         if 0 < y[:, c].sum() < y.shape[0]]
    return float(np.mean(v)) if v else np.nan


def macro_auroc(y, p):
    from sklearn.metrics import roc_auc_score
    v = [roc_auc_score(y[:, c], p[:, c]) for c in range(y.shape[1])
         if 0 < y[:, c].sum() < y.shape[0]]
    return float(np.mean(v)) if v else np.nan


def load_runs():
    runs = {}
    for f in TRAIN.glob("*.npz"):
        parts = f.stem.split("__")
        if len(parts) != 3:
            continue
        d = np.load(f)
        runs[tuple(parts)] = {"y": d["test_y"].astype(np.int8),
                              "p": d["test_prob"].astype(np.float64),
                              "params": int(d["params"]),
                              "peak": int(d["peak_act"]),
                              "macs": int(d["macs"]),
                              "width": float(d["width"])}
    return runs


def load_deploy():
    f = DEPLOY / "deploy.json"
    if not f.exists():
        return {}
    return {(r["model"], r["budget"]): r for r in json.loads(f.read_text())}


def models_present(runs, budget):
    return [m for m in ZOO if any(k[0] == m and k[1] == budget for k in runs)]


# ----------------------------------------------------------------- tables
def table1(runs):
    rows = []
    for s in SOURCES:
        k = next((k for k in runs if k[2] == s), None)
        if k:
            y = runs[k]["y"]
            rows.append({"held-out source": sname(s),
                         "test records": y.shape[0],
                         "label density": round(float(y.sum(1).mean()), 2)})
    emit(pd.DataFrame(rows).set_index("held-out source"), "table1_protocol",
         "Table 1. Leave-one-source-out protocol. Each row is one "
         "rotation: the model trains on the other three sources and is "
         "scored on every record of this one. Inputs are 10 s, 12-lead, "
         f"decimated to {FS_OUT} Hz ({12} x {FS_OUT * 10} samples), with "
         "13 harmonised labels. The cohort and partitions are imported "
         "unchanged from the cross-source study so the two cannot drift.")


def table2(runs, dep):
    rows = []
    for b in BUDGETS:
        for m in models_present(runs, b):
            k = next(k for k in runs if k[0] == m and k[1] == b)
            r = runs[k]
            d = dep.get((m, b), {})
            rows.append({
                "model": NICE[m], "budget": b, "year": YEAR[m],
                "defining block": BLOCK[m],
                "width": round(r["width"], 2),
                "params": r["params"],
                "int8 weights KB": round(r["params"] / 1024, 1),
                "peak act KB (graph)": round(r["peak"] / 1024, 1),
                "MACs (M)": round(r["macs"] / 1e6, 2),
                "MACC (M, ST tool)": round((d.get("macc") or 0) / 1e6, 2)})
    emit(pd.DataFrame(rows).set_index(["budget", "model"]),
         "table2_architectures",
         "Table 2. The ten families and the two size budgets. Each family "
         "is scaled to the largest width that satisfies both budgets "
         "rather than to a common width, so every design is given the "
         "biggest version of itself the part will accept. Parameters are "
         "the deployed count: RepViT is reparameterised first, so its "
         "folded training branches are not charged to Flash. Peak "
         "activation is derived from the graph as an SRAM proxy; the "
         "measured figure is in Table 4. MACC from the ST tool is "
         "reported beside the analytic count as a consistency check.")


def table3(runs, boot):
    rows = []
    for b in BUDGETS:
        for m in models_present(runs, b):
            rec = {"model": NICE[m], "budget": b}
            for s in SOURCES:
                k = (m, b, s)
                if k in runs:
                    rec[sname(s)] = round(macro_ap(runs[k]["y"],
                                                   runs[k]["p"]), 4)
            pt, lo, hi = boot[(m, b)]
            rec["mean AUPRC"] = round(pt, 4)
            rec["95% CI"] = f"({lo:.4f}, {hi:.4f})"
            rec["mean AUROC"] = round(np.mean(
                [macro_auroc(runs[(m, b, s)]["y"], runs[(m, b, s)]["p"])
                 for s in SOURCES if (m, b, s) in runs]), 4)
            rows.append(rec)
    emit(pd.DataFrame(rows).set_index(["budget", "model"]),
         "table3_external",
         "Table 3. External macro-AUPRC by held-out source, with the "
         "mean over the four rotations and a record-level bootstrap "
         f"interval ({REPS} replicates, seed 0, records resampled within "
         "each source and the equal-source mean formed inside every "
         "replicate). Macro AUROC is given alongside because average "
         "precision has a prevalence floor and the two do not always "
         "order the sources the same way. Intervals overlap for most "
         "pairs: this table is not a ranking.")


def table4(runs, dep):
    rows = []
    for b in BUDGETS:
        for m in models_present(runs, b):
            d = dep.get((m, b))
            if not d:
                continue
            fkb = (d.get("total_flash_b") or 0) / 1024
            rkb = (d.get("total_ram_b") or 0) / 1024
            ms = d.get("ms_median")
            rows.append({
                "model": NICE[m], "budget": b,
                "Flash KB": round(fkb, 1),
                "peak SRAM KB": round(rkb, 1),
                "SRAM free KB": round(SRAM_KB - rkb, 1),
                "MACC (M)": round((d.get("macc") or 0) / 1e6, 2),
                "latency ms": round(ms, 1) if ms else np.nan,
                "p95 ms": round(d["ms_p95"], 1) if d.get("ms_p95") else np.nan,
                "cycles": d.get("cycles_median"),
                "ms per MMACC": round(ms / ((d.get("macc") or 1) / 1e6), 1)
                if ms else np.nan,
                "RTF": round(d["rtf"], 4) if d.get("rtf") else np.nan,
                "deployed": "yes" if d.get("deployed") else "NO"})
    emit(pd.DataFrame(rows).set_index(["budget", "model"]),
         "table4_deployment",
         "Table 4. Measured on a physical STM32F411 at 100 MHz. Flash and "
         "peak SRAM are the totals the ST Edge AI toolchain reports for "
         "the linked image, including the generated runtime, not weights "
         "alone. Latency is the median of 32 timed inferences after a "
         "warm-up, read from the DWT cycle counter; p95 is from the same "
         "series. RTF is latency over the 10 s acquisition window, so RTF "
         "< 1 means inference finishes inside the recording. A model is "
         "marked deployed only if it linked, flashed and returned the "
         "expected output on the part. ms per MMACC is the column that "
         "matters for the proxy question: if MACs predicted latency it "
         "would be constant.")


def bootstrap(runs):
    """Equal-source mean AUPRC with a record-level interval."""
    rng = np.random.default_rng(0)
    out = {}
    for b in BUDGETS:
        for m in models_present(runs, b):
            srcs = [s for s in SOURCES if (m, b, s) in runs]
            pt = float(np.mean([macro_ap(runs[(m, b, s)]["y"],
                                         runs[(m, b, s)]["p"])
                                for s in srcs]))
            reps = np.empty(REPS)
            for i in range(REPS):
                vals = []
                for s in srcs:
                    r = runs[(m, b, s)]
                    idx = rng.integers(0, r["y"].shape[0], r["y"].shape[0])
                    vals.append(macro_ap(r["y"][idx], r["p"][idx]))
                reps[i] = np.mean(vals)
            out[(m, b)] = (pt, float(np.percentile(reps, 2.5)),
                           float(np.percentile(reps, 97.5)))
            print(f"    bootstrapped {m}/{b}", flush=True)
    return out


# ---------------------------------------------------------------- figures
def figure1(runs, dep):
    """Study design: one cohort, two budgets, ten families, one board.

    The diagram exists to make the control structure legible, because
    the benchmark's credibility rests on it: everything except the
    architecture is pinned, and the only concession -- channel width --
    is set by the budget rather than chosen, so each family is given the
    largest version of itself the part will accept.

    Counts are read from the artefacts rather than typed in, so the
    figure cannot drift away from the data it describes.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    n_runs = len(runs)
    n_dep = sum(1 for d in dep.values() if d.get("deployed"))
    # The partial-ratio ablation is a variant of FasterNet, not an
    # eleventh family, and must not be counted as one.
    fams = {k[0] for k in runs}
    n_abl = sum(1 for f in fams if "_d2" in f)
    n_fam = len(fams) - n_abl
    n_rec = sum(runs[k]["y"].shape[0] for k in runs
                if k[0] == next(iter(runs))[0] and k[1] == BUDGETS[0])

    FIG_H, YLO, YHI = 5.2, 10.0, 99.0
    YSPAN = YHI - YLO
    fig, ax = plt.subplots(figsize=(7.2, FIG_H), facecolor=SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(YLO, YHI)
    ax.axis("off")

    FILL = {"data": "#eaf2fc", "model": "#fdf0e6", "hw": "#e9f7f1"}
    EDGE = {"data": "#3f7cbf", "model": "#d9792b", "hw": "#2f9a6d"}
    ARR, TXT = "#5a5a55", "#2d2d2b"
    L, R = 16, 84
    MID = (L + R) / 2

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

    def label(x, y, t, ha="left"):
        ax.text(x, y, t, fontsize=6.6, color=INK2, ha=ha, va="center",
                style="italic", zorder=4)

    box(4, 88, 92, 10, "data", "Four clinical ECG sources",
        "PTB-XL | Georgia | Chapman | Ningbo" + NL
        + f"{n_rec:,} retained records, 13 harmonised labels, "
          "10 s at 100 Hz",
        fs_t=9.0, fs_d=7.4)
    down(MID, 88, 80.4)
    label(MID + 2, 84, "leave-one-source-out: train on three, "
                       "test on the fourth, four rotations")

    box(L, 69, R - L, 11, "model",
        f"{n_fam} encoder families x 2 size budgets"
        + (f"  (+{n_abl} ablation)" if n_abl else ""),
        "compact <= 32 KB int8 weights | standard <= 128 KB" + NL
        + "each family scaled to the largest width that fits," + NL
        + "everything else pinned: loss, optimiser, epochs, seed")
    down(MID, 69, 61.4)
    label(MID + 2, 65, f"{n_runs} trained models")

    box(L, 50, R - L, 11, "data", "RQ1   does the family decide accuracy?",
        "external macro-AUPRC and AUROC per held-out source," + NL
        + "float32 and int8, record-level bootstrap intervals")
    down(MID, 50, 42.4)
    label(MID + 2, 46, "int8 graphs, QDQ, calibrated on training sources")

    box(L, 31, R - L, 11, "hw", "RQ2   does the efficiency claim hold?",
        "ST Edge AI -> arm-none-eabi -> flash over SWD" + NL
        + f"measured Flash, peak SRAM and DWT latency; {n_dep} of "
          f"{len(dep)} ran")
    down(MID, 31, 23.4)
    label(MID + 2, 27, "measured against predicted")

    box(L, 12, R - L, 11, "hw", "RQ3   could we have known in advance?",
        "graph-derived activation proxy against the SRAM" + NL
        + "the allocator actually required")

    save(fig, "figure1_design")


def figure2(runs):
    """Cross-source AUPRC, both budgets, one heatmap each."""
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), facecolor=SURFACE)
    for ax, b in zip(axes, BUDGETS):
        ms = models_present(runs, b)
        M = np.array([[macro_ap(runs[(m, b, s)]["y"], runs[(m, b, s)]["p"])
                       if (m, b, s) in runs else np.nan for s in SOURCES]
                      for m in ms])
        im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0.38, vmax=0.64)
        ax.set_xticks(range(len(SOURCES)))
        ax.set_xticklabels([sname(s) for s in SOURCES], fontsize=6.4,
                           rotation=20, ha="right")
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels([NICE[m] for m in ms] if b == BUDGETS[0] else [],
                           fontsize=6.6)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        fontsize=5.6,
                        color="#ffffff" if M[i, j] > 0.57 else INK)
        ax.set_title(f"({'ab'[BUDGETS.index(b)]}) {b} budget",
                     fontsize=7.6, color=INK, pad=4)
        ax.tick_params(length=0, colors=INK2)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=6, colors=INK2, length=2)
    cb.set_label("external macro AUPRC", fontsize=6.4, color=INK2)
    save(fig, "figure2_cross_source")


def figure3(runs, boot):
    """Spread across sources, per family -- the ranking that is not one."""
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), facecolor=SURFACE,
                             sharex=True)
    cmap = plt.get_cmap("Dark2")
    for ax, b in zip(axes, BUDGETS):
        ms = sorted(models_present(runs, b), key=lambda m: boot[(m, b)][0])
        yy = np.arange(len(ms))
        for y_, m in zip(yy, ms):
            for k, s in enumerate(SOURCES):
                if (m, b, s) in runs:
                    ax.scatter(macro_ap(runs[(m, b, s)]["y"],
                                        runs[(m, b, s)]["p"]), y_,
                               s=16, color=cmap(k), edgecolor=SURFACE,
                               lw=0.4, zorder=3,
                               label=sname(s) if m == ms[0] else None)
            pt, lo, hi = boot[(m, b)]
            ax.plot([lo, hi], [y_, y_], lw=1.2, color=INK2, zorder=2)
            ax.scatter(pt, y_, s=36, color=INK, zorder=4, marker="|")
        ax.set_yticks(yy)
        ax.set_yticklabels([NICE[m] for m in ms] if b == BUDGETS[0] else [],
                           fontsize=6.8)
        ax.set_xlabel("external macro AUPRC", fontsize=7.0, color=INK)
        ax.set_title(f"({'ab'[BUDGETS.index(b)]}) {b} budget",
                     fontsize=7.6, color=INK, pad=4)
        _frame(ax)
        ax.grid(axis="x", color=GRID, lw=0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.2, frameon=False, labelcolor=INK2, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               title="held-out source  (bar = mean with bootstrap 95% CI)",
               title_fontsize=6.2)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.9, bottom=0.26,
                        wspace=0.08)
    save(fig, "figure3_distribution")


def figure4():
    """What int8 costs, per family and budget."""
    f = INT8 / "int8.json"
    if not f.exists():
        return
    d = pd.DataFrame(json.loads(f.read_text()))
    fig, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=SURFACE)
    cmap = plt.get_cmap("Dark2")
    order = (d.groupby("model").d_auprc.mean().sort_values().index.tolist())
    yy = np.arange(len(order))
    for i, b in enumerate(BUDGETS):
        g = d[d.budget == b]
        for y_, m in zip(yy, order):
            v = g[g.model == m].d_auprc
            if len(v):
                ax.scatter(v.mean(), y_ + (0.16 if i == 0 else -0.16), s=26,
                           color=cmap(i), edgecolor=SURFACE, lw=0.5,
                           zorder=3, label=b if m == order[0] else None)
                ax.plot([v.min(), v.max()],
                        [y_ + (0.16 if i == 0 else -0.16)] * 2, lw=0.9,
                        color=cmap(i), alpha=0.5, zorder=2)
    ax.axvline(0, color=INK, lw=1.0, ls="--", zorder=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([NICE[m] for m in order], fontsize=7.0)
    ax.set_xlabel("int8 minus float32 external macro AUPRC  "
                  "(point = mean, bar = range over the four sources)",
                  fontsize=6.8, color=INK)
    ax.set_title("quantisation costs less than the gap between families",
                 fontsize=7.6, color=INK, pad=4)
    ax.legend(fontsize=6.4, frameon=False, labelcolor=INK2, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, -0.38),
              title="budget", title_fontsize=6.4)
    _frame(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.subplots_adjust(left=0.26, right=0.98, top=0.9, bottom=0.31)
    save(fig, "figure4_int8_cost")


def figure5(runs, boot, dep):
    """Accuracy against measured latency; SRAM is the bubble."""
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), facecolor=SURFACE)
    cmap = plt.get_cmap("tab20")
    for ax, b in zip(axes, BUDGETS):
        ms = [m for m in models_present(runs, b) if (m, b) in dep
              and dep[(m, b)].get("ms_median")]
        for i, m in enumerate(ms):
            d = dep[(m, b)]
            rkb = (d.get("total_ram_b") or 0) / 1024
            ax.scatter(d["ms_median"], boot[(m, b)][0],
                       s=12 + 9 * rkb, color=cmap(list(ZOO).index(m) % 20),
                       alpha=0.78, edgecolor=SURFACE, lw=0.8, zorder=3,
                       label=NICE[m] if b == BUDGETS[0] else None)
        ax.set_xscale("log")
        # Minor log labels collide at this width; majors are enough.
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("measured latency on the F411 (ms, log)",
                      fontsize=7.0, color=INK)
        if b == BUDGETS[0]:
            ax.set_ylabel("external macro AUPRC", fontsize=7.0, color=INK)
        ax.set_title(f"({'ab'[BUDGETS.index(b)]}) {b} budget",
                     fontsize=7.6, color=INK, pad=4)
        ax.margins(0.18)
        _frame(ax)
        ax.grid(color=GRID, lw=0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.2, frameon=False, labelcolor=INK2, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.175, "bubble area is measured peak SRAM;  up and to "
             "the left is better", ha="center", fontsize=6.4, color=INK2,
             style="italic")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.30,
                        wspace=0.22)
    save(fig, "figure5_accuracy_latency")


def figure_s1(runs, dep):
    """MACs against measured latency -- the proxy, tested."""
    pts = [(d["macc"] / 1e6, d["ms_median"], m, b)
           for (m, b), d in dep.items()
           if d.get("ms_median") and d.get("macc")]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.4), facecolor=SURFACE)
    cmap = plt.get_cmap("Dark2")
    for i, b in enumerate(BUDGETS):
        g = [p for p in pts if p[3] == b]
        if not g:
            continue
        ax.scatter([p[0] for p in g], [p[1] for p in g], s=30,
                   color=cmap(i), edgecolor=SURFACE, lw=0.6, zorder=3,
                   label=b)
        for x, y, m, _ in g:
            ax.annotate(NICE[m], (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=5.0,
                        color=INK2)
    allp = np.array([(p[0], p[1]) for p in pts])
    # A correlation needs more than a couple of points; with a partial
    # deployment set this would otherwise print a divide-by-zero NaN
    # into the figure.
    r = (float(np.corrcoef(allp[:, 0], allp[:, 1])[0, 1])
         if allp.shape[0] >= 3 else float("nan"))
    ratio = (allp[:, 1] / allp[:, 0])
    ax.set_xlabel("MACC (millions, ST tool)", fontsize=7.2, color=INK)
    ax.set_ylabel("measured latency (ms)", fontsize=7.2, color=INK)
    ax.set_title("MACs is not latency", fontsize=7.8, color=INK, pad=4)
    rtxt = f"r = {r:+.2f} overall, but " if np.isfinite(r) else ""
    ax.annotate(rtxt + f"ms per MMACC spans "
                f"{ratio.min():.0f} to {ratio.max():.0f}" + chr(10)
                + f"-- a {ratio.max() / ratio.min():.1f}x range at equal "
                  f"arithmetic cost",
                xy=(0.03, 0.96), xycoords="axes fraction", va="top",
                fontsize=6.0, color=INK2, linespacing=1.4)
    ax.legend(fontsize=6.4, frameon=False, labelcolor=INK2,
              loc="lower right", title="budget", title_fontsize=6.4)
    ax.margins(0.16)
    _frame(ax)
    ax.grid(color=GRID, lw=0.5)
    save(fig, "figureS1_macs_vs_latency")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    runs, dep = load_runs(), load_deploy()
    print(f"{len(runs)} runs, {len(dep)} deployments")
    print("bootstrapping:")
    boot = bootstrap(runs)
    print("tables:")
    table1(runs)
    table2(runs, dep)
    table3(runs, boot)
    table4(runs, dep)
    print("figures:")
    figure1(runs, dep)
    figure2(runs)
    figure3(runs, boot)
    figure4()
    figure5(runs, boot, dep)
    figure_s1(runs, dep)
    print(f"\nfigures: {FIG}\ntables:  {TAB}")


if __name__ == "__main__":
    main()
