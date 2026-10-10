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


# M-identifiers and colours are defined once and shared, so a model is
# the same colour and the same label in every figure and in Table 2.
MID = {m: f"M{i + 1}" for i, m in enumerate(ZOO)}
_PAL = plt.get_cmap("tab10")
MODEL_COLOR = {m: _PAL(i % 10) for i, m in enumerate(ZOO)}


def sname(s):
    return {"PTBXL": "PTB-XL"}.get(s, s)


def _frame(ax):
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
        ax.spines[sp].set_linewidth(0.8)


def save(fig, name, dest=FIG, svg=False):
    dest.mkdir(parents=True, exist_ok=True)
    for ext in (("png", "pdf", "svg") if svg else ("png", "pdf")):
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
def table1(runs, boot):
    mean_by_src = {}
    for src in SOURCES:
        v = [macro_ap(runs[k]["y"], runs[k]["p"]) for k in runs if k[2] == src]
        if v:
            mean_by_src[src] = float(np.mean(v))
    best = max(mean_by_src, key=mean_by_src.get) if mean_by_src else None
    worst = min(mean_by_src, key=mean_by_src.get) if mean_by_src else None
    rows = []
    for src in SOURCES:
        k = next((k for k in runs if k[2] == src), None)
        if not k:
            continue
        y = runs[k]["y"]
        n_cls = int(sum(1 for c in range(y.shape[1])
                        if 0 < y[:, c].sum() < y.shape[0]))
        rows.append({
            "held-out source": sname(src),
            "test records": int(y.shape[0]),
            "label density": round(float(y.sum(1).mean()), 2),
            "positive rate": round(float(y.mean()), 4),
            "evaluated classes": n_cls,
            "mean AUPRC over models": round(mean_by_src.get(src, np.nan), 4),
            "note": ("lowest mean AUPRC across models" if src == worst else
                     "highest mean AUPRC across models" if src == best
                     else "")})
    emit(pd.DataFrame(rows).set_index("held-out source"), "table1_protocol",
         "Table 1. Held-out source characteristics under leave-one-source-"
         "out external evaluation. Each row is one rotation: models train "
         "on the other three sources and are scored on every record of "
         "this one. Label density is the mean number of positive labels "
         "per record, positive rate the overall fraction of positive "
         "label slots, both after harmonisation to the 13-label space. "
         "Evaluated classes counts those with at least one positive and "
         "one negative in that source, which is the set the macro average "
         f"is taken over. Inputs are 10 s, 12-lead, decimated to {FS_OUT} "
         "Hz. Patient counts are omitted: only one source repeats "
         "patients and the definitions are not comparable across the four.")


def table2(runs, dep):
    rows = []
    for b in BUDGETS:
        for m in models_present(runs, b):
            k = next(k for k in runs if k[0] == m and k[1] == b)
            r = runs[k]
            d = dep.get((m, b), {})
            rows.append({
                "id": MID[m],
                "model": NICE[m], "budget": b, "year": YEAR[m],
                "defining block": BLOCK[m],
                "width": round(r["width"], 2),
                "params": r["params"],
                "int8 weights KB": round(r["params"] / 1024, 1),
                "peak act KB (graph)": round(r["peak"] / 1024, 1),
                "MACC (M, ST tool)": round((d.get("macc") or 0) / 1e6, 2)})
    emit(pd.DataFrame(rows).set_index(["budget", "model"]),
         "table2_architectures",
         "Table 2. The ten families and the two size budgets. The "
         "identifier column is the short label used in Figures 5 and 6. Each family "
         "is scaled to the largest width that satisfies both budgets "
         "rather than to a common width, so every design is given the "
         "biggest version of itself the part will accept. Parameters are "
         "the deployed count: RepViT is reparameterised first, so its "
         "folded training branches are not charged to Flash. Peak "
         "activation is derived from the graph as an SRAM proxy; the "
         "measured figure is in Table 4, and the two differ "
         "substantially. MACC is taken from the deployment toolchain "
         "rather than counted analytically, so it is what the device "
         "executes. FasterNet is reported at its published partial ratio "
         "only.")


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
            rec["equal-source mean AUPRC"] = round(pt, 4)
            rec["95% CI"] = f"({lo:.4f}, {hi:.4f})"
            rec["mean AUROC"] = round(np.mean(
                [macro_auroc(runs[(m, b, s)]["y"], runs[(m, b, s)]["p"])
                 for s in SOURCES if (m, b, s) in runs]), 4)
            rows.append(rec)
    t = pd.DataFrame(rows)
    t["rank in budget"] = (t.groupby("budget")["equal-source mean AUPRC"]
                           .rank(ascending=False, method="min").astype(int))
    emit(t.set_index(["budget", "model"]),
         "table3_external",
         "Table 3. External macro-AUPRC by held-out source, with the "
         "mean over the four rotations and a record-level bootstrap "
         f"interval ({REPS} replicates, seed 0, records resampled within "
         "each source and the equal-source mean formed inside every "
         "replicate). Macro AUROC is given alongside because average "
         "precision has a prevalence floor and the two do not always "
         "order the sources the same way. Intervals overlap for most "
         "pairs, so the rank column orders point estimates and must not "
         "be read as a significance ordering. The intervals quantify "
         "uncertainty conditional on these four held-out datasets; they "
         "are not between-hospital transportability intervals.")


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
                "MACC (M)": round((d.get("macc") or 0) / 1e6, 2),
                "median latency ms": round(ms, 1) if ms else np.nan,
                "cycles": d.get("cycles_median"),
                "ms per MMACC": round(ms / ((d.get("macc") or 1) / 1e6), 1)
                if ms else np.nan,
                "RTF": round(d["rtf"], 4) if d.get("rtf") else np.nan,
                "deployed": "yes" if d.get("deployed") else "NO"})
    if not rows:
        # Allows assets to be built from training alone, before the
        # board stage has run, instead of failing the whole build.
        print("  (no deployments yet -- table4 skipped)")
        return
    emit(pd.DataFrame(rows).set_index(["budget", "model"]),
         "table4_deployment",
         "Table 4. Measured on a physical STM32F411 at 100 MHz. Flash and "
         "peak SRAM are the totals the ST Edge AI toolchain reports for "
         "the linked image, including the generated runtime, not weights "
         "alone. Latency is the median of 32 timed inferences after a "
         "warm-up, read from the DWT cycle counter. No p95 column is "
         "reported: across 32 runs the series is near-constant -- for a "
         "representative model, 7 distinct values spanning 4,188 cycles, "
         "0.007% of the median -- so p95 and the median coincide to the "
         "reported precision. Timing excludes preprocessing and input "
         "transfer. RTF is latency over the 10 s acquisition window, so "
         "RTF "
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
    """Benchmark design, as three labelled stages.

    Laid out to be read rather than narrated: a banded stage per logical
    step, one block per stage, and the three questions side by side with
    the claim each one is allowed to support ruled off beneath it. Every
    count is read from the artefacts, so the figure cannot drift from
    the data it describes.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    n_runs = len(runs)
    n_dep_ok = sum(1 for d in dep.values() if d.get("deployed"))
    n_fam = len({k[0] for k in runs})
    n_rec = sum(runs[k]["y"].shape[0] for k in runs
                if k[0] == next(iter(runs))[0] and k[1] == BUDGETS[0])

    FIG_W, FIG_H = 7.6, 5.4
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    BAND = "#f1efe8"
    FILL = {"data": "#e8f0fa", "model": "#eaf3ec", "rq": "#f7f6f2"}
    EDGE = {"data": "#3f7cbf", "model": "#4e8f69", "rq": "#9b9b93"}
    ARR = "#6b6b63"

    def band(y0, y1, label):
        ax.add_patch(plt.Rectangle((1, y0), 98, y1 - y0, facecolor=BAND,
                                   edgecolor="none", zorder=0))
        ax.text(3, y1 - 2.4, label, fontsize=6.4, color=INK2,
                fontweight="bold", va="center", zorder=3)

    def box(x, y, w, h, kind, title, body, rule=None,
            fs_t=8.2, fs_b=6.9):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6", linewidth=1.2,
            facecolor=FILL[kind], edgecolor=EDGE[kind], zorder=2))
        t_lines = title.count(NL) + 1
        ty = y + h - 2.2 - (t_lines - 1) * 1.6
        ax.text(x + w / 2, ty, title, ha="center", va="top",
                fontsize=fs_t, color=INK, fontweight="bold", zorder=4,
                linespacing=1.4)
        top = ty - t_lines * fs_t * 0.26 - 1.0
        ry = y + 5.2 if rule else y
        ax.text(x + w / 2, (top + ry) / 2, body, ha="center",
                va="center", fontsize=fs_b, color=INK, zorder=4,
                linespacing=1.55)
        if rule:
            ax.plot([x + 2.2, x + w - 2.2], [ry, ry], lw=0.8,
                    color=EDGE[kind], zorder=4)
            ax.text(x + w / 2, y + 2.6, rule, ha="center", va="center",
                    fontsize=6.1, color=INK2, style="italic", zorder=4,
                    linespacing=1.4)

    def arrow(x, y1, y2, side=None):
        ax.add_patch(FancyArrowPatch(
            (x, y1), (x, y2), arrowstyle="-|>", mutation_scale=10,
            color=ARR, lw=1.1, zorder=1, shrinkA=0, shrinkB=0))
        if side:
            ax.text(x + 2, (y1 + y2) / 2, side, fontsize=6.2, color=INK2,
                    ha="left", va="center", style="italic", zorder=4)

    # ---- stage 1 -------------------------------------------------------
    band(74, 99, "STAGE 1  ·  DATA AND PROTOCOL")
    box(8, 78, 84, 15, "data", "Four clinical ECG sources",
        "PTB-XL  |  Georgia  |  Chapman  |  Ningbo" + NL
        + f"{n_rec:,} retained records, 13 harmonised labels, 10 s at "
          "100 Hz")
    arrow(50, 78, 71.5,
          "leave-one-source-out: train on three, test on the fourth, "
          "four rotations")

    # ---- stage 2 -------------------------------------------------------
    band(46, 72, "STAGE 2  ·  MODELS AND CONTROLLED BUDGETS")
    box(8, 50, 84, 17, "model",
        f"{n_fam} encoder families  ×  2 size budgets",
        "compact ≤ 32 KB INT8 weights   |   standard ≤ 128 KB"
        + NL + "each family scaled to the largest width that fits" + NL
        + "everything else pinned: loss, optimiser, epochs, seed")
    arrow(50, 50, 43.5, f"{n_runs} trained models")

    # ---- stage 3 -------------------------------------------------------
    band(1, 44, "STAGE 3  ·  RESEARCH QUESTIONS")
    rq = [
        ("RQ1" + NL + "External discrimination",
         "external macro-AUPRC and" + NL + "AUROC per held-out source," + NL
         + "FP32 and INT8, record-level" + NL + "bootstrap intervals",
         "Record-level intervals ≠" + NL
         + "between-source uncertainty"),
        ("RQ2" + NL + "Hardware characterisation",
         "ST Edge AI → arm-none-eabi" + NL + "→ flash over SWD;"
         + NL + "measured Flash, peak SRAM" + NL
         + f"and DWT latency; {n_dep_ok} of {len(dep)} ran",
         "MACC ≠" + NL + "measured latency"),
        ("RQ3" + NL + "Performance–resource trade-offs",
         "observed trade-offs across" + NL + "families and budgets;" + NL
         + "non-dominated configurations" + NL + "among those tested",
         "Pareto frontier is descriptive," + NL + "not architecture search"),
    ]
    w, gap = 28.5, 3.2
    x0 = (100 - (3 * w + 2 * gap)) / 2
    for k, (title, body, rule) in enumerate(rq):
        box(x0 + k * (w + gap), 5, w, 33, "rq", title, body, rule,
            fs_t=7.0, fs_b=6.4)

    save(fig, "figure1_design", svg=True)


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
                best = i == int(np.nanargmax(M[:, j]))
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        fontsize=5.6,
                        fontweight="bold" if best else "normal",
                        color="#ffffff" if M[i, j] > 0.57 else INK)
                if best:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False,
                        edgecolor=INK, lw=1.3, zorder=4))
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
        ax.set_xlabel("external macro-AUPRC", fontsize=7.0, color=INK)
        ax.set_title(f"({'ab'[BUDGETS.index(b)]}) {b} budget",
                     fontsize=7.6, color=INK, pad=4)
        _frame(ax)
        ax.grid(axis="x", color=GRID, lw=0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.2, frameon=False, labelcolor=INK2, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               title="held-out source;  black bar = equal-source mean with "
                     "record-level bootstrap 95% CI",
               title_fontsize=6.2)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.9, bottom=0.26,
                        wspace=0.08)
    save(fig, "figure3_distribution")


def figure4():
    """Effect of INT8 quantisation, as a forest plot over every family.

    The interval is a paired record-level bootstrap of the equal-source
    mean difference, computed by b2b_int8_ci.py. It is not the range of
    the four per-source differences: that range describes how much the
    effect varies between these particular datasets and cannot narrow
    with more records, so it answers a different question and must not
    be drawn where a reader expects a confidence interval.

    Height scales with the number of rows rather than being fixed, so a
    ten-model figure does not inherit the spacing of a three-model one.
    """
    f = INT8 / "int8_ci.json"
    if not f.exists():
        print("  (no int8 CI -- run b2b_int8_ci.py)")
        return
    d = pd.DataFrame(json.loads(f.read_text()))
    order = (d.groupby("model").d_auprc.mean()
             .sort_values(ascending=False).index.tolist())
    cmap = plt.get_cmap("Dark2")
    h = 1.1 + 0.34 * len(order)
    fig, ax = plt.subplots(figsize=(6.6, h), facecolor=SURFACE)
    yy = np.arange(len(order))[::-1]
    off = {BUDGETS[0]: 0.17, BUDGETS[1]: -0.17}
    for y_, m in zip(yy, order):
        for i_, b in enumerate(BUDGETS):
            g = d[(d.model == m) & (d.budget == b)]
            if g.empty:
                continue
            r = g.iloc[0]
            yk = y_ + off[b]
            ax.plot([r.lo, r.hi], [yk, yk], lw=1.3, color=cmap(i_),
                    zorder=2, solid_capstyle="round")
            ax.scatter(r.d_auprc, yk, s=24, color=cmap(i_),
                       edgecolor=SURFACE, lw=0.6, zorder=3,
                       label=b if m == order[0] else None)
    ax.axvline(0, color=INK, lw=1.0, ls="--", zorder=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([NICE[m] for m in order], fontsize=7.2)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("INT8 minus FP32 external macro-AUPRC",
                  fontsize=7.4, color=INK)
    ax.legend(fontsize=6.6, frameon=False, labelcolor=INK2, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, -0.015 - 0.95 / h),
              title="size budget", title_fontsize=6.6)
    _frame(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.subplots_adjust(left=0.27, right=0.975, top=0.975,
                        bottom=0.055 + 0.62 / h)
    save(fig, "figure4_int8_cost")


def _label_points(ax, xs, ys, labels, sizes, *, fontsize=5.0,
                  inside_color=None, leader="#b5b5ad"):
    """Draw short identifiers next to points without letting them stack.

    Putting the identifier at the marker centre is legible only while no
    two markers overlap. Several architectures land within a few
    milliseconds and a thousandth of AUPRC of each other, so the centred
    labels pile up and none of them can be read.

    Each label is therefore tried at the centre first and, when that box
    intersects one already placed, at eight directions on rings of
    growing radius until it is clear. A leader line is drawn once the
    label has moved far enough to be ambiguous about which marker owns
    it. Larger markers are placed first, so the ones with room to hold a
    label keep it inside. Placement is deterministic -- same data, same
    figure size, same result -- so the figure is reproducible.

    `sizes` are matplotlib scatter areas in points squared, the same
    numbers passed to `ax.scatter`, so the rings clear the marker edge.
    """
    import math
    fig = ax.figure
    fig.canvas.draw()
    px = fig.dpi / 72.0                      # points -> display pixels
    xy = ax.transData.transform(list(zip(xs, ys)))

    # The text box is approximated rather than measured: a renderer query
    # per candidate offset would be hundreds of calls, and these labels
    # are two or three digits of one font size.
    def box(cx, cy, text):
        w = 0.68 * fontsize * len(text) * px
        h = 1.05 * fontsize * px
        return (cx - w / 2 - 1.0, cy - h / 2 - 1.0,
                cx + w / 2 + 1.0, cy + h / 2 + 1.0)

    def hits(a, placed):
        return any(not (a[2] <= b[0] or a[0] >= b[2] or
                        a[3] <= b[1] or a[1] >= b[3]) for b in placed)

    # Markers are obstacles too. Avoiding only other *labels* still let a
    # label come to rest on a neighbouring disc, where dark text on a
    # saturated fill is no easier to read than the pile-up it replaced.
    discs = [(xy[k][0], xy[k][1], math.sqrt(max(sizes[k], 1.0)) / 2.0 * px)
             for k in range(len(labels))]

    def on_other_marker(cx, cy, k):
        return any(math.hypot(cx - dx, cy - dy) < dr + 1.5
                   for j, (dx, dy, dr) in enumerate(discs) if j != k)

    angles = [math.radians(a) for a in
              (90, -90, 0, 180, 45, 135, -45, -135, 22, 158, -22, -158)]
    placed, out = [], [None] * len(labels)
    for k in sorted(range(len(labels)), key=lambda k: -sizes[k]):
        cx, cy = xy[k]
        r0 = discs[k][2]
        cands = [(0.0, 0.0)]
        for ring in (r0 + 5 * px, r0 + 10 * px, r0 + 16 * px,
                     r0 + 23 * px, r0 + 31 * px, r0 + 40 * px):
            cands += [(ring * math.cos(a), ring * math.sin(a)) for a in angles]
        pick = cands[-1]
        for dx, dy in cands:
            if (not hits(box(cx + dx, cy + dy, labels[k]), placed)
                    and not on_other_marker(cx + dx, cy + dy, k)):
                pick = (dx, dy)
                break
        placed.append(box(cx + pick[0], cy + pick[1], labels[k]))
        out[k] = pick

    for k, (dx, dy) in enumerate(out):
        moved = math.hypot(dx, dy) > math.sqrt(max(sizes[k], 1.0)) / 2.0 * px
        kw = {}
        if moved:
            kw["arrowprops"] = dict(arrowstyle="-", lw=0.4, color=leader,
                                    shrinkA=0.5, shrinkB=0.5)
        ax.annotate(labels[k], (xs[k], ys[k]), textcoords="offset points",
                    xytext=(dx / px, dy / px), ha="center", va="center",
                    fontsize=fontsize, zorder=6,
                    color=INK if moved or inside_color is None
                    else inside_color, **kw)


def _pareto(points):
    """Indices of non-dominated points for (minimise x, maximise y).

    Strict domination: a point is dropped only if another is at least as
    good on both axes and strictly better on one. Ties therefore survive
    rather than silently eliminating each other.
    """
    keep = []
    for a, (xa, ya) in enumerate(points):
        if not any((xb <= xa and yb >= ya) and (xb < xa or yb > ya)
                   for b, (xb, yb) in enumerate(points) if b != a):
            keep.append(a)
    return sorted(keep, key=lambda k: points[k][0])


def figure5(runs, boot, dep):
    """RQ3: observed accuracy, latency and memory relationships.

    Every benchmarked architecture is drawn and labelled. An earlier
    version outlined and named only the non-dominated points, which left
    the rest anonymous and made a benchmark read as a selection of
    winners. The frontier stays, as a thin secondary line.

    Dominance is computed on accuracy and latency only. SRAM is an extra
    encoding, not a third dominance axis: a three-dimensional frontier is
    a different object and is not what is drawn here.
    """
    if not any(d.get("ms_median") for d in dep.values()):
        print("  (no deployments yet -- figure5 skipped)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8), facecolor=SURFACE)
    for ax, b in zip(axes, BUDGETS):
        ms = [m for m in models_present(runs, b) if (m, b) in dep
              and dep[(m, b)].get("ms_median")]
        pts = [(dep[(m, b)]["ms_median"], boot[(m, b)][0]) for m in ms]
        front = set(_pareto(pts))
        areas = []
        for k, m in enumerate(ms):
            rkb = (dep[(m, b)].get("total_ram_b") or 0) / 1024
            areas.append(16 + 8 * rkb)
            ax.scatter(pts[k][0], pts[k][1], s=areas[k],
                       color=MODEL_COLOR[m], alpha=0.80,
                       edgecolor=INK if k in front else SURFACE,
                       lw=1.2 if k in front else 0.7, zorder=3)
        fr = sorted(front, key=lambda k: pts[k][0])
        if len(fr) > 1:
            ax.plot([pts[k][0] for k in fr], [pts[k][1] for k in fr],
                    lw=0.8, ls="--", color="#b5b5ad", zorder=1)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("measured STM32F411 latency (ms, log)",
                      fontsize=7.0, color=INK)
        if b == BUDGETS[0]:
            ax.set_ylabel("equal-source external macro-AUPRC",
                          fontsize=7.0, color=INK)
        ax.set_title(f"({'ab'[BUDGETS.index(b)]}) {b} budget",
                     fontsize=7.6, color=INK, pad=4)
        ax.margins(0.18)
        _frame(ax)
        ax.grid(color=GRID, lw=0.5)
        _label_points(ax, [p[0] for p in pts], [p[1] for p in pts],
                      [MID[m] for m in ms], areas, fontsize=5.0,
                      inside_color=INK)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=5,
                          color=MODEL_COLOR[m],
                          label=f"{MID[m]}  {NICE[m]}") for m in ZOO]
    fig.legend(handles=handles, fontsize=6.0, frameon=False,
               labelcolor=INK2, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.015))
    for kb in (20, 50, 90):
        axes[1].scatter([], [], s=16 + 8 * kb, color="#c3c3bb",
                        edgecolor=SURFACE, lw=0.6, label=f"{kb} KB")
    axes[1].legend(fontsize=5.8, frameon=False, labelcolor=INK2,
                   loc="lower right", labelspacing=1.0, handletextpad=0.9,
                   borderpad=0.4, title="peak SRAM", title_fontsize=5.8)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.30,
                        wspace=0.20)
    save(fig, "figure5_tradeoffs", svg=True)


def figure6(runs, dep):
    """RQ3B: arithmetic cost is informative about latency, not sufficient.

    Two panels. The regression residual that formerly occupied a third
    panel is specification dependent -- change the fit and every residual
    changes -- so a large value says a model is slower than *that* model
    predicts, not that it is intrinsically inefficient. It moves to the
    supplement. What remains is the association itself, the normalised
    cost, and rank statistics that need no fitted model at all.
    """
    pts = [(d["macc"] / 1e6, d["ms_median"], m, b)
           for (m, b), d in dep.items()
           if d.get("ms_median") and d.get("macc")]
    if len(pts) < 4:
        print("  (no deployments yet -- figure6 skipped)")
        return
    from scipy import stats as sps
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    rho, _ = sps.spearmanr(x, y)
    cmap = plt.get_cmap("Dark2")

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6), facecolor=SURFACE)
    ax = axes[0]
    for i_, b in enumerate(BUDGETS):
        sel = [k for k, q in enumerate(pts) if q[3] == b]
        if not sel:
            continue
        ax.scatter(x[sel], y[sel], s=46, color=cmap(i_),
                   edgecolor=SURFACE, lw=0.6, zorder=3, label=b)
    # Both axes span more than a decade, and on linear scaling the
    # compact budget occupied the bottom fifth of each, crushing nine of
    # the twenty points into one unreadable blob. Log-log gives that
    # cluster roughly 60% of the panel instead of 20%. The fit stays the
    # *linear* one -- the same regression Figure S2 takes its residuals
    # from -- so it is drawn as a dense polyline and simply renders as a
    # curve here. Spearman and Kendall are rank statistics and are
    # unchanged by the transform.
    fit = np.polyfit(x, y, 1)
    xg = np.linspace(x.min() * 0.9, x.max() * 1.05, 200)
    yg = np.polyval(fit, xg)
    # A line with a non-zero intercept dives towards it as x falls, and
    # on a log ordinate that dive occupies half the panel while
    # predicting latencies an order of magnitude below anything
    # measured. Draw the fit only where it stays inside the observed
    # range; outside it the extrapolation is not evidence of anything.
    keep = yg >= y.min() * 0.85
    ax.plot(xg[keep], yg[keep], lw=0.9, ls="--", color="#b5b5ad", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a_.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.margins(0.12)
    _label_points(ax, list(x), list(y), [MID[q[2]] for q in pts],
                  [46.0] * len(pts), fontsize=4.8, inside_color="#ffffff")
    taus = []
    for b in BUDGETS:
        sel = [k for k, q in enumerate(pts) if q[3] == b]
        if len(sel) > 2:
            t, _ = sps.kendalltau(x[sel], y[sel])
            taus.append(f"{b} Kendall tau = {t:+.2f}")
    ax.annotate(f"Spearman rho = {rho:+.2f} (pooled)" + NL + NL.join(taus),
                xy=(0.04, 0.96), xycoords="axes fraction", va="top",
                fontsize=6.0, color=INK2, linespacing=1.45)
    ax.set_xlabel("MACC (millions, ST tool, log)", fontsize=7.0, color=INK)
    ax.set_ylabel("measured latency (ms, log)", fontsize=7.0, color=INK)
    ax.set_title("(a) latency against arithmetic cost", fontsize=7.6,
                 color=INK, pad=4)
    ax.legend(fontsize=6.2, frameon=False, labelcolor=INK2,
              loc="lower right", title="budget", title_fontsize=6.2)
    _frame(ax)
    ax.grid(color=GRID, lw=0.5)

    ratio = {(m, b): d["ms_median"] / (d["macc"] / 1e6)
             for (m, b), d in dep.items()
             if d.get("ms_median") and d.get("macc")}
    fams = [m for m in ZOO if any((m, b) in ratio for b in BUDGETS)]
    ax = axes[1]
    yy = np.arange(len(fams))[::-1]
    for y_, m in zip(yy, fams):
        v = [(b, ratio[(m, b)]) for b in BUDGETS if (m, b) in ratio]
        if len(v) == 2:
            ax.plot([v[0][1], v[1][1]], [y_, y_], lw=0.9, color=INK2,
                    alpha=0.45, zorder=2)
        for b, r in v:
            ax.scatter(r, y_, s=30, color=cmap(BUDGETS.index(b)),
                       edgecolor=SURFACE, lw=0.5, zorder=3,
                       label=b if m == fams[0] else None)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{MID[m]}  {NICE[m]}" for m in fams], fontsize=6.2)
    ax.set_xlabel("ms per million MACC", fontsize=7.0, color=INK)
    ax.set_title("(b) latency normalised by arithmetic cost",
                 fontsize=7.6, color=INK, pad=4)
    _frame(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout(pad=0.6, w_pad=1.4)
    save(fig, "figure6_complexity_latency", svg=True)


def figure_s2_residual(runs, dep):
    """Supplementary: OLS residual latency, retained for completeness.

    Demoted from Figure 6 because the quantity depends on the regression
    chosen for it. The rank statistics in Figure 6(a) make the same point
    without needing one.
    """
    pts = [(d["macc"] / 1e6, d["ms_median"], m, b)
           for (m, b), d in dep.items()
           if d.get("ms_median") and d.get("macc")]
    if len(pts) < 4:
        return
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    resid = y - np.polyval(np.polyfit(x, y, 1), x)
    cmap = plt.get_cmap("Dark2")
    order = np.argsort(resid)
    fig, ax = plt.subplots(figsize=(5.8, 0.9 + 0.26 * len(order)),
                           facecolor=SURFACE)
    yy = np.arange(len(order))
    for y_, k in zip(yy, order):
        ax.barh(y_, resid[k], height=0.64,
                color=cmap(BUDGETS.index(pts[k][3])), zorder=2)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{MID[pts[k][2]]} ({pts[k][3][:4]})"
                        for k in order], fontsize=6.0)
    ax.set_xlabel("observed minus OLS-predicted latency (ms)",
                  fontsize=7.0, color=INK)
    _frame(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout(pad=0.5)
    save(fig, "figureS2_residual_latency")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    runs, dep = load_runs(), load_deploy()
    print(f"{len(runs)} runs, {len(dep)} deployments")
    print("bootstrapping:")
    boot = bootstrap(runs)
    print("tables:")
    table1(runs, boot)
    table2(runs, dep)
    table3(runs, boot)
    table4(runs, dep)
    print("figures:")
    figure1(runs, dep)
    figure2(runs)
    figure3(runs, boot)
    figure4()
    figure5(runs, boot, dep)
    figure6(runs, dep)
    figure_s2_residual(runs, dep)
    print(f"\nfigures: {FIG}\ntables:  {TAB}")


if __name__ == "__main__":
    main()
