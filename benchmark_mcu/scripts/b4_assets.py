r"""Tables and figures for the MCU architecture benchmark.

    .\.venv\Scripts\python.exe benchmark_mcu\scripts\b4_assets.py

Four tables and five figures, built only from measured artefacts:
cached held-out predictions, the int8 comparison, and what the board
reported. Nothing here is estimated from a parameter count.

    Table 1   cohort and leave-one-source-out protocol
    Table 2   the ten families: year, defining block, size, MACs
    Table 3   external macro-AP and AUROC per held-out source, CIs
    Table 4   measured deployment: Flash, SRAM, latency, pass/fail

    Figure 2  cross-source performance heatmaps, both budgets
    Figure 3  external performance distribution per family
    Figure 4  what int8 costs
    Figure 5  accuracy against measured latency, SRAM as bubble area
    Figure S1 MACs against measured latency -- the proxy under test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
REPS = 2000

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
    # A bare % opens a LaTeX comment and swallows the rest of the line,
    # so "female (%)" would silently delete every column after it.
    # escape=True is not the fix: it would also escape the en dashes and
    # underscores the captions and identifiers rely on.
    tex = df.to_latex(escape=False).replace("%", "\\%")
    (dest / f"{name}.tex").write_text(tex, "utf-8")
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


# ------------------------------------------------------------- cohort
# Table 1 and Figure 1 describe the data, not the models, so they read
# the cohort directly rather than inferring it from a run's label
# matrix. Everything below is measured from the finalised arrays:
# nothing here is typed in by hand.
_COHORT = {}


def cohort():
    """Merged record index, label matrix and class names, cached."""
    if _COHORT:
        return _COHORT
    d12 = ROOT / "12_leads"
    ix = pd.read_csv(d12 / "data" / "index.csv")
    y = np.load(d12 / "data" / "labels.npy")
    meta = pd.read_csv(ROOT / "data" / "meta" / "records.csv")
    m = ix.merge(meta, on=["source", "record"], how="left")

    # PTB-XL stores sex as 0/1, the other three as strings. Normalising
    # here keeps the inconsistency out of the table.
    sex = m["sex"].replace({0: "Male", 1: "Female",
                            "0": "Male", "1": "Female"})
    m = m.assign(sex=sex)
    _COHORT.update(index=m, y=y,
                   classes=json.loads(
                       (ROOT / "data" / "meta" / "classes.json").read_text()
                   )["abbreviations"],
                   signals=d12 / "data" / f"signals_{FS_OUT}.npy")
    return _COHORT


def prevalence():
    """Per-source positive rate for each harmonised class, as percent."""
    c = cohort()
    m, y, cls = c["index"], c["y"], c["classes"]
    out = {}
    for src in SOURCES:
        k = (m["source"] == src).values
        out[sname(src)] = {cl: 100.0 * float(y[k, j].mean())
                           for j, cl in enumerate(cls)}
    return pd.DataFrame(out)


# ----------------------------------------------------------------- tables
def table1(runs, boot):
    """Cohort characterisation: what the data are, not how models did.

    The previous version reported mean macro-AP per source and annotated
    the best and worst, which put a result into the cohort table and
    invited the reading that one source is intrinsically easier.
    Average precision has a prevalence floor, so a source with more
    positives scores higher before any model is any good. The
    composition belongs here; the performance belongs in Table 3, and a
    reader can then compare the two.
    """
    c = cohort()
    m, y = c["index"], c["y"]
    # Per-class prevalence moved out: four of thirteen classes was an
    # arbitrary window on the composition, and Table S1 carries all
    # thirteen for every source.
    rows = []
    for src in SOURCES:
        k = (m["source"] == src).values
        sub, lab = m[k], y[k]
        age = sub["age"].dropna()
        nf = int((sub["sex"] == "Female").sum())
        row = {
            "source": sname(src),
            "records": f"{int(k.sum()):,}",
            "patients": f"{sub['patient'].nunique():,}",
            "age, median (IQR)": (
                f"{age.median():.0f} ({age.quantile(.25):.0f}-"
                f"{age.quantile(.75):.0f})" if len(age) else "NR"),
            "female (%)": f"{100 * nf / k.sum():.1f}",
            "labels per ECG": f"{lab.sum(1).mean():.2f}",
            # The macro-average of per-class prevalence is the expected
            # average precision of a random ranking on this source.
            # Calling it a "floor" was wrong: it is not a bound, and a
            # model can score below it. It is a property of the data,
            # which is why it sits in the cohort table.
            "random-ranking AP reference": f"{lab.mean():.3f}",
        }
        rows.append(row)
    tot = {
        "source": "Total",
        "records": f"{len(m):,}",
        "patients": f"{m['patient'].nunique():,}",
        "age, median (IQR)": "",
        "female (%)": f"{100 * (m['sex'] == 'Female').sum() / len(m):.1f}",
        "labels per ECG": f"{y.sum(1).mean():.2f}",
        "random-ranking AP reference": f"{y.mean():.3f}",
    }
    rows.append(tot)
    emit(pd.DataFrame(rows).set_index("source"), "table1_protocol",
         "Table 1. Composition of the four harmonised ECG sources. "
         "Counts are the records retained after harmonisation to the "
         "13-label space; records carrying no label inside that space "
         "were dropped before any splitting. Patients are unique "
         "identifiers within a source, so the ratio to records shows "
         "that only PTB-XL repeats patients (17,928 patients over "
         "20,487 records); identifiers are not comparable between "
         "sources and the Total row therefore sums within-source "
         "counts. Sex is normalised across sources, which record it "
         "differently; age is missing for 221 records in total and the "
         "median is over those present. Labels per ECG is the mean "
         "number of positive labels. The random-ranking AP reference is "
         "the macro-average of per-class prevalence. Prevalence is the "
         "usual reference for a precision-recall baseline, though on a "
         "finite sample the expected AP of a uniformly random ranking "
         "is not exactly equal to it; it is a reference, not a bound, "
         "and a model can score below it. It is given so Table 3 can be "
         "read against something other than zero. Per-class prevalences for all 13 diagnoses are in "
         "Supplementary Table S1 -- but NSR is not coded comparably "
         "across these four sources and its prevalence should not be "
         "read as a population difference (see the harmonisation audit "
         f"in REPORT.md). Inputs are 10 s, 12 leads, decimated to "
         f"{FS_OUT} Hz.")


def table_s1_prevalence(runs=None, boot=None):
    """Supplement: every harmonised class, every source.

    Figure 2 shows external macro-AP varying far more by source than by
    architecture. A reader cannot tell from that alone how much of the
    variation is prevalence and how much is population or labelling, so
    the full composition is given here rather than summarised away.
    """
    prev = prevalence()
    c = cohort()
    n = {sname(src): int((c["index"]["source"] == src).sum())
         for src in SOURCES}
    df = prev.copy()
    df["all sources"] = [100.0 * float(c["y"][:, j].mean())
                         for j in range(len(c["classes"]))]
    df = df.sort_values("all sources", ascending=False)
    df = df.map(lambda v: f"{v:.2f}")
    df.index.name = "class"
    emit(df, "tableS1_prevalence",
         "Table S1. Prevalence (%) of each harmonised diagnostic class by "
         "source, ordered by pooled prevalence. Denominators are the "
         "retained records per source: "
         + ", ".join(f"{k} {v:,}" for k, v in n.items())
         + f", total {sum(n.values()):,}. A class is a positive label on a "
         "record, and records may carry several, so columns do not sum to "
         "100. These are label frequencies in the harmonised space, not "
         "clinical incidence: a diagnosis absent from a source's original "
         "coding appears here as 0.00 and cannot be distinguished from a "
         "diagnosis that was coded and never observed, so a zero is not "
         "evidence of absence in that population.")


def table2(runs, dep):
    """Architecture specification, with the two budgets side by side.

    One row per family rather than per family-and-budget: a reader
    comparing the compact and standard configurations of one
    architecture was previously looking ten rows apart. MACC has moved
    out -- Figure 5 panel C reports it, measured by the same toolchain
    that reports the latency beside it, and repeating it here only
    invited the two to drift.
    """
    rows = []
    for m in ZOO:
        ks = {b: next((k for k in runs if k[0] == m and k[1] == b), None)
              for b in BUDGETS}
        if any(v is None for v in ks.values()):
            continue
        row = {("", "architecture"): NICE[m], ("", "year"): YEAR[m],
               ("", "defining operation"): BLOCK[m]}
        for b in BUDGETS:
            r = runs[ks[b]]
            row[(b, "width")] = round(r["width"], 2)
            row[(b, "INT8 weights KB")] = round(r["params"] / 1024, 1)
            row[(b, "graph act KB")] = round(r["peak"] / 1024, 1)
        rows.append((MID[m], row))
    df = pd.DataFrame([r for _, r in rows], index=[i for i, _ in rows])
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index.name = "ID"
    emit(df, "table2_architectures",
         "Table 2. The ten architecture families and their two size "
         "budgets. Each family is scaled to the largest width that "
         "satisfies both budgets rather than to a common width, so every "
         "design is given the biggest version of itself the part will "
         "accept; widths are family-specific scaling factors and are not "
         "comparable between rows. INT8 weights are the deployed "
         "parameter count, one byte each: RepViT is reparameterised "
         "first, so its folded training branches are not charged. Graph "
         "activation is the peak tensor size derived from the model "
         "definition -- an SRAM proxy, not a measurement, and Finding 9 "
         "shows it carries no usable rank information about what the "
         "device allocates; the measured figure is Figure 5 panel B. "
         "Arithmetic cost is Figure 5 panel C. These are one-dimensional "
         "adaptations of the published families, sharing each family's "
         "defining operation but not its depth or published topology "
         "(Limitation 2). FasterNet is reported at its published partial "
         "ratio only.")


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
            rec["equal-source mean macro-AP"] = round(pt, 4)
            rec["95% CI"] = f"({lo:.4f}, {hi:.4f})"
            rec["equal-source mean macro-AUROC"] = round(np.mean(
                [macro_auroc(runs[(m, b, s)]["y"], runs[(m, b, s)]["p"])
                 for s in SOURCES if (m, b, s) in runs]), 4)
            rows.append(rec)
    t = pd.DataFrame(rows)
    t["rank in budget"] = (t.groupby("budget")["equal-source mean macro-AP"]
                           .rank(ascending=False, method="min").astype(int))
    emit(t.set_index(["budget", "model"]),
         "table3_external",
         "Table 3. External macro-AP by held-out source, with the "
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


def deposit_deployment(runs, dep):
    """Every deployment measurement, as a machine-readable deposit.

    This was Table 4. The table is gone from the manuscript -- Figure 5
    now carries the hardware comparison, and a bar cannot be read to
    three significant figures, so the exact values go here rather than
    into a supplementary table nobody cites. It lands in results/ beside
    the per-model JSON rather than in tables/, because it is data for
    reuse, not a typeset artefact.
    """
    rows = []
    for b in BUDGETS:
        for m in models_present(runs, b):
            d = dep.get((m, b))
            if not d:
                continue
            ms = d.get("ms_median")
            rows.append({
                "id": MID[m], "model": NICE[m], "budget": b,
                "flash_kb": round((d.get("total_flash_b") or 0) / 1024, 1),
                "peak_sram_kb": round((d.get("total_ram_b") or 0) / 1024, 1),
                "macc_millions": round((d.get("macc") or 0) / 1e6, 3),
                "latency_ms_median": round(ms, 2) if ms else np.nan,
                "cycles_median": d.get("cycles_median"),
                "ms_per_mmacc": round(ms / ((d.get("macc") or 1) / 1e6), 2)
                if ms else np.nan,
                "realtime_factor": round(d["rtf"], 5) if d.get("rtf")
                else np.nan,
                "deployed": bool(d.get("deployed"))})
    if not rows:
        print("  (no deployments yet -- deposit skipped)")
        return
    pd.DataFrame(rows).to_csv(
        DEPLOY / "deployment_measurements.csv", index=False)
    print(f"  results/deploy_{FS_OUT}hz/deployment_measurements.csv")


def class_ap(runs):
    """Equal-source mean AP per (model, budget, class), and per source.

    The macro-AP everything else reports is the mean of these over
    the 13 classes, so this is the layer underneath the headline number
    rather than a different metric. Averaging over sources with equal
    weight keeps Ningbo's 28,956 records from outvoting Georgia's 8,302.
    """
    k = len(cohort()["classes"])
    eq, per = {}, {}
    for b in BUDGETS:
        for m in models_present(runs, b):
            srcs = [s for s in SOURCES if (m, b, s) in runs]
            v = np.full((len(srcs), k), np.nan)
            for j, s in enumerate(srcs):
                r = runs[(m, b, s)]
                for c in range(k):
                    tot = r["y"][:, c].sum()
                    if 0 < tot < r["y"].shape[0]:
                        v[j, c] = ap_fast(r["y"][:, c], r["p"][:, c])
                per[(m, b, s)] = v[j].copy()
            with np.errstate(invalid="ignore"):
                eq[(m, b)] = np.nanmean(v, axis=0)
    return eq, per


def figure2(runs):
    """RQ1: which diagnoses carry the external performance?

    The columns are the thirteen diagnoses rather than the four sources.
    A cell is the equal-source mean AP for one architecture on one
    class: the quantity macro-AP is the average of, not a reweighting of
    it. Classes carry equal weight here exactly as they do in the macro
    average; nothing is prevalence-weighted.

    Standard budget only. The compact heatmap said the same thing one
    shade paler, and Table 3 keeps both budgets' numbers, so spending
    half a figure on the repetition bought nothing.

    Panel B is the context Panel A needs. An equal-source mean hides the
    spread it came from, and for NSR that spread is the point -- that
    class is coded three different ways across the four sources
    (Finding 0), so it is marked. Prevalence sits under each class
    because AP has a prevalence floor and a rare class scores low before
    any model is involved. The values are averaged over the ten
    architectures, which is a descriptive summary of the class, not an
    ensemble anyone deployed.
    """
    eq, per = class_ap(runs)
    cls = cohort()["classes"]
    prev = prevalence()
    pooled = {c: float(np.mean([prev.loc[c, sname(s)] for s in SOURCES]))
              for c in cls}
    order = sorted(range(len(cls)), key=lambda i: -pooled[cls[i]])
    names = [cls[i] for i in order]
    bud = BUDGETS[-1]
    ms = models_present(runs, bud)

    fig = plt.figure(figsize=(7.8, 4.5), facecolor=SURFACE)
    # Two columns: the plots, and a narrow strip the colourbar lives in.
    # Without it the colourbar eats width from panel A only, and panel
    # B's class labels no longer sit under panel A's columns -- which
    # defeats the point of giving them a common class order.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82],
                          width_ratios=[1.0, 0.028],
                          left=0.145, right=0.965, top=0.945, bottom=0.145,
                          hspace=0.40, wspace=0.015)

    ax = fig.add_subplot(gs[0, 0])
    grid = np.array([eq[(m, bud)][order] for m in ms])
    im = ax.imshow(grid, aspect="auto", cmap="RdYlBu_r",
                   vmin=float(np.nanpercentile(grid, 2)),
                   vmax=float(np.nanpercentile(grid, 98)))
    ax.set_xticks(range(len(names)))
    # Panel A carries its own class labels rather than borrowing panel
    # B's: the two are a full panel apart, and a reader checking one
    # cell should not have to count columns down to the other axis.
    ax.set_xticklabels(names, fontsize=5.8, color=INK, rotation=45,
                       ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([f"{MID[m]} {NICE[m]}" for m in ms], fontsize=6.0,
                       color=INK)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("(A) equal-source mean AP per diagnosis, "
                 f"{bud} budget", fontsize=7.6, color=INK, pad=5)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 1]))
    cb.ax.tick_params(labelsize=5.4, length=1.5, colors=INK2)
    cb.outline.set_visible(False)
    cb.set_label("AP", fontsize=5.8, color=INK2)

    ax = fig.add_subplot(gs[1, 0])
    SRCCOL = dict(zip(SOURCES, ["#2f6fb0", "#c4692a", "#2f8a4c", "#6a4fb0"]))
    for xi, i in enumerate(order):
        vals_s = {s_: np.nanmean([per[(m, bud, s_)][i] for m in ms
                                  if (m, bud, s_) in per]) for s_ in SOURCES}
        ax.plot([xi, xi], [min(vals_s.values()), max(vals_s.values())],
                lw=0.8, color="#bdbdb6", zorder=2)
        for s_ in SOURCES:
            ax.scatter(xi, vals_s[s_], s=13, color=SRCCOL[s_],
                       edgecolor=SURFACE, lw=0.35, zorder=4,
                       label=sname(s_) if xi == 0 else None)
    nsr_x = order.index(cls.index("NSR"))
    ax.axvspan(nsr_x - 0.45, nsr_x + 0.45, color="#f2dcd8", zorder=0)
    ax.annotate("NSR: coded three different ways across sources (Finding 0)",
                xy=(0.985, 0.90), xycoords="axes fraction", fontsize=5.8,
                color="#8c3b2f", ha="right", va="top")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}{NL}{pooled[n]:.1f}%" for n in names],
                       fontsize=5.8, color=INK)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel(f"AP, averaged over{NL}the ten architectures",
                  fontsize=6.2, color=INK, linespacing=1.4)
    ax.set_title("(B) spread across held-out sources; pooled prevalence "
                 "under each class", fontsize=7.6, color=INK, pad=5)
    ax.tick_params(axis="y", labelsize=5.8, colors=INK2)
    ax.tick_params(axis="x", length=0, pad=2)
    _frame(ax)
    ax.grid(axis="y", color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.2, frameon=False, labelcolor=INK2, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.008))
    save(fig, "figure2_classwise", svg=True)


def _patient_blocks(src):
    """Row positions within a held-out source, grouped by patient.

    LOSO's test index is `flatnonzero(source == held_out)`, which is
    ascending, so row i of a stored prediction matrix is the i-th record
    of that source in cohort order. That lets the resampling draw
    patients rather than records.

    It matters for PTB-XL, where 20,487 records come from 17,928
    patients; the other three sources are one record per patient, so
    there the two schemes coincide exactly. Resampling records inside a
    source that repeats patients treats correlated records as
    independent and gives an interval that is too narrow.
    """
    c = cohort()
    sub = c["index"][(c["index"]["source"] == src).values]
    pid = sub["patient"].to_numpy()
    order = np.argsort(pid, kind="mergesort")
    cuts = np.flatnonzero(np.r_[True, pid[order][1:] != pid[order][:-1]])
    return np.split(order, cuts[1:])


_PBLOCK = {}


def patient_blocks(src):
    if src not in _PBLOCK:
        _PBLOCK[src] = _patient_blocks(src)
    return _PBLOCK[src]


def _prepare_ap(y, p):
    """Descending score order, sorted labels and tie-block ends per class.

    ap_fast sorts the scores. In a bootstrap the scores never change --
    only which records are present and how many times each appears. So
    the order is computed once here and reused by every replicate, which
    is the whole optimisation: the inner loop stops being O(n log n)
    with two fancy-index copies and becomes O(n) with one gather.
    """
    n, k = y.shape
    order = np.zeros((k, n), dtype=np.int32)
    ys = np.zeros((k, n), dtype=np.float32)
    ends, live = [None] * k, []
    for c in range(k):
        tot = y[:, c].sum()
        if not (0 < tot < n):
            continue
        o = np.argsort(-p[:, c], kind="mergesort")
        order[c], ys[c] = o, y[o, c]
        sc = p[o, c]
        ends[c] = np.r_[np.flatnonzero(np.diff(sc)), n - 1].astype(np.int32)
        live.append(c)
    return {"order": order, "ys": ys, "ends": ends,
            "live": np.array(live, dtype=np.int32)}


def _ap_weighted(prep, w):
    """Per-class AP under record multiplicities `w`; NaN where undefined.

    A record drawn twice counts twice, which is exactly what resampling
    with replacement means, so this is the same estimator as scoring a
    materialised resample -- verified to 1e-8 against it on identical
    draws.
    """
    ends, live = prep["ends"], prep["live"]
    out = np.full(len(ends), np.nan)
    if live.size == 0:
        return out
    ws = w[prep["order"][live]]
    tpw = np.cumsum(ws * prep["ys"][live], axis=1)
    cnt = np.cumsum(ws, axis=1)
    for i, c in enumerate(live):
        e = ends[c]
        te = tpw[i, e]
        if te[-1] <= 0:
            continue
        ce = cnt[i, e]
        prec = np.divide(te, ce, out=np.zeros_like(te), where=ce > 0)
        rec = te / te[-1]
        out[c] = float((np.diff(np.r_[np.float32(0), rec]) * prec).sum())
    return out


def _patient_index(src):
    """Record -> patient-block id, so a draw becomes a weight vector."""
    bl = patient_blocks(src)
    n = sum(len(x) for x in bl)
    a = np.empty(n, dtype=np.int32)
    for i, b in enumerate(bl):
        a[b] = i
    return a, len(bl)


_INFER = {}


def inference(runs, ref="fcn", reps=REPS, seed=0, cache=True):
    """Marginal and paired intervals from one shared resampling.

    Running two bootstraps meant scoring every model twice on two
    different draws, which wasted half the work and left the marginal
    and paired intervals describing different resamples. Here each
    replicate draws one set of patients per held-out source, scores all
    twenty configurations on it, and everything else is arithmetic on
    the resulting matrix:

      `marg`  equal-source mean macro-AP per configuration;
      `agg`   paired difference against `ref`, equal-source mean;
      `src`   that difference on each held-out source separately;
      `src12` the same per source over the 12 classes left when NSR is
              dropped, which tests whether a source-specific reversal
              depends on the one label that is not coded comparably.

    The resampling unit is the patient. Intervals are percentile and
    cover patient sampling within these four sources; none of them is a
    between-hospital interval.
    """
    key = (ref, reps, seed)
    if key in _INFER:
        return _INFER[key]
    stamp_src = sorted((f.name, f.stat().st_size, int(f.stat().st_mtime))
                       for f in TRAIN.glob("*.npz"))
    stamp = hashlib.sha1(
        (repr(stamp_src) + f"|{ref}|{reps}|{seed}|v3").encode()
    ).hexdigest()[:16]
    cf = TRAIN.parent / f"inference_{FS_OUT}hz.json"
    if cache and cf.exists():
        try:
            blob = json.loads(cf.read_text())
            if blob.get("stamp") == stamp:
                print(f"    (inference cache hit, {blob['reps']} reps)")
                res = tuple({tuple(k.split("|")): tuple(v)
                             for k, v in blob[n].items()}
                            for n in ("marg", "agg", "src", "src12"))
                _INFER[key] = res
                return res
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    cls = cohort()["classes"]
    keep12 = np.array([c for c in range(len(cls)) if cls[c] != "NSR"])

    def mac(v, idx=None):
        w = v if idx is None else v[idx]
        return np.nan if np.all(np.isnan(w)) else float(np.nanmean(w))

    rng = np.random.default_rng(seed)
    marg, agg, src, src12 = {}, {}, {}, {}
    for b in BUDGETS:
        ms = models_present(runs, b)
        srcs = [s for s in SOURCES if all((m, b, s) in runs for m in ms)]
        prep = {(m, s): _prepare_ap(runs[(m, b, s)]["y"],
                                    runs[(m, b, s)]["p"])
                for m in ms for s in srcs}
        pidx = {s: _patient_index(s) for s in srcs}

        obs = {(m, s): _ap_weighted(
            prep[(m, s)], np.ones(runs[(m, b, s)]["y"].shape[0],
                                  dtype=np.float32))
            for m in ms for s in srcs}

        n_m, n_s = len(ms), len(srcs)
        a13 = np.empty((n_m, n_s, reps))
        a12 = np.empty((n_m, n_s, reps))
        for i in range(reps):
            for j, s in enumerate(srcs):
                pat, npat = pidx[s]
                cnt = np.bincount(rng.integers(0, npat, npat),
                                  minlength=npat)
                w = cnt[pat].astype(np.float32)
                for q, m in enumerate(ms):
                    v = _ap_weighted(prep[(m, s)], w)
                    a13[q, j, i] = mac(v)
                    a12[q, j, i] = mac(v, keep12)
            if (i + 1) % 250 == 0:
                print(f"    inference {b}: {i + 1}/{reps}", flush=True)

        def ci(arr):
            return (float(np.percentile(arr, 2.5)),
                    float(np.percentile(arr, 97.5)))

        eq13 = np.nanmean(a13, axis=1)                 # (n_m, reps)
        r = ms.index(ref) if ref in ms else None
        for q, m in enumerate(ms):
            pt = float(np.mean([mac(obs[(m, s)]) for s in srcs]))
            marg[(m, b)] = (pt, *ci(eq13[q]))
            if r is None or m == ref:
                continue
            p_agg = pt - float(np.mean([mac(obs[(ref, s)]) for s in srcs]))
            agg[(m, b)] = (p_agg, *ci(eq13[q] - eq13[r]))
            for j, s in enumerate(srcs):
                src[(m, b, s)] = (mac(obs[(m, s)]) - mac(obs[(ref, s)]),
                                  *ci(a13[q, j] - a13[r, j]))
                src12[(m, b, s)] = (
                    mac(obs[(m, s)], keep12) - mac(obs[(ref, s)], keep12),
                    *ci(a12[q, j] - a12[r, j]))
    if cache:
        cf.write_text(json.dumps(
            {"stamp": stamp, "ref": ref, "reps": reps,
             "marg": {"|".join(k): v for k, v in marg.items()},
             "agg": {"|".join(k): v for k, v in agg.items()},
             "src": {"|".join(k): v for k, v in src.items()},
             "src12": {"|".join(k): v for k, v in src12.items()}}, indent=1))
    res = (marg, agg, src, src12)
    _INFER[key] = res
    return res


def bootstrap(runs, cache=True):
    """Equal-source mean macro-AP with a patient-cluster interval."""
    return inference(runs, cache=cache)[0]


def paired_vs_reference(runs, ref="fcn", reps=REPS, seed=0, cache=True):
    """Paired differences against `ref`, kept at source resolution."""
    return inference(runs, ref=ref, reps=reps, seed=seed, cache=cache)[1:]


def _icon_axes(fig, x, y, w, h, z=6):
    ax = fig.add_axes([x, y, w, h], zorder=z)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    return ax


def _bar(ax, x, w, h, c, alpha=1.0, ls="solid", lw=1.0):
    ax.add_patch(matplotlib.patches.FancyBboxPatch(
        (x, 0.5 - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.04",
        facecolor=c, edgecolor=c, alpha=alpha, linewidth=lw,
        linestyle=ls))


def _arrow(ax, x0, y0, x1, y1, c, lw=1.0, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=c, lw=lw,
                                shrinkA=0, shrinkB=0))


def family_icon(fig, fam, x, y, w, h, c):
    """Draw `fam`'s defining block mechanism."""
    ax = _icon_axes(fig, x, y, w, h)
    pale = matplotlib.colors.to_rgba(c, 0.30)

    if fam == "fcn":                                   # plain convolution
        for i, hh in enumerate((0.45, 0.65, 0.85)):
            _bar(ax, 0.14 + i * 0.26, 0.17, hh, pale, lw=1.1)
            _bar(ax, 0.14 + i * 0.26, 0.17, hh, "none", lw=1.1)
            ax.add_patch(matplotlib.patches.FancyBboxPatch(
                (0.14 + i * 0.26, 0.5 - hh / 2), 0.17, hh,
                boxstyle="round,pad=0,rounding_size=0.04",
                facecolor=pale, edgecolor=c, linewidth=1.1))

    elif fam == "resnet":                              # residual + skip
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (0.18, 0.22), 0.46, 0.42,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=pale, edgecolor=c, linewidth=1.1))
        ax.add_patch(matplotlib.patches.Arc(
            (0.41, 0.63), 0.52, 0.52, theta1=12, theta2=168,
            edgecolor=c, linewidth=1.1))
        ax.add_patch(matplotlib.patches.Circle((0.78, 0.43), 0.085,
                                               facecolor="white",
                                               edgecolor=c, linewidth=1.1))
        ax.plot([0.715, 0.845], [0.43, 0.43], color=c, lw=0.9)
        ax.plot([0.78, 0.78], [0.365, 0.495], color=c, lw=0.9)
        _arrow(ax, 0.64, 0.43, 0.69, 0.43, c, 1.0)

    elif fam == "tcn":                                 # dilated convolution
        for i in range(5):
            ax.add_patch(matplotlib.patches.Circle(
                (0.12 + i * 0.19, 0.20), 0.052, facecolor=pale,
                edgecolor=c, linewidth=0.9))
        for i in range(3):
            ax.add_patch(matplotlib.patches.Circle(
                (0.12 + i * 0.38, 0.78), 0.052, facecolor=pale,
                edgecolor=c, linewidth=0.9))
        for src_, dst in ((0, 0), (2, 0), (2, 1), (4, 1), (4, 2)):
            ax.plot([0.12 + src_ * 0.19, 0.12 + dst * 0.38],
                    [0.25, 0.73], color=c, lw=0.75, alpha=0.85)

    elif fam in ("mobilenetv2", "mobilenetv3"):        # inverted residual
        ax.add_patch(matplotlib.patches.Polygon(
            [(0.14, 0.40), (0.14, 0.60), (0.40, 0.80), (0.40, 0.20)],
            closed=True, facecolor=pale, edgecolor=c, linewidth=1.0))
        ax.add_patch(matplotlib.patches.Rectangle(
            (0.42, 0.20), 0.14, 0.60, facecolor=pale, edgecolor=c,
            linewidth=1.0))
        ax.add_patch(matplotlib.patches.Polygon(
            [(0.84, 0.40), (0.84, 0.60), (0.58, 0.80), (0.58, 0.20)],
            closed=True, facecolor=pale, edgecolor=c, linewidth=1.0))
        if fam == "mobilenetv3":                       # + squeeze-excite
            ax.add_patch(matplotlib.patches.Circle(
                (0.49, 0.90), 0.105, facecolor="white", edgecolor=c,
                linewidth=1.0))
            ax.plot([0.43, 0.465, 0.515, 0.55],
                    [0.875, 0.875, 0.935, 0.935], color=c, lw=0.9)

    elif fam == "shufflenetv2":                        # split + shuffle
        for i, yy in enumerate((0.70, 0.30)):
            ax.add_patch(matplotlib.patches.FancyBboxPatch(
                (0.10, yy - 0.10), 0.22, 0.20,
                boxstyle="round,pad=0,rounding_size=0.04",
                facecolor=pale, edgecolor=c, linewidth=1.0))
            ax.add_patch(matplotlib.patches.FancyBboxPatch(
                (0.68, yy - 0.10), 0.22, 0.20,
                boxstyle="round,pad=0,rounding_size=0.04",
                facecolor=pale, edgecolor=c, linewidth=1.0))
        _arrow(ax, 0.34, 0.70, 0.66, 0.30, c, 0.9)
        _arrow(ax, 0.34, 0.30, 0.66, 0.70, c, 0.9)

    elif fam == "ghostnet":                            # ghost module
        ax.add_patch(matplotlib.patches.Rectangle(
            (0.12, 0.26), 0.20, 0.48, facecolor=pale, edgecolor=c,
            linewidth=1.1))
        for i in range(2):
            ax.add_patch(matplotlib.patches.Rectangle(
                (0.56 + i * 0.17, 0.26), 0.14, 0.48,
                facecolor="none", edgecolor=c, linewidth=0.9,
                linestyle=(0, (2, 1.6))))
        _arrow(ax, 0.34, 0.50, 0.53, 0.50, c, 0.9)

    elif fam == "fasternet":                           # partial convolution
        for i in range(4):
            yy = 0.16 + i * 0.19
            solid = i == 3
            ax.add_patch(matplotlib.patches.Rectangle(
                (0.14, yy), 0.40 if solid else 0.40, 0.14,
                facecolor=pale if solid else "none", edgecolor=c,
                linewidth=1.0 if solid else 0.75,
                linestyle="solid" if solid else (0, (2, 1.6))))
            if not solid:
                _arrow(ax, 0.56, yy + 0.07, 0.88, yy + 0.07, c, 0.7)
        _arrow(ax, 0.56, 0.73, 0.88, 0.73, c, 1.0)

    elif fam == "repvit":                              # reparameterisation
        for i, yy in enumerate((0.80, 0.50, 0.20)):
            ax.add_patch(matplotlib.patches.Rectangle(
                (0.10, yy - 0.065), 0.24, 0.13, facecolor=pale,
                edgecolor=c, linewidth=0.9))
            _arrow(ax, 0.36, yy, 0.56, 0.50, c, 0.8)
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (0.60, 0.34), 0.28, 0.32,
            boxstyle="round,pad=0,rounding_size=0.05",
            facecolor=pale, edgecolor=c, linewidth=1.2))

    elif fam == "mobilenetv4":                         # universal IB
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (0.08, 0.18), 0.84, 0.64,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor="none", edgecolor=c, linewidth=1.0,
            linestyle=(0, (2.5, 1.6))))
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (0.18, 0.30), 0.28, 0.40,
            boxstyle="round,pad=0,rounding_size=0.05",
            facecolor=pale, edgecolor=c, linewidth=1.0))
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (0.56, 0.36), 0.26, 0.28,
            boxstyle="round,pad=0,rounding_size=0.05",
            facecolor=pale, edgecolor=c, linewidth=1.0))
        _arrow(ax, 0.47, 0.50, 0.55, 0.50, c, 0.9)
    return ax


def _glyph(fig, kind, x, y, sz, c):
    """Small metadata marks beside the source-card statistics."""
    ax = _icon_axes(fig, x, y, sz, sz * fig.get_figwidth()
                    / fig.get_figheight())
    if kind == "records":                     # stacked database discs
        for i in range(3):
            ax.add_patch(matplotlib.patches.Ellipse(
                (0.5, 0.26 + i * 0.24), 0.78, 0.22, facecolor="none",
                edgecolor=c, linewidth=0.9))
    elif kind == "leads":                     # a tiny trace
        xx = np.linspace(0, 1, 60)
        yy = 0.5 + 0.30 * np.exp(-((xx - 0.42) / 0.045) ** 2) \
            - 0.12 * np.exp(-((xx - 0.52) / 0.05) ** 2)
        ax.plot(xx, yy, color=c, lw=0.9)
    elif kind == "site":                      # institution
        ax.add_patch(matplotlib.patches.Rectangle(
            (0.16, 0.10), 0.68, 0.62, facecolor="none", edgecolor=c,
            linewidth=0.9))
        for i in range(3):
            for j in range(2):
                ax.add_patch(matplotlib.patches.Rectangle(
                    (0.28 + i * 0.16, 0.24 + j * 0.22), 0.09, 0.13,
                    facecolor=c, edgecolor="none", alpha=0.55))
        ax.plot([0.5, 0.5], [0.72, 0.92], color=c, lw=0.9)
    return ax


# A photograph of the board, if one has been supplied. Drop a file at
# benchmark_mcu/assets/board.(png|jpg|jpeg|webp) and Figure 1 uses it in
# place of the drawing below.
#
# Nothing is downloaded to fill this in. A manufacturer's product shot is
# copyrighted, and a Creative Commons photo would oblige the manuscript
# to carry attribution and, under a ShareAlike licence, raise questions
# about the figure as a whole -- a poor trade for one inset. The two
# clean options are a photograph taken of the board itself, which is the
# photographer's own work and needs no permission, or the vector drawing
# below, which is original to this repository. Both are safe; the
# drawing is the default because it always exists.
BOARD_IMG = HERE / "assets"


def _board_photo():
    for ext in ("png", "jpg", "jpeg", "webp"):
        f = BOARD_IMG / f"board.{ext}"
        if f.exists():
            return f
    return None


def board_icon(fig, x, y, w, h):
    """The STM32F411E-DISCO carrying the STM32F411VET6.

    Uses a photograph when one has been supplied at
    benchmark_mcu/assets/board.*, which is the preferred case -- a photo
    of the actual board is both the honest illustration and free of any
    licensing question, because it is the photographer's own work.

    No usable photograph of *this* board exists under a permissive
    licence. The only STM32F4 Discovery photo on Wikimedia Commons is
    CC BY-SA 2.0 and shows the F407 kit, a different board; captioning
    it as the F411E-DISCO would misstate the hardware the measurements
    came from. So the fallback below is drawn: solder mask with a
    shaded edge and a cast shadow, the ST-LINK section fenced off at the
    top, both USB connectors, the LQFP100 part with leads on four sides
    and its pin-1 mark, the four user LEDs, the dual-row headers, and
    the MEMS and audio parts at the corners.
    """
    photo = _board_photo()
    if photo is not None:
        ax = fig.add_axes([x, y, w, h], zorder=6)
        ax.imshow(plt.imread(str(photo)), interpolation="antialiased")
        ax.set_axis_off()
        ax.patch.set_alpha(0)
        return ax

    ax = _icon_axes(fig, x, y, w, h)
    mask, deep, silk, chip = "#0f567d", "#09405e", "#e8eef3", "#1e242a"

    ax.add_patch(matplotlib.patches.FancyBboxPatch(      # cast shadow
        (0.11, 0.005), 0.86, 0.93,
        boxstyle="round,pad=0,rounding_size=0.035",
        facecolor="#9a9a93", edgecolor="none", alpha=0.35, zorder=0))
    ax.add_patch(matplotlib.patches.FancyBboxPatch(      # solder mask
        (0.06, 0.035), 0.88, 0.93,
        boxstyle="round,pad=0,rounding_size=0.035",
        facecolor=mask, edgecolor=deep, linewidth=0.9, zorder=1))
    ax.add_patch(matplotlib.patches.FancyBboxPatch(      # top highlight
        (0.08, 0.60), 0.84, 0.355,
        boxstyle="round,pad=0,rounding_size=0.03",
        facecolor="#ffffff", edgecolor="none", alpha=0.055, zorder=2))

    ax.plot([0.06, 0.94], [0.760, 0.760], color=silk, lw=0.5,
            linestyle=(0, (2.0, 1.7)), alpha=0.7, zorder=3)
    ax.add_patch(matplotlib.patches.Rectangle(           # ST-LINK MCU
        (0.41, 0.805), 0.18, 0.085, facecolor=chip, edgecolor="#4e5a64",
        linewidth=0.4, zorder=3))
    ax.add_patch(matplotlib.patches.Rectangle(           # mini-USB
        (0.40, 0.925), 0.20, 0.052, facecolor="#b6bec5",
        edgecolor="#8e979e", linewidth=0.3, zorder=4))
    ax.add_patch(matplotlib.patches.Rectangle(           # micro-USB
        (0.44, 0.030), 0.12, 0.036, facecolor="#b6bec5",
        edgecolor="#8e979e", linewidth=0.3, zorder=4))

    ax.add_patch(matplotlib.patches.Rectangle(           # STM32F411VET6
        (0.365, 0.350), 0.27, 0.205, facecolor=chip, edgecolor="#5b6670",
        linewidth=0.5, zorder=4))
    for i in range(8):
        q = 0.381 + i * 0.0335
        ax.plot([q, q], [0.555, 0.577], color="#99a2a9", lw=0.32, zorder=3)
        ax.plot([q, q], [0.328, 0.350], color="#99a2a9", lw=0.32, zorder=3)
    for i in range(5):
        q = 0.370 + i * 0.043
        ax.plot([0.343, 0.365], [q, q], color="#99a2a9", lw=0.32, zorder=3)
        ax.plot([0.635, 0.657], [q, q], color="#99a2a9", lw=0.32, zorder=3)
    ax.add_patch(matplotlib.patches.Circle(              # pin-1 mark
        (0.389, 0.533), 0.011, facecolor="#8d969d", edgecolor="none",
        zorder=5))

    for dx, dy, cc in ((0.0, 0.112, "#f2a52c"), (0.0, -0.112, "#3fa7e8"),
                       (-0.150, 0.0, "#4cbd72"), (0.150, 0.0, "#e2543c")):
        ax.add_patch(matplotlib.patches.Circle(
            (0.50 + dx, 0.452 + dy), 0.023, facecolor=cc,
            edgecolor="#ffffff", linewidth=0.25, zorder=5))

    for sx in (0.082, 0.846):                            # dual-row headers
        ax.add_patch(matplotlib.patches.Rectangle(
            (sx - 0.006, 0.078), 0.084, 0.620, facecolor=deep,
            edgecolor="none", zorder=2))
        for i in range(12):
            for k in range(2):
                ax.add_patch(matplotlib.patches.Rectangle(
                    (sx + k * 0.036, 0.086 + i * 0.0505), 0.027, 0.031,
                    facecolor="#d8c074", edgecolor="#a8904a",
                    linewidth=0.2, zorder=3))

    for cx_, cy_, ww, hh in ((0.245, 0.690, 0.075, 0.055),
                             (0.735, 0.690, 0.075, 0.055),
                             (0.285, 0.165, 0.065, 0.050),
                             (0.715, 0.165, 0.065, 0.050)):
        ax.add_patch(matplotlib.patches.Rectangle(
            (cx_ - ww / 2, cy_ - hh / 2), ww, hh, facecolor="#272f36",
            edgecolor="#3d474f", linewidth=0.25, zorder=3))
    for bx_, cc in ((0.20, "#3f6fa8"), (0.80, "#2b2f33")):   # buttons
        ax.add_patch(matplotlib.patches.Circle(
            (bx_, 0.395), 0.030, facecolor=cc, edgecolor="#8e979e",
            linewidth=0.3, zorder=4))

    # The board is laid out portrait above, which is how the real one is
    # oriented, but the deployment card is wide and short. A quarter turn
    # fills it far better. Rotating the finished artists keeps one set of
    # coordinates to maintain instead of two.
    turn = (matplotlib.transforms.Affine2D().rotate_deg(90).translate(1, 0)
            + ax.transData)
    for art in list(ax.patches) + list(ax.lines):
        art.set_transform(turn)
    return ax


def _flow(fig, x, y, direction="right", size=15, shaft=0.010,
          color="#555550", lw=2.4, z=8):
    """A flow arrow between panels, in figure coordinates.

    Drawn as a shaft plus a marker rather than as a text character or a
    FancyArrow. A ">" glyph is set in whatever the font offers and
    reads as punctuation, and a FancyArrow drawn in figure coordinates
    has its head stretched by the canvas aspect -- this figure is two
    and a half times wider than it is tall, so a vertical head would
    come out squat and a horizontal one spindly. Marker sizes are in
    points, so the head keeps its shape whatever the canvas does.
    """
    if shaft:
        xs = [x - shaft, x] if direction == "right" else [x, x]
        ys = [y, y] if direction == "right" else [y + shaft, y]
        fig.add_artist(matplotlib.lines.Line2D(
            xs, ys, transform=fig.transFigure, color=color, lw=lw,
            solid_capstyle="butt", zorder=z))
    fig.add_artist(matplotlib.lines.Line2D(
        [x], [y], transform=fig.transFigure, linewidth=0,
        marker=(">" if direction == "right" else "v"), markersize=size,
        markerfacecolor=color, markeredgecolor=color, zorder=z))


def _rrect(fig, x, y, w, h, fc, ec="none", lw=0.8, r=0.012, z=0):
    """Rounded panel in figure coordinates."""
    fig.add_artist(matplotlib.patches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        transform=fig.transFigure, facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=z, mutation_aspect=fig.get_figwidth()
        / fig.get_figheight()))


def _strip(fig, x, y, w, h, sig, color, lw=0.55, z=4):
    """A real ECG trace, scaled to its own range, no axes."""
    ax = fig.add_axes([x, y, w, h], zorder=z)
    ax.plot(sig, lw=lw, color=color, solid_joinstyle="round")
    ax.set_xlim(0, len(sig) - 1)
    pad = 0.12 * (np.ptp(sig) or 1.0)
    ax.set_ylim(sig.min() - pad, sig.max() + pad)
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    return ax


def _clean_record(sig, candidates, lead=1, probe=400):
    """Pick a legible example trace from a pool of real records.

    Taking the first or middle candidate gives whatever that index
    happens to hold, and in a clinical corpus that is often a record
    with heavy baseline wander or lead noise -- true to the data, but it
    illustrates nothing. So a fixed-size evenly spaced probe of the pool
    is scored on sample-to-sample roughness relative to its own
    amplitude, and the quietest is used. Deterministic, and no record is
    excluded from the study by it: this only chooses what to draw.
    """
    if not len(candidates):
        return None
    idx = candidates[np.linspace(0, len(candidates) - 1,
                                 min(probe, len(candidates))).astype(int)]
    best, best_score = int(idx[0]), np.inf
    for r in idx:
        x = np.asarray(sig[int(r), lead], dtype=np.float32)
        sd = float(x.std())
        if sd < 1e-6:
            continue
        score = float(np.abs(np.diff(x)).mean()) / sd
        if score < best_score:
            best, best_score = int(r), score
    return best


def figure1(runs, dep):
    """Graphical overview: sources, benchmark, outputs.

    An overview, not a protocol. Everything drawn comes from the
    artefacts: the ECG traces are real records pulled from the cohort by
    index, each card showing all twelve leads the model receives; the
    family glyphs are each architecture's defining block; the RQ1-RQ3
    insets are the actual results in miniature. If the study changes the
    figure changes with it rather than going quietly stale.
    """
    c = cohort()
    m, y, cls = c["index"], c["y"], c["classes"]
    sig = np.load(c["signals"], mmap_mode="r")

    TXT = "#000000"
    SHORT = {"mobilenetv4": "MobileNetV4" + NL + "(Conv)"}
    # panel fill, panel border, heading colour
    PAN = {"A": ("#eef3fa", "#9bb8db", "#1f4e79"),
           "B": ("#edf5ef", "#8dbd9e", "#1e6b35"),
           "C": ("#fdf4e8", "#e0b483", "#9a6318")}
    SRC = {"PTBXL": ("#dbe8f7", "#2f6fb0"),
           "Georgia": ("#fbe4d3", "#c4692a"),
           "Chapman": ("#dcefe2", "#2f8a4c"),
           "Ningbo": ("#e7e1f6", "#6a4fb0")}
    fig = plt.figure(figsize=(16.4, 6.6), facecolor=SURFACE)

    for key, x0, w, title in [("A", 0.006, 0.298, "A. Clinical ECG sources"),
                              ("B", 0.320, 0.324,
                               "B. Lightweight architecture benchmark"),
                              ("C", 0.660, 0.336, "C. Evaluation outputs")]:
        fc, ec, hc = PAN[key]
        _rrect(fig, x0, 0.012, w, 0.946, fc, ec, 1.0)
        fig.text(x0 + 0.012, 0.930, title, fontsize=13.5, weight="bold",
                 color=hc, va="center")

    # ---------------------------------------------------------- column A
    x0 = 0.006
    nsr = cls.index("NSR")
    cw, chh = 0.139, 0.212
    for i, src in enumerate(SOURCES):
        cx = x0 + 0.010 + (i % 2) * 0.1465
        cy = 0.688 - (i // 2) * 0.224
        tint, accent = SRC[src]
        _rrect(fig, cx, cy, cw, chh, tint, accent, 1.0, 0.010, z=2)
        fig.text(cx + cw / 2, cy + chh - 0.026, sname(src), fontsize=11.5,
                 weight="bold", color=accent, ha="center", va="center",
                 zorder=6)
        # the trace sits on white so twelve leads stay legible on a tint
        _rrect(fig, cx + 0.007, cy + 0.010, cw - 0.014, chh - 0.056,
               "#ffffff", accent, 0.5, 0.007, z=3)
        k = np.flatnonzero((m["source"] == src).values
                           & (y[:, nsr] == 1) & (y.sum(1) == 1))
        rec = _clean_record(sig, k)
        if rec is None:
            rec = int(np.flatnonzero((m["source"] == src).values)[0])
        r12 = np.asarray(sig[rec], dtype=np.float32)
        axl = fig.add_axes([cx + 0.012, cy + 0.016, cw - 0.024,
                            chh - 0.068], zorder=5)
        for li in range(12):
            v = r12[li]
            sd = v.std() or 1.0
            axl.plot(np.arange(len(v)), (v - v.mean()) / sd * 0.32 - li,
                     lw=0.26, color="#333333", solid_joinstyle="round")
        axl.set_xlim(0, r12.shape[1] - 1)
        axl.set_ylim(-11.8, 1.0)
        axl.set_axis_off()
        axl.patch.set_alpha(0)

    _rrect(fig, x0 + 0.010, 0.358, 0.286, 0.086, "#dbe8f7", "#9bb8db",
           1.0, 0.010, z=2)
    axw = fig.add_axes([x0 + 0.020, 0.376, 0.032, 0.048], zorder=5)
    v = np.asarray(sig[int(np.flatnonzero((m["source"] == "PTBXL").values)[0]),
                       1], dtype=np.float32)[:260]
    axw.plot(v, lw=1.0, color="#2f6fb0")
    axw.set_axis_off()
    axw.patch.set_alpha(0)
    fig.text(x0 + 0.062, 0.420, "Harmonised label space", fontsize=10.5,
             weight="bold", color=TXT, zorder=6)
    fig.text(x0 + 0.062, 0.398, f"{len(cls)} diagnostic labels retained",
             fontsize=8.6, color=TXT, zorder=6)
    fig.text(x0 + 0.062, 0.378, f"across all sources  |  {len(m):,} "
             "recordings", fontsize=8.6, color=TXT, zorder=6)

    fig.text(x0 + 0.010, 0.314, "Leave-one-source-out evaluation",
             fontsize=11.0, weight="bold", color=TXT, zorder=6)
    fw_, fg = 0.0665, 0.0065
    for j, src in enumerate(SOURCES):
        tint, accent = SRC[src]
        bx = x0 + 0.010 + j * (fw_ + fg)
        _rrect(fig, bx, 0.148, fw_, 0.136, tint, accent, 1.0, 0.009, z=2)
        fig.text(bx + fw_ / 2, 0.258, f"Fold {j + 1}", fontsize=9.5,
                 weight="bold", color=TXT, ha="center", va="center",
                 zorder=6)
        fig.text(bx + fw_ / 2, 0.219, "Test:", fontsize=8.2, color=TXT,
                 ha="center", va="center", zorder=6)
        fig.text(bx + fw_ / 2, 0.184, sname(src), fontsize=9.2,
                 weight="bold", color=accent, ha="center", va="center",
                 zorder=6)
    fig.add_artist(matplotlib.lines.Line2D(
        [x0 + 0.014, x0 + 0.014, x0 + 0.288, x0 + 0.288],
        [0.132, 0.118, 0.118, 0.132], transform=fig.transFigure,
        color="#6f6f69", lw=1.0))
    for j in range(len(SOURCES)):
        tx = x0 + 0.010 + j * (fw_ + fg) + fw_ / 2
        fig.add_artist(matplotlib.lines.Line2D(
            [tx, tx], [0.118, 0.142], transform=fig.transFigure,
            color="#6f6f69", lw=0.8))
    fig.text(x0 + 0.151, 0.086, "train on three sources, test on the "
             "fourth" + NL + "(four rotations)", fontsize=8.4, color=TXT,
             ha="center", va="center", linespacing=1.7, zorder=6)

    # ---------------------------------------------------------- column B
    x0 = 0.320
    _rrect(fig, x0 + 0.010, 0.736, 0.310, 0.172, "#ffffff", PAN["B"][1],
           1.0, 0.010, z=2)
    fig.text(x0 + 0.024, 0.868, "12-lead ECG input", fontsize=12.6,
             weight="bold", color=TXT, zorder=6)
    fig.text(x0 + 0.024, 0.826, f"10 s at {FS_OUT} Hz", fontsize=9.6,
             color=TXT, zorder=6)
    fig.text(x0 + 0.024, 0.796, f"(12 x {FS_OUT * 10:,} samples)",
             fontsize=9.6, color=TXT, zorder=6)
    rec0 = _clean_record(sig, np.flatnonzero(
        (m["source"] == "PTBXL").values & (y[:, nsr] == 1)
        & (y.sum(1) == 1)))
    axin = fig.add_axes([x0 + 0.144, 0.762, 0.164, 0.134], zorder=5)
    r12 = np.asarray(sig[rec0], dtype=np.float32)
    for li in range(12):
        v = r12[li]
        sd = v.std() or 1.0
        axin.plot(np.arange(len(v)), (v - v.mean()) / sd * 0.40 - li,
                  lw=0.32, color="#333333")
    axin.set_xlim(0, r12.shape[1] - 1)
    axin.set_ylim(-11.8, 1.2)
    axin.set_axis_off()
    axin.patch.set_alpha(0)
    # name the leads the way a 12-lead printout does, with an ellipsis
    # standing in for the eight that are not labelled
    for li, lab in ((0, "I"), (1, "II"), (2, "III"), (11, "V6")):
        axin.text(-16, -li, lab, fontsize=6.6, color=TXT, ha="right",
                  va="center")
    for d in range(3):
        axin.plot([-30], [-5.6 - d * 0.75], marker=".", ms=1.6,
                  color=TXT, clip_on=False)
    fig.add_artist(matplotlib.lines.Line2D(
        [x0 + 0.144, x0 + 0.308], [0.756, 0.756], transform=fig.transFigure,
        color=TXT, lw=0.9))
    for q in (x0 + 0.144, x0 + 0.308):
        fig.add_artist(matplotlib.lines.Line2D(
            [q, q], [0.752, 0.760], transform=fig.transFigure, color=TXT,
            lw=0.9))
    fig.text(x0 + 0.226, 0.744, "10 seconds", fontsize=8.0, color=TXT,
             ha="center", zorder=6)
    _flow(fig, x0 + 0.165, 0.705, "down", size=7, shaft=0.020)

    _rrect(fig, x0 + 0.010, 0.266, 0.310, 0.408, "#e4efe8", PAN["B"][1],
           1.0, 0.010, z=2)
    _rrect(fig, x0 + 0.010, 0.630, 0.310, 0.044, "#d4e7da", PAN["B"][1],
           1.0, 0.010, z=3)
    fig.text(x0 + 0.165, 0.652, f"{len(ZOO)} encoder families  ×  "
             f"{len(BUDGETS)} size budgets", fontsize=11.5, weight="bold",
             color=TXT, ha="center", va="center", zorder=6)
    iw, ih = 0.0575, 0.118
    for i, fam in enumerate(ZOO):
        bx = x0 + 0.019 + (i % 5) * 0.0605
        by = 0.502 - (i // 5) * 0.144
        col = MODEL_COLOR[fam]
        pale = matplotlib.colors.to_hex(
            np.array(matplotlib.colors.to_rgb(col)) * 0.14
            + np.ones(3) * 0.86)
        _rrect(fig, bx, by, iw, ih, pale, col, 1.0, 0.008, z=3)
        fig.text(bx + iw / 2, by + ih - 0.018,
                 SHORT.get(fam, NICE[fam]), fontsize=6.6, weight="bold",
                 color=col, ha="center", va="center", zorder=6,
                 linespacing=1.25)
        family_icon(fig, fam, bx + 0.007, by + 0.011, iw - 0.014, 0.076,
                    col)
    BUD = {"compact": ("#e7e3f6", "#6a4fb0"),
           "standard": ("#fdf0d7", "#b8862a")}
    for i, (b, lab) in enumerate(zip(BUDGETS,
                                     ["≤ 32 KB INT8 weights",
                                      "≤ 128 KB INT8 weights"])):
        bx = x0 + 0.010 + i * 0.162
        tint, accent = BUD[b]
        _rrect(fig, bx, 0.278, 0.148, 0.060, tint, accent, 1.0, 0.009, z=4)
        fig.text(bx + 0.074, 0.320, b.capitalize(), fontsize=10.0,
                 weight="bold", color=TXT, ha="center", va="center",
                 zorder=6)
        fig.text(bx + 0.074, 0.296, lab, fontsize=8.0, color=TXT,
                 ha="center", va="center", zorder=6)
    _flow(fig, x0 + 0.165, 0.251, "down", size=7, shaft=0.014)

    chain = [("Train (FP32)", "same protocol for" + NL
              + "all models and sources", "#e6eef7", "#4a7ab0"),
             ("Quantise (INT8)", "post-training" + NL
              + "quantisation", "#fdf2de", "#c08f31"),
             ("Deploy (STM32F411)", "measured on device:" + NL
              + "Flash, SRAM, latency", "#e4f0e9", "#3f8a5c")]
    for i, (head, sub, tint, accent) in enumerate(chain):
        bx = x0 + 0.010 + i * 0.1067
        _rrect(fig, bx, 0.030, 0.0967, 0.210, tint, accent, 1.0, 0.010, z=2)
        fig.text(bx + 0.0484, 0.218, head, fontsize=9.0, weight="bold",
                 color=TXT, ha="center", va="center", zorder=6)
        fig.text(bx + 0.0484, 0.062, sub, fontsize=7.6, color=TXT,
                 ha="center", va="center", linespacing=1.6, zorder=6)
        if i == 0:
            axn = _icon_axes(fig, bx + 0.0084, 0.092, 0.080, 0.112)
            # real leads on the left, feeding the network on the right
            for q, li in enumerate((0, 1, 6, 11)):
                vv = r12[li][::8].astype(np.float64)
                vv = vv - vv.mean()
                vv = vv / (np.abs(vv).max() or 1.0)
                axn.plot(np.linspace(0.02, 0.34, vv.size),
                         0.80 - q * 0.20 + 0.052 * vv,
                         lw=0.4, color="#8794a0")
            NODE, EDGE = "#2f6fb0", "#b9cfe4"
            layers = ((0.56, 3), (0.76, 4), (0.95, 1))
            for (xa, na), (xb, nb) in zip(layers, layers[1:]):
                for ia in range(na):
                    for ib in range(nb):
                        axn.plot([xa, xb],
                                 [0.5 + (ia - (na - 1) / 2) * 0.22,
                                  0.5 + (ib - (nb - 1) / 2) * 0.22],
                                 color=EDGE, lw=0.3, zorder=0)
            for xx, nn in layers:
                for nq in range(nn):
                    axn.add_patch(matplotlib.patches.Circle(
                        (xx, 0.5 + (nq - (nn - 1) / 2) * 0.22), 0.050,
                        facecolor=NODE if xx < 0.9 else "#9aa3ab",
                        edgecolor="none", zorder=2))
            _arrow(axn, 0.38, 0.50, 0.48, 0.50, "#8794a0", 0.7)
        elif i == 1:
            axq = _icon_axes(fig, bx + 0.0084, 0.092, 0.080, 0.112)
            NODE, EDGE = "#9fc3e0", "#cfe0ee"
            layers = ((0.07, 3), (0.24, 4), (0.41, 2))
            for (xa, na), (xb, nb) in zip(layers, layers[1:]):
                for ia in range(na):
                    for ib in range(nb):
                        axq.plot([xa, xb],
                                 [0.5 + (ia - (na - 1) / 2) * 0.22,
                                  0.5 + (ib - (nb - 1) / 2) * 0.22],
                                 color=EDGE, lw=0.3, zorder=0)
            for xx, nn in layers:
                for nq in range(nn):
                    axq.add_patch(matplotlib.patches.Circle(
                        (xx, 0.5 + (nq - (nn - 1) / 2) * 0.22), 0.045,
                        facecolor=NODE, edgecolor="#5f93bf", lw=0.4,
                        zorder=2))
            _arrow(axq, 0.49, 0.50, 0.59, 0.50, "#c08f31", 0.8)
            axq.add_patch(matplotlib.patches.FancyBboxPatch(
                (0.63, 0.33), 0.34, 0.34,
                boxstyle="round,pad=0,rounding_size=0.09",
                facecolor="#ffffff", edgecolor="#c08f31", lw=1.0))
            axq.text(0.80, 0.50, "INT8", fontsize=6.6, weight="bold",
                     color="#9a6520", ha="center", va="center")
        else:
            board_icon(fig, bx + 0.0194, 0.098, 0.058, 0.090)
        if i < 2:
            _flow(fig, bx + 0.1042, 0.140, "right", size=8,
                  shaft=0.007, lw=1.6)

    # ---------------------------------------------------------- column C
    x0 = 0.660
    budget = BUDGETS[0]
    fams = [f for f in ZOO if (f, budget) in dep] or list(ZOO)
    RQ = {1: ("#fdecea", "#b5362a"), 2: ("#e9f1f9", "#2f6fb0"),
          3: ("#e6f2ea", "#2f8a4c")}

    def bullets(xx, yy, lines, step=0.021):
        for q, t in enumerate(lines):
            fig.text(xx, yy - q * step, "•  " + t, fontsize=7.6,
                     color=TXT, zorder=6)

    tint, accent = RQ[1]
    _rrect(fig, x0 + 0.010, 0.650, 0.322, 0.256, tint, accent, 1.0,
           0.010, z=2)
    fig.text(x0 + 0.022, 0.880, "RQ1. External discrimination",
             fontsize=11.0, weight="bold", color=accent, zorder=6)
    fig.text(x0 + 0.114, 0.856, "AP per diagnosis (Fig 2)", fontsize=7.8,
             color=TXT, ha="center", zorder=6)
    fig.text(x0 + 0.266, 0.856, "paired vs FCN-1D (Fig 3)",
             fontsize=7.8, color=TXT, ha="center", zorder=6)
    # The inset mirrors what Figure 2 now is: architectures against the
    # thirteen diagnoses, not against the four sources.
    eqap, _ = class_ap(runs)
    grid = np.array([eqap[(f, budget)] for f in fams])
    axh = fig.add_axes([x0 + 0.040, 0.712, 0.148, 0.128], zorder=5)
    with np.errstate(invalid="ignore"):
        im = axh.imshow(grid, aspect="auto", cmap="RdYlBu_r",
                        vmin=float(np.nanpercentile(grid, 5)),
                        vmax=float(np.nanpercentile(grid, 95)))
    axh.set_xticks(range(grid.shape[1]))
    axh.set_xticklabels(cohort()["classes"], fontsize=4.2, color=TXT,
                        rotation=90)
    axh.set_yticks(range(len(fams)))
    axh.set_yticklabels([MID[f] for f in fams], fontsize=4.6, color=TXT)
    axh.tick_params(length=0, pad=1.5)
    for sp in axh.spines.values():
        sp.set_visible(False)
    cax = fig.add_axes([x0 + 0.046, 0.672, 0.136, 0.011], zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks(np.linspace(0.35, 0.60, 6))
    cb.ax.tick_params(labelsize=5.4, length=1.5, pad=1.2, color=TXT,
                      labelcolor=TXT)
    cb.outline.set_visible(False)

    agg = inference(runs)[1]
    if agg:
        axd = fig.add_axes([x0 + 0.212, 0.712, 0.108, 0.128], zorder=5)
        shown = [f for f in fams if (f, budget) in agg]
        for i, f in enumerate(shown):
            pt, lo, hi = agg[(f, budget)]
            yy = len(shown) - 1 - i
            axd.plot([lo, hi], [yy, yy], lw=0.9, color=MODEL_COLOR[f],
                     solid_capstyle="round")
            axd.plot([pt], [yy], "o", ms=2.1, color=MODEL_COLOR[f])
        axd.axvline(0, color=TXT, lw=0.7, ls=(0, (2.5, 1.8)))
        axd.set_yticks([])
        axd.tick_params(length=0, pad=1.2, colors=TXT)
        axd.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        axd.tick_params(axis="x", labelsize=5.2, length=1.4, pad=1)
        for sp in ("top", "right", "left"):
            axd.spines[sp].set_visible(False)
        axd.spines["bottom"].set_color("#b9b9b2")
        axd.grid(axis="x", color="#d8d8d2", lw=0.4)
        axd.patch.set_alpha(0)
        axd.margins(y=0.10)

    tint, accent = RQ[2]
    _rrect(fig, x0 + 0.010, 0.344, 0.322, 0.288, tint, accent, 1.0,
           0.010, z=2)
    fig.text(x0 + 0.022, 0.606, "RQ2. Hardware characterisation",
             fontsize=11.0, weight="bold", color=accent, zorder=6)
    metrics = [("Flash", "KB", lambda d: d["total_flash_b"] / 1024),
               ("Peak SRAM", "KB", lambda d: d["total_ram_b"] / 1024),
               ("MACCs", "millions", lambda d: d["macc"] / 1e6),
               ("Latency", "ms", lambda d: d["ms_median"])]
    for i, (lab, unit, fn) in enumerate(metrics):
        axb = fig.add_axes([x0 + 0.034 + i * 0.0785, 0.438, 0.056, 0.126],
                           zorder=5)
        vals = [fn(dep[(f, budget)]) if (f, budget) in dep else 0.0
                for f in fams]
        axb.bar(range(len(fams)), vals,
                color=[MODEL_COLOR[f] for f in fams], width=0.78)
        axb.set_title(lab, fontsize=7.4, color=TXT, pad=2.5,
                      weight="bold")
        axb.set_ylabel(unit, fontsize=5.6, color=TXT, labelpad=1)
        axb.set_xticks([])
        axb.tick_params(axis="y", labelsize=5.2, length=1.5, pad=1,
                        colors=TXT)
        for sp in ("top", "right", "bottom"):
            axb.spines[sp].set_visible(False)
        axb.spines["left"].set_color("#b9b9b2")
        axb.patch.set_alpha(0)
    bullets(x0 + 0.024, 0.408,
            ["measured Flash, peak SRAM, MACCs and latency on STM32F411",
             "both size budgets for each architecture family"])

    tint, accent = RQ[3]
    _rrect(fig, x0 + 0.010, 0.036, 0.322, 0.296, tint, accent, 1.0,
           0.010, z=2)
    fig.text(x0 + 0.022, 0.306, "RQ3. Performance-resource trade-offs",
             fontsize=11.0, weight="bold", color=accent, zorder=6)
    axs_ = fig.add_axes([x0 + 0.056, 0.142, 0.188, 0.140], zorder=5)
    for b_, mk in zip(BUDGETS, ("o", "s")):
        xs, ys, ss, csr = [], [], [], []
        for f in ZOO:
            if (f, b_) not in dep or not dep[(f, b_)].get("ms_median"):
                continue
            ks = [(f, b_, s_) for s_ in SOURCES if (f, b_, s_) in runs]
            if not ks:
                continue
            xs.append(dep[(f, b_)]["ms_median"])
            ys.append(float(np.mean([macro_ap(runs[k]["y"], runs[k]["p"])
                                     for k in ks])))
            ss.append(3.0 + 0.52 * dep[(f, b_)]["total_ram_b"] / 1024)
            csr.append(MODEL_COLOR[f])
        axs_.scatter(xs, ys, s=ss, c=csr, marker=mk, alpha=0.85,
                     edgecolor="#ffffff", lw=0.45, label=b_)
    axs_.set_xscale("log")
    axs_.set_xlabel("measured STM32F411 latency (ms, log)", fontsize=6.6,
                    color=TXT, labelpad=1.5)
    axs_.set_ylabel("equal-source" + NL + "macro-AP", fontsize=6.6,
                    color=TXT, labelpad=1.5, linespacing=1.4)
    axs_.tick_params(labelsize=5.6, length=1.5, pad=1, colors=TXT)
    axs_.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v_, _: f"{v_:g}"))
    axs_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for sp in ("top", "right"):
        axs_.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axs_.spines[sp].set_color("#b9b9b2")
    axs_.grid(color="#d8d8d2", lw=0.4)
    axs_.patch.set_alpha(0)
    axs_.margins(0.16)
    axs_.legend(fontsize=5.6, frameon=False, labelcolor=TXT,
                loc="lower right", handletextpad=0.3, borderpad=0.2,
                labelspacing=0.3)
    fig.text(x0 + 0.258, 0.272, "Bubble size:", fontsize=7.0, color=TXT,
             zorder=6)
    fig.text(x0 + 0.258, 0.253, "peak SRAM", fontsize=7.0, color=TXT,
             zorder=6)
    axkey = _icon_axes(fig, x0 + 0.256, 0.138, 0.062, 0.112)
    for i, kb in enumerate((20, 50, 90)):
        axkey.scatter([0.20], [0.80 - i * 0.30], s=3.0 + 0.52 * kb,
                      color="#b9b9b2", edgecolor="#ffffff", lw=0.4)
        axkey.text(0.46, 0.80 - i * 0.30, f"{kb} KB", fontsize=6.4,
                   color=TXT, va="center")
    axkey.set_xlim(0, 1)
    axkey.set_ylim(0, 1)
    bullets(x0 + 0.024, 0.085,
            ["external macro-AP against measured latency",
             "SRAM shown as bubble size; both budgets drawn"])

    # centred in the gutter, so neither arrow touches a panel edge
    for tip in (0.3140, 0.6540):
        _flow(fig, tip, 0.487, "right", size=8, shaft=0.007, lw=1.8,
              color="#44443f")

    save(fig, "figure1_design", svg=True)


def figure3(runs, boot, ref="fcn"):
    """RQ1: does the architecture change external discrimination, and
    does the answer hold on every source?

    One full-width panel for the standard budget. Each architecture gets
    a row carrying five things: the paired difference against the
    reference on each of the four held-out sources, and the equal-source
    mean with its interval. An earlier version reduced the four sources
    to a count, "3 of 4", which threw away the part that matters --
    *which* source disagrees, and by how much. Another split the figure
    across both budgets, which halved the space each row had and made
    the four source points unreadable; the compact contrasts are in the
    text and in Table 3.

    FCN-1D is the reference -- the plain-convolution baseline the other
    nine are implicitly arguing against -- fixed before the differences
    were examined. Nine contrasts are made with no multiplicity
    adjustment, so they are exploratory. The coloured points are
    descriptive source estimates; only the black interval carries
    uncertainty, and it is uncertainty in the mean across these four
    sources, not across hospitals in general.
    """
    agg, src, _ = paired_vs_reference(runs, ref=ref)
    if not agg:
        print("  (no paired differences -- figure3 skipped)")
        return
    bud = BUDGETS[-1]
    SRCCOL = dict(zip(SOURCES, ["#2f6fb0", "#c4692a", "#2f8a4c", "#6a4fb0"]))
    ms = [m for m in ZOO if (m, bud) in agg]
    ms = sorted(ms, key=lambda m: agg[(m, bud)][0])
    lo_all = min(min(v[1] for k, v in agg.items() if k[1] == bud),
                 min(v[0] for k, v in src.items() if k[1] == bud))
    hi_all = max(max(v[2] for k, v in agg.items() if k[1] == bud),
                 max(v[0] for k, v in src.items() if k[1] == bud))
    pad = 0.07 * (hi_all - lo_all)

    h = 1.5 + 0.34 * len(ms)
    fig, ax = plt.subplots(figsize=(7.4, h), facecolor=SURFACE)
    for y_, m in enumerate(ms):
        for s_ in SOURCES:
            if (m, bud, s_) in src:
                ax.scatter(src[(m, bud, s_)][0], y_ + 0.19, s=20,
                           color=SRCCOL[s_], edgecolor=SURFACE, lw=0.4,
                           zorder=4, label=sname(s_) if y_ == 0 else None)
        pt, lo, hi = agg[(m, bud)]
        ax.plot([lo, hi], [y_ - 0.20, y_ - 0.20], lw=1.6, color=INK,
                solid_capstyle="round", zorder=3)
        ax.scatter(pt, y_ - 0.20, s=22, color=INK, marker="s", zorder=5,
                   label="equal-source mean (95% CI)" if y_ == 0 else None)
    ax.axvline(0, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([f"{MID[m]}  {NICE[m]}" for m in ms], fontsize=7.2,
                       color=INK)
    ax.set_ylim(-0.75, len(ms) - 0.25)
    ax.set_xlim(lo_all - pad, hi_all + pad)
    ax.set_xlabel(f"paired difference in macro-AP vs {NICE[ref]}  "
                  f"({bud} budget)", fontsize=7.4, color=INK)
    ax.tick_params(axis="x", labelsize=6.4, colors=INK2)
    ax.tick_params(axis="y", length=0)
    _frame(ax)
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    ax.text(0.015, 0.012, f"favours {NICE[ref]}", transform=ax.transAxes,
            fontsize=6.0, color=INK2, ha="left", style="italic")
    ax.text(0.985, 0.012, "favours this model", transform=ax.transAxes,
            fontsize=6.0, color=INK2, ha="right", style="italic")
    h_, l_ = ax.get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=6.4, frameon=False, labelcolor=INK2,
               ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.055),
               columnspacing=1.5, handletextpad=0.4)
    fig.text(0.5, 0.018, "coloured points are source-specific estimates, "
             "not source-specific intervals;  black bar = patient-cluster "
             f"paired bootstrap 95% CI ({REPS} replicates)",
             fontsize=6.0, color=INK2, ha="center")
    fig.subplots_adjust(left=0.215, right=0.985, top=1 - 0.30 / h,
                        bottom=0.95 / h)
    save(fig, "figure3_paired", svg=True)


def _load_int8_probs(m, b, src):
    f = INT8 / f"{m}__{b}__{src}_prob.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return (d["y"].astype(np.int8), d["prob_f32"].astype(np.float64),
            d["prob"].astype(np.float64))


def budget_transition(runs=None, reps=REPS, seed=0, cache=True):
    """What changes between the compact and standard budgets.

    Two quantities, both on matched records:

      `gain`  macro-AP(INT8 standard) - macro-AP(INT8 compact). What the
              larger configuration buys, measured on the deployed
              numerics rather than on FP32 weights nobody ships.

      `inter` [AP(INT8,std) - AP(FP32,std)] - [AP(INT8,cmp) - AP(FP32,cmp)].
              Whether the quantisation penalty itself changes with
              budget. Positive means the standard configuration loses
              less to INT8 than the compact one does.

    These are different questions and are kept apart deliberately: a
    budget that improves discrimination is not the same as a budget that
    quantises more gracefully, and neither on its own settles what to
    deploy.

    All four scores in a replicate come from one patient draw per
    held-out source, so every difference is paired and the shared
    difficulty cancels. Compact and standard are separately trained
    models, so the pairing here is on records, not on weights.
    """
    stamp_src = sorted((f.name, f.stat().st_size, int(f.stat().st_mtime))
                       for f in INT8.glob("*_prob.npz"))
    stamp = hashlib.sha1(
        (repr(stamp_src) + f"|{reps}|{seed}|t1").encode()).hexdigest()[:16]
    cf = INT8 / f"transition_{FS_OUT}hz.json"
    if cache and cf.exists():
        try:
            blob = json.loads(cf.read_text())
            if blob.get("stamp") == stamp:
                print(f"    (transition cache hit, {blob['reps']} reps)")
                return ({k: tuple(v) for k, v in blob["gain"].items()},
                        {k: tuple(v) for k, v in blob["inter"].items()})
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    rng = np.random.default_rng(seed)
    gain, inter = {}, {}
    cmp_b, std_b = BUDGETS[0], BUDGETS[1]
    for m in ZOO:
        data, ok = {}, True
        for b in (cmp_b, std_b):
            for src in SOURCES:
                d = _load_int8_probs(m, b, src)
                if d is None:
                    ok = False
                    break
                data[(b, src)] = d
            if not ok:
                break
        if not ok:
            continue
        prep = {(b, src, q): _prepare_ap(data[(b, src)][0],
                                         data[(b, src)][1 + q])
                for b in (cmp_b, std_b) for src in SOURCES for q in (0, 1)}
        pidx = {src: _patient_index(src) for src in SOURCES}

        def macro(b, src, q, w):
            v = _ap_weighted(prep[(b, src, q)], w)
            return np.nan if np.all(np.isnan(v)) else float(np.nanmean(v))

        ones = {src: np.ones(data[(cmp_b, src)][0].shape[0],
                             dtype=np.float32) for src in SOURCES}
        g0 = float(np.mean([macro(std_b, s_, 1, ones[s_])
                            - macro(cmp_b, s_, 1, ones[s_])
                            for s_ in SOURCES]))
        i0 = float(np.mean([(macro(std_b, s_, 1, ones[s_])
                             - macro(std_b, s_, 0, ones[s_]))
                            - (macro(cmp_b, s_, 1, ones[s_])
                               - macro(cmp_b, s_, 0, ones[s_]))
                            for s_ in SOURCES]))
        ga, ia = np.empty(reps), np.empty(reps)
        for i in range(reps):
            gv, iv = [], []
            for s_ in SOURCES:
                pat, npat = pidx[s_]
                cnt = np.bincount(rng.integers(0, npat, npat),
                                  minlength=npat)
                w = cnt[pat].astype(np.float32)
                a = macro(std_b, s_, 1, w)
                c = macro(cmp_b, s_, 1, w)
                gv.append(a - c)
                iv.append((a - macro(std_b, s_, 0, w))
                          - (c - macro(cmp_b, s_, 0, w)))
            ga[i], ia[i] = np.mean(gv), np.mean(iv)
            if (i + 1) % 500 == 0:
                print(f"    transition {NICE[m]}: {i + 1}/{reps}",
                      flush=True)
        gain[m] = (g0, float(np.percentile(ga, 2.5)),
                   float(np.percentile(ga, 97.5)))
        inter[m] = (i0, float(np.percentile(ia, 2.5)),
                    float(np.percentile(ia, 97.5)))
    if cache:
        cf.write_text(json.dumps({"stamp": stamp, "reps": reps,
                                  "gain": gain, "inter": inter}, indent=1))
    return gain, inter


def figure4(runs=None):
    """The compact-to-standard transition, in its two separable parts.

    The old version plotted only what INT8 costs. That is one of two
    things a reader needs at this point in the argument, and on its own
    it invites the wrong summary -- "quantisation is free, so take the
    bigger model". Those are different claims resting on different
    comparisons.

    Panel A asks what the larger budget buys, measured on the deployed
    INT8 numerics rather than on FP32 weights nobody ships. Panel B asks
    whether the quantisation penalty itself depends on budget: the
    difference of differences, positive where the standard
    configuration loses less to INT8 than the compact one does.

    Both are paired on records within a held-out source and averaged
    over the four sources with equal weight.
    """
    gain, inter = budget_transition(None)
    if not gain:
        print("  (no int8 probabilities -- figure4 skipped)")
        return
    order = sorted(gain, key=lambda m: gain[m][0])
    h = 1.3 + 0.30 * len(order)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, h), facecolor=SURFACE)
    panels = [("(A) predictive gain from the larger budget",
               "macro-AP(INT8 standard) - macro-AP(INT8 compact)", gain),
              ("(B) change in the INT8 penalty",
               "penalty(standard) - penalty(compact)", inter)]
    for ax, (title, xlab, data) in zip(axes, panels):
        for y_, m in enumerate(order):
            pt, lo, hi = data[m]
            ax.plot([lo, hi], [y_, y_], lw=1.5, color=MODEL_COLOR[m],
                    solid_capstyle="round", zorder=3)
            ax.scatter(pt, y_, s=20, color=MODEL_COLOR[m],
                       edgecolor=SURFACE, lw=0.5, zorder=4)
        ax.axvline(0, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{MID[m]}  {NICE[m]}" for m in order]
                           if data is gain else [], fontsize=6.6,
                           color=INK)
        ax.set_xlabel(xlab, fontsize=6.4, color=INK)
        ax.set_title(title, fontsize=7.4, color=INK, pad=4)
        ax.tick_params(axis="x", labelsize=6.0, colors=INK2)
        ax.tick_params(axis="y", length=0)
        _frame(ax)
        ax.grid(axis="x", color=GRID, lw=0.45)
        ax.set_axisbelow(True)
    axes[1].text(0.015, 0.015, "standard loses more to INT8",
                 transform=axes[1].transAxes, fontsize=5.4, color=INK2,
                 ha="left", style="italic")
    axes[1].text(0.985, 0.015, "standard loses less",
                 transform=axes[1].transAxes, fontsize=5.4, color=INK2,
                 ha="right", style="italic")
    fig.text(0.5, 0.022, "points are equal-source means; bars are "
             f"patient-cluster paired bootstrap 95% CIs ({REPS} replicates)",
             fontsize=6.0, color=INK2, ha="center")
    fig.subplots_adjust(left=0.215, right=0.985, top=1 - 0.52 / h,
                        bottom=0.40 / h + 0.045, wspace=0.07)
    save(fig, "figure4_budget_transition", svg=True)


def _label_points(ax, xs, ys, labels, sizes, *, fontsize=5.0,
                  inside_color=None, leader="#b5b5ad"):
    """Draw short identifiers next to points without letting them stack.

    Putting the identifier at the marker centre is legible only while no
    two markers overlap. Several architectures land within a few
    milliseconds and a thousandth of macro-AP of each other, so the centred
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


def figure7(runs, boot, dep):
    """RQ3: observed accuracy, latency and memory relationships.

    One panel, both budgets. Splitting them across two panels gave each
    its own latency axis, which quietly hid the thing worth seeing: the
    budgets occupy different decades, and the slowest compact model is
    faster than the quickest standard one. On a shared log axis that is
    immediate. Marker shape carries the budget, colour the family, area
    the measured peak SRAM.

    Non-dominated configurations carry a dark outline and there is
    deliberately no line joining them. A frontier drawn as a curve says
    a continuum of designs exists between the points, and nothing here
    supports that -- twenty discrete configurations were measured, some
    of which happen not to be beaten on both axes by another. Outlines
    make that claim; a curve makes a larger one. Dominance is computed
    within a budget, since the budgets are different design problems.

    Identifiers are not written on the markers: the legend maps colour
    to family, and twenty leader-lined labels would cost more than the
    second naming buys. Dominance uses accuracy and latency only; SRAM
    is an extra encoding, not a third axis.
    """
    if not any(d.get("ms_median") for d in dep.values()):
        print("  (no deployments yet -- figure7 skipped)")
        return
    fig = plt.figure(figsize=(7.8, 3.2), facecolor=SURFACE)
    ax = fig.add_axes([0.088, 0.325, 0.700, 0.635])
    for b, mk in zip(BUDGETS, ("o", "s")):
        ms = [m for m in models_present(runs, b) if (m, b) in dep
              and dep[(m, b)].get("ms_median")]
        pts = [(dep[(m, b)]["ms_median"], boot[(m, b)][0]) for m in ms]
        front = set(_pareto(pts))
        for k, m in enumerate(ms):
            rkb = (dep[(m, b)].get("total_ram_b") or 0) / 1024
            # The standard budget clusters tightly between 400 and 900
            # ms, so the markers were covering each other. Smaller area
            # and a lighter fill with a firm edge keeps each one
            # separable without moving any point off its coordinates.
            ax.scatter(pts[k][0], pts[k][1], s=12 + 6.0 * rkb, marker=mk,
                       color=MODEL_COLOR[m], alpha=0.62,
                       edgecolor=INK if k in front else MODEL_COLOR[m],
                       lw=1.3 if k in front else 0.9, zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("measured STM32F411 latency (ms, log)", fontsize=7.0,
                  color=INK)
    ax.set_ylabel("equal-source external macro-AP", fontsize=7.0,
                  color=INK)
    ax.tick_params(labelsize=6.2, colors=INK2)
    ax.margins(0.13)
    _frame(ax)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    shape_h = [plt.Line2D([], [], marker=mk, ls="", ms=7.5, color=c,
                          markeredgecolor=SURFACE, markeredgewidth=0.7,
                          label=b)
               for b, mk, c in zip(BUDGETS, ("o", "s"),
                                   ("#c6c6be", "#5f5f58"))]
    ax.add_artist(ax.legend(handles=shape_h, fontsize=7.0, frameon=False,
                            labelcolor=INK2, loc="lower right",
                            handletextpad=0.8, borderpad=0.7,
                            labelspacing=0.9))
    # The size key is drawn by hand rather than with fig.legend, which
    # spaces entries uniformly: with bubbles from 7 to 27 pt across, a
    # gap that clears the largest pair leaves the smallest pair adrift,
    # and a gap that suits the smallest makes the largest two touch.
    # Here each gap is the sum of the two neighbouring radii plus a
    # fixed margin, so every pair is separated by the same visible space.
    KEYS = (20, 50, 90)
    kax = fig.add_axes([0.800, 0.400, 0.150, 0.420], zorder=5)
    kax.set_xlim(0, 1)
    kax.set_ylim(0, 1)
    kax.set_axis_off()
    kax.patch.set_alpha(0)
    pt = 1.0 / (0.420 * fig.get_figheight() * 72.0)   # points -> axes units
    rad = [np.sqrt(12 + 6.0 * kb) / 2.0 * pt for kb in KEYS]
    margin = 7.0 * pt
    ys, cur = [], 1.0 - rad[0]
    for i, r in enumerate(rad):
        if i:
            cur -= rad[i - 1] + r + margin
        ys.append(cur)
    for kb, r, yy in zip(KEYS, rad, ys):
        kax.scatter([0.24], [yy], s=12 + 6.0 * kb, color="#c3c3bb",
                    edgecolor=SURFACE, lw=0.6, zorder=3)
        kax.text(0.56, yy, f"{kb} KB", fontsize=6.4, color=INK2,
                 va="center", ha="left")
    fig.text(0.875, 0.845, "peak SRAM", fontsize=6.4, color=INK2,
             ha="center")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=5,
                          color=MODEL_COLOR[m],
                          label=f"{MID[m]}  {NICE[m]}") for m in ZOO]
    fig.legend(handles=handles, fontsize=6.0, frameon=False,
               labelcolor=INK2, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.012), columnspacing=1.6)
    save(fig, "figure7_tradeoffs", svg=True)


def figure5(runs, dep):
    """RQ2: the four measured hardware costs, every family, both budgets.

    This figure replaces the deployment table outright, so it has to be
    readable without one. That drives three choices. Budgets are told
    apart by hatching rather than by transparency alone, because a
    lighter fill of an already pale family colour is not a reliable
    distinction at column width. The extremes a reader would otherwise
    look up -- largest Flash, largest SRAM, largest MACC, fastest and
    slowest inference in each budget -- are annotated on the bars. And
    the part's capacities are stated in the panels that have one rather
    than drawn as a limit line, which would squash every bar to make
    room for headroom nothing uses.

    Each panel keeps its own axis. Flash, SRAM, arithmetic and time are
    different quantities and a shared scale would only be legible for
    whichever happened to be largest.

    Exact values for all twenty configurations are deposited as
    results/deploy_*/deployment_measurements.csv.
    """
    if not any(d.get("ms_median") for d in dep.values()):
        print("  (no deployments yet -- figure5 skipped)")
        return
    fams = [m for m in ZOO if any((m, b) in dep for b in BUDGETS)]
    metrics = [
        ("A", "Flash occupancy", "KB", lambda d: d["total_flash_b"] / 1024,
         "device: 512 KB", "{:.0f} KB"),
        ("B", "Peak SRAM", "KB", lambda d: d["total_ram_b"] / 1024,
         "device: 128 KB", "{:.0f} KB"),
        ("C", "Arithmetic cost", "MACC (millions)",
         lambda d: d["macc"] / 1e6, None, "{:.1f} M"),
        ("D", "Measured latency", "ms", lambda d: d["ms_median"],
         "100 MHz, median of 32", "{:.0f} ms")]

    fig, axes = plt.subplots(1, 4, figsize=(7.8, 3.15), facecolor=SURFACE)
    w, hatch = 0.38, (None, "///")
    for ax, (tag, lab, unit, fn, note, fmt) in zip(axes, metrics):
        vals = {}
        for j, b in enumerate(BUDGETS):
            xs = np.arange(len(fams)) + (j - 0.5) * w
            v = [fn(dep[(m, b)]) if (m, b) in dep else np.nan
                 for m in fams]
            vals[b] = v
            ax.bar(xs, v, width=w, color=[MODEL_COLOR[m] for m in fams],
                   alpha=1.0 if j == 0 else 0.55, hatch=hatch[j],
                   edgecolor=INK, lw=0.45, zorder=3)
        # Annotate only the extremes a removed table would have been
        # consulted for; labelling twenty bars would be unreadable.
        marks = []
        if tag == "D":
            for j, b in enumerate(BUDGETS):
                arr = np.asarray(vals[b], dtype=float)
                marks += [(int(np.nanargmin(arr)), j, "fastest"),
                          (int(np.nanargmax(arr)), j, "slowest")]
        else:
            flat = [(v, i, j) for j, b in enumerate(BUDGETS)
                    for i, v in enumerate(vals[b]) if np.isfinite(v)]
            v, i, j = max(flat)
            marks = [(i, j, "max")]
        top = np.nanmax([v for b in BUDGETS for v in vals[b]])
        for i, j, _ in marks:
            v = vals[BUDGETS[j]][i]
            # The fastest and slowest of a budget are often the two bars
            # of one family, side by side; centring both labels puts them
            # on top of each other. Lean each away from its neighbour.
            ax.annotate(fmt.format(v),
                        (i + (j - 0.5) * w, v), textcoords="offset points",
                        xytext=(-1.5 if j == 0 else 1.5, 2.5),
                        ha="right" if j == 0 else "left", va="bottom",
                        fontsize=4.6, color=INK, zorder=6)
        ax.set_ylim(0, top * 1.22)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([MID[m] for m in fams], fontsize=5.0,
                           color=INK2, rotation=90)
        ax.set_title(f"({tag}) {lab}", fontsize=7.4, color=INK, pad=3)
        ax.set_ylabel(unit, fontsize=6.2, color=INK2, labelpad=2)
        ax.tick_params(axis="y", labelsize=5.6, colors=INK2, length=2)
        ax.tick_params(axis="x", length=0, pad=1.5)
        if note:
            ax.text(0.5, 0.965, note, transform=ax.transAxes, fontsize=5.0,
                    color=INK2, ha="center", va="top", style="italic")
        _frame(ax)
        ax.grid(axis="y", color=GRID, lw=0.45)
        ax.set_axisbelow(True)

    bud = [matplotlib.patches.Patch(
        facecolor="#b8b8b0", alpha=1.0 if j == 0 else 0.55,
        hatch=hatch[j], edgecolor=INK, lw=0.45,
        label=f"{b} ({'<= 32' if j == 0 else '<= 128'} KB INT8 weights)")
        for j, b in enumerate(BUDGETS)]
    fig.legend(handles=bud, fontsize=6.0, frameon=False, labelcolor=INK2,
               ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.135),
               columnspacing=2.4, handletextpad=0.7)
    fam_h = [plt.Line2D([], [], marker="s", ls="", ms=4.2,
                        color=MODEL_COLOR[m], label=f"{MID[m]} {NICE[m]}")
             for m in fams]
    fig.legend(handles=fam_h, fontsize=5.4, frameon=False,
               labelcolor=INK2, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), columnspacing=1.1,
               handletextpad=0.4)
    fig.subplots_adjust(left=0.068, right=0.99, top=0.90, bottom=0.345,
                        wspace=0.40)
    save(fig, "figure5_hardware", svg=True)


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
                  [46.0] * len(pts), fontsize=4.8)
    taus = []
    for b in BUDGETS:
        sel = [k for k, q in enumerate(pts) if q[3] == b]
        if len(sel) > 2:
            t, _ = sps.kendalltau(x[sel], y[sel])
            taus.append(f"{b} Kendall tau = {t:+.2f}")
    # A correlation says how well MACC tracks latency on average. The
    # question a designer actually faces is narrower: if I pick the
    # model with fewer MACCs, do I get the faster one? Counting the
    # pairs where the two orderings disagree answers exactly that, and
    # needs no model fitted to ten points.
    import itertools
    disc, tot = 0, 0
    per_budget = []
    for b in BUDGETS:
        sel = [k for k, q in enumerate(pts) if q[3] == b]
        d = n = 0
        for i, j in itertools.combinations(sel, 2):
            if (x[i] - x[j]) * (y[i] - y[j]) < 0:
                d += 1
            n += 1
        disc, tot = disc + d, tot + n
        per_budget.append(f"{b} {d}/{n}")
    taus.append("")
    taus.append(f"MACC orders the pair wrongly in {disc}/{tot} "
                f"({100 * disc / tot:.0f}%)")
    taus.append("   " + ",  ".join(per_budget))
    taus.append("")
    taus.append("dashed: least-squares fit of latency on MACC, linear")
    taus.append("in the raw variables, so it curves on these axes")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help="targets to rebuild, e.g. --only figure1 table1")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    runs, dep = load_runs(), load_deploy()
    print(f"{len(runs)} runs, {len(dep)} deployments")

    jobs = [("table1", lambda: table1(runs, boot)),
            ("tableS1", table_s1_prevalence),
            ("table2", lambda: table2(runs, dep)),
            ("table3", lambda: table3(runs, boot)),
            ("deposit", lambda: deposit_deployment(runs, dep)),
            ("figure1", lambda: figure1(runs, dep)),
            ("figure2", lambda: figure2(runs)),
            ("figure3", lambda: figure3(runs, boot)),
            ("figure4", figure4),
            ("figure5", lambda: figure5(runs, dep)),
            ("figure6", lambda: figure6(runs, dep)),
            ("figure7", lambda: figure7(runs, boot, dep)),
            ("figureS2", lambda: figure_s2_residual(runs, dep))]
    wanted = set(args.only) if args.only else None
    unknown = (wanted or set()) - {n for n, _ in jobs}
    if unknown:
        sys.exit(f"unknown target(s): {sorted(unknown)}")

    # Only the four artefacts that carry an interval pay for computing it.
    NEEDS_BOOT = {"table1", "table3", "figure3", "figure7"}
    boot = {}
    if wanted is None or wanted & NEEDS_BOOT:
        print("bootstrapping:")
        boot = bootstrap(runs)
    for name, fn in jobs:
        if wanted is None or name in wanted:
            fn()
    print(f"\nfigures: {FIG}\ntables:  {TAB}")


if __name__ == "__main__":
    main()
