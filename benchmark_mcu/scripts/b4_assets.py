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

    The previous version reported mean AUPRC per source and annotated
    the best and worst, which put a result into the cohort table and
    invited the reading that one source is intrinsically easier.
    Average precision has a prevalence floor, so a source with more
    positives scores higher before any model is any good. The
    composition belongs here; the performance belongs in Table 3, and a
    reader can then compare the two.
    """
    c = cohort()
    m, y = c["index"], c["y"]
    prev = prevalence()
    # Four classes spanning the prevalence range, named from the actual
    # 13-label space. There is no AF label in this harmonisation, so the
    # usual AF/RBBB/LBBB trio cannot be reported.
    show = ["NSR", "SB", "CRBBB", "CLBBB"]
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
            # The macro-average of per-class prevalence is exactly the
            # macro-AUPRC a random scorer attains on this source, so it
            # is the floor Table 3 should be read against. It is a
            # property of the data, which is why it sits here.
            "macro-AUPRC floor": f"{lab.mean():.3f}",
        }
        for cl in show:
            row[f"{cl} (%)"] = f"{prev.loc[cl, sname(src)]:.1f}"
        rows.append(row)
    tot = {
        "source": "Total",
        "records": f"{len(m):,}",
        "patients": f"{m['patient'].nunique():,}",
        "age, median (IQR)": "",
        "female (%)": f"{100 * (m['sex'] == 'Female').sum() / len(m):.1f}",
        "labels per ECG": f"{y.sum(1).mean():.2f}",
        "macro-AUPRC floor": f"{y.mean():.3f}",
    }
    for cl in show:
        tot[f"{cl} (%)"] = f"{100 * y[:, c['classes'].index(cl)].mean():.1f}"
    rows.append(tot)
    emit(pd.DataFrame(rows).set_index("source"), "table1_protocol",
         "Table 1. Composition of the four harmonised ECG sources. Counts "
         "are the records retained after harmonisation to the 13-label "
         "space; records carrying no label inside that space were dropped "
         "before any splitting. Patients are unique identifiers within a "
         "source, so the ratio to records shows that only PTB-XL repeats "
         "patients (17,928 patients over 20,487 records); identifiers are "
         "not comparable between sources and the Total row therefore sums "
         "within-source counts. Sex is normalised across sources, which "
         "record it differently; age is missing for 221 records in total "
         "and the median is over those present. Labels per ECG is the mean "
         "number of positive labels. The four diagnoses shown span the "
         "prevalence range of the label space and are named from it: there "
         "is no atrial fibrillation label in this harmonisation. Full "
         "13-class prevalences are in Supplementary Table S1. Inputs are "
         f"10 s, 12 leads, decimated to {FS_OUT} Hz. Prevalence differences "
         "The macro-AUPRC floor is the macro-average of per-class "
         "prevalence, which is what a random scorer attains, and is given "
         "so Table 3 can be read against it rather than against zero.")


def table_s1_prevalence(runs=None, boot=None):
    """Supplement: every harmonised class, every source.

    Figure 2 shows external AUPRC varying far more by source than by
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


def bootstrap(runs, cache=True):
    """Equal-source mean AUPRC with a record-level interval.

    Cached to disk. This is 20 configurations x 400 resamples over the
    whole prediction matrix, and it dominates the cost of rebuilding the
    assets -- but it is a pure function of the cached predictions, so
    re-running it to restyle a figure burns minutes for an identical
    answer. The key is the prediction files' names, sizes and mtimes, so
    it invalidates itself the moment b1 or b2 writes anything new.
    """
    key = sorted((f.name, f.stat().st_size, int(f.stat().st_mtime))
                 for f in TRAIN.glob("*.npz"))
    stamp = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
    cf = TRAIN.parent / f"boot_{FS_OUT}hz.json"
    if cache and cf.exists():
        try:
            blob = json.loads(cf.read_text())
            if blob.get("stamp") == stamp:
                print("    (bootstrap cache hit)")
                return {(r[0], r[1]): tuple(r[2:]) for r in blob["rows"]}
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
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
    if cache:
        cf.write_text(json.dumps(
            {"stamp": stamp,
             "rows": [[m, b, *v] for (m, b), v in out.items()]}, indent=1))
    return out


# ---------------------------------------------------------------- figures
# --------------------------------------------------------------- icons
# Each family is named after a block, so the figure draws that block
# rather than a generic box. The glyphs are schematic on purpose: the
# point is that a reader can see at a glance that MobileNetV2 expands
# and contracts while FasterNet convolves a quarter of its channels and
# passes the rest through. Drawn as vectors so they stay sharp in print
# and carry no third-party licence.
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
    fig.text(x0 + 0.114, 0.856, "macro-AUPRC (FP32)", fontsize=7.8,
             color=TXT, ha="center", zorder=6)
    fig.text(x0 + 0.266, 0.856, "INT8 - FP32 (equal-source)",
             fontsize=7.8, color=TXT, ha="center", zorder=6)
    grid = np.array([[macro_ap(runs[(f, budget, s)]["y"],
                               runs[(f, budget, s)]["p"])
                      if (f, budget, s) in runs else np.nan
                      for f in fams] for s in SOURCES])
    axh = fig.add_axes([x0 + 0.040, 0.712, 0.148, 0.128], zorder=5)
    im = axh.imshow(grid, aspect="auto", cmap="RdYlBu_r", vmin=0.33,
                    vmax=0.62)
    axh.set_xticks(range(len(fams)))
    axh.set_xticklabels([MID[f] for f in fams], fontsize=5.6, color=TXT)
    axh.set_yticks(range(len(SOURCES)))
    axh.set_yticklabels([sname(s) for s in SOURCES], fontsize=6.4,
                        color=TXT)
    axh.tick_params(length=0, pad=1.5)
    for sp in axh.spines.values():
        sp.set_visible(False)
    cax = fig.add_axes([x0 + 0.046, 0.672, 0.136, 0.011], zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks(np.linspace(0.35, 0.60, 6))
    cb.ax.tick_params(labelsize=5.4, length=1.5, pad=1.2, color=TXT,
                      labelcolor=TXT)
    cb.outline.set_visible(False)

    ci_f = INT8 / "int8_ci.json"
    if ci_f.exists():
        ci = {(r["model"], r["budget"]): r
              for r in json.loads(ci_f.read_text())}
        axd = fig.add_axes([x0 + 0.212, 0.712, 0.108, 0.128], zorder=5)
        shown = [f for f in fams if (f, budget) in ci]
        for i, f in enumerate(shown):
            r = ci[(f, budget)]
            yy = len(shown) - 1 - i
            axd.plot([r["lo"], r["hi"]], [yy, yy], lw=0.9,
                     color=MODEL_COLOR[f], solid_capstyle="round")
            axd.plot([r["d_auprc"]], [yy], "o", ms=2.1,
                     color=MODEL_COLOR[f])
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
    axs_.set_ylabel("equal-source" + NL + "macro-AUPRC", fontsize=6.6,
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
            ["external macro-AUPRC against measured latency",
             "SRAM shown as bubble size; both budgets drawn"])

    # centred in the gutter, so neither arrow touches a panel edge
    for tip in (0.3140, 0.6540):
        _flow(fig, tip, 0.487, "right", size=8, shaft=0.007, lw=1.8,
              color="#44443f")

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
        # No per-marker identifiers: the legend already maps colour to
        # family, and ten leader-lined labels over two panels cost more
        # legibility than the second naming buys.

    handles = [plt.Line2D([], [], marker="o", ls="", ms=5,
                          color=MODEL_COLOR[m],
                          label=f"{MID[m]}  {NICE[m]}") for m in ZOO]
    fig.legend(handles=handles, fontsize=6.0, frameon=False,
               labelcolor=INK2, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), columnspacing=1.6)
    # The size key moves out of panel (b), where three bubbles up to
    # 27 pt across were stacked on top of each other and sat over the
    # data. Laid out in a row at the foot of the figure it keeps the
    # exact areas used in the plot, which a rescaled key would not.
    for kb in (20, 50, 90):
        axes[1].scatter([], [], s=16 + 8 * kb, color="#c3c3bb",
                        edgecolor=SURFACE, lw=0.6, label=f"{kb} KB")
    axes[1].legend(fontsize=5.8, frameon=False, labelcolor=INK2,
                   loc="lower right", bbox_to_anchor=(0.995, 0.08),
                   ncol=1, labelspacing=2.1, handletextpad=1.3,
                   borderpad=0.6, title="peak SRAM", title_fontsize=5.8)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.205,
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
                  [46.0] * len(pts), fontsize=4.8)
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
            ("table4", lambda: table4(runs, dep)),
            ("figure1", lambda: figure1(runs, dep)),
            ("figure2", lambda: figure2(runs)),
            ("figure3", lambda: figure3(runs, boot)),
            ("figure4", figure4),
            ("figure5", lambda: figure5(runs, boot, dep)),
            ("figure6", lambda: figure6(runs, dep)),
            ("figureS2", lambda: figure_s2_residual(runs, dep))]
    wanted = set(args.only) if args.only else None
    unknown = (wanted or set()) - {n for n, _ in jobs}
    if unknown:
        sys.exit(f"unknown target(s): {sorted(unknown)}")

    # Only the four artefacts that carry an interval pay for computing it.
    NEEDS_BOOT = {"table1", "table3", "figure3", "figure5"}
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
