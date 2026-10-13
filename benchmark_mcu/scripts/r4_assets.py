r"""Figures and tables for the reconstructed benchmark.

    .\.venv\Scripts\python.exe benchmark_mcu\scripts\r4_assets.py
    .\.venv\Scripts\python.exe benchmark_mcu\scripts\r4_assets.py --only table2

Built incrementally: the architecture and configuration tables depend
only on the model definitions, so they can be produced and checked
before any training finishes. Everything that needs predictions is added
as the runs land.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

from mcubench.footprint import (float_fallback_frac,           # noqa: E402
                                peak_activation_bytes, weight_bytes)
from mcubench.models import macs, n_params                     # noqa: E402
from mcubench.reference import (PUBLISHED_DEPTH, build,        # noqa: E402
                                fuse)
from mcubench.scaling import ALPHA, BETA, BUDGETS, fit         # noqa: E402
import r_data as D                                             # noqa: E402
import r_fig1 as _fig1                                         # noqa: E402

#: (published variant, year, the block the family is defined by)
FAMILY = {
    "FCN-1D": ("Wang et al., FCN", 2017, "plain convolution"),
    "ResNet-1D": ("He et al., ResNet-18", 2016, "residual block"),
    "TCN": ("Bai et al., 8 levels", 2018, "dilated causal convolution"),
    "MobileNetV2": ("Sandler et al., 1.0x", 2018, "inverted residual"),
    "ShuffleNetV2": ("Ma et al., 1.0x", 2018, "channel split and shuffle"),
    "GhostNet": ("Han et al., 1.0x", 2020, "ghost module"),
    "MobileNetV3": ("Howard et al., Small", 2019,
                    "inverted residual, SE, hard-swish"),
    "FasterNet": ("Chen et al., T0", 2023, "partial convolution"),
    "RepViT": ("Wang et al., M0.9", 2024,
               "reparameterised depthwise and FFN"),
    "MobileNetV4": ("Qin et al., Conv-S", 2024,
                    "universal inverted bottleneck"),
}


def _blocks(model) -> int:
    """Block count, by the class each family is built from."""
    import torch.nn as nn
    from mcubench import reference as R
    kinds = (R._BasicBlock, R._TemporalBlock, R._InvertedResidual,
             R._ShuffleBlock, R._GhostBottleneck, R._V3Bneck,
             R._FasterBlock, R._RepViTBlock, R._UIB)
    n = sum(isinstance(x, kinds) for x in model.modules())
    return n or sum(isinstance(x, nn.Conv1d) for x in model.features)


def save_table(df, name, caption):
    D.TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(D.TAB / f"{name}.csv", index=False)
    # A bare % opens a LaTeX comment and eats the rest of the line.
    tex = df.to_latex(escape=False, index=False).replace("%", "\\%")
    (D.TAB / f"{name}.tex").write_text(tex, "utf-8")
    (D.TAB / f"{name}.txt").write_text(
        caption + "\n" + "-" * 78 + "\n" + df.to_string(index=False), "utf-8")
    print(f"  tables/{name}")


def table_architectures():
    """The ten families at published topology, with no device constraint.

    This is the reference tier's specification. Every block count here is
    the published one, which is the property the reconstruction exists to
    restore: the earlier benchmark truncated nine of the ten families to
    between 15% and 100% of their published depth, so a design was being
    judged partly on how much of it had been kept.
    """
    rows = []
    for m in D.ORDER:
        variant, year, block = FAMILY[m]
        mo = fuse(build(m))
        rows.append({
            "ID": D.LABEL[m], "Family": m, "Variant": variant,
            "Year": year, "Defining block": block,
            "Blocks": _blocks(mo),
            "Published": PUBLISHED_DEPTH[m],
            "Params (k)": round(n_params(mo) / 1e3, 1),
            "MMACC": round(macs(mo, 1000) / 1e6, 2),
            "INT8 (KB)": round(weight_bytes(mo) / 1024, 1),
            "Peak act (KB)": round(peak_activation_bytes(mo, 1000) / 1024, 1)})
    df = pd.DataFrame(rows)
    save_table(df, "table2_architectures",
               "Table 2. The ten encoder families at published topology, "
               "adapted to 1-D. Blocks equals Published for every family: "
               "the reference tier carries no device constraint, so depth "
               "is never truncated. Footprint columns describe the float32 "
               "reference and are given for scale, not as a deployment "
               "claim.")
    return df


def table_configurations():
    """The forty deployed configurations.

    Width and depth come from the frozen manifest where one exists, so
    the table describes what was trained rather than what the search
    would return today.
    """
    import json
    man = {}
    mf = HERE / "results" / "config_manifest.json"
    if mf.exists():
        man = json.loads(mf.read_text())

    rows = []
    for m in D.ORDER:
        for cfg in D.DEPLOYED:
            key = f"{m}__{cfg}"
            budget = D.BUDGET_OF[cfg]
            cap = BUDGETS[budget]
            if key in man:
                w, d = man[key]["width"], man[key]["depth"]
                verified = man[key]["verified"]
            else:
                w, d = fit(m, cap, D.METHOD_OF[cfg])
                verified = None
            mo = fuse(build(m, w, d))
            wb = weight_bytes(mo)
            rows.append({
                "ID": D.LABEL[m], "Family": m,
                "Method": D.METHOD_OF[cfg].capitalize(),
                "Budget": budget,
                "Width": round(w, 3),
                "Depth mult": round(d, 3),
                "Blocks": _blocks(mo),
                "Published": PUBLISHED_DEPTH[m],
                "INT8 (KB)": round(wb / 1024, 1),
                "Budget used (%)": round(100 * wb / cap, 1),
                "Peak act (KB)": round(peak_activation_bytes(mo, 1000) / 1024,
                                       1),
                "float32 weights (%)": round(100 * float_fallback_frac(mo)),
                "Links": "" if verified is None else
                         ("yes" if verified else "no")})
    df = pd.DataFrame(rows)
    save_table(df, "table3_configurations",
               "Table 3. The forty deployed configurations: two scaling "
               "methods crossed with two int8 weight budgets. Uniform keeps "
               f"published depth and scales width alone; joint scales both "
               f"by compound coefficients (depth {ALPHA}, width {BETA}, "
               "resolution dropped because the input length is fixed), "
               "never below one block per published stage. Budget used "
               "below 100% means the 96 KB activation ceiling bound first. "
               "float32 weights is the share the int8 path refuses to "
               "quantise, which is non-zero only where a family uses "
               "dilated convolutions.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    jobs = [("figure1", _fig1.figure1),
            ("table2", table2),
            ("table3", lambda: table3(D.load_runs())),
            ("figure2", lambda: figure2(D.load_runs())),
            ("figure3", lambda: figure3(D.load_runs())),
            ("figure4", lambda: figure4(D.load_runs())),
            ("figure5", figure5),
            ("figure6", figure6),
            ("figure7", lambda: figure7(D.load_runs()))]
    wanted = set(args.only) if args.only else None
    unknown = (wanted or set()) - {n for n, _ in jobs}
    if unknown:
        sys.exit(f"unknown target(s): {sorted(unknown)}")
    for name, fn in jobs:
        if wanted is None or name in wanted:
            fn()
    print(f"\ntables: {D.TAB}")



# --------------------------------------------------------- inference
# The resampling primitives are imported from the earlier asset builder
# rather than restated: the sort-free weighted AP, the patient blocking
# and the cohort index are identical questions here, and they were
# verified against sklearn to 1.7e-16 including ties. Only the driver
# below is new, because it ranges over configurations instead of two
# budgets -- and because it must not write to the earlier benchmark's
# cache file, which sits in the same directory.
import hashlib                                                  # noqa: E402
import json                                                     # noqa: E402

import numpy as np                                              # noqa: E402

import b4_assets as B                                           # noqa: E402
from ecgmcu.data import SOURCES                                 # noqa: E402

REPS = 2000
CACHE = HERE / "results" / "recon_inference_100hz.json"
_INFER = {}


def inference(runs, configs=None, ref="FCN-1D", reps=REPS, seed=0,
              cache=True):
    """Marginal and paired intervals from one shared resampling.

    Each replicate draws one set of patients per held-out source, scores
    every configuration on that draw, and everything else is arithmetic
    on the resulting matrix. Two separate bootstraps would score each
    model twice on different draws and leave the marginal and paired
    intervals describing different resamples.

    Returns, keyed by (family, config) or (family, config, source):

      marg   equal-source mean macro-AP
      agg    paired difference against `ref`, equal-source mean
      src    that difference on each held-out source separately

    The resampling unit is the patient, which matters for PTB-XL, where
    20,487 records come from 17,928 patients. Intervals are percentile
    and cover patient sampling within these four sources; none of them
    is a between-hospital interval.
    """
    configs = list(configs or D.CONFIGS)
    key = (tuple(configs), ref, reps, seed)
    if key in _INFER:
        return _INFER[key]
    stamp_src = sorted((f.name, f.stat().st_size, int(f.stat().st_mtime))
                       for f in D.TRAIN.glob("*.npz"))
    stamp = hashlib.sha1(
        (repr(stamp_src) + repr(key)).encode()).hexdigest()[:16]
    if cache and CACHE.exists():
        try:
            blob = json.loads(CACHE.read_text())
            if blob.get("stamp") == stamp:
                print(f"    (inference cache hit, {blob['reps']} reps)")
                res = tuple({tuple(k.split("|")): tuple(v)
                             for k, v in blob[n].items()}
                            for n in ("marg", "agg", "src"))
                _INFER[key] = res
                return res
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def mac(v):
        return np.nan if np.all(np.isnan(v)) else float(np.nanmean(v))

    rng = np.random.default_rng(seed)
    marg, agg, src = {}, {}, {}
    for cfg in configs:
        ms = D.complete(runs, cfg)
        if not ms:
            continue
        srcs = [s for s in SOURCES if all((m, cfg, s) in runs for m in ms)]
        prep = {(m, s): B._prepare_ap(runs[(m, cfg, s)]["y"],
                                      runs[(m, cfg, s)]["p"])
                for m in ms for s in srcs}
        pidx = {s: B._patient_index(s) for s in srcs}
        obs = {(m, s): B._ap_weighted(
            prep[(m, s)],
            np.ones(runs[(m, cfg, s)]["y"].shape[0], dtype=np.float32))
            for m in ms for s in srcs}

        a13 = np.empty((len(ms), len(srcs), reps))
        for i in range(reps):
            for j, s in enumerate(srcs):
                pat, npat = pidx[s]
                cnt = np.bincount(rng.integers(0, npat, npat),
                                  minlength=npat)
                w = cnt[pat].astype(np.float32)
                for q, m in enumerate(ms):
                    a13[q, j, i] = mac(B._ap_weighted(prep[(m, s)], w))
            if (i + 1) % 500 == 0:
                print(f"    inference {cfg}: {i + 1}/{reps}", flush=True)

        def ci(arr):
            return (float(np.percentile(arr, 2.5)),
                    float(np.percentile(arr, 97.5)))

        eq13 = np.nanmean(a13, axis=1)
        r = ms.index(ref) if ref in ms else None
        for q, m in enumerate(ms):
            pt = float(np.mean([mac(obs[(m, s)]) for s in srcs]))
            marg[(m, cfg)] = (pt, *ci(eq13[q]))
            if r is None or m == ref:
                continue
            base = float(np.mean([mac(obs[(ref, s)]) for s in srcs]))
            agg[(m, cfg)] = (pt - base, *ci(eq13[q] - eq13[r]))
            for j, s in enumerate(srcs):
                src[(m, cfg, s)] = (mac(obs[(m, s)]) - mac(obs[(ref, s)]),
                                    *ci(a13[q, j] - a13[r, j]))
    if cache:
        CACHE.write_text(json.dumps(
            {"stamp": stamp, "ref": ref, "reps": reps,
             "marg": {"|".join(k): v for k, v in marg.items()},
             "agg": {"|".join(k): v for k, v in agg.items()},
             "src": {"|".join(k): v for k, v in src.items()}}, indent=1))
    res = (marg, agg, src)
    _INFER[key] = res
    return res


def class_ap(runs, configs=None):
    """Equal-source mean AP per (family, config, class), and per source.

    The macro-AP everything else reports is the mean of these over the
    thirteen classes, so this is the layer underneath the headline
    number rather than a different metric. Sources carry equal weight,
    which keeps Ningbo's 28,956 records from outvoting Georgia's 8,302.
    """
    k = len(B.cohort()["classes"])
    eq, per = {}, {}
    for cfg in (configs or D.CONFIGS):
        for m in D.models_present(runs, cfg):
            srcs = [s for s in SOURCES if (m, cfg, s) in runs]
            v = np.full((len(srcs), k), np.nan)
            for j, s in enumerate(srcs):
                r = runs[(m, cfg, s)]
                for c in range(k):
                    tot = r["y"][:, c].sum()
                    if 0 < tot < r["y"].shape[0]:
                        v[j, c] = B.ap_fast(r["y"][:, c], r["p"][:, c])
                per[(m, cfg, s)] = v[j].copy()
            with np.errstate(invalid="ignore"):
                eq[(m, cfg)] = np.nanmean(v, axis=0)
    return eq, per


# ------------------------------------------------------------ figures
import matplotlib                                              # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402

SURFACE, INK, INK2, GRID = B.SURFACE, B.INK, B.INK2, B.GRID
NL = chr(10)


def figure2(runs):
    """RQ1: which diagnoses carry the external performance?

    Reference tier only. These are the published designs at published
    depth with no device constraint, so a difference between rows is a
    difference between architectures rather than between the widths a
    budget happened to allow them -- which is the separation the earlier
    benchmark could not make.

    The columns are the thirteen diagnoses, not the four sources. A cell
    is the equal-source mean AP for one architecture on one class: the
    quantity macro-AP is the average of, not a reweighting of it.

    Panel B is the context Panel A needs. An equal-source mean hides the
    spread it came from, and for NSR that spread is the point -- the
    class is coded three different ways across the four sources, so it
    is marked. Prevalence sits under each class because AP has a
    prevalence floor and a rare class scores low before any model is
    involved.
    """
    cfg = "REF"
    eq, per = class_ap(runs, [cfg])
    ms = D.complete(runs, cfg)
    if not ms:
        print("  figure2 skipped: no complete reference runs")
        return
    cls = B.cohort()["classes"]
    prev = B.prevalence()
    pooled = {c: float(np.mean([prev.loc[c, B.sname(s)] for s in SOURCES]))
              for c in cls}
    order = sorted(range(len(cls)), key=lambda i: -pooled[cls[i]])
    names = [cls[i] for i in order]

    fig = plt.figure(figsize=(7.8, 4.5), facecolor=SURFACE)
    # A narrow second column holds the colourbar. Without it the
    # colourbar eats width from panel A alone, and panel B's class
    # labels stop lining up under panel A's columns -- which defeats
    # the point of giving the two a common class order.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82],
                          width_ratios=[1.0, 0.028],
                          left=0.145, right=0.965, top=0.945, bottom=0.145,
                          hspace=0.40, wspace=0.015)

    ax = fig.add_subplot(gs[0, 0])
    grid = np.array([eq[(m, cfg)][order] for m in ms])
    im = ax.imshow(grid, aspect="auto", cmap="RdYlBu_r",
                   vmin=float(np.nanpercentile(grid, 2)),
                   vmax=float(np.nanpercentile(grid, 98)))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=5.8, color=INK, rotation=45,
                       ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([f"{D.LABEL[m]} {m}" for m in ms], fontsize=6.0,
                       color=INK)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("(A) equal-source mean AP per diagnosis, reference tier "
                 "(float32, published depth, unconstrained)",
                 fontsize=7.6, color=INK, pad=5)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 1]))
    cb.ax.tick_params(labelsize=5.4, length=1.5, colors=INK2)
    cb.outline.set_visible(False)
    cb.set_label("AP", fontsize=5.8, color=INK2)

    ax = fig.add_subplot(gs[1, 0])
    SRCCOL = dict(zip(SOURCES, ["#2f6fb0", "#c4692a", "#2f8a4c", "#6a4fb0"]))
    for xi, i in enumerate(order):
        vals = {s: np.nanmean([per[(m, cfg, s)][i] for m in ms
                               if (m, cfg, s) in per]) for s in SOURCES}
        ax.plot([xi, xi], [min(vals.values()), max(vals.values())],
                lw=0.8, color="#bdbdb6", zorder=2)
        for s in SOURCES:
            ax.scatter(xi, vals[s], s=13, color=SRCCOL[s],
                       edgecolor=SURFACE, lw=0.35, zorder=4,
                       label=B.sname(s) if xi == 0 else None)
    nsr_x = order.index(cls.index("NSR"))
    ax.axvspan(nsr_x - 0.45, nsr_x + 0.45, color="#f2dcd8", zorder=0)
    ax.annotate("NSR: coded three different ways across sources",
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
    B._frame(ax)
    ax.grid(axis="y", color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.2, frameon=False, labelcolor=INK2, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.008))
    B.save(fig, "figure2_classwise", dest=D.FIG, svg=True)


# ------------------------------------------------------------ figure 1
COL_A, COL_B, COL_C = "#2f6fb0", "#2f8a4c", "#c4692a"
TINT = {"A": "#eef4fb", "B": "#eef7f1", "C": "#fdf3ec"}
SRC_COL = {"PTBXL": "#2f6fb0", "Georgia": "#c4692a",
           "Chapman": "#2f8a4c", "Ningbo": "#6a4fb0"}
UNI_C, JNT_C = "#2f6fb0", "#c4692a"


def _hdr(fig, letter, x, y, w, title, colour):
    """Column header: a lettered disc and a title, on the tinted panel."""
    fig.add_artist(matplotlib.patches.Circle(
        (x + 0.012, y), 0.011, transform=fig.transFigure, facecolor=colour,
        edgecolor="none", zorder=12))
    fig.text(x + 0.012, y, letter, ha="center", va="center", fontsize=8.5,
             color="white", fontweight="bold", zorder=13)
    fig.text(x + 0.030, y, title, ha="left", va="center", fontsize=9.2,
             color=INK, fontweight="bold", zorder=13)


def _card(fig, x, y, w, h, fc, ec, r=0.009, lw=0.9, z=3):
    _rr(fig, x, y, w, h, fc, ec=ec, lw=lw, r=r, z=z)


def _rr(fig, x, y, w, h, fc, ec="none", lw=0.8, r=0.010, z=0):
    fig.add_artist(matplotlib.patches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        transform=fig.transFigure, facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=z,
        mutation_aspect=fig.get_figwidth() / fig.get_figheight()))


def figure1(runs=None):
    """The experimental framework, in three columns.

    Panel B carries the one structural change the design needs to make
    visible: a single set of reference models branches into two scaling
    methods, and each method into two weight budgets. That branch is the
    experiment -- without it the figure says only that forty models were
    trained, not how the forty were arrived at.
    """
    c = B.cohort()
    ix, ycls = c["index"], c["y"]
    sig = np.load(c["signals"], mmap_mode="r")
    cls = c["classes"]

    fig = plt.figure(figsize=(13.6, 5.35), facecolor=SURFACE)
    for key, (x0, w0, col) in {
            "A": (0.004, 0.322, COL_A), "B": (0.336, 0.324, COL_B),
            "C": (0.670, 0.326, COL_C)}.items():
        _rr(fig, x0, 0.015, w0, 0.965, TINT[key], ec=col, lw=1.1, r=0.013,
            z=1)
    _hdr(fig, "A", 0.014, 0.945, 0.32,
         "ECG sources and evaluation protocol", COL_A)
    _hdr(fig, "B", 0.346, 0.945, 0.32,
         "Reference models and MCU scaling", COL_B)
    _hdr(fig, "C", 0.680, 0.945, 0.32,
         "Deployment and research questions", COL_C)

    # ---------------------------------------------------------- panel A
    stats = {}
    for s in SOURCES:
        k = (ix["source"] == s).values
        age = pd.to_numeric(ix.loc[k, "age"], errors="coerce")
        stats[s] = (int(k.sum()), int(ix.loc[k, "patient"].nunique())
                    if "patient" in ix.columns else int(k.sum()),
                    float(np.nanmedian(age)),
                    100.0 * float((ix.loc[k, "sex"] == "Female").mean()))
    for i, s in enumerate(SOURCES):
        cx = 0.016 + (i % 2) * 0.157
        cy = 0.745 - (i // 2) * 0.150
        _card(fig, cx, cy, 0.147, 0.135, "#ffffff", SRC_COL[s])
        fig.text(cx + 0.0735, cy + 0.113, B.sname(s), ha="center",
                 va="center", fontsize=8.0, color=SRC_COL[s],
                 fontweight="bold", zorder=6)
        rec = B._clean_record(sig, np.flatnonzero((ix["source"] == s).values))
        if rec is not None:
            B._strip(fig, cx + 0.010, cy + 0.058, 0.127, 0.044,
                     np.asarray(sig[rec, 1], dtype=np.float32)[:600],
                     "#4a4a46", lw=0.5, z=6)
        n, npat, age, fem = stats[s]
        fig.text(cx + 0.0735, cy + 0.037, f"{n:,} ECGs | {npat:,} patients",
                 ha="center", va="center", fontsize=6.3, color=INK, zorder=6)
        fig.text(cx + 0.0735, cy + 0.016,
                 f"median age {age:.0f} | {fem:.1f}% female",
                 ha="center", va="center", fontsize=6.3, color=INK2, zorder=6)

    _card(fig, 0.016, 0.505, 0.304, 0.072, "#ffffff", "#c9d6e6")
    fig.text(0.168, 0.555, f"{len(cls)} diagnostic labels (SNOMED-CT mapped)",
             ha="center", va="center", fontsize=7.2, color=INK, zorder=6)
    fig.text(0.168, 0.527, f"{len(ix):,} ECG recordings across all sources",
             ha="center", va="center", fontsize=7.2, color=INK, zorder=6)

    fig.text(0.168, 0.455, "Leave-one-source-out evaluation (4 folds)",
             ha="center", va="center", fontsize=8.2, color=INK,
             fontweight="bold", zorder=6)
    for i, s in enumerate(SOURCES):
        fx = 0.018 + i * 0.0765
        _card(fig, fx, 0.300, 0.070, 0.125, "#ffffff", SRC_COL[s])
        fig.text(fx + 0.035, 0.396, f"Fold {i + 1}", ha="center",
                 va="center", fontsize=7.0, color=INK, fontweight="bold",
                 zorder=6)
        fig.text(fx + 0.035, 0.368, "Test:", ha="center", va="center",
                 fontsize=6.3, color=INK2, zorder=6)
        fig.text(fx + 0.035, 0.344, B.sname(s), ha="center", va="center",
                 fontsize=6.8, color=SRC_COL[s], fontweight="bold", zorder=6)
        fig.text(fx + 0.035, 0.317, "train on\nthe other 3", ha="center",
                 va="center", fontsize=5.6, color=INK2, zorder=6,
                 linespacing=1.3)
    fig.text(0.168, 0.265,
             "Every model sees the same folds, loss, schedule and seed",
             ha="center", va="center", fontsize=6.4, color=INK2, zorder=6)
    _card(fig, 0.016, 0.040, 0.304, 0.205, "#ffffff", "#c9d6e6")
    fig.text(0.168, 0.222, "Scoring", ha="center", va="center", fontsize=7.8,
             color=INK, fontweight="bold", zorder=6)
    for j, line in enumerate([
            "macro average precision over 13 classes",
            "equal weight per held-out source",
            "patient-cluster bootstrap, 2,000 replicates",
            "paired differences against FCN-1D"]):
        fig.text(0.030, 0.190 - j * 0.034, "•", fontsize=7,
                 color=COL_A, va="center", zorder=6)
        fig.text(0.044, 0.190 - j * 0.034, line, fontsize=6.6, color=INK,
                 va="center", zorder=6)
    _panel_b(fig)
    _panel_c(fig)
    B.save(fig, "figure1_design", dest=D.FIG, svg=True)


def _mini_scale(fig, x, y, w, h, colour, mode):
    """A cartoon of what each scaling method does to a network.

    Uniform thins every block and keeps the count; joint thins less and
    drops blocks. Drawn rather than described, because the difference is
    geometric and one glance settles it.
    """
    ax = fig.add_axes([x, y, w, h], zorder=6)
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    for i in range(4):
        ax.add_patch(matplotlib.patches.Rectangle(
            (0.25 + i * 0.95, 0.31), 0.62, 0.38,
            facecolor=colour, alpha=0.30, edgecolor=colour, lw=0.6))
    ax.annotate("", xy=(5.25, 0.5), xytext=(4.35, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.1))
    keep = 4 if mode == "uniform" else 2
    hh = 0.15 if mode == "uniform" else 0.27
    for i in range(keep):
        ax.add_patch(matplotlib.patches.Rectangle(
            (5.6 + i * 0.95, 0.5 - hh / 2), 0.62, hh,
            facecolor=colour, edgecolor=colour, lw=0.6))
    return ax


def _panel_b(fig):
    _card(fig, 0.346, 0.770, 0.304, 0.145, "#ffffff", "#bcd9c6")
    fig.text(0.498, 0.893, "10 reference models (FP32, unconstrained)",
             ha="center", va="center", fontsize=8.0, color=INK,
             fontweight="bold", zorder=6)
    show = ["FCN-1D", "ResNet-1D", "TCN", "MobileNetV2", "ShuffleNetV2"]
    for i, m in enumerate(show):
        ix = 0.360 + i * 0.0475
        B.family_icon(fig, D.ICON[m], ix, 0.824, 0.034, 0.038,
                      B.MODEL_COLOR[D.ICON[m]])
        fig.text(ix + 0.017, 0.805, D.LABEL[m], ha="center", va="center",
                 fontsize=6.0, color=INK, fontweight="bold", zorder=7)
        fig.text(ix + 0.017, 0.788, m, ha="center", va="center",
                 fontsize=5.0, color=INK2, zorder=7)
    fig.text(0.597, 0.840, "...", ha="center", va="center", fontsize=10,
             color=INK2, zorder=7)
    B.family_icon(fig, "mobilenetv4", 0.610, 0.824, 0.034, 0.038,
                  B.MODEL_COLOR["mobilenetv4"])
    fig.text(0.627, 0.805, "M10", ha="center", va="center", fontsize=6.0,
             color=INK, fontweight="bold", zorder=7)
    fig.text(0.627, 0.788, "MobileNetV4", ha="center", va="center",
             fontsize=5.0, color=INK2, zorder=7)

    B._flow(fig, 0.498, 0.752, direction="down", size=11, shaft=0.013,
            color=COL_B, lw=2.0)

    _card(fig, 0.346, 0.395, 0.304, 0.330, "#ffffff", "#bcd9c6")
    fig.text(0.498, 0.700,
             "Two scaling methods, each at two INT8 weight budgets",
             ha="center", va="center", fontsize=8.0, color=INK,
             fontweight="bold", zorder=6)
    for name, sub, colour, mode, x0 in [
            ("Uniform width", "published depth kept,\nchannels scaled",
             UNI_C, "uniform", 0.355),
            ("Joint width-depth",
             "depth and width together,\nat least 1 block per stage",
             JNT_C, "joint", 0.505)]:
        _card(fig, x0, 0.505, 0.136, 0.180, "#fbfbf7", colour, lw=1.0)
        fig.text(x0 + 0.068, 0.663, name, ha="center", va="center",
                 fontsize=7.4, color=colour, fontweight="bold", zorder=7)
        fig.text(x0 + 0.068, 0.628, sub, ha="center", va="center",
                 fontsize=5.6, color=INK2, zorder=7, linespacing=1.35)
        _mini_scale(fig, x0 + 0.011, 0.522, 0.114, 0.060, colour, mode)
        for j, (bname, cap, shape) in enumerate(
                [("UC-128", "<= 128 KB", "o"), ("C-256", "<= 256 KB", "s")]):
            bx = x0 + 0.004 + j * 0.067
            _card(fig, bx, 0.412, 0.062, 0.072, "#ffffff", colour, lw=0.9)
            fig.text(bx + 0.031, 0.466, bname, ha="center", va="center",
                     fontsize=6.8, color=colour, fontweight="bold", zorder=7)
            fig.text(bx + 0.031, 0.445, cap, ha="center", va="center",
                     fontsize=5.8, color=INK, zorder=7)
            fig.add_artist(matplotlib.lines.Line2D(
                [bx + 0.031], [0.425], transform=fig.transFigure, lw=0,
                marker=shape, markersize=4.0, markerfacecolor=colour,
                markeredgecolor=colour, zorder=7))

    B._flow(fig, 0.498, 0.378, direction="down", size=11, shaft=0.013,
            color=COL_B, lw=2.0)
    _card(fig, 0.346, 0.268, 0.304, 0.085, "#ffffff", COL_B, lw=1.1)
    fig.text(0.498, 0.325, "40 INT8 configurations", ha="center",
             va="center", fontsize=8.6, color=INK, fontweight="bold",
             zorder=6)
    fig.text(0.498, 0.297, "10 architectures x 2 methods x 2 budgets",
             ha="center", va="center", fontsize=6.8, color=INK2, zorder=6)
    _card(fig, 0.346, 0.040, 0.304, 0.205, "#ffffff", "#bcd9c6")
    fig.text(0.498, 0.222, "Every configuration verified before training",
             ha="center", va="center", fontsize=7.8, color=INK,
             fontweight="bold", zorder=6)
    for j, line in enumerate([
            "budget enforced on predicted deployed bytes",
            "dilated layers charged at float32: not quantisable",
            "activation ceiling settled by building the firmware",
            "widths frozen in a manifest, read by every stage"]):
        fig.text(0.360, 0.190 - j * 0.034, "-", fontsize=7,
                 color=COL_B, va="center", zorder=6)
        fig.text(0.372, 0.190 - j * 0.034, line, fontsize=6.3, color=INK,
                 va="center", zorder=6)


def _panel_c(fig):
    _card(fig, 0.680, 0.740, 0.306, 0.175, "#ffffff", "#f0cdb0")
    fig.text(0.833, 0.893, "RQ1   Reference architectures (FP32)",
             ha="center", va="center", fontsize=8.0, color="#9c4a1a",
             fontweight="bold", zorder=6)
    for j, line in enumerate(["10 reference models at published topology",
                              "4 held-out sources, no device constraint",
                              "outputs: Figures 2-3, Table 3 (REF column)"]):
        fig.text(0.694, 0.852 - j * 0.034, "-", fontsize=7,
                 color=COL_C, va="center", zorder=6)
        fig.text(0.706, 0.852 - j * 0.034, line, fontsize=6.5, color=INK,
                 va="center", zorder=6)

    _card(fig, 0.680, 0.330, 0.306, 0.385, "#ffffff", "#f0cdb0")
    fig.text(0.833, 0.690, "RQ2-RQ3   Deployment on STM32F411",
             ha="center", va="center", fontsize=8.0, color="#9c4a1a",
             fontweight="bold", zorder=6)
    B.board_icon(fig, 0.690, 0.425, 0.112, 0.215)
    fig.text(0.746, 0.402, "STM32F411VET6", ha="center", va="center",
             fontsize=6.6, color=INK, fontweight="bold", zorder=7)
    fig.text(0.746, 0.382, "Cortex-M4F, 100 MHz", ha="center", va="center",
             fontsize=6.0, color=INK2, zorder=7)
    fig.text(0.746, 0.363, "512 KB Flash, 128 KB SRAM", ha="center",
             va="center", fontsize=6.0, color=INK2, zorder=7)
    fig.text(0.820, 0.648, "Measured per configuration:", ha="left",
             va="center", fontsize=6.8, color=INK, fontweight="bold",
             zorder=6)
    for j, line in enumerate(["Flash occupancy (KB)", "peak SRAM (KB)",
                              "arithmetic cost (MACC)",
                              "inference latency (ms)",
                              "macro-AP of the INT8 model"]):
        fig.text(0.822, 0.610 - j * 0.036, "-", fontsize=7,
                 color=COL_C, va="center", zorder=6)
        fig.text(0.834, 0.610 - j * 0.036, line, fontsize=6.4, color=INK,
                 va="center", zorder=6)
    fig.text(0.903, 0.400, "32 timed inferences per model,",
             ha="center", va="center", fontsize=5.9, color=INK2, zorder=6)
    fig.text(0.903, 0.381, "DWT cycle counter at 100 MHz;",
             ha="center", va="center", fontsize=5.9, color=INK2, zorder=6)
    fig.text(0.903, 0.362, "deployed means it ran on the board",
             ha="center", va="center", fontsize=5.9, color=INK2, zorder=6)

    fig.text(0.833, 0.295, "Main outputs", ha="center", va="center",
             fontsize=8.2, color=INK, fontweight="bold", zorder=6)
    for j, (tag, body, colour) in enumerate([
            ("RQ1", "reference\ngeneralisation\n\nFig. 2-3", "#b03a4a"),
            ("RQ2", "scaling method\nand budget\n\nFig. 4-6", COL_B),
            ("RQ3", "discrimination\nvs resource\n\nFig. 7", "#6a4fb0")]):
        bx = 0.684 + j * 0.1005
        _card(fig, bx, 0.040, 0.094, 0.225, "#fbfbf7", colour, lw=1.0)
        fig.text(bx + 0.047, 0.233, tag, ha="center", va="center",
                 fontsize=8.0, color=colour, fontweight="bold", zorder=7)
        fig.text(bx + 0.047, 0.142, body, ha="center", va="center",
                 fontsize=6.2, color=INK, zorder=7, linespacing=1.5)

    B._flow(fig, 0.3335, 0.50, direction="right", size=12, shaft=0.010,
            color="#8a8a84", lw=2.2)
    B._flow(fig, 0.6675, 0.50, direction="right", size=12, shaft=0.010,
            color="#8a8a84", lw=2.2)


def figure3(runs, ref="FCN-1D"):
    """RQ1: does the architecture change external discrimination, and
    does the answer hold on every source?

    Reference tier only, so a row is a property of the design rather
    than of the width a budget allowed it. Each architecture carries
    five things: the paired difference against the reference on each of
    the four held-out sources, and the equal-source mean with its
    interval. An earlier version reduced the four sources to a count,
    "3 of 4", which threw away the part that matters -- which source
    disagrees, and by how much.

    FCN-1D is the reference: the plain-convolution baseline the other
    nine are implicitly arguing against, fixed before the differences
    were examined. Nine contrasts with no multiplicity adjustment, so
    they are exploratory. The coloured points are descriptive source
    estimates; only the black interval carries uncertainty, and it is
    uncertainty in the mean across these four sources, not across
    hospitals in general.
    """
    cfg = "REF"
    _, agg, src = inference(runs, configs=[cfg], ref=ref)
    if not agg:
        print("  figure3 skipped: no paired differences")
        return
    # Fixed M1-M10 order, as in Figures 4 to 7: a family sits in the
    # same row everywhere, so a reader can carry it between figures.
    # The ranking is already in the x positions and does not need the
    # row order to repeat it.
    ms = [m for m in D.ORDER if (m, cfg) in agg]
    lo_all = min(min(v[1] for k, v in agg.items() if k[1] == cfg),
                 min(v[0] for k, v in src.items() if k[1] == cfg))
    hi_all = max(max(v[2] for k, v in agg.items() if k[1] == cfg),
                 max(v[0] for k, v in src.items() if k[1] == cfg))
    pad = 0.07 * (hi_all - lo_all)

    h = 1.5 + 0.34 * len(ms)
    fig, ax = plt.subplots(figsize=(7.4, h), facecolor=SURFACE)
    for y_, m in enumerate(ms):
        for s in SOURCES:
            if (m, cfg, s) in src:
                ax.scatter(src[(m, cfg, s)][0], y_ + 0.19, s=20,
                           color=SRC_COL[s], edgecolor=SURFACE, lw=0.4,
                           zorder=4, label=B.sname(s) if y_ == 0 else None)
        pt, lo, hi = agg[(m, cfg)]
        ax.plot([lo, hi], [y_ - 0.20, y_ - 0.20], lw=1.6, color=INK,
                solid_capstyle="round", zorder=3)
        ax.scatter(pt, y_ - 0.20, s=22, color=INK, marker="s", zorder=5,
                   label="equal-source mean (95% CI)" if y_ == 0 else None)
    ax.axvline(0, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([f"{D.LABEL[m]}  {m}" for m in ms], fontsize=7.2,
                       color=INK)
    ax.set_ylim(len(ms) - 0.25, -0.75)
    ax.set_xlim(lo_all - pad, hi_all + pad)
    ax.set_xlabel(f"paired difference in macro-AP vs {ref}   "
                  "(reference tier, float32, unconstrained)",
                  fontsize=7.4, color=INK)
    ax.tick_params(axis="x", labelsize=6.4, colors=INK2)
    ax.tick_params(axis="y", length=0)
    B._frame(ax)
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    ax.text(0.015, 0.012, f"favours {ref}", transform=ax.transAxes,
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
    B.save(fig, "figure3_paired", dest=D.FIG, svg=True)


_METHOD = {}


def method_contrast(runs, reps=REPS, seed=0, cache=True):
    """Joint minus uniform at a matched weight budget, per family.

        delta(m, b) = macroAP(joint, m, b) - macroAP(uniform, m, b)

    The two models being differenced were trained on the same folds and
    are scored on the same records, so the difference is paired at the
    record level and the resampling must be too: one draw of patients
    per replicate, both models scored on it. Differencing two
    independently drawn bootstraps would widen the interval by the
    variance the pairing is there to remove.
    """
    key = (reps, seed)
    if key in _METHOD:
        return _METHOD[key]
    cf = HERE / "results" / "recon_method_contrast.json"
    stamp_src = sorted((f.name, f.stat().st_size, int(f.stat().st_mtime))
                       for f in D.TRAIN.glob("*.npz"))
    import hashlib as _h
    stamp = _h.sha1((repr(stamp_src) + repr(key)).encode()).hexdigest()[:16]
    if cache and cf.exists():
        try:
            blob = json.loads(cf.read_text())
            if blob.get("stamp") == stamp:
                print(f"    (method-contrast cache hit, {blob['reps']} reps)")
                out = {tuple(k.split("|")): tuple(v)
                       for k, v in blob["delta"].items()}
                _METHOD[key] = out
                return out
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    rng = np.random.default_rng(seed)
    out = {}
    for budget, (ucfg, jcfg) in {"UC-128": ("U-UC128", "J-UC128"),
                                 "C-256": ("U-C256", "J-C256")}.items():
        ms = [m for m in D.complete(runs, ucfg) if m in D.complete(runs, jcfg)]
        if not ms:
            continue
        srcs = [s for s in SOURCES
                if all((m, c, s) in runs for m in ms for c in (ucfg, jcfg))]
        prep = {(m, c, s): B._prepare_ap(runs[(m, c, s)]["y"],
                                         runs[(m, c, s)]["p"])
                for m in ms for c in (ucfg, jcfg) for s in srcs}
        pidx = {s: B._patient_index(s) for s in srcs}

        def mac(v):
            return np.nan if np.all(np.isnan(v)) else float(np.nanmean(v))

        obs = {}
        for m in ms:
            vals = []
            for s in srcs:
                one = np.ones(runs[(m, ucfg, s)]["y"].shape[0],
                              dtype=np.float32)
                vals.append(mac(B._ap_weighted(prep[(m, jcfg, s)], one))
                            - mac(B._ap_weighted(prep[(m, ucfg, s)], one)))
            obs[m] = float(np.mean(vals))

        draw = np.empty((len(ms), len(srcs), reps))
        for i in range(reps):
            for j, s in enumerate(srcs):
                pat, npat = pidx[s]
                w = np.bincount(rng.integers(0, npat, npat),
                                minlength=npat)[pat].astype(np.float32)
                for q, m in enumerate(ms):
                    draw[q, j, i] = (mac(B._ap_weighted(prep[(m, jcfg, s)], w))
                                     - mac(B._ap_weighted(prep[(m, ucfg, s)],
                                                          w)))
            if (i + 1) % 500 == 0:
                print(f"    method contrast {budget}: {i + 1}/{reps}",
                      flush=True)
        eq = np.nanmean(draw, axis=1)
        for q, m in enumerate(ms):
            out[(m, budget)] = (obs[m],
                                float(np.percentile(eq[q], 2.5)),
                                float(np.percentile(eq[q], 97.5)))
    if cache:
        cf.write_text(json.dumps(
            {"stamp": stamp, "reps": reps,
             "delta": {"|".join(k): v for k, v in out.items()}}, indent=1))
    _METHOD[key] = out
    return out


def figure4(runs):
    """RQ2: at a matched weight budget, does joint scaling beat uniform?

    This is the bridge from the reference tier to the device. The
    comparison that matters is not how far each scaled model falls below
    its own reference -- that confounds the method with the family's
    headroom -- but which method wins when both are given the same
    number of bytes. So the quantity plotted is the paired difference
    between the two methods at a fixed budget, one panel per budget, on
    a shared horizontal scale so the two panels can be read against each
    other.

    A family whose budget never binds cannot answer the question: FCN-1D
    and TCN are activation-bound, so their uniform and joint models sit
    at the same ceiling and the contrast is near-vacuous by
    construction. They are kept and marked rather than dropped, because
    the reason they cannot answer is itself a deployment result.
    """
    delta = method_contrast(runs)
    if not delta:
        print("  figure4 skipped: no matched pairs")
        return
    budgets = [b for b in ("UC-128", "C-256")
               if any(k[1] == b for k in delta)]
    # Fixed M1-M10 order, not sorted by effect. The two budgets rank
    # the families differently, so no single sort is correct for both
    # panels: ordering by the mean of the two produces rows that are
    # genuinely sorted for neither, which invites a reader to trust a
    # vertical position that means nothing. The ranking is already in
    # the x positions; row order is spent instead on identity, so a
    # family sits in the same place here as in Figures 5 to 7.
    ms = [m for m in D.ORDER if any((m, b) in delta for b in budgets)]
    lo = min(v[1] for v in delta.values())
    hi = max(v[2] for v in delta.values())
    pad = 0.10 * (hi - lo)

    h = 1.45 + 0.30 * len(ms)
    fig, axes = plt.subplots(1, len(budgets), figsize=(8.6, h),
                             facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    man = json.loads((HERE / "results" / "config_manifest.json").read_text()) \
        if (HERE / "results" / "config_manifest.json").exists() else {}
    for k, (bud, ax) in enumerate(zip(budgets, axes)):
        cap = D.BUDGET_KB[bud]
        for y_, m in enumerate(ms):
            if (m, bud) not in delta:
                continue
            pt, l_, h_ = delta[(m, bud)]
            colour = JNT_C if pt > 0 else UNI_C
            ax.plot([l_, h_], [y_, y_], lw=1.7, color=colour,
                    solid_capstyle="round", zorder=3, alpha=0.9)
            ax.scatter(pt, y_, s=26, color=colour,
                       marker="o" if bud == "UC-128" else "s",
                       edgecolor=SURFACE, lw=0.5, zorder=5)
            # mark the families whose weight budget never binds
            cfg = ("U-UC128" if bud == "UC-128" else "U-C256")
            rec = man.get(f"{m}__{cfg}")
            if rec and rec["weight_bytes"] / 1024 < 0.5 * cap:
                ax.text(hi + pad * 0.55, y_, "*", fontsize=9, color="#8c3b2f",
                        ha="center", va="center", zorder=6)
        ax.axvline(0, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.set_xlim(lo - pad, hi + pad * 1.2)
        ax.set_ylim(len(ms) - 0.3, -0.7)
        ax.set_title(f"({chr(65 + k)}) {bud}  (≤ {cap} KB INT8 weights)",
                     fontsize=8.0, color=INK, pad=6)
        ax.set_xlabel("macro-AP:  joint − uniform", fontsize=7.2,
                      color=INK)
        ax.tick_params(axis="x", labelsize=6.4, colors=INK2)
        ax.tick_params(axis="y", length=0)
        B._frame(ax)
        ax.grid(axis="x", color=GRID, lw=0.45)
        ax.set_axisbelow(True)
        ax.text(0.02, 0.012, "uniform better", transform=ax.transAxes,
                fontsize=6.0, color=UNI_C, ha="left", style="italic")
        ax.text(0.98, 0.012, "joint better", transform=ax.transAxes,
                fontsize=6.0, color=JNT_C, ha="right", style="italic")
    axes[0].set_yticks(range(len(ms)))
    axes[0].set_yticklabels([f"{D.LABEL[m]}  {m}" for m in ms],
                            fontsize=7.2, color=INK)
    fig.text(0.5, 0.020,
             "paired patient-cluster bootstrap, "
             f"{REPS} replicates, equal weight per held-out source;   "
             "* weight budget never binds (activation-bound)",
             fontsize=6.1, color=INK2, ha="center")
    fig.subplots_adjust(left=0.175, right=0.985, top=1 - 0.42 / h,
                        bottom=0.92 / h, wspace=0.07)
    B.save(fig, "figure4_scaling_method", dest=D.FIG, svg=True)


# ------------------------------------------- deployment, real or stand-in
def deploy_table(allow_synthetic=True):
    """Measured deployment records, or a clearly-labelled stand-in.

    Figures 5 to 7 describe the device, so until the board has been
    flashed there is nothing real to draw. Rather than leave them empty
    while the layout is still being settled, a stand-in is derived from
    quantities that *are* known for all forty configurations -- the
    exact MAC count, the predicted weight bytes and the predicted peak
    activation -- plus a flat runtime allowance and a constant
    throughput. It reproduces the shape of the figure, not its findings.

    Returns (records, synthetic). Every figure that receives
    synthetic=True stamps itself, so a placeholder cannot be mistaken
    for a measurement.
    """
    real = D.load_deploy()
    if real or not allow_synthetic:
        return real, False
    man = json.loads((HERE / "results" / "config_manifest.json").read_text())
    out = {}
    for key, rec in man.items():
        m, cfg = rec["model"], rec["config"]
        mo = fuse(build(m, rec["width"], rec["depth"]))
        mm = macs(mo, 1000)
        # a flat runtime allowance and one throughput for everybody: the
        # stand-in must not imply a per-family hardware result
        out[(m, cfg)] = {
            "model": m, "config": cfg,
            "weights_b": rec["weight_bytes"],
            "activations_b": rec["pred_act_b"],
            "total_flash_b": rec["weight_bytes"] + 40 * 1024,
            "total_ram_b": rec["pred_act_b"] + 20 * 1024,
            "macc": mm,
            "ms_median": mm / 1e6 * 48.0,
            "deployed": True, "synthetic": True}
    return out, True


def deployed_ap(runs, int8=None):
    """Equal-source mean macro-AP of the model that was deployed.

    The int8 scores where quantisation has run, the float32 scores
    otherwise, because a figure about deployment should describe the
    graph the device executes rather than the one it was distilled from.
    """
    int8 = D.load_int8() if int8 is None else int8
    # All int8 or none of it. Quantisation costs a few thousandths of
    # macro-AP, which is the same order as the differences this figure
    # is used to read, so a panel holding some int8 points and some
    # float32 ones would put a precision effect on the same axis as an
    # architecture effect and label the result neither.
    whole = all(all((m, cfg, s) in int8 for s in SOURCES)
                for cfg in D.DEPLOYED for m in D.complete(runs, cfg))
    out = {}
    for cfg in D.DEPLOYED:
        for m in D.complete(runs, cfg):
            vals = []
            for s in SOURCES:
                k = (m, cfg, s)
                if whole:
                    vals.append(B.macro_ap(int8[k]["y"], int8[k]["p8"]))
                elif k in runs:
                    vals.append(B.macro_ap(runs[k]["y"], runs[k]["p"]))
            if vals:
                out[(m, cfg)] = float(np.mean(vals))
    return out, whole


def _stamp(fig, text="SYNTHETIC PLACEHOLDER - layout only, not results"):
    fig.text(0.5, 0.5, text, fontsize=22, color="#c0392b", alpha=0.13,
             ha="center", va="center", rotation=24, zorder=50,
             fontweight="bold")


#: the stable encoding, used by every deployment figure
MCOL = {"uniform": UNI_C, "joint": JNT_C}
BSHAPE = {"UC-128": "o", "C-256": "s"}


def _cfg_style(cfg):
    return MCOL[D.METHOD_OF[cfg]], BSHAPE[D.BUDGET_OF[cfg]]


def _deploy_legend(fig, y=0.020):
    hs = [plt.Line2D([], [], lw=0, marker=BSHAPE[b], markersize=5,
                     markerfacecolor=MCOL[meth], markeredgecolor="none",
                     label=f"{meth} {b}")
          for meth in ("uniform", "joint") for b in ("UC-128", "C-256")]
    fig.legend(handles=hs, fontsize=6.3, frameon=False, labelcolor=INK2,
               ncol=4, loc="lower center", bbox_to_anchor=(0.5, y),
               columnspacing=1.6, handletextpad=0.4)


def figure5(runs=None):
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
        # Right-aligned against the line: left-aligned it runs off
        # the panel, and there is empty space inside the limit.
        ax.text(limit, -0.56, lname + " " + str(limit) + " KB  ",
                fontsize=6.4, color="#b03a4a", ha="right", va="center")
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


def figure6(runs=None):
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

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), facecolor=SURFACE,
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
    fig.subplots_adjust(left=0.100, right=0.985, top=0.950, bottom=0.200,
                        hspace=0.26)
    if synth:
        _stamp(fig)
    B.save(fig, "figure6_complexity_latency", dest=D.FIG, svg=True)


def figure7(runs):
    """RQ3: which deployed configurations trade discrimination against
    resource well?

    One panel per scaling method so forty bubbles are not piled onto one
    pair of axes, but the Pareto front is computed **once over the
    combined set**. Fronts computed per method would be two different
    questions wearing one name: a configuration that another method
    dominates outright would still appear non-dominated within its own
    panel, and the figure would recommend it.

    Axes are shared so the panels can be compared directly. Bubble area
    is peak SRAM, the resource that is scarcest here and the one that
    decided two of the ten families.
    """
    dep, synth = deploy_table()
    ap, used_int8 = deployed_ap(runs)
    if not dep or not ap:
        print("  figure7 skipped: nothing deployed yet")
        return
    pts = []
    for (m, cfg), r in dep.items():
        if not (r.get("deployed", True) and r.get("ms_median")):
            continue
        if (m, cfg) not in ap:
            continue
        pts.append({"m": m, "cfg": cfg, "ms": r["ms_median"],
                    "ap": ap[(m, cfg)],
                    "sram": (r.get("total_ram_b") or 0) / 1024})
    if not pts:
        print("  figure7 skipped: no scored deployments")
        return

    # Non-dominated over the combined set: faster and no worse, or
    # better and no slower.
    front = {(p["m"], p["cfg"]) for p in pts
             if not any(q["ms"] <= p["ms"] and q["ap"] >= p["ap"]
                        and (q["ms"], -q["ap"]) != (p["ms"], -p["ap"])
                        for q in pts)}

    lo_ms = min(p["ms"] for p in pts) * 0.75
    hi_ms = max(p["ms"] for p in pts) * 1.35
    lo_ap = min(p["ap"] for p in pts) - 0.02
    hi_ap = max(p["ap"] for p in pts) + 0.02
    smax = max(p["sram"] for p in pts) or 1.0

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), facecolor=SURFACE,
                             sharey=True)
    for k, (meth, ax) in enumerate(zip(("uniform", "joint"), axes)):
        sel = [p for p in pts if D.METHOD_OF[p["cfg"]] == meth]
        fr = sorted([p for p in sel if (p["m"], p["cfg"]) in front],
                    key=lambda p: p["ms"])
        if len(fr) > 1:
            ax.plot([p["ms"] for p in fr], [p["ap"] for p in fr], lw=1.0,
                    color="#8a8a84", ls=(0, (4, 2)), zorder=2)
        for p in sel:
            colour = B.MODEL_COLOR[D.ICON[p["m"]]]
            on = (p["m"], p["cfg"]) in front
            ax.scatter(p["ms"], p["ap"],
                       s=28 + 300 * (p["sram"] / smax),
                       marker=BSHAPE[D.BUDGET_OF[p["cfg"]]],
                       color=colour, alpha=0.55 if not on else 0.9,
                       edgecolor=INK if on else SURFACE,
                       lw=1.0 if on else 0.4, zorder=5 if on else 4)
            ax.annotate(D.LABEL[p["m"]], (p["ms"], p["ap"]),
                        textcoords="offset points", xytext=(0, 0),
                        ha="center", va="center", fontsize=4.8,
                        color=INK, zorder=7)
        ax.set_xscale("log")
        ax.set_xlim(lo_ms, hi_ms)
        ax.set_ylim(lo_ap, hi_ap)
        # Decade ticks only: the default minor labels collide at this
        # width and turn the axis into a smear.
        ax.set_xticks([100, 200, 500, 1000, 2000, 5000])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"({chr(65 + k)}) {meth} scaling   "
                     f"({len(sel)} deployed)", fontsize=8.0, color=INK,
                     pad=5)
        ax.set_xlabel("measured latency on STM32F411 (ms, log)",
                      fontsize=7.2, color=INK)
        ax.tick_params(labelsize=6.2, colors=INK2)
        B._frame(ax)
        ax.grid(color=GRID, lw=0.4, which="both")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("equal-source mean macro-AP"
                       + (" of the deployed INT8 model" if used_int8
                          else " (float32; INT8 scoring not finished)"),
                       fontsize=7.2, color=INK)

    hs = [plt.Line2D([], [], lw=0, marker=BSHAPE[b], markersize=5,
                     markerfacecolor="#8a8a84", markeredgecolor="none",
                     label=b) for b in ("UC-128", "C-256")]
    hs.append(plt.Line2D([], [], lw=0, marker="o", markersize=7,
                         markerfacecolor="none", markeredgecolor=INK,
                         label="non-dominated (both methods pooled)"))
    for s_kb in (np.percentile([p["sram"] for p in pts], 20),
                 np.percentile([p["sram"] for p in pts], 80)):
        hs.append(plt.Line2D(
            [], [], lw=0, marker="o",
            markersize=np.sqrt(28 + 300 * (s_kb / smax)) * 0.62,
            markerfacecolor="#cfcfc8", markeredgecolor="none",
            label=f"{s_kb:.0f} KB SRAM"))
    fig.legend(handles=hs, fontsize=6.3, frameon=False, labelcolor=INK2,
               ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               columnspacing=1.5, handletextpad=0.5)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.925, bottom=0.185,
                        wspace=0.06)
    if synth:
        _stamp(fig)
    B.save(fig, "figure7_tradeoffs", dest=D.FIG, svg=True)


def table2(runs=None):
    """Architecture definitions and scaling specifications.

    Section A is the reference tier: what each family is, and how much
    of it there is at published topology. Section B is what survives a
    budget, two rows per family so the two scaling methods can be read
    against each other on one line each.

    No Flash or SRAM measurement appears here. Those are device results
    and belong to Figure 5; mixing them in would make a specification
    table look like a finding.
    """
    man = json.loads((HERE / "results" / "config_manifest.json").read_text())
    rows_a = []
    for m in D.ORDER:
        variant, year, block = FAMILY[m]
        mo = fuse(build(m))
        rows_a.append({"ID": D.LABEL[m], "Family": m,
                       "Published variant": variant, "Year": year,
                       "Defining operation": block,
                       "Blocks": PUBLISHED_DEPTH[m],
                       "Parameters (k)": round(n_params(mo) / 1e3, 1),
                       "MMACC": round(macs(mo, 1000) / 1e6, 2)})
    df_a = pd.DataFrame(rows_a)
    save_table(df_a, "table2a_reference",
               "Table 2, Section A. The ten reference architectures at "
               "published topology, adapted to 1-D. Blocks is the published "
               "count and is reproduced exactly; the reference tier carries "
               "no device constraint, so no family is truncated.")

    rows_b = []
    for m in D.ORDER:
        for meth in ("uniform", "joint"):
            r = {"ID": D.LABEL[m], "Family": m, "Method": meth.capitalize()}
            for bud, cfg in (("UC-128", "U-" if meth == "uniform" else "J-"),
                             ("C-256", "U-" if meth == "uniform" else "J-")):
                key = f"{m}__{cfg}{'UC128' if bud == 'UC-128' else 'C256'}"
                rec = man[key]
                mo = fuse(build(m, rec["width"], rec["depth"]))
                tag = "UC-128" if bud == "UC-128" else "C-256"
                r[f"{tag} width"] = round(rec["width"], 3)
                r[f"{tag} blocks"] = _blocks(mo)
                r[f"{tag} weights (KB)"] = round(rec["weight_bytes"] / 1024,
                                                 1)
            rows_b.append(r)
    df_b = pd.DataFrame(rows_b)
    save_table(df_b, "table2b_scaled",
               "Table 2, Section B. The forty deployed configurations, two "
               "rows per family. Uniform keeps the published block count and "
               "scales channel width alone; joint scales width and depth "
               "together and reports the block count that survives. Per-stage "
               "block counts for joint scaling are in "
               "results/config_manifest.json, since one global depth "
               "multiplier does not determine them. Every configuration was "
               "compiled and linked for the target before training.")
    return df_a, df_b


def table3(runs):
    """External discrimination across the reference and scaled tiers.

    Ten rows, five configuration columns, each cell an equal-source mean
    macro-AP with a patient-cluster bootstrap interval. Source-specific
    estimates and macro-AUROC are supplementary: carrying them here
    would turn one readable table into about fifty rows.
    """
    marg, _, _ = inference(runs, configs=D.CONFIGS)
    if not marg:
        print("  table3 skipped: no runs")
        return
    rows = []
    for m in D.ORDER:
        r = {"ID": D.LABEL[m], "Family": m}
        for cfg in D.CONFIGS:
            v = marg.get((m, cfg))
            r[D.CONFIG_LABEL[cfg].split(" (")[0]] = (
                f"{v[0]:.3f} ({v[1]:.3f}-{v[2]:.3f})" if v else "--")
        rows.append(r)
    df = pd.DataFrame(rows)
    save_table(df, "table3_discrimination",
               "Table 3. Equal-source mean macro-AP with 95% patient-cluster "
               f"bootstrap intervals ({REPS} replicates), for the float32 "
               "reference tier and the four deployed INT8 configurations. "
               "Sources carry equal weight. Intervals cover patient sampling "
               "within these four sources and are not between-hospital "
               "intervals.")
    return df

if __name__ == "__main__":
    main()
