"""Assemble the paper's two tables and two figures from the finished runs.

    Table I   overall performance: AUROC, AUPRC, full risk, AURC, C@R<=0.20
    Table II  diagnosis-level behaviour at the headline risk level, one column
              per representation, each cell "retention / accepted sensitivity"
    Fig. 1    study design schematic
    Fig. 2    risk-coverage curves per held-out source, with the oracle ranking

A per-source breakdown is also written as supplementary output, but the paper
carries two tables.

Tables are written as plain text and LaTeX so they drop into the manuscript.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

TARGETS = ("PTBXL", "Georgia", "Chapman", "Ningbo")
TARGET_LABEL = {"PTBXL": "PTB-XL", "Georgia": "Georgia",
                "Chapman": "Chapman", "Ningbo": "Ningbo"}
REPRESENTATIONS = ("raw", "vcg", "fft")
REP_LABEL = {"raw": "Raw 12-lead", "vcg": "VCG X,Y,Z", "fft": "FFT magnitude"}
REP_VIEW = {"raw": "temporal", "vcg": "spatial", "fft": "spectral"}
REP_MODEL = {"raw": "ResNet1d18", "vcg": "ResNet1d18", "fft": "ResNet1d18"}

# alpha = 0.20 is the headline operating point; 0.10 and 0.30 are the stringent
# and lenient points. This is selective Jaccard risk, not a patient-harm
# threshold.
HEADLINE_ALPHA = 0.20
ALPHA_LEVELS = (0.20, 0.10, 0.30)

# Display floor for Fig. 2: below this coverage the accepted subset is too small
# for the empirical risk to mean anything, and the resulting spike compresses
# the informative part of every curve. AURC still uses the whole curve.
MIN_DISPLAY_COVERAGE = 0.01
RISK_DISPLAY_MAX = 0.65

# Colour-blind safe, and distinguishable in greyscale by linestyle too.
REP_STYLE = {"raw": ("#0072B2", "-"), "vcg": ("#D55E00", "--"),
             "fft": ("#009E73", "-.")}


def load_results(runs: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for rep in REPRESENTATIONS:
        for target in TARGETS:
            path = runs / f"{rep}__{target}" / "eval.json"
            if path.exists():
                out[(rep, target)] = json.loads(path.read_text(encoding="utf-8"))
    return out


def model_label(results: dict, rep: str) -> str:
    for (r, _), res in results.items():
        if r == rep and res.get("backbone"):
            return res["backbone"]
    return REP_MODEL[rep]


def _sweep_entry(result: dict, alpha: float) -> dict | None:
    return next((s for s in result["alpha_sweep"]
                 if abs(s["alpha"] - alpha) < 1e-9), None)


def certified_sources(results: dict, rep: str, alpha: float) -> list[str]:
    """Held-out sources where this pipeline certifies non-zero coverage."""
    out = []
    for t in TARGETS:
        r = results.get((rep, t))
        e = _sweep_entry(r, alpha) if r else None
        if e and e["test_coverage"] > 0:
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
#  Table I -- overall performance
# --------------------------------------------------------------------------- #
def table_one(results: dict, out_dir: Path,
              alpha: float = HEADLINE_ALPHA) -> str:
    """Discrimination and selective performance in one table.

    AURC is the primary selective metric: it summarises the whole
    risk-coverage ranking without depending on a chosen risk level. The
    fixed-risk coverage follows as the interpretable operating point. The oracle
    row is the best any confidence score could achieve on these predictions --
    neither a certified coverage nor a statistical bound.
    """
    lines = ["Table I. Overall predictive and selective performance across four "
             "held-out sources.",
             f"Cert C@R<={alpha:.2f} is the coverage retained at a certified "
             f"selective Jaccard risk of {alpha:.2f}",
             "(delta = 0.05), averaged over the four leave-one-source-out "
             "experiments.", ""]
    header = (f"{'Representation':<16}{'AUROC':>8}{'AUPRC':>8}{'Full risk':>11}"
              f"{'AURC':>8}{'Cert C@R<=' + format(alpha, '.2f'):>16}")
    lines += [header, "-" * len(header)]

    rows, oracle_aurc, oracle_cov = [], [], []
    for rep in REPRESENTATIONS:
        got = [results[(rep, t)] for t in TARGETS if (rep, t) in results]
        if not got:
            continue
        auroc = float(np.mean([g["classification"]["macro_auroc"] for g in got]))
        auprc = float(np.mean([g["classification"]["macro_auprc"] for g in got]))
        risk = float(np.mean([g["classification"]["full_coverage_risk"] for g in got]))
        a = float(np.mean([g["selective"]["aurc"] for g in got]))
        entries = [e for e in (_sweep_entry(g, alpha) for g in got) if e]
        cov = float(np.mean([e["test_coverage"] for e in entries])) if entries else np.nan
        oracle_aurc.append(np.mean([g["selective"].get("oracle_aurc", np.nan)
                                    for g in got]))
        oracle_cov.append(np.nanmean([e.get("oracle_coverage", np.nan)
                                      for e in entries]) if entries else np.nan)
        rows.append((rep, auroc, auprc, risk, a, cov, len(got)))
        lines.append(f"{REP_LABEL[rep]:<16}{auroc:>8.3f}{auprc:>8.3f}"
                     f"{risk:>11.3f}{a:>8.3f}{cov:>16.3f}"
                     + ("" if len(got) == 4 else f"   [{len(got)}/4]"))

    if rows:
        lines.append("-" * len(header))
        lines.append(f"{'Oracle ranking':<16}{'--':>8}{'--':>8}{'--':>11}"
                     f"{np.nanmean(oracle_aurc):>8.3f}"
                     f"{np.nanmean(oracle_cov):>16.3f}")
        lines += ["", "Oracle ranking: records ordered by their realised loss. "
                  "The best any confidence",
                  "score could achieve on these predictions; not a certified "
                  "coverage and not a bound."]

    text = "\n".join(lines)
    (out_dir / "table1.txt").write_text(text + "\n", encoding="utf-8")

    tex = [r"\caption{Overall predictive and selective performance across four "
           r"held-out sources. AURC is the primary selective metric; "
           rf"Cert.\ C@R$\leq${alpha:.2f} is the coverage retained at a certified "
           r"selective Jaccard risk ($\delta=0.05$), averaged over the four "
           r"leave-one-source-out experiments. The oracle row orders records by "
           r"their realised loss: it is the best any confidence score could "
           r"achieve on these predictions, and is neither a certified coverage "
           r"nor a statistical bound.}",
           r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Representation & AUROC & AUPRC & Full risk & AURC $\downarrow$ & "
           rf"Cert.\ C@R$\leq${alpha:.2f} $\uparrow$ \\", r"\midrule"]
    for rep, auroc, auprc, risk, a, cov, _ in rows:
        tex.append(f"{REP_LABEL[rep]} ({REP_VIEW[rep]}) & {auroc:.3f} & "
                   f"{auprc:.3f} & {risk:.3f} & {a:.3f} & {cov:.3f} " + r"\\")
    if rows:
        tex += [r"\midrule",
                r"\textit{Oracle ranking} & -- & -- & -- & "
                + f"{np.nanmean(oracle_aurc):.3f} & "
                + f"{np.nanmean(oracle_cov):.3f} " + r"\\"]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "table1.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return text


# --------------------------------------------------------------------------- #
#  Table II -- diagnosis-level behaviour
# --------------------------------------------------------------------------- #
def table_two(results: dict, out_dir: Path,
              alpha: float = HEADLINE_ALPHA) -> str:
    """Per-diagnosis retention and accepted sensitivity, one column per
    representation, macro-averaged over all four held-out sources.

    A K x K confusion matrix is not defined here: a record may carry several
    diagnoses at once, so "AF confused with RBBB" has no meaning when both can
    be simultaneously correct.  Each diagnosis instead gets

        retention = P(accepted | y_k = 1)
        sens_acc  = P(yhat_k = 1 | y_k = 1, accepted)

    Two aggregation decisions matter here.

    *All four sources count.*  A source with zero certified coverage contributes
    zero retention rather than being dropped.  Restricting the table to the
    sources where the method happened to work would condition the analysis on
    its own success and make it optimistic -- the hard sources would disappear
    precisely because they were hard.

    *Macro over sources, not micro over records.*  Ningbo's test split is
    27,924 records against Chapman's 8,198, so a record-pooled average would let
    one institution dominate the diagnosis-level summary -- which would sit
    badly with a paper whose central claim is that source variation dominates.
    Each held-out source therefore gets equal weight.

    Accepted sensitivity is undefined where a source accepted no positive
    records, and undefined is not zero: it is averaged only over the sources
    that contributed accepted positives, and that count is reported.
    """
    abbrev = next(iter(results.values()))["abbreviations"]
    lines = [f"Table II. Diagnosis-level selective performance at R <= {alpha:.2f}, "
             "macro-averaged across the four",
             "leave-one-source-out test sets. Entries are positive-class "
             "retention / sensitivity among",
             "accepted positive records; [n] is the number of sources "
             "contributing accepted positives.",
             "Sources with zero certified coverage contribute zero retention; "
             "accepted sensitivity is",
             "averaged only over sources containing accepted positive records.", ""]
    header = (f"{'Diagnosis':<11}{'n pos':>8}"
              + "".join(f"{REP_LABEL[r]:>22}" for r in REPRESENTATIONS))
    lines += [header, "-" * len(header)]

    tex_rows, payload = [], {}
    for k, name in enumerate(abbrev):
        n_pos = sum(results[(REPRESENTATIONS[0], t)]["per_class"][k]["n_positive"]
                    for t in TARGETS if (REPRESENTATIONS[0], t) in results)
        if n_pos == 0:
            continue
        cells, tex_cells = [], []
        for rep in REPRESENTATIONS:
            pcs = [results[(rep, t)]["per_class"][k] for t in TARGETS
                   if (rep, t) in results]
            pcs = [p for p in pcs if p["n_positive"] > 0]
            if not pcs:
                cells.append("--")
                tex_cells.append("--")
                continue
            # Retention: every source counts, zero-coverage ones included.
            ret = float(np.mean([p["retention"] for p in pcs]))
            # Accepted sensitivity: only sources that accepted a positive.
            contributing = [p["sens_accepted"] for p in pcs
                            if np.isfinite(p["sens_accepted"])]
            sa = float(np.mean(contributing)) if contributing else np.nan
            n_src = len(contributing)
            payload.setdefault(name, {})[rep] = {
                "retention_macro": ret, "sens_accepted_macro": sa,
                "n_sources_with_accepted_positives": n_src,
                "n_sources": len(pcs)}
            body = f"{ret:.2f} / " + (f"{sa:.2f}" if np.isfinite(sa) else "--")
            cells.append(f"{body} [{n_src}]")
            tex_cells.append(body)
        lines.append(f"{name:<11}{n_pos:>8d}"
                     + "".join(f"{c:>22}" for c in cells))
        tex_rows.append(f"{name} & " + " & ".join(tex_cells) + r" \\")

    text = "\n".join(lines)
    (out_dir / "table2.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / "table2_detail.json").write_text(
        json.dumps({"alpha": alpha, "aggregation": "source-macro over 4 targets",
                    "classes": payload}, indent=2), encoding="utf-8")

    caption = (
        rf"\caption{{Diagnosis-level selective performance at $R\leq{alpha:.2f}$, "
        r"macro-averaged across the four leave-one-source-out test sets. Entries "
        r"report positive-class retention / sensitivity among accepted positive "
        r"records. Each held-out source uses an independently trained model and "
        r"an abstention threshold certified on its disjoint target calibration "
        r"subset. PTB-XL and Georgia yielded zero certified coverage and "
        r"therefore contribute zero to retention; accepted sensitivity is "
        r"averaged only over sources containing accepted positive records.}")
    tex = [caption, r"\begin{tabular}{l" + "c" * len(REPRESENTATIONS) + "}",
           r"\toprule",
           "Diagnosis & "
           + " & ".join(REP_LABEL[r] for r in REPRESENTATIONS) + r" \\",
           r"\midrule"]
    tex += tex_rows
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "table2.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return text


# --------------------------------------------------------------------------- #
#  Supplementary -- per-source detail
# --------------------------------------------------------------------------- #
def table_supplementary(results: dict, out_dir: Path,
                        alphas: tuple[float, ...] = ALPHA_LEVELS) -> str:
    lines = ["Supplementary. Certified coverage (observed test risk) "
             "[oracle ranking coverage] per held-out source.", ""]
    for alpha in alphas:
        lines.append(f"alpha = {alpha:.2f}")
        header = (f"  {'Representation':<16}"
                  + "".join(f"{TARGET_LABEL[t]:>21}" for t in TARGETS))
        lines += [header, "  " + "-" * (len(header) - 2)]
        for rep in REPRESENTATIONS:
            cells = []
            for t in TARGETS:
                r = results.get((rep, t))
                e = _sweep_entry(r, alpha) if r else None
                if e is None:
                    cells.append("--")
                else:
                    flag = "" if e["risk_respected"] else "*"
                    cells.append(f"{e['test_coverage']:.2f} "
                                 f"({e['test_risk']:.3f}){flag} "
                                 f"[{e.get('oracle_coverage', float('nan')):.2f}]")
            lines.append(f"  {REP_LABEL[rep]:<16}"
                         + "".join(f"{c:>21}" for c in cells))
        lines.append("")
    lines.append("* observed test risk exceeded alpha.")
    text = "\n".join(lines)
    (out_dir / "supplementary_per_source.txt").write_text(text + "\n",
                                                          encoding="utf-8")
    return text


# --------------------------------------------------------------------------- #
#  Figure 2 -- risk-coverage curves
# --------------------------------------------------------------------------- #
def figure_two(runs: Path, results: dict, out_dir: Path,
               alpha: float = HEADLINE_ALPHA) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharex=True, sharey=True)

    for ax, target, panel in zip(axes.ravel(), TARGETS, "abcd"):
        oracle_best = None
        zero_certified: list[str] = []
        for rep in REPRESENTATIONS:
            npz_path = runs / f"{rep}__{target}" / "risk_coverage.npz"
            if not npz_path.exists():
                continue
            d = np.load(npz_path)
            colour, style = REP_STYLE[rep]
            cov, risk = d["coverage"], d["risk"]
            keep = cov >= MIN_DISPLAY_COVERAGE
            step = max(1, int(keep.sum()) // 1500)
            ax.plot(cov[keep][::step], risk[keep][::step], style,
                    color=colour, lw=1.4, label=REP_LABEL[rep])

            e = _sweep_entry(results[(rep, target)], alpha)
            if e and e["test_coverage"] > 0:
                ax.plot(e["test_coverage"], e["test_risk"], "o", color=colour,
                        ms=5, mec="white", mew=0.8, zorder=5)
            elif e:
                zero_certified.append(REP_LABEL[rep].split()[0])

            if "oracle_risk" in d.files:
                cur = np.asarray(d["oracle_risk"])
                if oracle_best is None or cur.mean() < oracle_best.mean():
                    oracle_best = cur

        if oracle_best is not None:
            ocov = np.linspace(1 / len(oracle_best), 1.0, len(oracle_best))
            okeep = ocov >= MIN_DISPLAY_COVERAGE
            ostep = max(1, int(okeep.sum()) // 1500)
            ax.plot(ocov[okeep][::ostep], oracle_best[okeep][::ostep], ":",
                    color="0.25", lw=1.0, label="oracle ranking")

        if zero_certified:
            ax.text(0.97, 0.05, "no certified coverage: "
                    + ", ".join(zero_certified), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6.5, color="0.35")

        ax.axhline(alpha, color="0.55", lw=0.8, ls=(0, (1, 3)), zorder=1)
        ax.set_title(f"({panel}) {TARGET_LABEL[target]} held out", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, RISK_DISPLAY_MAX)

    for ax in axes[-1]:
        ax.set_xlabel("Coverage")
    for ax in axes[:, 0]:
        ax.set_ylabel("Selective risk (Jaccard)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="0.55", lw=0.8, ls=(0, (1, 3))))
    labels.append(rf"$\alpha={alpha:g}$")
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Risk-coverage on the unseen clinical source "
                 "(markers: calibrated operating point)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig2_risk_coverage.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 1 -- study design
# --------------------------------------------------------------------------- #
def figure_one(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EAF2F8", ec="#2C3E50", fs=8):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#2C3E50"))

    for i, rep in enumerate(REPRESENTATIONS):
        box(0.06 + i * 0.30, 0.78, 0.24, 0.13,
            f"{REP_LABEL[rep]}\n({REP_VIEW[rep]})", fc="#FDEBD0")
        arrow(0.18 + i * 0.30, 0.78, 0.50, 0.70)

    box(0.28, 0.58, 0.44, 0.12,
        "ResNet1d18  (shared backbone)\n13 sigmoid outputs", fc="#EAF2F8")
    arrow(0.50, 0.58, 0.50, 0.50)
    box(0.20, 0.38, 0.60, 0.12,
        r"confidence  $s(x)$  from per-class decision margins", fc="#EAF2F8")
    arrow(0.50, 0.38, 0.50, 0.30)
    box(0.22, 0.18, 0.56, 0.12,
        "Learn-then-Test calibration on the target source\n"
        r"$\max_\tau C(\tau)$  s.t.  risk certified $\leq\alpha$", fc="#D5F5E3")
    arrow(0.50, 0.18, 0.50, 0.10)
    box(0.34, 0.00, 0.32, 0.09, "accept  /  abstain", fc="#FADBD8")

    ax.text(0.5, 0.955, "Three training sources  $\\rightarrow$  model    |    "
                        "held-out source  $\\rightarrow$  calibrate + test",
            ha="center", va="center", fontsize=8.5, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.0)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig1_design.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "artifacts" / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--alpha", type=float, default=HEADLINE_ALPHA)
    args = ap.parse_args()

    tables = args.out / "tables"
    figures = args.out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    results = load_results(args.runs)
    if not results:
        raise SystemExit(f"no eval.json found under {args.runs}")
    print(f"Loaded {len(results)}/{len(REPRESENTATIONS) * len(TARGETS)} "
          "evaluated pipelines.\n")

    print(table_one(results, tables, args.alpha), "\n")
    print(table_two(results, tables, args.alpha), "\n")
    print(table_supplementary(results, tables), "\n")

    figure_one(figures)
    figure_two(args.runs, results, figures, args.alpha)
    print(f"Wrote tables to {tables} and figures to {figures}")


if __name__ == "__main__":
    main()
