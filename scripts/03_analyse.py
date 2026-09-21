r"""The whole study: one figure, the numbers behind it, nothing else.

    .\.venv\Scripts\python.exe scripts\03_analyse.py

Reads only cached predictions, so every post-hoc decision -- a different
coverage grid, more bootstrap replicates, a changed uncertainty definition --
costs seconds and never a retrain.

    data/ensembles/code15/ensemble.npz   5 members' val+test probabilities,
                                         thresholds frozen on validation
    data/prepared/code_test/meta.npz     adjudicated gold, reader 1, reader 2

The question
------------
Do cardiologist-disputed ECG labels remain disproportionately represented
among the errors that survive uncertainty-based referral?

    A   humans disagree            ->  the model is more uncertain
    B   refer uncertain predictions ->  overall risk falls
    C   but what remains?           ->  disputed labels may be enriched

One uncertainty throughout: ``U = -|logit(pbar) - logit(t_c)|``.  The same
numbers rank predictions for referral and populate panel A, so the supporting
result and the headline cannot come from two differently behaved scores.

``D`` is computed here, after prediction, and never touches selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

from ecguq.bootstrap import clustered_ci, fmt                     # noqa: E402
from ecguq.data import LABELS                                     # noqa: E402
from ecguq.metrics import macro_auprc, macro_auroc                # noqa: E402
from ecguq.selective import (retained_mask, disagreement_enrichment,             # noqa: E402
                             patient_errors_at, patient_risk_at,
                             disputed_error_share, disputed_prevalence,
                             macro_aurc, macro_risk_at,
                             retained_disputed_prevalence)
from ecguq.uncertainty import (ensemble_mean, model_uncertainty,  # noqa: E402
                               predict, predictive_uncertainty,
                               reader_disagreement, uncertainty)

# Validated categorical slots 1-2 (all six colour checks pass on the light
# surface): identity is also carried by position and direct labels, never by
# colour alone.
C_AGREE, C_DISPUTE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SURFACE = "#fcfcfb"

# Ascending: risk-coverage reads left to right, full coverage on the right.
# q = 0 retains nothing and is undefined, so evaluation starts at 0.01
# while the axis still spans the full [0, 1].
COVERAGES = np.round(np.arange(0.01, 1.0001, 0.01), 2)
XVIEW = (0.50, 1.00)          # displayed span of the risk-coverage panel
XTICKS = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
# Analysis runs the full grid; the figure shows the clinically
# informative high-coverage region. 1 - q is the referral fraction, so
# this window spans 50% referred (left) to none referred (right).
# Deliberately wider than the region where the curve moves: showing only
# the high-coverage rise invites the charge that the window was chosen
# to make the effect look dramatic. A flat left half is itself a result.
# Not extended to q = 0, where the retained sample becomes tiny and the
# panel gains nothing but empty space.
REPORT_AT = (1.00, 0.90, 0.80, 0.50)
N_BOOT = 1000
# Below this many residual errors an enrichment ratio is arithmetic, not
# evidence: one disputed error out of six gives rho_D = 0.167 regardless
# of whether disagreement has anything to do with it.
MIN_EVENTS = 10


def load(ens_dir: Path, test_dir: Path) -> dict:
    e = np.load(ens_dir / "ensemble.npz")
    m = np.load(test_dir / "meta.npz")
    pbar = ensemble_mean(e["test"])
    t = e["thresholds"]
    y = m["y"].astype(np.int8)
    return {
        "probs": e["test"],
        "pbar": pbar, "thresholds": t, "y": y,
        "U": uncertainty(pbar, t),
        "loss": (predict(pbar, t) != y).astype(np.float64),
        "D": reader_disagreement(m["reader1"], m["reader2"]),
        "members": e["test"].shape[0],
        "y_val": e["y_val"], "pbar_val": ensemble_mean(e["val"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensembles", type=Path,
                    default=ROOT / "data" / "ensembles" / "code15")
    ap.add_argument("--test", type=Path,
                    default=ROOT / "data" / "prepared" / "code_test")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--replicates", type=int, default=N_BOOT)
    args = ap.parse_args()
    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)

    d = load(args.ensembles, args.test)
    U, loss, D, y = d["U"], d["loss"], d["D"], d["y"]
    conf = -U                                     # referral ranks by confidence
    n = y.shape[0]
    L = list(LABELS)

    # ---- descriptive -----------------------------------------------------
    pairs = D.size
    n_dis = int(D.sum())
    per_class_dis = D.sum(axis=0)
    lines = ["CARDIOLOGIST DISAGREEMENT AND PREDICTIVE UNCERTAINTY", "=" * 74, "",
             f"CODE-test: {n} ECGs x {len(L)} diagnoses = {pairs} label-record "
             f"pairs", f"Ensemble of {d['members']} independently initialised "
             f"plain 1-D CNNs trained on CODE-15%.", "",
             f"Reader disagreement: {n_dis} of {pairs} pairs "
             f"({n_dis / pairs:.2%})", ""]
    hdr = (f"{'Diagnosis':<10}{'gold +':>9}{'disputed':>10}"
           f"{'AUROC':>9}{'AUPRC':>9}")
    lines += [hdr, "-" * len(hdr)]
    from sklearn.metrics import average_precision_score, roc_auc_score
    for j, lab in enumerate(L):
        yc, pc = y[:, j], d["pbar"][:, j]
        ro = roc_auc_score(yc, pc) if 0 < yc.sum() < n else float("nan")
        pr = average_precision_score(yc, pc) if yc.sum() else float("nan")
        lines.append(f"{lab:<10}{int(yc.sum()):>9}{int(per_class_dis[j]):>10}"
                     f"{ro:>9.3f}{pr:>9.3f}")
    lines += ["-" * len(hdr),
              f"{'macro':<10}{'':>9}{'':>10}"
              f"{macro_auroc(y, d['pbar']):>9.3f}"
              f"{macro_auprc(y, d['pbar']):>9.3f}", ""]

    # ---- A: uncertainty vs disagreement ----------------------------------
    def delta_u(idx):
        u, dd = U[idx], D[idx].astype(bool)
        if dd.sum() == 0 or (~dd).sum() == 0:
            return float("nan")
        return float(np.median(u[dd]) - np.median(u[~dd]))

    du = clustered_ci(delta_u, n, args.replicates, seed=1)
    lines += ["A. DECISION UNCERTAINTY BY READER AGREEMENT", "-" * 74,
              f"  median U | agreement    {np.median(U[D == 0]):+.3f}   "
              f"(n = {pairs - n_dis})",
              f"  median U | disagreement {np.median(U[D == 1]):+.3f}   "
              f"(n = {n_dis})",
              f"  delta U                 {fmt(*du)}",
              "", "  U = -|logit(pbar) - logit(t_c)|; higher means less "
              "certain.", ""]

    # ---- B: referral -----------------------------------------------------
    lines += ["B. UNCERTAINTY-BASED REFERRAL", "-" * 74,
              f"{'coverage':>10}{'macro risk (95% CI)':>34}", "-" * 44]
    riskq = {}
    for q in REPORT_AT:
        r = clustered_ci(lambda i, q=q: macro_risk_at(loss[i], conf[i], q),
                         n, args.replicates, seed=2)
        riskq[q] = r
        lines.append(f"{q:>10.0%}{fmt(*r):>34}")
    au = clustered_ci(lambda i: macro_aurc(loss[i], conf[i]), n,
                      args.replicates, seed=3)
    lines += ["", f"  macro AURC {fmt(*au)}", ""]

    # ---- C: enrichment ---------------------------------------------------
    lines += ["C. DISAGREEMENT AMONG RESIDUAL ERRORS", "-" * 74,
              f"{'coverage':>9}{'n_ret':>8}{'n_err':>7}{'n_D':>6}"
              f"{'n_errD':>8}{'rho_D':>8}{'pi_D':>9}"
              f"{'enrichment E_D (95% CI)':>30}", "-" * 85]
    enr = {}
    counts = {}
    for q in REPORT_AT:
        rho = disputed_error_share(loss, conf, D, q)
        pi = retained_disputed_prevalence(conf, D, q)
        # Raw counts behind the two ratios. With 33 disagreements in the whole
        # test set, referral can drive these into single digits, and a ratio
        # built on 3 errors should not be read like one built on 30.
        keep = retained_mask(conf, q)
        Db = D.astype(bool)
        n_err_D = int((loss.astype(bool) & keep & Db).sum())
        n_D = int((keep & Db).sum())
        counts[q] = {"retained": int(keep.sum()),
                     "errors": int((loss.astype(bool) & keep).sum()),
                     "disputed_retained": n_D,
                     "disputed_errors_retained": n_err_D}
        e = clustered_ci(
            lambda i, q=q: disagreement_enrichment(loss[i], conf[i], D[i], q),
            n, args.replicates, seed=4)
        enr[q] = e
        c_ = counts[q]
        lines.append(f"{q:>9.0%}{c_['retained']:>8d}{c_['errors']:>7d}"
                     f"{n_D:>6d}{n_err_D:>8d}{rho:>8.3f}"
                     f"{pi:>9.4f}{fmt(*e, nd=2):>30}")
    lines += ["", "  n_ret  = retained (record, diagnosis) pairs summed over diagnoses",
              "  n_err  = errors among them        n_D = retained disputed labels",
              "  n_errD = retained errors on disputed labels (numerator of rho_D)",
              "  rho_D = disputed share of errors surviving referral",
              "  pi_D  = disputed share of retained predictions",
              "  E_D   = rho_D / pi_D; 1 means no enrichment.",
              "",
              "  pi_D is the retained prevalence, not the full-cohort "
              f"{disputed_prevalence(D):.2%}: referral itself removes",
              "  disputed labels, so the fixed baseline would credit the "
              "policy with an enrichment", "  it did not produce.", ""]

    # ---- prespecified secondary: 1dAVb -----------------------------------
    j = L.index("1dAVb")
    lines += ["PRESPECIFIED SECONDARY: 1dAVb", "-" * 74,
              f"  disputed {int(per_class_dis[j])} of {n_dis} disagreements "
              f"overall, against {int(y[:, j].sum())} adjudicated positives.",
              "  Counts are reported separately; the two denominators are not "
              "interchangeable.", ""]

    # ---- D: sensitivity to the uncertainty score -------------------------
    # Prespecified as sensitivity only: the decision-margin score was fixed
    # before the test set was read and does not change because an alternative
    # happens to rank better here.  Reported as numbers, not a second figure.
    alt = {"decision margin (primary)": conf,
           "ensemble variance": -model_uncertainty(d["probs"]),
           "predictive entropy": -predictive_uncertainty(d["pbar"])}
    sens = {}
    lines += ["D. SENSITIVITY TO THE UNCERTAINTY SCORE", "-" * 74,
              f"{'score':<28}{'macro AURC':>12}{'R(0.80)':>10}"
              f"{'E_D(0.80)':>12}", "-" * 62]
    for name, cf in alt.items():
        sens[name] = {"macro_aurc": macro_aurc(loss, cf),
                      "risk_0.80": macro_risk_at(loss, cf, 0.80),
                      "E_D_0.80": disagreement_enrichment(loss, cf, D, 0.80)}
        s = sens[name]
        lines.append(f"{name:<28}{s['macro_aurc']:>12.4f}"
                     f"{s['risk_0.80']:>10.4f}{s['E_D_0.80']:>12.2f}")
    lines += ["", "  The primary score is the one used everywhere else in "
              "this report; it was",
              "  fixed before the test set was read and is not revised here. "
              "The alternatives",
              "  are reported as specified, whether or not they agree with "
              "it.", ""]

    # ---- curves and proportions for the figure ---------------------
    risk_c = np.array([patient_risk_at(loss, conf, q) for q in COVERAGES])
    band_r = np.array([clustered_ci(
        lambda i, q=q: patient_risk_at(loss[i], conf[i], q), n,
        max(200, args.replicates // 4), seed=5)[1:] for q in COVERAGES])

    # Clinically legible landmarks for panel A: what referring 0%, 5% and
    # 10% of the least confident predictions actually buys, as both a
    # selective risk and a residual error count.
    marks = [(q, patient_risk_at(loss, conf, q),
              patient_errors_at(loss, conf, q))
             for q in (1.00, 0.95, 0.90)]

    # Panel B is patient-level: the clinically interpretable unit. Each
    # CODE-test ECG is a distinct patient, and an ECG counts as disputed if
    # the readers disagreed on any of its six diagnoses, as erroneous if the
    # model was wrong on any.
    #
    # The label-level version of the same comparison gives a much larger
    # ratio (25x), but part of that is arithmetic: six label opportunities
    # per ECG against a consensus baseline of 1.08%. Patient level is the
    # more conservative framing and is what the figure shows; the
    # diagnosis-level result stays in the text.
    pe = err_any = (loss.astype(bool)).sum(axis=1) > 0
    pd_ = dis_any = D.astype(bool).sum(axis=1) > 0

    def _safe(v):
        return float(v) if np.isfinite(v) else float('nan')

    def _rate_p(i, disputed):
        s = pd_[i] if disputed else ~pd_[i]
        return _safe(pe[i][s].mean()) if s.any() else float('nan')

    def _rr_p(i):
        a, b = _rate_p(i, True), _rate_p(i, False)
        return a / b if np.isfinite(a) and np.isfinite(b) and b > 0 \
            else float('nan')

    def _prev_err(i):
        return _safe(pd_[i][pe[i]].mean()) if pe[i].any() else float('nan')

    def _enrich_p(i):
        a, b = _prev_err(i), _safe(pd_[i].mean())
        return a / b if np.isfinite(a) and b > 0 else float('nan')

    panelc = {
        "prev_all": clustered_ci(lambda i: _safe(pd_[i].mean()), n,
                                 args.replicates, seed=8),
        "prev_err": clustered_ci(_prev_err, n, args.replicates, seed=9),
        "enrichment": clustered_ci(_enrich_p, n, args.replicates, seed=13),
        "rate_con": clustered_ci(lambda i: _rate_p(i, False), n,
                                 args.replicates, seed=10),
        "rate_dis": clustered_ci(lambda i: _rate_p(i, True), n,
                                 args.replicates, seed=11),
        "ratio": clustered_ci(_rr_p, n, args.replicates, seed=12),
        "n_dis": int(pd_.sum()), "n_pairs": n,
        "n_err": int(pe.sum()),
        "n_errD": int((pe & pd_).sum()),
        "n_errC": int((pe & ~pd_).sum()), "n_C": int((~pd_).sum()),
    }
    figure(COVERAGES, risk_c, band_r, marks, panelc,
           args.out / "figures")

    text = "\n".join(lines)
    (args.out / "tables" / "results.txt").write_text(text + "\n",
                                                     encoding="utf-8")
    (args.out / "tables" / "results.json").write_text(json.dumps({
        "n_ecg": int(n), "n_pairs": int(pairs), "n_disputed": int(n_dis),
        "disputed_per_class": dict(zip(L, per_class_dis.tolist())),
        "macro_auroc": macro_auroc(y, d["pbar"]),
        "macro_auprc": macro_auprc(y, d["pbar"]),
        "delta_U": du, "macro_aurc": au,
        "risk": {f"{q:.2f}": riskq[q] for q in REPORT_AT},
        "enrichment": {f"{q:.2f}": enr[q] for q in REPORT_AT},
        "counts": {f"{q:.2f}": counts[q] for q in REPORT_AT},
        "score_sensitivity": sens,
        # Patient-level quantities behind Figure 1B, so every number in the
        # figure and in the Results paragraph has a recorded source.
        "patient_level": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in panelc.items()},
        "disputed_prevalence_full": disputed_prevalence(D),
        "replicates": args.replicates,
    }, indent=2), encoding="utf-8")
    print(text)

    # ---- the four sentences ----------------------------------------------
    # The lowest reported coverage at which enrichment is actually defined.
    # With only a few dozen disagreements in 827 ECGs, aggressive referral can
    # retain no disputed label at all; printing "nan-fold" into a manuscript
    # sentence would be worse than saying the quantity is undefined.
    q_star = next((q for q in sorted(REPORT_AT)
                   if np.isfinite(enr[q][0])
                   and counts[q]["errors"] >= MIN_EVENTS), None)
    if q_star is None:
        n1 = counts[1.00]["errors"]
        q_lo = min(REPORT_AT)
        last = (f"Referral removed almost all error before enrichment could "
                f"be assessed: {n1} residual errors at full coverage fell to "
                f"{counts[q_lo]['errors']} at {q_lo:.0%} coverage, so E_D at "
                f"reduced coverage rests on too few events to interpret "
                f"(see results.txt).")
    elif q_star >= 1.0:
        # Only full coverage carries enough events. That is not a post-referral
        # result, so it must not be worded as one: at q = 1 nothing has been
        # referred, and "despite referral" would be simply false.
        q_lo = min(REPORT_AT)
        last = (f"Before referral, disputed labels accounted for "
                f"{disputed_error_share(loss, conf, D, 1.0):.1%} of all errors "
                f"against {retained_disputed_prevalence(conf, D, 1.0):.1%} of "
                f"predictions, an enrichment of {fmt(*enr[1.00], nd=1)}-fold; "
                f"referral then removed nearly all error "
                f"({counts[1.00]['errors']} residual errors at full coverage "
                f"to {counts[q_lo]['errors']} at {q_lo:.0%}), leaving too few "
                f"events to test whether the enrichment persists.")
    else:
        last = (f"Despite this, disputed labels accounted for "
                f"{disputed_error_share(loss, conf, D, q_star):.1%} of residual "
                f"errors at {q_star:.0%} coverage against "
                f"{retained_disputed_prevalence(conf, D, q_star):.1%} of "
                f"retained predictions, an enrichment of "
                f"{fmt(*enr[q_star], nd=1)}-fold.")
    para = (
        f"Reader disagreement occurred in {n_dis} of {pairs} diagnosis-record "
        f"pairs ({n_dis / pairs:.2%}) and was concentrated in "
        f"{L[int(np.argmax(per_class_dis))]} "
        f"({int(per_class_dis.max())} disagreements). Disputed labels showed "
        f"greater decision uncertainty than consensus labels "
        f"(delta U = {fmt(*du)}). Uncertainty-based referral reduced macro "
        f"classification risk from {riskq[1.00][0]:.3f} at full coverage to "
        f"{riskq[0.50][0]:.3f} at 50% coverage. " + last)
    (args.out / "tables" / "paragraph.txt").write_text(para + "\n",
                                                       encoding="utf-8")

    # ---- figure caption --------------------------------------------
    caption = (
        f"Figure 1. Uncertainty-based referral and cardiologist "
        f"disagreement among model errors, CODE-test ({n} ECGs x "
        f"{len(L)} diagnoses = {pairs} diagnosis-record pairs). "
        f"(A) Macro selective risk against coverage over "
        f"q in [{XVIEW[0]:.2f}, {XVIEW[1]:.2f}], with 95% ECG-level "
        f"bootstrap interval; coverage denotes the proportion of "
        f"predictions retained, so 1 - q is the referral fraction and "
        f"q = {XVIEW[0]:.2f} corresponds to "
        f"{(1 - XVIEW[0]) * 100:.0f}% referred. Risk-coverage curves were "
        f"computed over the full range q in (0, 1]. "
        f"(B) Patient-level analysis of the {panelc['n_pairs']:,} "
        f"CODE-test ECGs, each from a distinct patient; an ECG is "
        f"classified as disputed if the two cardiologists disagreed on any "
        f"of its {len(L)} diagnoses and as erroneous if the model was "
        f"incorrect on any. Left, reader disagreement among all ECGs "
        f"({panelc['n_dis']:,} of {panelc['n_pairs']:,}) and among ECGs "
        f"containing any model error ({panelc['n_errD']:,} of "
        f"{panelc['n_err']:,}), a {panelc['enrichment'][0]:.1f}-fold "
        f"enrichment. Right, model error among ECGs with complete reader "
        f"consensus ({panelc['n_errC']:,} of {panelc['n_C']:,}) and with "
        f"any disagreement ({panelc['n_errD']:,} of {panelc['n_dis']:,}), "
        f"risk ratio {panelc['ratio'][0]:.1f}. Points are percentages and "
        f"error bars are 95% ECG-level bootstrap intervals. The "
        f"diagnosis-level analysis is reported in the text.")
    (args.out / "tables" / "caption.txt").write_text(
        caption + chr(10), encoding="utf-8")
    print("\n" + "-" * 74 + "\nRESULTS PARAGRAPH\n" + "-" * 74)
    print(para)


def _inview(cov):
    return (cov >= XVIEW[0] - 1e-9) & (cov <= XVIEW[1] + 1e-9)


def _ytop(vals, pad=1.08):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.max() * pad) if v.size and v.max() > 0 else 1.0


def figure(cov, risk, band_r, marks, panelc, out_dir: Path) -> None:
    """Two panels, in the order the argument runs.

    A  confidence identifies the error-prone tail
    B  those errors land disproportionately on disputed labels

    The uncertainty-by-agreement comparison that used to open the figure
    replicates a known effect and reads perfectly well as one sentence; it
    stays in results.txt section A and in the results paragraph rather
    than spending a panel. Per-diagnosis disagreement counts likewise
    survive in the descriptive table.
    """
    # A carries a curve whose shape is the message and needs horizontal
    # room; B is a two-point comparison. Equal widths would spend the same
    # space on both.
    # Equal widths: B holds four points in two pairs plus three lines of
    # annotation under each, so it needs as much room as the curve.
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 4.6), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=10, length=3)

    # A -- one series, so no legend box; the title names it.
    ax = axes[0]
    ax.fill_between(cov, band_r[:, 0] * 100, band_r[:, 1] * 100,
                    color=C_AGREE, alpha=0.18, lw=0)
    ax.plot(cov, risk * 100, color=C_AGREE, lw=2.0,
            solid_capstyle="round")
    ax.set_xlabel("ECG coverage", fontsize=11.0, color=INK)
    ax.set_ylabel(r"ECGs with $\geq$1 error (%)", fontsize=11.0, color=INK)
    ax.set_title("A   Uncertainty-based referral" + chr(10) +
                 "of whole ECGs",
                 fontsize=10.5, color=INK, loc="left", pad=14)
    ax.set_xlim(*XVIEW)
    ax.set_xticks(XTICKS)
    # Scale y to the visible window only: the full grid runs to q = 0.01,
    # and autoscaling over off-screen points would flatten the display.
    ax.set_ylim(0, _ytop(band_r[:, 1][_inview(cov)] * 100))
    # A few operating points, so the curve reads as a decision rather
    # than a shape: how many errors survive at 0%, 5% and 10% referral.
    for q, r, ne in marks:
        ax.plot(q, r * 100, "o", color=C_AGREE, ms=6.5, mec=SURFACE,
                mew=1.5, zorder=4)
        right = q >= 0.95
        ax.annotate(f"{r * 100:.1f}%" + chr(10) + f"{ne} ECGs",
                    xy=(q, r * 100),
                    xytext=(-9, 5) if right else (9, -4),
                    textcoords="offset points",
                    ha="right" if right else "left",
                    va="bottom" if right else "top",
                    fontsize=9.0, color=INK2, linespacing=1.3)
    sec = ax.secondary_xaxis("top", functions=(
        lambda x: (1.0 - x) * 100.0, lambda x: 1.0 - x / 100.0))
    sec.set_xlabel("ECGs referred (%)", fontsize=10.0, color=INK2,
                   labelpad=3)
    sec.tick_params(colors=INK2, labelsize=10, length=3)
    sec.spines["top"].set_color(GRID)

    # B -- two paired comparisons, separated by a gap.
    #
    # Left pair answers 'how much of X was disputed', right pair answers
    # 'how often was the model wrong in group X'. Running all four points
    # as one sequence would invite the reader to compare across the two,
    # which is meaningless; the gap and the sub-headings keep them apart.
    ax = axes[1]
    nl = chr(10)
    XL, XR = (0.6, 2.3), (4.1, 5.8)
    groups = (
        (XL, "Disagreement" + nl + "prevalence",
         ((panelc["prev_all"], C_AGREE, "All" + nl + "ECGs",
           f"{panelc['n_dis']:,} of {panelc['n_pairs']:,}"),
          (panelc["prev_err"], C_DISPUTE, "With" + nl + "any error",
           f"{panelc['n_errD']:,} of {panelc['n_err']:,}")),
         (f"PR = {panelc['enrichment'][0]:.1f}",
          f"({panelc['enrichment'][1]:.1f}-{panelc['enrichment'][2]:.1f})")),
        (XR, "Patient-level" + nl + "model error",
         ((panelc["rate_con"], C_AGREE, "Complete" + nl + "consensus",
           f"{panelc['n_errC']:,} of {panelc['n_C']:,}"),
          (panelc["rate_dis"], C_DISPUTE, "Any" + nl + "disagreement",
           f"{panelc['n_errD']:,} of {panelc['n_dis']:,}")),
         (f"RR = {panelc['ratio'][0]:.1f}",
          f"({panelc['ratio'][1]:.1f}-{panelc['ratio'][2]:.1f})")),
    )
    top = max(g[2][k][0][2] for g in groups for k in (0, 1)) * 100 * 1.34
    ticks, labels = [], []
    for (xs, heading, pts, ratio) in groups:
        for x, ((pt, lo, hi), col, name, cnt) in zip(xs, pts):
            ax.plot([x, x], [lo * 100, hi * 100], color=col, lw=2.2,
                    solid_capstyle="round", zorder=2)
            ax.plot(x, pt * 100, "o", color=col, ms=8.5, mec=SURFACE,
                    mew=1.6, zorder=3)
            ax.annotate(f"{pt * 100:.2f}%" if pt < 0.02
                        else f"{pt * 100:.1f}%",
                        xy=(x, pt * 100), xytext=(9, 0),
                        textcoords="offset points", va="center",
                        fontsize=10.0, color=col, fontweight="bold")
            ax.annotate(cnt, xy=(x, -0.175), xycoords=("data",
                        "axes fraction"), ha="center", va="top",
                        fontsize=8.0, color=INK2)
            ticks.append(x)
            labels.append(name)
        mid = sum(xs) / 2
        ax.annotate(heading, xy=(mid, 0.965), xycoords=("data",
                    "axes fraction"), ha="center", va="top",
                    fontsize=10.0, color=INK)
        y_mid = (pts[0][0][0] + pts[1][0][0]) / 2 * 100
        box = dict(facecolor=SURFACE, edgecolor="none",
                   boxstyle="round,pad=0.18")
        ax.annotate(ratio[0], xy=(mid, y_mid), xytext=(0, 1),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=11.0, color=INK, fontweight="bold", zorder=6,
                    bbox=box)
        ax.annotate(ratio[1], xy=(mid, y_mid), xytext=(0, -2),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=8.0, color=INK2, zorder=6, bbox=box)
        ax.annotate("", xy=(xs[1], pts[1][0][0] * 100),
                    xytext=(xs[0], pts[0][0][0] * 100),
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=1.1,
                                    linestyle=(0, (3, 3))))
    ax.axvline(3.2, color=GRID, lw=1.0, zorder=1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8.5, color=INK2)
    ax.set_xlim(-0.25, 6.65)
    ax.set_ylim(0, top)
    ax.set_ylabel("percentage (%)", fontsize=11.0, color=INK)
    ax.set_title("B   Reader disagreement and" + nl +
                 "patient-level model error",
                 fontsize=10.5, color=INK, loc="left", pad=40)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"figure1.{ext}", dpi=300,
                    bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
