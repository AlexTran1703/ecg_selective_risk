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
                             patient_group_size, patient_risk_in_group,
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
C_RR = "#1baf7a"        # validated slot 3, for the ratio strip
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SURFACE = "#fcfcfb"

# Ascending: risk-coverage reads left to right, full coverage on the right.
# q = 0 retains nothing and is undefined, so evaluation starts at 0.01
# while the axis still spans the full [0, 1].
COVERAGES = np.round(np.arange(0.01, 1.0001, 0.01), 2)
XVIEW = (0.80, 1.00)          # displayed span of both panels
XTICKS = (0.80, 0.85, 0.90, 0.95, 1.00)
MARKS = (1.00, 0.95, 0.90, 0.80)   # annotated operating points
# RR(c) spans the same window as the risk curves. Below 0.90 coverage
# fewer than ~15 disagreement ECGs remain and the lower confidence
# bound reaches zero, so the band is drawn across the whole range
# rather than the point estimate alone -- the widening is the warning.
RRVIEW = XVIEW
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
             for q in MARKS]

    # Panel B: the same referral policy as A, split by whether the two
    # cardiologists disagreed anywhere on the ECG.
    #
    # Ranking is global. Each ECG keeps the confidence it had in A, the
    # top q fraction is retained, and only then are retained ECGs split
    # into strata. Ranking within strata would make coverage mean a
    # different thing in each curve.
    dis_ecg = D.astype(bool).any(axis=1)
    err_ecg = loss.astype(bool).any(axis=1)

    def _rr(i):
        a = patient_risk_in_group(loss[i], conf[i], dis_ecg[i], 1.0, True)
        b = patient_risk_in_group(loss[i], conf[i], dis_ecg[i], 1.0, False)
        return a / b if np.isfinite(a) and np.isfinite(b) and b > 0 \
            else float('nan')

    reps_band = max(200, args.replicates // 4)
    panelc = {}
    for nm, ing, sd in (("dis", True, 20), ("con", False, 21)):
        panelc[nm] = np.array([
            patient_risk_in_group(loss, conf, dis_ecg, q, ing)
            for q in COVERAGES])
        panelc[nm + "_band"] = np.array([clustered_ci(
            lambda i, q=q, g=ing: patient_risk_in_group(
                loss[i], conf[i], dis_ecg[i], q, g),
            n, reps_band, seed=sd)[1:] for q in COVERAGES])
        panelc[nm + "_n"] = np.array([
            patient_group_size(conf, dis_ecg, q, ing) for q in COVERAGES])
    panelc["rr"] = clustered_ci(_rr, n, args.replicates, seed=12)

    # RR at every coverage. The ratio is formed inside each replicate;
    # dividing the two risk curves' confidence limits would be wrong.
    def _rr_at(q):
        def f(i):
            a = patient_risk_in_group(loss[i], conf[i], dis_ecg[i],
                                      q, True)
            b = patient_risk_in_group(loss[i], conf[i], dis_ecg[i],
                                      q, False)
            return (a / b if np.isfinite(a) and np.isfinite(b) and b > 0
                    else float('nan'))
        return clustered_ci(f, n, reps_band, seed=30)

    rr_c = np.array([_rr_at(q) if RRVIEW[0] - 1e-9 <= q <= RRVIEW[1]
                     else (np.nan, np.nan, np.nan) for q in COVERAGES])
    panelc["rr_curve"] = rr_c[:, 0]
    panelc["rr_band"] = rr_c[:, 1:]
    panelc["n_all"] = n
    panelc["n_dis"] = int(dis_ecg.sum())
    panelc["n_con"] = int((~dis_ecg).sum())
    panelc["n_err"] = int(err_ecg.sum())
    panelc["n_errD"] = int((err_ecg & dis_ecg).sum())
    panelc["n_errC"] = int((err_ecg & ~dis_ecg).sum())
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
        # The stratified curves as well as the scalars, so every point in
        # panel B is recoverable without rerunning the figure.
        "coverage_grid": COVERAGES,
        "patient_level": panelc,
        "disputed_prevalence_full": disputed_prevalence(D),
        "replicates": args.replicates,
    }, indent=2, default=lambda o: (o.tolist() if hasattr(o, "tolist")
                                    else list(o))), encoding="utf-8")
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
        f"(B) The same referral policy, with retained ECGs split by "
        f"whether the 2 cardiologists disagreed on any of the "
        f"{len(L)} diagnoses. Both curves use the single global "
        f"confidence ranking of (A); strata are formed only after "
        f"retention, so coverage means the same thing in both panels. "
        f"At full coverage, any model error occurred in "
        f"{panelc['n_errD']} of {panelc['n_dis']} ECGs with any "
        f"disagreement ({panelc['dis'][-1]:.1%}) versus "
        f"{panelc['n_errC']} of {panelc['n_con']} with complete "
        f"consensus ({panelc['con'][-1]:.1%}), a risk ratio of "
        f"{panelc['rr'][0]:.2f} (95% CI, {panelc['rr'][1]:.2f} to "
        f"{panelc['rr'][2]:.2f}). Shading in both panels represents 95% "
        f"ECG-level bootstrap intervals. The lower strip of (B) shows the "
        f"risk ratio between the 2 strata at each coverage, with each "
        f"ratio formed within the bootstrap replicate rather than from "
        f"the limits of the 2 risk curves, and a reference line at 1. It "
        f"Below 90% coverage fewer than 15 disagreement ECGs remain "
        f"and the lower confidence bound reaches zero, so the ratio "
        f"there should be read as unstable rather than rising. "
        f"Disagreement ECGs retained fall from {panelc['n_dis']} at full "
        f"coverage to "
        f"{int(panelc['dis_n'][int(round(XVIEW[0] * 100)) - 1])} at "
        f"{XVIEW[0]:.0%} coverage. RR = risk ratio.")
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


def _annotate_marks(ax, cov, curve, qs, col, above=True, fs=8.5):
    """Mark the reported operating points and print their values."""
    for q in qs:
        i = int(np.argmin(np.abs(cov - q)))
        v = curve[i]
        if not np.isfinite(v):
            continue
        ax.plot(cov[i], v * 100, "o", color=col, ms=5.5, mec=SURFACE,
                mew=1.2, zorder=4)
        edge = "left" if q <= XVIEW[0] + 1e-9 else (
            "right" if q >= XVIEW[1] - 1e-9 else "center")
        dx = {"left": 5, "right": -5, "center": 0}[edge]
        ax.annotate(f"{v * 100:.1f}", xy=(cov[i], v * 100),
                    xytext=(dx, 7 if above else -7),
                    textcoords="offset points", ha=edge,
                    va="bottom" if above else "top", fontsize=fs,
                    color=col, zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none",
                              alpha=0.8, boxstyle="round,pad=0.12"))


def figure(cov, risk, band_r, marks, panelc, out_dir: Path) -> None:
    """Two panels on one coverage axis: overall referral, then stratified."""
    nl = chr(10)
    fig = plt.figure(figsize=(7.9, 4.6), facecolor=SURFACE)
    # Panel B is two stacked axes sharing one coverage axis: the risk
    # curves, and a shallow RR strip. Formally still one part.
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1.0],
                          hspace=0.12, wspace=0.28,
                          left=0.085, right=0.985, top=0.86,
                          bottom=0.115)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axR = fig.add_subplot(gs[1, 1], sharex=axB)
    axes = [axA, axB]
    for ax in (axA, axB, axR):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9.5, length=3)
        ax.set_xlim(*XVIEW)
        ax.set_xticks(XTICKS)
    for ax in axes:
        ax.set_ylabel(r"ECGs with $\geq$1 error (%)", fontsize=10.5,
                      color=INK)
    axA.set_xlabel("ECG coverage", fontsize=10.5, color=INK)
    axR.set_xlabel("ECG coverage", fontsize=10.5, color=INK)
    plt.setp(axB.get_xticklabels(), visible=False)

    # ---- A: overall ------------------------------------------------
    ax = axes[0]
    ax.fill_between(cov, band_r[:, 0] * 100, band_r[:, 1] * 100,
                    color=C_AGREE, alpha=0.18, lw=0)
    ax.plot(cov, risk * 100, color=C_AGREE, lw=2.0,
            solid_capstyle="round", zorder=3)
    ax.set_ylim(0, _ytop(band_r[:, 1][_inview(cov)] * 100))
    for q, r, ne in marks:
        ax.plot(q, r * 100, "o", color=C_AGREE, ms=6, mec=SURFACE, mew=1.3,
                zorder=4)
        edge = "left" if q <= XVIEW[0] + 1e-9 else (
            "right" if q >= XVIEW[1] - 1e-9 else "center")
        dx = {"left": 5, "right": -5, "center": 0}[edge]
        ax.annotate(f"{r * 100:.1f}%" + nl + f"{ne} ECGs",
                    xy=(q, r * 100),
                    xytext=(dx, 9), textcoords="offset points",
                    ha=edge, va="bottom", fontsize=8.5, color=INK2,
                    linespacing=1.3, zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none",
                              alpha=0.8, boxstyle="round,pad=0.12"))
    ax.set_title("A   Uncertainty-based referral" + nl + "of whole ECGs",
                 fontsize=10.5, color=INK, loc="left", pad=12)

    # ---- B: stratified ----------------------------------------------
    ax = axes[1]
    vis = _inview(cov)
    top = 0.0
    for nm, col, lab, up in (
            ("dis", C_DISPUTE, "reader disagreement", True),
            ("con", C_AGREE, "reader consensus", True)):
        r, band = panelc[nm], panelc[nm + "_band"]
        ok = np.isfinite(r) & vis
        fb = np.isfinite(band[:, 1]) & ok
        ax.fill_between(cov[fb], band[fb, 0] * 100, band[fb, 1] * 100,
                        color=col, alpha=0.16, lw=0)
        ax.plot(cov[ok], r[ok] * 100, color=col, lw=2.0, label=lab,
                solid_capstyle="round", zorder=3)
        _annotate_marks(ax, cov, r, MARKS, col, above=up)
        top = max(top, float(np.nanmax(band[fb, 1])) * 100 if fb.any() else 0)
    ax.set_ylim(0, top * 1.16)
    # RR rides in the legend title: anywhere inside the axes it would
    # land on a curve or inside the disagreement band.
    ax.legend(loc="upper left", frameon=True, facecolor=SURFACE,
              edgecolor=GRID, framealpha=0.92, fontsize=8.5,
              labelcolor=INK2, handlelength=1.6, borderpad=0.5)
    ax.set_title("B   Error by reader agreement" + nl + "across referral",
                 fontsize=10.5, color=INK, loc="left", pad=12)

    # ---- B, lower strip: RR across coverage -------------------------
    rrc, rrb = panelc["rr_curve"], panelc["rr_band"]
    ok = np.isfinite(rrc) & _inview(cov)
    axR.axhline(1.0, color=INK2, lw=1.0, ls=(0, (3, 3)), zorder=2)
    fb = ok & np.isfinite(rrb[:, 1])
    axR.fill_between(cov[fb], rrb[fb, 0], rrb[fb, 1], color=C_RR,
                     alpha=0.16, lw=0)
    axR.plot(cov[ok], rrc[ok], color=C_RR, lw=1.8,
             solid_capstyle="round", zorder=3)
    axR.set_ylabel("RR", fontsize=10.5, color=INK)
    # Scale to the point estimates: the interval at low coverage runs to
    # about 75 and would flatten the region that carries the result.
    ytop = float(np.nanmax(rrc[ok])) * 1.55 if ok.any() else 30.0
    axR.set_ylim(0, ytop)
    for q in MARKS:
        i = int(np.argmin(np.abs(cov - q)))
        if not np.isfinite(rrc[i]):
            continue
        axR.plot(cov[i], rrc[i], "o", color=C_RR, ms=5, mec=SURFACE,
                 mew=1.1, zorder=4)
        edge = "left" if q <= XVIEW[0] + 1e-9 else (
            "right" if q >= XVIEW[1] - 1e-9 else "center")
        dx = {"left": 4, "right": -4, "center": 0}[edge]
        # Value on its own line with the interval beneath it, so a
        # reader tracking the curve reads the estimate first. All
        # labels sit above the curve -- below the last one would run
        # into the axis -- and the 0.95 label is lifted clear of the
        # 1.00 label, which is close by on a flat stretch.
        dy = 20 if q >= XVIEW[1] - 1e-9 else 7
        axR.annotate(f"{rrc[i]:.1f}" + nl +
                     f"({rrb[i, 0]:.1f}-{rrb[i, 1]:.1f})",
                     xy=(cov[i], rrc[i]), xytext=(dx, dy),
                     textcoords="offset points", ha=edge,
                     va="bottom",
                     fontsize=7.2, color=C_RR, zorder=5,
                     linespacing=1.25,
                     bbox=dict(facecolor=SURFACE, edgecolor="none",
                               alpha=0.85, boxstyle="round,pad=0.12"))
    axR.annotate("RR = 1", xy=(XVIEW[0] + 0.004, 1.0), ha="left",
                 va="bottom", fontsize=7.5, color=INK2)


    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"figure1.{ext}", dpi=300,
                    bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
