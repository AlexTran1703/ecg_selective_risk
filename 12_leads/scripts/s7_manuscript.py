r"""Driver for the 100 Hz manuscript set: tables, then the five figures.

    .\.venv\Scripts\python.exe 12_leads\scripts\s7_manuscript.py

Figures live in s7_assets.py; the tables and the ordering live here, so
the two can be edited without stepping on each other.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

_spec = importlib.util.spec_from_file_location(
    "s7a", Path(__file__).with_name("s7_assets.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

from ecgmcu.data import SOURCES, FS_OUT, SECONDS, N_LEADS      # noqa: E402
from ecgmcu.models import build, n_params                      # noqa: E402
from ecgmcu.specs import SPECS                                 # noqa: E402

ORDER, NICE, TAB, STAB = A.ORDER, A.NICE, A.TAB, A.STAB
PAPER_FS = 100          # the reported rate; 250 Hz is archived only
SRAM_KB, FLASH_KB = A.SRAM_KB, A.FLASH_KB
RESERVE_FLASH, RESERVE_SRAM = A.RESERVE_FLASH, A.RESERVE_SRAM


def table1():
    rec = pd.read_csv(HERE.parent / "data" / "meta" / "records.csv")
    idx = pd.read_csv(HERE / "data" / "index.csv")
    lab = np.load(HERE / "data" / "labels.npy")
    npos = lab.sum(1)
    rows = []
    for s in SOURCES:
        sel = (idx.source == s).to_numpy()
        rows.append({
            "source": A.sname(s),
            "records, original": int((rec.source == s).sum()),
            "retained": int(sel.sum()),
            "labels per record": round(float(npos[sel].mean()), 2),
            "original Fs (Hz)": 500,
            "duration (s)": SECONDS,
            "model input Fs (Hz)": FS_OUT,
            "model input": f"{N_LEADS} leads x {FS_OUT * SECONDS} samples"})
    A.emit(pd.DataFrame(rows).set_index("source"), "table1_datasets",
           "Table I. The four clinical sources. A record is retained when "
           "at least one of its labels falls inside the harmonised "
           "13-class space. Signal-quality characteristics are reported in "
           "Figure 2 and are measured on the native 500 Hz signal, not on "
           "the decimated model input.", TAB)


def table2(dep):
    import torch.nn as nn
    rows = []
    for m in ORDER:
        mod = build(m)
        rows.append({
            "model": NICE[m], "width": "1.0",
            "parameters": n_params(mod),
            "learnable conv layers": sum(1 for x in mod.modules()
                                         if isinstance(x, nn.Conv1d)),
            "blocks": SPECS[m]["blocks"],
            "core operation": SPECS[m]["operator"],
            "MACs (M)": round((dep.get(m, {}).get("macc") or 0) / 1e6, 2)})
    A.emit(pd.DataFrame(rows).set_index("model"), "table2_architectures",
           f"Table II. The six encoder families at their reference width. "
           f"All share the same input ({SECONDS} s, {N_LEADS} leads, "
           f"{FS_OUT} Hz), the same global-average-pooled linear head and "
           f"the same training recipe, so differences are attributable to "
           f"the body. MACs are measured by ST Edge AI on the quantised "
           f"graph, and count multiply-accumulate operations per "
           f"inference. Width-scaled variants are supplementary.", TAB)


def table3(runs):
    """Bootstrap the cross-source mean for figure 2.

    This used to be Table III. Figure 2 now carries both halves of it --
    the per-source values in the heatmap, the mean and interval in the dot
    plot -- so printing the same 24 numbers underneath would be a second
    container for evidence the reader just saw.
    """
    rng = np.random.default_rng(0)
    reps = 300
    rows, boot = {}, {}
    for m in ORDER:
        srcs = [s for s in SOURCES if (m, s) in runs]
        if not srcs:
            continue
        cells = {}
        per_src = []
        for s in srcs:
            r = runs[(m, s)]
            a = A.macro_ap(r["y"], r["p"])
            cells[A.sname(s)] = f"{a:.3f}"
            per_src.append(a)
        # Record-level bootstrap on the mean, resampling within each source.
        draws = np.empty(reps)
        for b in range(reps):
            v = []
            for s in srcs:
                r = runs[(m, s)]
                n = r["y"].shape[0]
                i_ = rng.integers(0, n, n)
                v.append(A.macro_ap(r["y"][i_], r["p"][i_]))
            draws[b] = np.mean(v)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        cells["mean (95% CI)"] = f"{np.mean(per_src):.3f} ({lo:.3f}-{hi:.3f})"
        rows[NICE[m]] = cells
        boot[m] = (float(np.mean(per_src)), float(lo), float(hi))
        print(f"    bootstrapped {m}", flush=True)
    t = pd.DataFrame(rows).T
    t.index.name = "model"
    return boot


def table4(qr):
    """RQ3: per-diagnosis prevalence-matched contrasts.

    The support rule means different indicators rest on different numbers
    of held-out sources, so that count travels with the table rather than
    being buried in the methods.
    """
    if qr is None or qr.empty:
        return
    qm = qr[qr.model != "__pooled__"]          # pooled is a summary, not a model
    qp = qr[qr.model == "__pooled__"]
    d = qm.pivot(index="model", columns="label", values="delta")
    lo = qm.pivot(index="model", columns="label", values="lo")
    hi = qm.pivot(index="model", columns="label", values="hi")
    cols = [c for c in A.FACTOR_ORDER if c in d.columns]
    rows = {}
    for m in [x for x in ORDER if x in d.index]:
        cells = {c: f"{d.loc[m, c]:+.3f} ({lo.loc[m, c]:+.3f}, "
                    f"{hi.loc[m, c]:+.3f})" for c in cols}
        rows[NICE[m]] = cells
    # Pooled row and the support each column actually had.
    if not qp.empty:
        pv = qp.set_index("label")
        rows["pooled"] = {
            **{c: f"{pv.loc[c, 'delta']:+.3f} ({pv.loc[c, 'lo']:+.3f}, "
                  f"{pv.loc[c, 'hi']:+.3f})" for c in cols},
        }
        rows["held-out sources"] = {
            c: str(int(pv.loc[c, "sources_used"])) for c in cols}
    t = pd.DataFrame(rows).T[cols]
    t.index.name = "model"
    A.emit(t, "table3_quality_association",
           "Table III. Prevalence-matched association between signal-quality "
           "strata and diagnostic discrimination. Values are dAUPRC = "
           "higher-impairment minus lower-impairment stratum, with "
           "record-level bootstrap 95% intervals in parentheses (B = 1000, "
           "seed 0). Impairment direction is fixed in advance from what "
           "each metric measures, never from the outcome. Strata are formed "
           "within each held-out source; for the three continuous "
           "indicators they are the within-source quartiles (<=Q25 against "
           ">=Q75), for the three structural indicators absent against "
           "present. For every diagnosis independently the two strata are "
           "sampled to a common n+ = min(N+_H, N+_L) and n- = min(N-_H, "
           "N-_L), so the quantity reported is the mean of "
           "prevalence-matched class-specific AP contrasts rather than "
           "macro AUPRC on one common matched set. A diagnosis enters only "
           "where both strata supply 20 positives and 20 negatives, which "
           "is why the last row differs by column: flat/clipped leads are "
           "too rare outside Ningbo to support the contrast there. "
           "Negative indicates lower discrimination in the more impaired "
           "stratum. These are associations, not causal effects.", TAB)


def supplementary(runs, qr):
    cls = json.loads((HERE.parent / "data" / "meta" / "classes.json")
                     .read_text())["abbreviations"]
    rows = []
    for m in ORDER:
        rec = {"model": NICE[m]}
        for c, nm in enumerate(cls):
            v = []
            for s in SOURCES:
                if (m, s) not in runs:
                    continue
                y, p = runs[(m, s)]["y"], runs[(m, s)]["p"]
                if 0 < y[:, c].sum() < y.shape[0]:
                    v.append(A.ap_fast(y[:, c], p[:, c]))
            rec[nm] = round(float(np.mean(v)), 3) if v else np.nan
        rows.append(rec)
    A.emit(pd.DataFrame(rows).set_index("model"), "tableS1_per_diagnosis",
           "Supplementary Table S1. AUPRC for each harmonised diagnosis, "
           "averaged over the four held-out sources. A macro average can "
           "hide a collapsed rare class; this is where to check.", STAB)

    rows = {}
    for m in ORDER:
        cells, v = {}, []
        for s in SOURCES:
            if (m, s) not in runs:
                continue
            a = A.macro_auroc(runs[(m, s)]["y"], runs[(m, s)]["p"])
            cells[A.sname(s)] = round(a, 3)
            v.append(a)
        if cells:
            cells["mean"] = round(float(np.mean(v)), 3)
            rows[NICE[m]] = cells
    t = pd.DataFrame(rows).T
    t.index.name = "model"
    A.emit(t, "tableS6_auroc",
           "Supplementary Table S6. Macro AUROC by held-out source. AUPRC "
           "in Table III is the primary endpoint; AUROC is reported "
           "because it is the more familiar number, not because it is the "
           "more informative one under this class imbalance.", STAB)

    if qr is not None and not qr.empty:
        # Schema follows s6_quality_robustness.py: prevalence is now
        # matched per diagnosis, so there is no residual label-count
        # gap to report; what matters instead is how much support
        # each contrast had.
        keep = ["model", "label", "delta", "lo", "hi", "baseline",
                "relative", "n_low", "n_high", "sources_used",
                "contrasts"]
        t = qr[[c for c in keep if c in qr.columns]].copy()
        t = t[t.model != "__pooled__"]
        t["model"] = [NICE.get(x, x) for x in t["model"]]
        A.emit(t.set_index(["model", "label"]).round(4),
               "tableS4_quality_robustness_full",
               "Supplementary Table S4. Full quality-robustness results: "
               "the clean-stratum baseline AUPRC, the relative change, "
               "stratum sizes, and the residual difference in labels per "
               "record that remains after matching. The relative column "
               "matters because an encoder with a lower baseline has less "
               "accuracy available to lose.", STAB)


def supplementary_rq3(qr):
    """Support behind each RQ3 column, and the support-rule sensitivity.

    How many source x diagnosis contrasts actually contribute is the
    difference between a four-source average and a single-source estimate,
    and it is not visible from the effect size alone.
    """
    if qr is None or qr.empty:
        return
    qp = qr[qr.model == "__pooled__"].set_index("label")
    cols = [c for c in A.FACTOR_ORDER if c in qp.index]
    rows = []
    for c in cols:
        rows.append({
            "indicator": c,
            "dAUPRC": round(float(qp.loc[c, "delta"]), 4),
            "lo": round(float(qp.loc[c, "lo"]), 4),
            "hi": round(float(qp.loc[c, "hi"]), 4),
            "held-out sources": int(qp.loc[c, "sources_used"]),
            "source x diagnosis contrasts": int(qp.loc[c, "contrasts"]),
            "sources": qp.loc[c, "sources"]})
    t = pd.DataFrame(rows).set_index("indicator")

    # Sensitivity: the same analysis at the looser support rule.
    f10 = A.S2D / "quality_robustness_s10.json"
    if f10.exists():
        q10 = pd.DataFrame(json.loads(f10.read_text()))
        p10 = q10[q10.model == "__pooled__"].set_index("label")
        t["dAUPRC (support 10)"] = [
            round(float(p10.loc[c, "delta"]), 4) if c in p10.index else np.nan
            for c in cols]
        t["sources (support 10)"] = [
            int(p10.loc[c, "sources_used"]) if c in p10.index else -1
            for c in cols]
    A.emit(t, "tableS5_rq3_support",
           "Supplementary Table S5. Support behind each RQ3 column. The "
           "primary analysis requires 20 positives and 20 negatives per "
           "diagnosis in both strata; the final two columns repeat it at a "
           "threshold of 10, which is reported because the threshold "
           "changes how many held-out sources can contribute. Structural "
           "and rhythm defects are rare outside Ningbo, so their contrasts "
           "rest on fewer sources than the spectral indicators, and the "
           "effect sizes should be read with that in mind.", STAB)


def _fit(L):
    """Practical fit: does it leave room for the rest of the firmware?

    Flash and RAM here already include the generated runtime, so the only
    thing reserved is the application: 64 KB Flash and 16 KB SRAM for
    drivers, acquisition buffers and a bootloader.
    """
    fkb = (L.get("total_flash_b") or 0) / 1024
    rkb = (L.get("total_ram_b") or 0) / 1024
    if fkb >= FLASH_KB or rkb >= SRAM_KB:
        return "no"
    if fkb >= FLASH_KB - RESERVE_FLASH:
        return "Flash-limited"
    if rkb >= SRAM_KB - RESERVE_SRAM:
        return "SRAM-limited"
    return "yes"


def supplementary_deployment():
    """S7: every deployment number, so the main text can stay prose.

    Figure 5 carries the argument. This is what somebody reproducing it
    needs: the measured footprint and latency beside what int8 did to
    both discrimination and selective reliability.
    """
    lat = A.load_latency()
    fp = A.S2D / "int8_preservation.json"
    if not lat:
        return
    pres = {}
    if fp.exists():
        pres = {r["model"]: r for r in json.loads(fp.read_text())}
    rows = []
    for m in ORDER:
        L = lat.get(m)
        if not L:
            continue
        r = pres.get(m, {})
        rows.append({
            "model": NICE[m],
            "Flash KB": round((L.get("total_flash_b") or 0) / 1024, 1),
            "RAM KB": round((L.get("total_ram_b") or 0) / 1024, 1),
            "MACs (M)": round((L.get("macc") or 0) / 1e6, 2),
            "latency ms": round(L["ms_median"], 1)
            if L.get("ms_median") else np.nan,
            "RTF": round(L["rtf"], 4) if L.get("rtf") else np.nan,
            "float32 AUPRC": round(r["auprc_f32"], 4) if r else np.nan,
            "int8 AUPRC": round(r["auprc_int8"], 4) if r else np.nan,
            "dAUPRC": round(r["d_auprc"], 4) if r else np.nan,
            "float32 E-AURC": round(r["eaurc_f32"], 4) if r else np.nan,
            "int8 E-AURC": round(r["eaurc_int8"], 4) if r else np.nan,
            "dE-AURC": round(r["d_eaurc"], 4) if r else np.nan,
            "practical fit": _fit(L),
        })
    A.emit(pd.DataFrame(rows).set_index("model"),
           "tableS7_deployment_measurements",
           "Supplementary Table S7. Complete STM32F411 deployment "
           "measurements. Flash and RAM are totals including the "
           "generated ST Edge AI runtime. Latency is the median of 32 "
           "timed runs on the part at 100 MHz after a warm-up, and RTF is "
           "that latency over the 10 s acquisition window, so RTF < 1 "
           "means inference finishes inside the recording. The last six "
           "columns are the quantisation comparison plotted in Figure "
           "5(a-b), on identical records: dAUPRC is int8 minus float32 "
           "discrimination, dE-AURC is int8 minus float32 selective "
           "reliability, where a negative dE-AURC means the confidence "
           "ordering was not degraded. Latency excludes preprocessing, "
           "which is performed off-device. Practical fit reserves "
           f"{RESERVE_FLASH} KB Flash and {RESERVE_SRAM} KB SRAM for "
           "application firmware on top of the measured totals.", STAB)


def supplementary_operating_points():
    """S8: selective risk at fixed coverage, which E-AURC cannot express."""
    fp = A.S2D / "int8_preservation.json"
    if not fp.exists():
        return
    pres = {r["model"]: r for r in json.loads(fp.read_text())}
    covs = ["1.0", "0.9", "0.8", "0.7"]
    rows = []
    for m in ORDER:
        if m not in pres:
            continue
        rf = pres[m]["risk_f32"]
        r8 = pres[m]["risk_int8"]
        rec = {"model": NICE[m]}
        for c in covs:
            rec[f"float32 R({float(c):.2f})"] = round(rf[c], 4)
        rec["float32 dR(0.90)"] = round(rf["1.0"] - rf["0.9"], 4)
        rec["float32 dR(0.80)"] = round(rf["1.0"] - rf["0.8"], 4)
        rec["int8 dR(0.90)"] = round(r8["1.0"] - r8["0.9"], 4)
        rows.append(rec)
    A.emit(pd.DataFrame(rows).set_index("model"),
           "tableS8_operating_points",
           "Supplementary Table S8. Selective risk at fixed retained "
           "coverage, averaged over the four held-out sources. R(1.00) is "
           "risk with every record reported; R(0.90) is risk on the 90% "
           "the model is most confident about, so dR(0.90) = R(1.00) - "
           "R(0.90) is what withholding the least-confident tenth for "
           "reacquisition or review would buy on the records still "
           "reported. The int8 column shows the same quantity survives "
           "quantisation. These are observational quantities computed on "
           "held-out predictions; no abstention workflow was clinically "
           "validated here.", STAB)


def supplementary_stratum_sensitivity():
    """S9: do the quality associations survive a different stratum cut?"""
    base = A.S2D / "quality_robustness.json"
    alt = A.S2D / "quality_robustness_q20.json"
    if not (base.exists() and alt.exists()):
        print("  (no stratum sensitivity yet)")
        return
    b = pd.DataFrame(json.loads(base.read_text()))
    a = pd.DataFrame(json.loads(alt.read_text()))
    bp = b[b.model == "__pooled__"].set_index("label")
    apd = a[a.model == "__pooled__"].set_index("label")
    cols = [c for c in A.FACTOR_ORDER if c in bp.index]
    rows = []
    for c in cols:
        rec = {
            "indicator": c,
            "Q25/Q75 dAUPRC": round(float(bp.loc[c, "delta"]), 4),
            "Q25/Q75 lo": round(float(bp.loc[c, "lo"]), 4),
            "Q25/Q75 hi": round(float(bp.loc[c, "hi"]), 4),
            "sources": int(bp.loc[c, "sources_used"]),
        }
        if c in apd.index:
            rec["Q20/Q80 dAUPRC"] = round(float(apd.loc[c, "delta"]), 4)
            rec["Q20/Q80 lo"] = round(float(apd.loc[c, "lo"]), 4)
            rec["Q20/Q80 hi"] = round(float(apd.loc[c, "hi"]), 4)
        rows.append(rec)
    A.emit(pd.DataFrame(rows).set_index("indicator"),
           "tableS9_stratum_sensitivity",
           "Supplementary Table S9. Sensitivity of the "
           "quality-performance associations to how the strata are cut. "
           "The primary analysis contrasts the within-source quartiles "
           "(<=Q25 against >=Q75) for the three continuous indicators; "
           "this repeats it at Q20 against Q80. The structural indicators "
           "are present-versus-absent in both and so are unchanged by "
           "the cut. The purpose is to show the principal associations "
           "are not an artefact of one arbitrary threshold.", STAB)


def main():
    # Results are rate-tagged but figures/ and tables/ are not, so a build
    # launched without ECG_FS quietly refills the paper with the archived
    # 250 Hz numbers and still exits zero. Refuse instead.
    if FS_OUT != PAPER_FS:
        sys.exit(f"refusing to build at {FS_OUT} Hz: the paper reports "
                 f"{PAPER_FS} Hz and the asset folders are shared across "
                 f"rates. Re-run with ECG_FS={PAPER_FS}.")
    for d in (A.FIG, TAB, A.SFIG, STAB):
        d.mkdir(parents=True, exist_ok=True)
    q = A.quality()
    runs = A.load_runs()
    dep, f32, acc, qr = {}, {}, pd.DataFrame(), None
    if (A.S3B / "deployment_int8.json").exists():
        dep = {d["model"]: d for d in
               json.loads((A.S3B / "deployment_int8.json").read_text())}
    if (A.S3D / "deployment.json").exists():
        f32 = {d["model"]: d for d in
               json.loads((A.S3D / "deployment.json").read_text())
               if d.get("fs") == FS_OUT}
    if (A.S3B / "int8.json").exists():
        acc = pd.DataFrame(json.loads((A.S3B / "int8.json").read_text()))
    if (A.S2D / "quality_robustness.json").exists():
        qr = pd.DataFrame(json.loads(
            (A.S2D / "quality_robustness.json").read_text()))

    # Tables first: the bootstrap table III computes is also what
    # figure 2 plots, so computing it once stops the two disagreeing.
    print("tables:")
    table1()
    table2(dep)
    boot = table3(runs) if runs else {}
    table4(qr)
    supplementary_rq3(qr)
    supplementary_deployment()
    supplementary_operating_points()
    supplementary_stratum_sensitivity()

    print("main figures:")
    A.figure1()                      # design, three RQ blocks
    A.figure2_generalisation(boot)   # RQ1  external generalisation
    A.figure2(q)                     # RQ2a -> figure3_signal_quality
    A.figure4_reliability(qr)        # RQ2c  risk-coverage + E-AURC
    if runs:
        A.figure5(runs, dep)         # RQ3  -> figure5_operating_points

    print("supplementary:")
    A.figure3_reliability()          # E-AURC by held-out source
    A.figure4(dep, acc, f32)         # float32 vs int8 memory detail
    A.figureS2_risk_coverage()
    A.figureS5_quality_reliability()
    A.figureS8_abstention()          # was figure 4(f)
    if runs:
        supplementary(runs, qr)
    print(f"\nmain: {A.FIG} | {TAB}\nsupp: {A.SFIG} | {STAB}")


if __name__ == "__main__":
    main()
