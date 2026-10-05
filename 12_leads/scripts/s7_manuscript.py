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


def table5(dep, acc):
    lat = A.load_latency()
    if not lat:
        return
    rows = []
    for m in ORDER:
        L = lat.get(m)
        if not L:
            continue
        a = acc[acc.model == m] if not acc.empty else None
        fkb = (L.get("total_flash_b") or 0) / 1024
        rkb = (L.get("total_ram_b") or 0) / 1024
        ms = L.get("ms_median")
        # Two distinct questions. Physical fit is whether the image is
        # smaller than the part, which is what actually ran. Practical fit
        # asks whether a real product could also hold its application,
        # drivers, acquisition buffers and bootloader, so it works against
        # a reserved budget. A model occupying 508.8 of 512 KB passes the
        # first and fails the second, and the second is the useful answer.
        phys = fkb < FLASH_KB and rkb < SRAM_KB
        prac = (fkb < FLASH_KB - RESERVE_FLASH
                and rkb < SRAM_KB - RESERVE_SRAM)
        if not phys:
            v = "no"
        elif prac:
            v = "yes"
        else:
            v = "Flash-limited" if fkb >= FLASH_KB - RESERVE_FLASH \
                else "SRAM-limited"
        rows.append({
            "model": NICE[m],
            "Flash KB": round(fkb, 1),
            "RAM KB": round(rkb, 1),
            "MACs (M)": round((L.get("macc") or 0) / 1e6, 2),
            "latency ms": round(ms, 1) if ms else np.nan,
            "inference RTF": round(L["rtf"], 4) if L.get("rtf") else np.nan,
            "float32 AUPRC": round(a.auprc_f32.mean(), 4)
            if a is not None and len(a) else np.nan,
            "int8 AUPRC": round(a.auprc_int8.mean(), 4)
            if a is not None and len(a) else np.nan,
            "dAUPRC": round(a.delta.mean(), 4)
            if a is not None and len(a) else np.nan,
            "practical fit": v})
    A.emit(pd.DataFrame(rows).set_index("model"),
           "tableS7_stm32f411_deployment",
           f"Supplementary Table S7. STM32F411 deployment at {FS_OUT} Hz "
           f"in int8. Figure 5 carries the scientific result; this is "
           f"the numerical detail behind it. "
           f"({SRAM_KB} KB SRAM, {FLASH_KB} KB Flash). Flash and RAM are "
           f"totals including the generated ST Edge AI runtime, not "
           f"weights alone. Latency is measured on the part: each model "
           f"was built into a bare-metal firmware, flashed over SWD and "
           f"timed with the DWT cycle counter at 100 MHz over 32 runs "
           f"after a warm-up, and the median is reported. Inference RTF "
           f"is that latency over the {SECONDS} s acquisition window; it "
           f"excludes preprocessing, which is performed off-device. "
           f"Practical fit is assessed against a reserved budget of "
           f"{FLASH_KB - RESERVE_FLASH} KB Flash and "
           f"{SRAM_KB - RESERVE_SRAM} KB SRAM, leaving {RESERVE_FLASH} KB "
           f"and {RESERVE_SRAM} KB for application firmware, drivers, "
           f"acquisition buffers and a bootloader. All six models are "
           f"physically smaller than the part and all six ran on it.",
           STAB)


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


def main():
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
    table5(dep, acc)   # -> supplementary S7

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
    if runs:
        supplementary(runs, qr)
    print(f"\nmain: {A.FIG} | {TAB}\nsupp: {A.SFIG} | {STAB}")


if __name__ == "__main__":
    main()
