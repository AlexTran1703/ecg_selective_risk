r"""Study I: cross-source signal-quality characterisation.

    .\.venv\Scripts\python.exe 12_leads\scripts\s1_signal_quality.py [--limit N]

Computes interpretable, individually reported quality indicators for every
record in the four-source cohort. Deliberately no composite "SQI = 0.742":
a single number would hide arbitrary weights behind a scientific-looking
decimal. The indicators stay separate, and the only derived category uses
explicit thresholds stated here.

Indicators, per record (aggregated over the 12 leads):

    flat_leads      leads whose amplitude never leaves a narrow band
    clip_frac       samples pinned at a lead's own extreme value
    p2p_mv          median lead peak-to-peak amplitude
    bw_rel          relative spectral power below 0.5 Hz   (baseline wander)
    hf_rel          relative power 40-100 Hz, mains bands excluded (EMG)
    pli_rel         mains: max of the 50 Hz and 60 Hz band power
    n_qrs           R peaks detected on the best limb lead
    rr_implaus      fraction of RR intervals outside 0.3-2.0 s
    usable_leads    leads that are neither flat nor heavily clipped

Writes results/study1/quality.parquet (per record) plus the dataset-level
table. Reads only the cached signals, so it never touches the raw sources.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

HERE = Path(__file__).resolve().parents[1]   # 12_leads/
ROOT = HERE.parent                           # repo root: shared data
CACHE = ROOT / "data" / "cache"
OUT = HERE / "results" / "study1"

FS = 500.0
FLAT_MV = 0.02          # a lead that never exceeds this is not carrying signal
CLIP_TOL = 1e-3         # distance from a lead's own extreme counted as pinned
CLIP_BAD = 0.01         # >1% of samples pinned marks the lead unusable
RR_LO, RR_HI = 0.3, 2.0  # plausible RR seconds (30-200 bpm)


def band_powers(x: np.ndarray) -> dict[str, np.ndarray]:
    """Relative power in the bands that correspond to known artefacts."""
    n = x.shape[-1]
    freq = np.fft.rfftfreq(n, d=1.0 / FS)
    # Hann window: ECG segments are not periodic, and leakage from the large
    # low-frequency content would otherwise contaminate the mains bands.
    w = np.hanning(n).astype(np.float32)
    p = np.abs(np.fft.rfft(x * w, axis=-1)) ** 2
    total = p.sum(axis=-1) + 1e-12

    def rel(lo, hi, mask=None):
        m = (freq >= lo) & (freq < hi)
        if mask is not None:
            m = m & ~mask
        return (p[..., m].sum(axis=-1) / total).mean(axis=-1)

    # The mains bands sit inside 40-100 Hz, so a plain high-frequency
    # measure double-counts them and the two indicators stop being
    # separable. HF therefore excludes them explicitly.
    b50 = (freq >= 48) & (freq < 52)
    b60 = (freq >= 58) & (freq < 62)
    mains_band = b50 | b60

    # max(P50, P60) rather than the sum: the sum bakes the recording
    # region into the metric, which for a cross-source audit would make
    # the indicator partly a label for the dataset itself.
    p50 = (p[..., b50].sum(axis=-1) / total).mean(axis=-1)
    p60 = (p[..., b60].sum(axis=-1) / total).mean(axis=-1)
    return {"bw_rel": rel(0.0, 0.5),
            "hf_rel": rel(40.0, 100.0, mask=mains_band),
            "pli_rel": np.maximum(p50, p60),
            "pli_50": p50, "pli_60": p60}


def qrs_metrics(sig: np.ndarray, sos) -> tuple[int, float]:
    """R peaks and implausible-RR fraction on the most plausible limb lead.

    Pan-Tompkins in miniature: band-pass, square, smooth, then peak-pick.
    Lead II is the usual choice but is sometimes the dead one, so the lead
    with the largest filtered energy among I/II/aVF is used instead.
    """
    cand = sig[[0, 1, 5]]                       # I, II, aVF
    filt = sosfiltfilt(sos, cand, axis=-1)
    lead = filt[int(np.argmax(filt.std(axis=-1)))]
    env = np.convolve(lead ** 2, np.ones(int(0.12 * FS)) / (0.12 * FS), "same")
    if env.max() <= 0:
        return 0, 1.0
    peaks, _ = find_peaks(env, height=0.35 * np.percentile(env, 99),
                          distance=int(0.3 * FS))
    if peaks.size < 3:
        return int(peaks.size), 1.0
    rr = np.diff(peaks) / FS
    return int(peaks.size), float(((rr < RR_LO) | (rr > RR_HI)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="records, 0 = all")
    ap.add_argument("--chunk", type=int, default=400)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rec = pd.read_csv(ROOT / "data" / "meta" / "records.csv")
    sig = np.load(CACHE / "signals.npy", mmap_mode="r")
    n = args.limit or sig.shape[0]
    assert len(rec) == sig.shape[0], "index and cache disagree"
    sos = butter(3, [5.0, 15.0], btype="band", fs=FS, output="sos")

    rows = []
    for s in range(0, n, args.chunk):
        e = min(s + args.chunk, n)
        x = np.asarray(sig[s:e], dtype=np.float32)
        lead_std = x.std(axis=-1)
        lead_p2p = x.max(axis=-1) - x.min(axis=-1)
        flat = lead_p2p < FLAT_MV

        hi = x.max(axis=-1, keepdims=True)
        lo = x.min(axis=-1, keepdims=True)
        pinned = ((np.abs(x - hi) < CLIP_TOL) | (np.abs(x - lo) < CLIP_TOL))
        clip_lead = pinned.mean(axis=-1)

        bands = band_powers(x)
        for k in range(e - s):
            nq, rri = qrs_metrics(x[k], sos)
            rows.append({
                "idx": s + k,
                "flat_leads": int(flat[k].sum()),
                "clip_frac": float(clip_lead[k].mean()),
                "p2p_mv": float(np.median(lead_p2p[k])),
                "lead_std": float(np.median(lead_std[k])),
                "bw_rel": float(bands["bw_rel"][k]),
                "hf_rel": float(bands["hf_rel"][k]),
                "pli_rel": float(bands["pli_rel"][k]),
                "n_qrs": nq,
                "rr_implaus": rri,
                "usable_leads": int((~flat[k] & (clip_lead[k] < CLIP_BAD)).sum()),
            })
        if (s // args.chunk) % 20 == 0:
            print(f"  {e}/{n}", flush=True)

    q = pd.DataFrame(rows)
    q["source"] = rec["source"].to_numpy()[q["idx"].to_numpy()]
    q["record"] = rec["record"].to_numpy()[q["idx"].to_numpy()]
    q.to_parquet(OUT / "quality.parquet", index=False)

    # Explicit, stated criteria -- not a learned or weighted score.
    bad = ((q.usable_leads < 10) | (q.hf_rel > 0.25) | (q.rr_implaus > 0.25)
           | (q.p2p_mv < 0.1))
    mod = ~bad & ((q.usable_leads < 12) | (q.hf_rel > 0.10)
                  | (q.bw_rel > 0.40) | (q.pli_rel > 0.05)
                  | (q.rr_implaus > 0.05))
    q["grade"] = np.where(bad, "poor", np.where(mod, "moderate", "clean"))
    q.to_parquet(OUT / "quality.parquet", index=False)

    agg = q.groupby("source").agg(
        n=("idx", "size"),
        flat_lead_pct=("flat_leads", lambda s: 100 * (s > 0).mean()),
        clip_pct=("clip_frac", lambda s: 100 * (s > CLIP_BAD).mean()),
        p2p_mv=("p2p_mv", "median"),
        bw_rel=("bw_rel", "median"),
        hf_rel=("hf_rel", "median"),
        pli_rel=("pli_rel", "median"),
        hf_high_pct=("hf_rel", lambda s: 100 * (s > 0.10).mean()),
        rr_bad_pct=("rr_implaus", lambda s: 100 * (s > 0.05).mean()),
        usable_leads=("usable_leads", "median"),
    ).round(4)
    grade = (q.groupby("source").grade.value_counts(normalize=True)
             .unstack(fill_value=0).mul(100).round(1))

    txt = ["STUDY I. CROSS-SOURCE SIGNAL-QUALITY CHARACTERISATION", "=" * 78,
           "", f"{len(q):,} records, {FS:.0f} Hz, 12 leads, 10 s.", "",
           "1A. Dataset-level quality indicators (medians, or % of records)",
           "-" * 78, agg.to_string(), "",
           "Quality grade, % of records (explicit criteria, see script)",
           "-" * 78, grade.to_string(), ""]
    (OUT / "study1.txt").write_text("\n".join(txt), encoding="utf-8")
    agg.to_json(OUT / "study1.json", indent=2)
    print("\n".join(txt))


if __name__ == "__main__":
    main()
