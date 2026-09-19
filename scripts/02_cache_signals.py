"""Filter every record once and write a single float16 signal cache.

Produces ``data/cache/signals.npy`` with shape ``(N, 12, 5000)`` in millivolts,
mains-notched and band-limited to 0.5-50 Hz, lead order fixed to
``STANDARD_LEADS``, length standardised to 10 s at 500 Hz.  Training reads only
this file, so the raw archives are touched exactly once.

Amplitude is deliberately left in physical units: per-record normalisation would
discard QRS voltage, which is itself diagnostic (LQRSV is one of the 13 classes).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgsr.ingest import (GAIN_OVERRIDE, MAINS_HZ, PLAUSIBLE_LIMB_P2P_MV,   # noqa: E402
                          STANDARD_LEADS, TARGET_FS, TARGET_SAMPLES,
                          limb_lead_p2p, parse_header, read_signal_mv,
                          reorder_leads, resample_to, standardise_length)
from ecgsr.representations import bandpass_notch                                # noqa: E402

_DATASET: Path | None = None


def _init(dataset: str) -> None:
    global _DATASET
    _DATASET = Path(dataset)


def process(task: tuple[int, str, str]) -> tuple[int, np.ndarray, str, float]:
    """Load, filter and standardise one record.

    Returns ``(row, signal, warning, limb_p2p)``; the last value feeds the
    per-source amplitude-scale check in ``main``.
    """
    row, source, rel_path = task
    stem = _DATASET / rel_path
    try:
        hdr = parse_header(stem.with_suffix(".hea").read_text(
            encoding="utf-8", errors="ignore"), stem.name)
        sig = read_signal_mv(stem, hdr, source)
        sig = reorder_leads(sig, hdr.leads)
        sig = resample_to(sig, hdr.fs, TARGET_FS)
        sig = bandpass_notch(sig, TARGET_FS, MAINS_HZ[source])
        sig = standardise_length(sig, TARGET_SAMPLES)
        warn = ""
        peak = float(np.abs(sig).max()) if sig.size else 0.0
        if not np.isfinite(peak) or peak > 100.0:
            warn = f"{rel_path}: implausible peak amplitude {peak:.1f} mV"
        return row, sig.astype(np.float16), warn, limb_lead_p2p(sig)
    except Exception as exc:                       # noqa: BLE001
        return (row, np.zeros((12, TARGET_SAMPLES), np.float16),
                f"{rel_path}: {exc}", float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument("--meta", type=Path, default=ROOT / "data" / "meta")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "cache")
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count - 1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with (args.meta / "records.csv").open(newline="", encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))
    n = len(records)

    cache_path = args.out / "signals.npy"
    gib = n * 12 * TARGET_SAMPLES * 2 / 2**30
    print(f"Caching {n} records -> {cache_path} ({gib:.2f} GiB)")

    cache = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float16, shape=(n, 12, TARGET_SAMPLES))

    import multiprocessing as mp
    workers = args.workers or max(1, (mp.cpu_count() or 2) - 1)
    tasks = [(i, r["source"], r["path"]) for i, r in enumerate(records)]

    warnings: list[str] = []
    p2p_by_source: dict[str, list[float]] = {}
    t0 = time.time()
    with mp.Pool(workers, initializer=_init, initargs=(str(args.dataset),)) as pool:
        for done, (row, sig, warn, p2p) in enumerate(
                pool.imap_unordered(process, tasks, chunksize=64), start=1):
            cache[row] = sig
            p2p_by_source.setdefault(records[row]["source"], []).append(p2p)
            if warn:
                warnings.append(warn)
            if done % 5000 == 0 or done == n:
                rate = done / (time.time() - t0)
                print(f"  {done:6d}/{n}  {rate:6.0f} rec/s  "
                      f"eta {(n - done) / rate / 60:5.1f} min", flush=True)

    cache.flush()
    del cache

    # --- amplitude-scale quality control (mandatory gate) --------------------
    # A wrong ADC gain produces a perfectly clean cache that is silently off by
    # a constant factor, surfacing only as mysteriously bad cross-source
    # transfer. Since low QRS voltage is itself one of the labels, this cannot
    # be normalised away later -- absolute voltage has to be right. Training is
    # therefore gated on every source passing a physiological amplitude range.
    lo, hi = PLAUSIBLE_LIMB_P2P_MV
    qc: dict[str, dict] = {}
    print("\nsource-wise amplitude QC: limb-lead QRS peak-to-peak (mV)")
    print(f"  {'source':<9}{'median':>9}{'q05':>9}{'q95':>9}"
          f"{'P(<0.5mV)':>11}   status")
    for source, vals in sorted(p2p_by_source.items()):
        v = np.asarray(vals, dtype=np.float64)
        v = v[np.isfinite(v)]
        med = float(np.median(v))
        entry = {
            "median": med,
            "q05": float(np.quantile(v, 0.05)),
            "q95": float(np.quantile(v, 0.95)),
            "p_below_0.5mV": float((v < 0.5).mean()),
            "gain_override": GAIN_OVERRIDE.get(source),
            "passed": bool(lo <= med <= hi),
        }
        qc[source] = entry
        print(f"  {source:<9}{entry['median']:>9.3f}{entry['q05']:>9.3f}"
              f"{entry['q95']:>9.3f}{entry['p_below_0.5mV']:>11.3f}"
              f"   {'ok' if entry['passed'] else 'FAIL'}")

    (args.out / "amplitude_qc.json").write_text(json.dumps(
        {"plausible_range_mv": [lo, hi], "all_passed": all(
            e["passed"] for e in qc.values()), "sources": qc}, indent=2),
        encoding="utf-8")

    failed = [s for s, e in qc.items() if not e["passed"]]
    if failed:
        raise SystemExit(
            f"\nAMPLITUDE QC FAILED for {', '.join(failed)}: median limb-lead "
            f"QRS peak-to-peak outside [{lo}, {hi}] mV.\n"
            "  The signals are not in physical millivolts. Check the declared "
            "ADC gain for that\n  source and add an entry to "
            "ecgsr.ingest.GAIN_OVERRIDE if it disagrees with the stored data.\n"
            "  Do not work around this by normalising per record: low QRS "
            "voltage is one of the labels.")

    meta = {
        "n_records": n, "fs": TARGET_FS, "n_samples": TARGET_SAMPLES,
        "gain_overrides": GAIN_OVERRIDE,
        "amplitude_qc": qc,
        "leads": list(STANDARD_LEADS), "dtype": "float16", "units": "mV",
        "filtering": {"band_hz": [0.5, 50.0], "notch_hz": MAINS_HZ,
                      "order": 3, "zero_phase": True},
        "n_warnings": len(warnings),
    }
    (args.out / "signals_meta.json").write_text(json.dumps(meta, indent=2),
                                                encoding="utf-8")
    if warnings:
        (args.out / "cache_warnings.txt").write_text("\n".join(warnings),
                                                     encoding="utf-8")
        print(f"\n{len(warnings)} records raised warnings -> "
              f"{args.out / 'cache_warnings.txt'}")
        for w in warnings[:10]:
            print("  ", w)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
