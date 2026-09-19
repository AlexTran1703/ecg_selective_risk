"""Source-wise physical-unit quality control, computed from an existing cache.

``02_cache_signals.py`` runs this check inline and refuses to finish if a source
fails.  This standalone version recomputes ``amplitude_qc.json`` from a cache
that is already on disk, which is useful when the cache is fine but the QC
record is missing or stale.  It only reads the cache, so it is safe to run while
training is in progress.

The quantity checked is the largest limb-lead QRS peak-to-peak amplitude per
record: the clinical low-QRS-voltage criterion is stated on the limb leads, so
that is what a source's amplitude scale should be judged against.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgsr.ingest import GAIN_OVERRIDE, PLAUSIBLE_LIMB_P2P_MV        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "cache" / "signals.npy")
    ap.add_argument("--meta", type=Path, default=ROOT / "data" / "meta")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any source fails")
    args = ap.parse_args()

    with (args.meta / "records.csv").open(newline="", encoding="utf-8") as fh:
        sources = np.array([r["source"] for r in csv.DictReader(fh)])

    X = np.load(args.cache, mmap_mode="r")
    if len(X) != len(sources):
        raise SystemExit(f"cache has {len(X)} records but the index has "
                         f"{len(sources)}; rebuild the cache.")

    p2p = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), args.chunk):
        blk = np.asarray(X[start:start + args.chunk, :6, :], dtype=np.float32)
        p2p[start:start + len(blk)] = (blk.max(axis=2) - blk.min(axis=2)).max(axis=1)

    lo, hi = PLAUSIBLE_LIMB_P2P_MV
    qc: dict[str, dict] = {}
    print("source-wise amplitude QC: limb-lead QRS peak-to-peak (mV)")
    print(f"  {'source':<9}{'median':>9}{'q05':>9}{'q95':>9}"
          f"{'P(<0.5mV)':>11}   status")
    for source in sorted(set(sources.tolist())):
        v = p2p[sources == source]
        v = v[np.isfinite(v)].astype(np.float64)
        med = float(np.median(v))
        entry = {
            "n": int(v.size),
            "median": med,
            "q05": float(np.quantile(v, 0.05)),
            "q95": float(np.quantile(v, 0.95)),
            "p_below_0.5mV": float((v < 0.5).mean()),
            "gain_override": GAIN_OVERRIDE.get(source),
            "passed": bool(lo <= med <= hi),
        }
        qc[source] = entry
        print(f"  {source:<9}{med:>9.3f}{entry['q05']:>9.3f}{entry['q95']:>9.3f}"
              f"{entry['p_below_0.5mV']:>11.3f}"
              f"   {'ok' if entry['passed'] else 'FAIL'}")

    all_passed = all(e["passed"] for e in qc.values())
    out = args.cache.parent / "amplitude_qc.json"
    out.write_text(json.dumps({"plausible_range_mv": [lo, hi],
                               "all_passed": all_passed,
                               "sources": qc}, indent=2), encoding="utf-8")
    print(f"\nall_passed = {all_passed}   ->  {out}")
    if args.strict and not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
