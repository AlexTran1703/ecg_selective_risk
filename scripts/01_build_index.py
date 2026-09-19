"""Build the harmonised record index and fix the common label space.

Scans all four databases, converts every record to a SNOMED-CT label set,
verifies the PTB-XL SCP -> SNOMED rules against the official Challenge counts,
then applies the pre-registered class-selection rule (>= ``--min-per-source``
positives in every source, after merging Challenge equivalence pairs).

Outputs
    data/meta/records.csv    one row per record: source, path, labels, n_samples
    data/meta/classes.json   the retained label space and its per-source counts
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgsr import labels as L                                  # noqa: E402
from ecgsr.ingest import parse_header                          # noqa: E402

# Where each source lives, relative to the dataset directory.
SOURCE_DIRS = {
    "Georgia": "georgia",
    "Chapman": "WFDB_ChapmanShaoxing",
    "Ningbo": "WFDB_Ningbo",
}


def scan_wfdb_source(source: str, directory: Path) -> list[dict]:
    """Read every ``.hea`` in a CinC-format directory."""
    records = []
    for hea in sorted(directory.glob("*.hea")):
        hdr = parse_header(hea.read_text(encoding="utf-8", errors="ignore"), hea.stem)
        records.append({
            "source": source,
            "record": hdr.name,
            "path": str(hea.with_suffix("").relative_to(directory.parent)),
            "patient": f"{source}:{hdr.name}",     # one record per patient
            "fs": hdr.fs,
            "n_samples": hdr.n_samples,
            "age": hdr.age,
            "sex": hdr.sex,
            "snomed": set(hdr.dx),
        })
    return records


def scan_ptbxl(directory: Path, configs: Path) -> list[dict]:
    """Read ``ptbxl_database.csv`` and convert SCP statements to SNOMED."""
    csv_text = (directory / "ptbxl_database.csv").read_text(encoding="utf-8")

    problems = L.verify_ptbxl_mapping(csv_text, configs)
    if problems:
        raise SystemExit("PTB-XL SCP -> SNOMED mapping disagrees with the "
                         "official Challenge counts:\n  " + "\n  ".join(problems))
    print("  PTB-XL SCP -> SNOMED mapping reproduces all official counts exactly.")

    records = []
    for row in csv.DictReader(csv_text.splitlines()):
        stem = row["filename_hr"]                   # records500/00000/00001_hr
        records.append({
            "source": "PTBXL",
            "record": Path(stem).name,
            "path": str(Path("ptbxl") / stem).replace("\\", "/"),
            "patient": f"PTBXL:{row['patient_id']}",
            "fs": 500.0,
            "n_samples": 5000,
            "age": row["age"] or "",
            "sex": row["sex"] or "",
            "snomed": L.ptbxl_record_labels(row["scp_codes"], row["heart_axis"]),
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument("--configs", type=Path, default=ROOT / "configs")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "meta")
    ap.add_argument("--min-per-source", type=int, default=100)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Scanning sources...")
    records = scan_ptbxl(args.dataset / "ptbxl", args.configs)
    print(f"  PTBXL   {len(records):6d}")
    for source, sub in SOURCE_DIRS.items():
        found = scan_wfdb_source(source, args.dataset / sub)
        if not found:
            raise SystemExit(f"no records found for {source} in {args.dataset / sub}")
        print(f"  {source:7s} {len(found):6d}")
        records += found
    print(f"  total   {len(records):6d}")

    # --- pre-registered common label space -----------------------------------
    by_source: dict[str, list[set[str]]] = {s: [] for s in L.SOURCES}
    for r in records:
        by_source[r["source"]].append(r["snomed"])

    keep, counts = L.select_common_classes(
        by_source, L.scored_codes(args.configs), args.min_per_source)
    abbrev = L.abbreviation_map(args.configs)
    equiv = L.build_equivalence_map()

    print(f"\nRetained {len(keep)} classes (>= {args.min_per_source} positives "
          "in every source, equivalence pairs merged):")
    header = f"  {'SNOMED':<16}{'abbr':<10}" + "".join(f"{s:>9}" for s in L.SOURCES)
    print(header)
    for code in keep:
        row = "".join(f"{counts[code][s]:>9d}" for s in L.SOURCES)
        print(f"  {code:<16}{abbrev.get(code, '?'):<10}{row}")

    dropped = [(c, counts[c]) for c in counts if c not in keep]
    print(f"\nDropped {len(dropped)} scored classes; the closest misses were:")
    for code, per in sorted(dropped, key=lambda kv: -min(kv[1].values()))[:5]:
        row = "".join(f"{per[s]:>9d}" for s in L.SOURCES)
        print(f"  {code:<16}{abbrev.get(code, '?'):<10}{row}")

    # --- write index ---------------------------------------------------------
    keep_index = {c: i for i, c in enumerate(keep)}
    rows_out = []
    for r in records:
        canon = L.canonicalise(r["snomed"], equiv)
        positives = sorted(keep_index[c] for c in canon if c in keep_index)
        rows_out.append({
            "source": r["source"], "record": r["record"], "path": r["path"],
            "patient": r["patient"], "fs": r["fs"], "n_samples": r["n_samples"],
            "age": r["age"], "sex": r["sex"],
            "y": " ".join(map(str, positives)),
            "n_pos": len(positives),
        })

    records_csv = args.out / "records.csv"
    with records_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    (args.out / "classes.json").write_text(json.dumps({
        "classes": keep,
        "abbreviations": [abbrev.get(c, c) for c in keep],
        "min_per_source": args.min_per_source,
        "counts": {c: counts[c] for c in keep},
        "equivalence_pairs": [list(p) for p in L.EQUIVALENT_SNOMED],
    }, indent=2), encoding="utf-8")

    # A record with no retained label still carries information (it is a valid
    # all-negative target), so it is kept, but report how common that is.
    empty = sum(1 for r in rows_out if r["n_pos"] == 0)
    print(f"\nWrote {records_csv} ({len(rows_out)} records; "
          f"{empty} have no retained label).")
    print(f"Wrote {args.out / 'classes.json'}")


if __name__ == "__main__":
    main()
