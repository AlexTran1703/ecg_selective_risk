"""SNOMED-CT label harmonisation across the four source databases.

Chapman-Shaoxing, Ningbo and Georgia ship in PhysioNet/CinC 2021 WFDB format and
already carry ``#Dx:`` SNOMED codes.  PTB-XL is used here in its native 1.0.1
release, whose labels are SCP-ECG statements, so we reproduce the Challenge's
SCP -> SNOMED conversion.  Every rule in ``PTBXL_SCP_TO_SNOMED`` and
``PTBXL_AXIS_TO_SNOMED`` was verified by exact record-count agreement with the
``PTB_XL`` column of the official ``dx_mapping_{scored,unscored}.csv`` tables;
see ``verify_ptbxl_mapping``.
"""

from __future__ import annotations

import ast
import csv
import io
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SOURCES = ("PTBXL", "Georgia", "Chapman", "Ningbo")

# --- PTB-XL SCP-ECG statement -> SNOMED-CT ----------------------------------
# A SNOMED code is assigned to a record if *any* of its listed SCP codes is
# present; the Challenge treats these as a union, not a co-occurrence.
PTBXL_SCP_TO_SNOMED: dict[str, tuple[str, ...]] = {
    "426783006": ("SR", "NORM"),                 # sinus rhythm
    "164889003": ("AFIB",),                      # atrial fibrillation
    "164890007": ("AFLT",),                      # atrial flutter
    "270492004": ("1AVB",),                      # 1st degree AV block
    "195042002": ("2AVB",),                      # 2nd degree AV block
    "27885002": ("3AVB",),                       # complete heart block
    "713427006": ("CRBBB",),                     # complete RBBB
    "713426002": ("IRBBB",),                     # incomplete RBBB
    "164909002": ("CLBBB",),                     # complete LBBB
    "251120003": ("ILBBB",),                     # incomplete LBBB
    "445118002": ("LAFB",),                      # left anterior fascicular block
    "445211001": ("LPFB",),                      # left posterior fascicular block
    "698252002": ("IVCD",),                      # nonspecific intraventricular conduction block
    "251146004": ("LVOLT",),                     # low QRS voltages
    "164917005": ("QWAVE",),                     # Q wave abnormal
    "426177001": ("SBRAD",),                     # sinus bradycardia
    "427084000": ("STACH",),                     # sinus tachycardia
    "427393009": ("SARRH",),                     # sinus arrhythmia
    "284470004": ("PAC",),                       # premature atrial contraction
    "63593006": ("SVARR",),                      # supraventricular premature beats
    "10370003": ("PACE",),                       # pacing rhythm
    "164947007": ("LPR",),                       # prolonged PR interval
    "111975006": ("LNGQT",),                     # prolonged QT interval
    "59931005": ("INVT",),                       # T wave inversion
    "164934002": ("NDT", "NT_", "TAB_", "LOWT"), # T wave abnormal
    "429622005": ("STD_",),                      # ST depression
    "164931005": ("STE_",),                      # ST elevation
    "55930002": ("NST_",),                       # ST changes
    "164873001": ("LVH", "VCLVH"),               # left ventricular hypertrophy
    "89792004": ("RVH",),                        # right ventricular hypertrophy
    "266249003": ("SEHYP",),                     # ventricular hypertrophy
    "67741000119109": ("LAO/LAE",),              # left atrial enlargement
    "446358003": ("RAO/RAE",),                   # right atrial hypertrophy
    "11157007": ("BIGU",),                       # ventricular bigeminy
    "251180001": ("TRIGU",),                     # ventricular trigeminy
    "74390002": ("WPW",),                        # Wolff-Parkinson-White
    "67198005": ("PSVT",),                       # paroxysmal supraventricular tachycardia
    "426761007": ("SVTAC",),                     # supraventricular tachycardia
    "164951009": ("ABQRS", "HVOLT"),             # abnormal QRS
    "54329005": ("AMI",),                        # anterior myocardial infarction
    "426434006": ("ISCAN",),                     # anterior ischemia
    "425419005": ("ISCIN",),                     # inferior ischaemia
    "425623009": ("ISCLA",),                     # lateral ischaemia
    "164861001": ("ISC_", "ISCAL", "ISCIL", "ISCAS"),  # myocardial ischemia
}

# Unscored umbrella codes whose PTB-XL record counts we could not reproduce from
# the published SCP statements by any union of SCP codes (the Challenge appears
# to have used additional metadata columns when building them).  None of these
# is a scored diagnosis, so none can enter the common label space; they are
# therefore left unmapped and excluded from the audit rather than guessed at.
UNRECONSTRUCTED_PTBXL: frozenset[str] = frozenset({
    "164865005",   # MI      -- myocardial infarction (umbrella)
    "164884008",   # VEB     -- ventricular ectopics
    "428750005",   # NSSTTA  -- nonspecific ST-T abnormality
})

# PTB-XL codes the Challenge derives from the ``heart_axis`` column instead.
PTBXL_AXIS_TO_SNOMED: dict[str, tuple[str, ...]] = {
    "39732003": ("LAD", "ALAD"),                 # left axis deviation
    "47665007": ("RAD", "ARAD"),                 # right axis deviation
    "251200008": ("AXR", "AXL", "SAG"),          # indeterminate cardiac axis
}

# --- Challenge equivalence classes ------------------------------------------
# Official scoring treats each pair below as a single diagnosis.
EQUIVALENT_SNOMED: tuple[tuple[str, str], ...] = (
    ("733534002", "164909002"),   # CLBBB == LBBB
    ("713427006", "59118001"),    # CRBBB == RBBB
    ("284470004", "63593006"),    # PAC   == SVPB
    ("427172004", "17338001"),    # PVC   == VPB
)


def build_equivalence_map() -> dict[str, str]:
    """Map every SNOMED code in an equivalence pair onto its canonical member."""
    canonical: dict[str, str] = {}
    for primary, alias in EQUIVALENT_SNOMED:
        canonical[primary] = primary
        canonical[alias] = primary
    return canonical


def canonicalise(codes: set[str], equiv: dict[str, str] | None = None) -> set[str]:
    equiv = build_equivalence_map() if equiv is None else equiv
    return {equiv.get(c, c) for c in codes}


# --- Challenge mapping tables -----------------------------------------------
@dataclass(frozen=True)
class DxEntry:
    snomed: str
    abbreviation: str
    name: str
    scored: bool
    counts: dict[str, int]


_SOURCE_COLUMN = {
    "PTBXL": "PTB_XL",
    "Georgia": "Georgia",
    "Chapman": "Chapman_Shaoxing",
    "Ningbo": "Ningbo",
}


def load_dx_tables(configs_dir: Path) -> dict[str, DxEntry]:
    """Read ``dx_mapping_scored.csv`` and ``dx_mapping_unscored.csv``."""
    table: dict[str, DxEntry] = {}
    for fname, scored in (("dx_mapping_scored.csv", True),
                          ("dx_mapping_unscored.csv", False)):
        with (Path(configs_dir) / fname).open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                code = row["SNOMEDCTCode"].strip()
                table[code] = DxEntry(
                    snomed=code,
                    abbreviation=row["Abbreviation"].strip(),
                    name=row["Dx"].strip(),
                    scored=scored,
                    counts={s: int(row[c] or 0) for s, c in _SOURCE_COLUMN.items()},
                )
    return table


def scored_codes(configs_dir: Path) -> set[str]:
    return {c for c, e in load_dx_tables(configs_dir).items() if e.scored}


def abbreviation_map(configs_dir: Path) -> dict[str, str]:
    return {c: e.abbreviation for c, e in load_dx_tables(configs_dir).items()}


# --- PTB-XL conversion -------------------------------------------------------
def ptbxl_record_labels(scp_codes: str | dict, heart_axis: str) -> set[str]:
    """Convert one PTB-XL database row into its SNOMED-CT label set."""
    present = set(ast.literal_eval(scp_codes) if isinstance(scp_codes, str) else scp_codes)
    axis = (heart_axis or "").strip()
    labels = {snomed for snomed, scps in PTBXL_SCP_TO_SNOMED.items()
              if present.intersection(scps)}
    labels |= {snomed for snomed, axes in PTBXL_AXIS_TO_SNOMED.items()
               if axis in axes}
    return labels


def verify_ptbxl_mapping(ptbxl_csv_text: str, configs_dir: Path) -> list[str]:
    """Cross-check the SCP -> SNOMED rules against the official PTB-XL counts.

    Returns a list of human-readable discrepancies; an empty list means every
    rule reproduces the Challenge's per-code record count exactly.
    """
    table = load_dx_tables(configs_dir)
    counts: dict[str, int] = defaultdict(int)
    for row in csv.DictReader(io.StringIO(ptbxl_csv_text)):
        for code in ptbxl_record_labels(row["scp_codes"], row["heart_axis"]):
            counts[code] += 1

    problems: list[str] = []
    audited = (set(counts) | {c for c, e in table.items() if e.counts["PTBXL"]}
               ) - UNRECONSTRUCTED_PTBXL
    for code in sorted(audited):
        if code not in table:
            problems.append(f"{code}: produced {counts[code]} records but code is "
                            "unknown to the Challenge tables")
            continue
        expected, got = table[code].counts["PTBXL"], counts.get(code, 0)
        if expected != got:
            problems.append(f"{code} ({table[code].abbreviation}): "
                            f"expected {expected}, produced {got}")
    return problems


# --- Common-label selection --------------------------------------------------
def select_common_classes(
    label_sets_by_source: dict[str, list[set[str]]],
    scored: set[str],
    min_per_source: int = 100,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Pre-registered rule: keep scored diagnoses with at least
    ``min_per_source`` positives in *every* source, after collapsing the
    Challenge equivalence pairs.
    """
    equiv = build_equivalence_map()
    scored_canon = {equiv.get(c, c) for c in scored}
    sources = list(label_sets_by_source)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in sources})
    for source, label_sets in label_sets_by_source.items():
        for labels in label_sets:
            for code in canonicalise(labels, equiv) & scored_canon:
                counts[code][source] += 1

    keep = sorted(
        (c for c, per in counts.items() if all(per[s] >= min_per_source for s in sources)),
        key=lambda c: -sum(counts[c].values()),
    )
    return keep, {c: dict(per) for c, per in counts.items()}
