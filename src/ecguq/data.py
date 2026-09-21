"""Loading CODE and PTB-XL, each on its own terms.

The two datasets are kept entirely separate.  Models are trained
independently on each, and PTB-XL serves as an independent replication of the
phenomenon rather than a transfer target -- otherwise domain shift, label
mapping, reference uncertainty and model uncertainty all vary at once and no
single result can be attributed to any of them.

Label spaces
------------
Both are reduced to the six CODE abnormalities::

    1dAVb  RBBB  LBBB  SB  AF  ST

For PTB-XL these come from SCP statements: ``1AVB``, ``CRBBB``, ``CLBBB``,
``SBRAD``, ``AFIB``, ``STACH``.

Reference uncertainty differs in kind, which is the point of using both:

    CODE-test   two independent cardiologists plus an adjudicated reference,
                giving a per-label disagreement flag.
    PTB-XL      a cardiologist-assigned likelihood per statement.

**The PTB-XL likelihood is not available for all six.** It is attached to
*diagnostic* statements, not *rhythm* statements, so only ``1AVB``, ``CRBBB``
and ``CLBBB`` carry it; ``AFIB``, ``SBRAD`` and ``STACH`` store 0, which means
"not specified" rather than "certain" or "impossible".  Only the three
diagnostic statements may enter the likelihood analysis.
"""

from __future__ import annotations

import ast
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Shared six-label space, in CODE's column order.
LABELS = ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST")

# PTB-XL SCP statement per label.
PTBXL_SCP = {"1dAVb": "1AVB", "RBBB": "CRBBB", "LBBB": "CLBBB",
             "SB": "SBRAD", "AF": "AFIB", "ST": "STACH"}

# Only these carry a likelihood in PTB-XL; the rest are rhythm statements.
PTBXL_DIAGNOSTIC = ("1dAVb", "RBBB", "LBBB")

CODE_FS = 400.0
PTBXL_FS = 500.0


@dataclass
class Split:
    x: np.ndarray               # (N, 12, T) float32, mV
    y: np.ndarray               # (N, 6) int8
    ids: np.ndarray             # (N,) record identifiers
    extra: dict                 # dataset-specific side information


# --------------------------------------------------------------------------- #
#  CODE
# --------------------------------------------------------------------------- #
def code15_index(root: Path):
    """Label table for CODE-15%, one row per exam."""
    import pandas as pd
    d = pd.read_csv(Path(root) / "exams.csv")
    # A handful of exams are listed but absent from the released parts.
    return d[d["trace_file"].notna()].reset_index(drop=True)


def code15_patient_split(df, val_frac: float = 0.10, seed: int = 0):
    """Split at the patient level, never the exam level.

    CODE-15% contains repeat exams per patient, so an exam-level split would
    put the same patient on both sides and inflate validation performance --
    which in turn would bias the decision thresholds fitted there.
    """
    rng = np.random.default_rng(seed)
    pats = np.unique(df["patient_id"].to_numpy())
    rng.shuffle(pats)
    n_val = max(1, int(round(val_frac * pats.size)))
    val = set(pats[:n_val].tolist())
    is_val = df["patient_id"].isin(val).to_numpy()
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def code15_signals(root: Path, df, rows: np.ndarray,
                   cache: Path | None = None) -> np.ndarray:
    """Gather tracings for ``rows`` from the part archives.

    Reads each HDF5 part once and pulls every requested exam from it, rather
    than reopening per record; the parts are ~1 GB decompressed and random
    access across them is what makes naive loading slow.
    """
    import h5py

    root = Path(root)
    sub = df.iloc[rows]
    out = np.zeros((len(rows), 12, 4096), dtype=np.float32)
    pos = {int(e): i for i, e in enumerate(sub["exam_id"].to_numpy())}

    for part in sorted(sub["trace_file"].unique()):
        want = {int(e) for e in sub.loc[sub["trace_file"] == part, "exam_id"]}
        zpath = root / (Path(part).stem + ".zip")
        with zipfile.ZipFile(zpath) as z, z.open(part) as fh:
            with h5py.File(fh, "r") as f:
                ids = np.asarray(f["exam_id"])
                keep = np.flatnonzero(np.isin(ids, list(want)))
                trac = f["tracings"]
                for j in keep:
                    out[pos[int(ids[j])]] = np.asarray(
                        trac[j], dtype=np.float32).T
    return out


def code_test(root: Path) -> Split:
    """CODE-test with both readers and the adjudicated reference."""
    import h5py
    import pandas as pd

    root = Path(root)
    ann = root / "annotations"
    with h5py.File(root / "ecg_tracings.hdf5", "r") as f:
        x = np.asarray(f["tracings"], dtype=np.float32).transpose(0, 2, 1)

    def read(name):
        return pd.read_csv(ann / f"{name}.csv")[list(LABELS)].to_numpy().astype(np.int8)

    gold = read("gold_standard")
    return Split(x=x, y=gold, ids=np.arange(x.shape[0]),
                 extra={"reader1": read("cardiologist1"),
                        "reader2": read("cardiologist2"),
                        "residents_cardio": read("cardiology_residents"),
                        "residents_emerg": read("emergency_residents"),
                        "students": read("medical_students"),
                        "fs": CODE_FS})


# --------------------------------------------------------------------------- #
#  PTB-XL
# --------------------------------------------------------------------------- #
def ptbxl_table(csv_path: Path):
    """PTB-XL database rows with the six labels and their likelihoods.

    Returns the dataframe plus ``(N, 6)`` label and likelihood arrays. A label
    is positive when its SCP statement is present at all; the likelihood is
    what the annotator attached to it, with 0 meaning unspecified.
    """
    import pandas as pd

    d = pd.read_csv(csv_path, index_col="ecg_id")
    codes = [ast.literal_eval(s) for s in d["scp_codes"]]
    y = np.zeros((len(d), len(LABELS)), dtype=np.int8)
    lk = np.zeros((len(d), len(LABELS)), dtype=np.float32)
    for i, entry in enumerate(codes):
        for j, lab in enumerate(LABELS):
            scp = PTBXL_SCP[lab]
            if scp in entry:
                y[i, j] = 1
                lk[i, j] = float(entry[scp])
    return d, y, lk


def ptbxl_folds(d) -> dict[str, np.ndarray]:
    """The recommended split: folds 1-8 train, 9 validation, 10 test.

    Folds 9 and 10 were human-validated to a higher standard than the rest,
    which is why the reference-uncertainty analysis runs on fold 10.
    """
    f = d["strat_fold"].to_numpy()
    return {"train": np.flatnonzero(f <= 8),
            "val": np.flatnonzero(f == 9),
            "test": np.flatnonzero(f == 10)}


def ptbxl_signals_from_cache(meta_dir: Path, cache_dir: Path,
                             ecg_ids: np.ndarray) -> np.ndarray:
    """Reuse the preprocessed signal cache rather than re-reading WFDB.

    The cache holds every record at 500 Hz in mV with the same filtering, so
    taking PTB-XL's rows out of it keeps preprocessing identical to the
    earlier study and costs one memmap read.
    """
    import csv as _csv

    with (Path(meta_dir) / "records.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    row_of: dict[int, int] = {}
    for i, r in enumerate(rows):
        if r["source"] != "PTBXL":
            continue
        digits = "".join(c for c in Path(r["record"]).name if c.isdigit())
        if digits:
            row_of[int(digits)] = i

    sig = np.load(Path(cache_dir) / "signals.npy", mmap_mode="r")
    missing = [int(e) for e in ecg_ids if int(e) not in row_of]
    if missing:
        raise KeyError(f"{len(missing)} PTB-XL ids absent from the cache, "
                       f"first few: {missing[:5]}")
    idx = np.asarray([row_of[int(e)] for e in ecg_ids])
    return np.asarray(sig[idx], dtype=np.float32)
