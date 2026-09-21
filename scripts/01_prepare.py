r"""Preprocess CODE-15%, CODE-test and PTB-XL once into memmaps.

    .\.venv\Scripts\python.exe scripts\01_prepare.py --dataset code15
    .\.venv\Scripts\python.exe scripts\01_prepare.py --dataset code_test
    .\.venv\Scripts\python.exe scripts\01_prepare.py --dataset ptbxl

Five seeds over thirty epochs on 345,779 exams is roughly 52 million reads.
Decompressing HDF5 on every one of them would leave the GPU waiting on zlib,
so each source archive is read exactly once here and written into a
contiguous array the dataloader can memmap.

Two rules this file exists to enforce
-------------------------------------
**Nothing training-derived is baked into the array.** Only deterministic
preprocessing -- lead order, length, dtype -- is applied. Normalisation
statistics are computed from the training partition afterwards and applied by
the dataloader, so the stored signal can never leak validation or test
information into training.

**The split is frozen before preprocessing and saved.** All five seeds then
train on identical partitions, and only initialisation and batch order differ.
If each seed saw a different split, ``Var_m p^(m)`` would mix data-split
variance into what is meant to be model uncertainty, and the central
comparison against reader disagreement would be reading partly the wrong
quantity.

Every output carries a manifest recording sampling rate, length, lead order,
dtype, record count, split seed and the source file each record came from, so
a later reader can tell what was done without re-deriving it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecguq.data import (CODE_FS, LABELS, PTBXL_DIAGNOSTIC,          # noqa: E402
                        PTBXL_FS, code15_index, code15_patient_split,
                        ptbxl_folds, ptbxl_signals_from_cache, ptbxl_table)

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6")
CODE_LEN = 4096
PTBXL_LEN = 5000
DTYPE = np.float16
SPLIT_SEED = 0


def sha256_head(path: Path, n: int = 1 << 20) -> str:
    """Hash of the first MB -- enough to detect a swapped file, cheap on 34 GB."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(n))
    return h.hexdigest()[:16]


def write_manifest(out: Path, **fields) -> None:
    (out / "manifest.json").write_text(json.dumps(fields, indent=2),
                                       encoding="utf-8")


# --------------------------------------------------------------------------- #
#  CODE-15%
# --------------------------------------------------------------------------- #
def prepare_code15(src: Path, out: Path, val_frac: float) -> None:
    import h5py

    out.mkdir(parents=True, exist_ok=True)
    df = code15_index(src)
    # Freeze the split first, on patients, before a single sample is written.
    tr, va = code15_patient_split(df, val_frac=val_frac, seed=SPLIT_SEED)
    is_val = np.zeros(len(df), dtype=np.int8)
    is_val[va] = 1

    n = len(df)
    xpath = out / "signals.npy"
    x = np.lib.format.open_memmap(xpath, mode="w+", dtype=DTYPE,
                                  shape=(n, 12, CODE_LEN))
    row_of = {int(e): i for i, e in enumerate(df["exam_id"].to_numpy())}
    written = np.zeros(n, dtype=bool)

    zips = sorted(src.glob("exams_part*.zip"))
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for zi, zpath in enumerate(zips, 1):
            with zipfile.ZipFile(zpath) as zf:
                name = zf.namelist()[0]
                # Extract to a real file: HDF5 needs seeks, and seeking inside
                # a deflate stream means re-decompressing from the start.
                local = Path(zf.extract(name, tmp))
            with h5py.File(local, "r") as f:
                ids = np.asarray(f["exam_id"])
                trac = f["tracings"]
                keep = [j for j, e in enumerate(ids) if int(e) in row_of]
                for j in keep:
                    sig = np.asarray(trac[j], dtype=np.float32).T   # (12, T)
                    t = sig.shape[1]
                    if t < CODE_LEN:
                        sig = np.pad(sig, ((0, 0), (0, CODE_LEN - t)))
                    r = row_of[int(ids[j])]
                    x[r] = sig[:, :CODE_LEN].astype(DTYPE)
                    written[r] = True
            local.unlink()
            print(f"[{zi}/{len(zips)}] {zpath.name}  "
                  f"{written.sum()}/{n} rows  "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
    x.flush()
    del x

    # Exams listed in the table but absent from the released parts are dropped
    # rather than left as zeros, which would train the model on silence.
    ok = np.flatnonzero(written)
    if ok.size != n:
        print(f"  {n - ok.size} exams listed but not present in any part; "
              f"excluded from the index")

    y = df[list(LABELS)].to_numpy().astype(np.int8)
    np.savez_compressed(
        out / "meta.npz",
        exam_id=df["exam_id"].to_numpy(),
        patient_id=df["patient_id"].to_numpy(),
        row_index=np.arange(n),
        y=y, is_val=is_val, present=written,
        trace_file=df["trace_file"].to_numpy().astype("U32"))

    write_manifest(
        out, dataset="CODE-15%", fs=CODE_FS, length=CODE_LEN,
        leads=list(LEADS), dtype=str(np.dtype(DTYPE)), n_records=int(n),
        n_present=int(ok.size), labels=list(LABELS),
        split="patient-level, frozen before preprocessing",
        split_seed=SPLIT_SEED, val_fraction=val_frac,
        n_train=int((is_val[ok] == 0).sum()), n_val=int((is_val[ok] == 1).sum()),
        n_patients=int(np.unique(df["patient_id"]).size),
        normalisation="not applied; computed from train partition at load",
        source_files=[z.name for z in zips],
        signals_sha256_head=sha256_head(xpath))
    print(f"wrote {xpath} ({xpath.stat().st_size / 1e9:.1f} GB)")


# --------------------------------------------------------------------------- #
#  CODE-test
# --------------------------------------------------------------------------- #
def prepare_code_test(src: Path, out: Path) -> None:
    import h5py
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(src / "ecg_tracings.hdf5", "r") as f:
        sig = np.asarray(f["tracings"], dtype=np.float32).transpose(0, 2, 1)
    n = sig.shape[0]
    xpath = out / "signals.npy"
    x = np.lib.format.open_memmap(xpath, mode="w+", dtype=DTYPE,
                                  shape=(n, 12, CODE_LEN))
    t = sig.shape[2]
    x[:, :, :min(t, CODE_LEN)] = sig[:, :, :CODE_LEN].astype(DTYPE)
    x.flush()
    del x

    ann = src / "annotations"

    def read(name):
        return pd.read_csv(ann / f"{name}.csv")[list(LABELS)].to_numpy().astype(np.int8)

    # Reader annotations are stored but D is deliberately NOT computed here:
    # it must not exist anywhere the model or the selection can reach it.
    np.savez_compressed(
        out / "meta.npz", row_index=np.arange(n),
        y=read("gold_standard"),
        reader1=read("cardiologist1"), reader2=read("cardiologist2"),
        residents_cardio=read("cardiology_residents"),
        residents_emerg=read("emergency_residents"),
        students=read("medical_students"))

    write_manifest(
        out, dataset="CODE-test", fs=CODE_FS, length=CODE_LEN,
        leads=list(LEADS), dtype=str(np.dtype(DTYPE)), n_records=int(n),
        labels=list(LABELS), role="held-out test only; never used for "
        "training, thresholds or calibration",
        reference="adjudicated gold standard; two independent cardiologists "
        "stored separately",
        disagreement="NOT precomputed -- derived after prediction only",
        normalisation="not applied; train-partition statistics applied at load",
        signals_sha256_head=sha256_head(xpath))
    print(f"wrote {xpath} ({xpath.stat().st_size / 1e6:.0f} MB), {n} records")


# --------------------------------------------------------------------------- #
#  PTB-XL
# --------------------------------------------------------------------------- #
def prepare_ptbxl(csv_path: Path, meta_dir: Path, cache_dir: Path,
                  out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    d, y, lk = ptbxl_table(csv_path)
    folds = ptbxl_folds(d)
    ecg_ids = d.index.to_numpy()

    sig = ptbxl_signals_from_cache(meta_dir, cache_dir, ecg_ids)
    n = sig.shape[0]
    xpath = out / "signals.npy"
    x = np.lib.format.open_memmap(xpath, mode="w+", dtype=DTYPE,
                                  shape=(n, 12, PTBXL_LEN))
    x[:] = sig[:, :, :PTBXL_LEN].astype(DTYPE)
    x.flush()
    del x

    split = np.full(n, "train", dtype="U5")
    split[folds["val"]] = "val"
    split[folds["test"]] = "test"

    np.savez_compressed(
        out / "meta.npz", ecg_id=ecg_ids,
        patient_id=d["patient_id"].to_numpy(),
        row_index=np.arange(n), y=y, likelihood=lk, split=split,
        strat_fold=d["strat_fold"].to_numpy(),
        validated_by_human=d["validated_by_human"].to_numpy())

    write_manifest(
        out, dataset="PTB-XL", fs=PTBXL_FS, length=PTBXL_LEN,
        leads=list(LEADS), dtype=str(np.dtype(DTYPE)), n_records=int(n),
        labels=list(LABELS),
        split="official strat_fold: 1-8 train, 9 val, 10 test",
        n_train=int((split == "train").sum()), n_val=int((split == "val").sum()),
        n_test=int((split == "test").sum()),
        likelihood_scale=[15, 35, 50, 80, 100],
        likelihood_zero_means="not specified, not zero probability",
        likelihood_available_for=list(PTBXL_DIAGNOSTIC),
        likelihood_note="attached to diagnostic statements only; SB, AF and "
                        "ST are rhythm statements and store 0",
        normalisation="not applied; train-partition statistics applied at load",
        signals_sha256_head=sha256_head(xpath))
    print(f"wrote {xpath} ({xpath.stat().st_size / 1e9:.2f} GB), {n} records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["code15", "code_test", "ptbxl", "all"])
    ap.add_argument("--raw", type=Path, default=ROOT / "dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "prepared")
    ap.add_argument("--val-frac", type=float, default=0.10)
    args = ap.parse_args()

    which = (["code_test", "ptbxl", "code15"] if args.dataset == "all"
             else [args.dataset])
    for name in which:
        print(f"=== {name} ===", flush=True)
        if name == "code15":
            prepare_code15(args.raw / "code" / "code15",
                           args.out / "code15", args.val_frac)
        elif name == "code_test":
            prepare_code_test(args.raw / "code" / "code_test" / "data",
                              args.out / "code_test")
        else:
            prepare_ptbxl(args.raw / "ptbxl" / "ptbxl_database.csv",
                          ROOT / "data" / "meta", ROOT / "data" / "cache",
                          args.out / "ptbxl")


if __name__ == "__main__":
    main()
