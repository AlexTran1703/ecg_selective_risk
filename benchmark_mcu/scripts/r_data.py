r"""Loading layer for the reconstructed benchmark.

One vocabulary, used by every downstream script, so the tier a number
belongs to is never inferred from a filename.

Two tiers, and the distinction is the point of the design:

    REF        float32, published topology, no device constraint.
               Answers what the designs are worth as designs (RQ1).
    deployed   int8 under a weight budget and an activation ceiling,
               four configurations per family (RQ2, RQ3).

A figure that mixes the two is comparing an architecture question with a
deployment question, so the tier is carried explicitly rather than left
implicit in a budget name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(ROOT / "12_leads" / "src"))

from ecgmcu.data import FS_OUT                                 # noqa: E402
from mcubench.reference import PUBLISHED_DEPTH, REFERENCE      # noqa: E402

TRAIN = HERE / "results" / f"recon_{FS_OUT}hz"
INT8 = HERE / "results" / f"recon_int8_{FS_OUT}hz"
DEPLOY = HERE / "results" / f"recon_deploy_{FS_OUT}hz"
FIG, TAB = HERE / "figures", HERE / "tables"

#: every configuration, reference tier first
CONFIGS = ["REF", "U-UC128", "U-C256", "J-UC128", "J-C256"]
DEPLOYED = CONFIGS[1:]
BUDGET_OF = {"U-UC128": "UC-128", "J-UC128": "UC-128",
             "U-C256": "C-256", "J-C256": "C-256"}
METHOD_OF = {"U-UC128": "uniform", "U-C256": "uniform",
             "J-UC128": "joint", "J-C256": "joint"}
BUDGET_KB = {"UC-128": 128, "C-256": 256}

#: families in publication order, with the label each figure uses
ORDER = list(REFERENCE)
LABEL = {n: f"M{i + 1}" for i, n in enumerate(ORDER)}


#: family name -> the key the shared icon library and palette use
ICON = {"FCN-1D": "fcn", "ResNet-1D": "resnet", "TCN": "tcn",
        "MobileNetV2": "mobilenetv2", "ShuffleNetV2": "shufflenetv2",
        "GhostNet": "ghostnet", "MobileNetV3": "mobilenetv3",
        "FasterNet": "fasternet", "RepViT": "repvit",
        "MobileNetV4": "mobilenetv4"}

CONFIG_LABEL = {"REF": "Reference (float32, unconstrained)",
                "U-UC128": "Uniform, UC-128",
                "U-C256": "Uniform, C-256",
                "J-UC128": "Joint, UC-128",
                "J-C256": "Joint, C-256"}

FLASH_KB, SRAM_KB = 512, 128
ACT_CAP_KB = 96


def load_runs(tier: str | None = None) -> dict:
    """Float32 predictions, keyed (family, config, held-out source)."""
    runs = {}
    for f in sorted(TRAIN.glob("*.npz")):
        if f.stem.count("__") != 2:
            continue
        d = np.load(f)
        key = (str(d["model"]), str(d["config"]), str(d["held_out"]))
        if tier == "REF" and key[1] != "REF":
            continue
        if tier == "deployed" and key[1] == "REF":
            continue
        runs[key] = {
            "y": d["test_y"].astype(np.int8),
            "p": d["test_prob"].astype(np.float64),
            "rows": d["test_rows"],
            "width": float(d["width"]), "depth": float(d["depth"]),
            "params": int(d["params"]),
            "weight_bytes": int(d["weight_bytes"]),
            "peak_act": int(d["peak_act"]),
            "macs": int(d["macs"]),
            "fp32_frac": float(d["fp32_frac"]),
            "val_auprc": float(d["val_auprc"])}
    return runs


def load_int8() -> dict:
    """Quantised predictions and the float32 pair scored on the same rows."""
    out = {}
    for f in sorted(INT8.glob("*_prob.npz")):
        tag = f.stem[:-5]
        if tag.count("__") != 2:
            continue
        meta = INT8 / f"{tag}.json"
        if not meta.exists():
            continue
        rec = json.loads(meta.read_text())
        d = np.load(f)
        out[(rec["model"], rec["config"], rec["source"])] = {
            "y": d["y"].astype(np.int8),
            "p8": d["prob"].astype(np.float64),
            "p32": d["prob_f32"].astype(np.float64),
            **rec}
    return out


def load_deploy() -> dict:
    """Measured footprint and latency, keyed (family, config)."""
    f = DEPLOY / "deploy.json"
    if not f.exists():
        return {}
    return {(r["model"], r["config"]): r
            for r in json.loads(f.read_text())}


def models_present(runs: dict, config: str) -> list:
    """Families with at least one fold for this configuration."""
    return [m for m in ORDER
            if any(k[0] == m and k[1] == config for k in runs)]


def complete(runs: dict, config: str, n_sources: int = 4) -> list:
    """Families with every LOSO rotation present, which is what a mean
    over sources needs; a partial family would otherwise be averaged
    over a different set of sources than the rest."""
    return [m for m in ORDER
            if sum(1 for k in runs if k[0] == m and k[1] == config)
            == n_sources]


def summary() -> None:
    runs = load_runs()
    print(f"{len(runs)} runs in {TRAIN.name}")
    print("{:<13} {:>5}".format("family", "pub")
          + "".join("{:>9}".format(c) for c in CONFIGS))
    for m in ORDER:
        counts = [sum(1 for k in runs if k[0] == m and k[1] == c)
                  for c in CONFIGS]
        print("{:<13} {:>5}".format(m, PUBLISHED_DEPTH[m])
              + "".join("{:>9}".format(c) for c in counts))


if __name__ == "__main__":
    summary()
