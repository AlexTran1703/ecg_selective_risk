r"""Compile, flash and time the forty deployed configurations.

    .\.venv\Scripts\python.exe benchmark_mcu\scripts\r3_deploy.py
    .\.venv\Scripts\python.exe benchmark_mcu\scripts\r3_deploy.py --no-flash

Everything up to this point is a projection: predicted weight bytes, a
graph-derived peak tensor, a MAC count. This stage replaces all three
with what the part reports and what the cycle counter measures.

    stedgeai analyze     weights, activations, MACC, runtime overhead
    stedgeai generate    CMSIS-NN C for the network
    arm-none-eabi-gcc    link against the ST runtime
    flash over SWD       and run it

A configuration counts as **deployed** only if it links, flashes and
returns the expected result block from the board. One fold's weights are
deployed per configuration: footprint and latency do not depend on which
source was held out, so flashing all four would measure the same numbers
four times.

The toolchain stages are imported from `b3_deploy` so the firmware, the
compiler flags and the cycle-counting protocol are identical to the
earlier benchmark's; only the model that gets exported is different.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "benchmark_mcu"
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(ROOT / "12_leads" / "src"))

import b3_deploy as T                                          # noqa: E402
from ecgmcu.data import Cohort, FS_OUT                         # noqa: E402
from mcubench.footprint import (float_fallback_frac,           # noqa: E402
                                weight_bytes)
from mcubench.models import n_params                           # noqa: E402
from mcubench.reference import REFERENCE, build, fuse          # noqa: E402
from r1_train import CONFIGS, SLUG                             # noqa: E402

TRAIN = HERE / "results" / f"recon_{FS_OUT}hz"
OUT = HERE / "results" / f"recon_deploy_{FS_OUT}hz"
DEPLOYED = [c for c in CONFIGS if c != "REF"]
LENGTH, N_LEADS = 1000, 12
FLASH_KB, SRAM_KB = 512, 128
CALIB_N = 256


def export_int8_batch1(name, config, src, coh, seed, path_f32, path_i8):
    """Deployment graph: batch pinned to 1, same calibration as scoring."""
    from onnxruntime.quantization import (quantize_static, QuantFormat,
                                          QuantType, CalibrationMethod)
    run = np.load(TRAIN / f"{SLUG[name]}__{config}__{src}.npz")
    w, d = float(run["width"]), float(run["depth"])
    model = fuse(build(name, w, d))
    assert n_params(model) == int(run["params"]), (
        f"{name}/{config}: rebuilt model has {n_params(model)} "
        f"parameters, checkpoint has {int(run['params'])}")
    model.load_state_dict(torch.load(
        TRAIN / f"{SLUG[name]}__{config}__{src}.pt", map_location="cpu"))
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, torch.zeros(1, N_LEADS, LENGTH),
                          str(path_f32), input_names=["x"],
                          output_names=["y"], opset_version=13, dynamo=False)

    tr, _, _ = coh.loso(src, seed=seed)
    mean, std = coh.stats(tr)
    rng = np.random.default_rng(seed)
    calib = np.sort(rng.choice(tr, min(CALIB_N, tr.size), replace=False))
    sig = np.load(coh.sig_path, mmap_mode="r")
    data = (np.asarray(sig[calib], dtype=np.float32) - mean) / std

    class R:
        def __init__(self):
            self.i = 0

        def get_next(self):
            if self.i >= data.shape[0]:
                return None
            x = data[self.i:self.i + 1]
            self.i += 1
            return {"x": x}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantize_static(str(path_f32), str(path_i8), R(),
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.Percentile,
                        extra_options={"CalibPercentile": 99.99,
                                       "CalibNumBins": 2048},
                        per_channel=True)
    return w, d, weight_bytes(model), float_fallback_frac(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(REFERENCE))
    ap.add_argument("--configs", nargs="+", default=DEPLOYED)
    ap.add_argument("--fold", default="PTBXL",
                    help="which fold's weights to deploy (footprint and "
                         "latency are identical across folds)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-flash", action="store_true",
                    help="analyze/generate/build only, do not touch the board")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for exe in (T.STEDGEAI, T.PROG,
                T.GCC_BIN / "arm-none-eabi-gcc.exe", T.AI_LIB):
        if not Path(exe).exists():
            sys.exit(f"missing toolchain component: {exe}")
    OUT.mkdir(parents=True, exist_ok=True)
    coh = Cohort()

    for config in args.configs:
        for name in args.models:
            tag = f"{SLUG[name]}__{config}"
            dst = OUT / f"{tag}.json"
            if dst.exists() and not args.force:
                continue
            if not (TRAIN / f"{SLUG[name]}__{config}__{args.fold}.pt").exists():
                print(f"[{tag}] not trained, skipped", flush=True)
                continue

            work = OUT / tag
            work.mkdir(parents=True, exist_ok=True)
            f32, i8 = work / "model_f32.onnx", work / "model_int8.onnx"
            w, d, wb, fp = export_int8_batch1(name, config, args.fold, coh,
                                              args.seed, f32, i8)

            rec = {"model": name, "config": config, "width": w, "depth": d,
                   "fold": args.fold, "predicted_weight_b": wb,
                   "fp32_frac": fp}
            rec |= T.analyze(i8, work)
            rec |= T.generate(i8, work / "gen")
            ok, err = (False, "generate failed")
            if rec.get("generate_ok"):
                ok, err = T.build(work / "gen", work / "fw")
            rec["build_ok"] = ok
            if not ok:
                rec["build_error"] = err
            if ok and not args.no_flash:
                rec |= T.flash_and_time(work / "fw")
            rec["deployed"] = bool(rec.get("ran"))

            fkb = (rec.get("total_flash_b") or 0) / 1024
            rkb = (rec.get("total_ram_b") or 0) / 1024
            rec["fits_flash"] = fkb < FLASH_KB
            rec["fits_sram"] = 0 < rkb < SRAM_KB
            # The budget is on model weights, not on total flash: the ST
            # runtime adds 20-120 KB that belongs to the tooling rather
            # than to the architecture, and is reported separately.
            meas = rec.get("weights_b")
            rec["weight_budget_err"] = (
                None if not meas else (wb - meas) / meas)
            dst.write_text(json.dumps(rec, indent=1))
            ms = rec.get("ms_median")
            print(f"[{tag:28s}] weights {(meas or 0) / 1024:6.1f} KB  "
                  f"flash {fkb:6.1f} KB  sram {rkb:5.1f} KB  "
                  f"macc {(rec.get('macc') or 0) / 1e6:5.2f}M  "
                  + (f"{ms:7.1f} ms" if ms else "NOT DEPLOYED")
                  + ("" if rec["deployed"] else
                     f"   <- {rec.get('reason', err or 'see log')}"),
                  flush=True)

    # Rebuilt from every record on disk, never from this run's rows: a
    # scoped --models run would otherwise rewrite the aggregate with
    # only the subset it touched, which is data loss rather than a
    # filter.
    allrec = []
    for f in sorted(OUT.glob("*__*.json")):
        try:
            allrec.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    (OUT / "deploy.json").write_text(json.dumps(allrec, indent=1))
    print(f"\n-> {OUT}  ({len(allrec)} records)", flush=True)


if __name__ == "__main__":
    main()
