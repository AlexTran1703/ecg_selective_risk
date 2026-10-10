r"""Compile, flash and time every encoder on a physical STM32F411.

    .\.venv\Scripts\python.exe benchmark_mcu\scripts\b3_deploy.py

This is the measurement the benchmark exists for. Everything up to here
is a projection: parameter counts, a graph-derived peak tensor, MACs.
None of them is Flash, none is SRAM as the allocator sees it, and MACs
is a poor proxy for latency on a Cortex-M4 where depthwise kernels are
memory-bound and plain convolutions vectorise.

So each model is put through the path a product would take:

    stedgeai analyze    weights, activations, MACC, runtime overhead
    stedgeai generate   C sources for the target
    arm-none-eabi-gcc   link against the CM4 runtime, real Flash/SRAM
    flash over SWD      and run it
    DWT cycle counter   32 timed inferences after a warm-up, median

A model counts as **deployed** only if it links, flashes and returns the
expected output on the part. Fitting a budget on paper is not the same
claim and is reported separately, because the gap between the two is one
of the things worth knowing.

Footprint and latency depend on the architecture and the budget, not on
which source was held out -- the graph is identical across folds. So one
fold's weights are deployed per (model, budget) and the fold is recorded,
rather than burning 88 flash cycles to measure the same 22 numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "benchmark_mcu"
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(ROOT / "12_leads" / "src"))

from ecgmcu.data import Cohort, SOURCES, FS_OUT                  # noqa: E402
from mcubench.models import (ZOO, NICE, n_params,                # noqa: E402
                             build as build_model)

PACK = Path("C:/Users/Alith/STM32Cube/Repository/Packs/STMicroelectronics"
            "/X-CUBE-AI/10.2.0")
STEDGEAI = PACK / "Utilities/windows/stedgeai.exe"
AI_INC = PACK / "Middlewares/ST/AI/Inc"
AI_LIB = (PACK / "Middlewares/ST/AI/Lib/GCC/ARMCortexM4"
          / "NetworkRuntime1020_CM4_GCC.a")
GCC_BIN = Path("F:/ST/STM32CubeIDE_1.18.1/STM32CubeIDE/plugins"
               "/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32"
               ".13.3.rel1.win32_1.0.0.202411081344/tools/bin")
PROG = Path("F:/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin"
            "/STM32_Programmer_CLI.exe")

FW = ROOT / "12_leads" / "firmware"          # the timing firmware is generic
BUDGETS = {"compact": (32 * 1024, 96 * 1024),
           "standard": (128 * 1024, 96 * 1024)}
TRAIN = HERE / "results" / f"train_{FS_OUT}hz"
OUT = HERE / "results" / f"deploy_{FS_OUT}hz"
LENGTH, N_LEADS = 1000, 12
FLASH_KB, SRAM_KB = 512, 128
WINDOW_S = 10.0
MAGIC = 0xEC61BEEF
RESULT_ADDR = 0x2001FE00
CALIB_N = 256


def run(cmd, **kw):
    env = dict(os.environ)
    env["PATH"] = str(GCC_BIN) + os.pathsep + env.get("PATH", "")
    return subprocess.run(cmd, capture_output=True, text=True, env=env, **kw)


def _grab(txt, pat, cast=int):
    m = re.search(pat, txt, re.I)
    return cast(m.group(1).replace(",", "")) if m else None


def analyze(onnx: Path, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    r = run([str(STEDGEAI), "analyze", "--model", str(onnx),
             "--target", "stm32", "--workspace", str(work / "ws")])
    txt = r.stdout + r.stderr
    (work / "analyze.log").write_text(txt, encoding="utf-8", errors="ignore")
    return {"analyze_ok": r.returncode == 0,
            "weights_b": _grab(txt, r"weights \(ro\)\s*:\s*([\d,]+)"),
            "activations_b": _grab(txt, r"activations \(rw\)\s*:\s*([\d,]+)"),
            "macc": _grab(txt, r"macc\s*:\s*([\d,]+)")}


def generate(onnx: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    r = run([str(STEDGEAI), "generate", "--model", str(onnx),
             "--target", "stm32", "--name", "network", "--c-api", "legacy",
             "--input-data-type", "int8", "--allocate-inputs",
             "--allocate-outputs", "--workspace", str(out / "ws"),
             "--output", str(out)])
    txt = r.stdout + r.stderr
    (out / "generate.log").write_text(txt, encoding="utf-8", errors="ignore")
    rt = re.search(r"RT total\s+([\d,]+)\s+[\d.]+%\s+([\d,]+)\s+[\d.]+%", txt)
    tot = re.search(r"TOTAL\s+([\d,]+)\s+([\d,]+)\s*$", txt, re.M)
    return {
        "generate_ok": r.returncode == 0,
        "rt_flash_b": int(rt.group(1).replace(",", "")) if rt else None,
        "rt_ram_b": int(rt.group(2).replace(",", "")) if rt else None,
        "total_flash_b": int(tot.group(1).replace(",", "")) if tot else None,
        "total_ram_b": int(tot.group(2).replace(",", "")) if tot else None}


def build(src: Path, work: Path) -> tuple[bool, str]:
    work.mkdir(parents=True, exist_ok=True)
    elf, binf = work / "fw.elf", work / "fw.bin"
    cflags = ["-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16",
              "-mfloat-abi=hard", "-O2", "-ffunction-sections",
              "-fdata-sections", "-std=c11", "-Wall",
              f"-I{src}", f"-I{AI_INC}", f"-I{FW}"]
    sources = [str(FW / "main.c"), str(FW / "startup.c"),
               str(src / "network.c"), str(src / "network_data.c"),
               str(src / "network_data_params.c")]
    cmd = ([str(GCC_BIN / "arm-none-eabi-gcc.exe")] + cflags + sources +
           [str(AI_LIB), "-T", str(FW / "stm32f411.ld"),
            "-Wl,--gc-sections", "-Wl,-Map," + str(work / "fw.map"),
            "--specs=nosys.specs", "-lm", "-o", str(elf)])
    r = run(cmd)
    if r.returncode != 0:
        (work / "build.log").write_text(r.stdout + r.stderr,
                                        encoding="utf-8", errors="ignore")
        return False, (r.stdout + r.stderr)[-1200:]
    run([str(GCC_BIN / "arm-none-eabi-objcopy.exe"), "-O", "binary",
         str(elf), str(binf)])
    return True, ""


def flash_and_time(work: Path, timeout_s: int = 180) -> dict:
    """Flash, then poll the fixed result block until the firmware finishes.

    Layout written by firmware/main.c, in words:
        0 magic   1 stage   2 core_hz   3 n_runs
        4 ai_error   5 macc_hint   6.. cycles[N_RUNS]
    stage is the diagnostic: 6 means it completed, 99 means the network
    failed to initialise, anything else means it stopped partway and the
    value says where.
    """
    n_runs_max = 32
    nwords = 6 + n_runs_max
    binf = work / "fw.bin"
    r = run([str(PROG), "-c", "port=SWD", "mode=UR",
             "-d", str(binf), "0x08000000", "-v", "-rst"])
    (work / "flash.log").write_text(r.stdout + r.stderr, encoding="utf-8",
                                    errors="ignore")
    log = r.stdout + r.stderr
    if "Download verified successfully" not in log and r.returncode != 0:
        return {"flashed": False, "ran": False, "reason": "flash failed"}

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        # Sleep before the first read, not after. Attaching over SWD
        # while the core is still coming out of reset leaves the probe
        # in a state that every later attach inherits, which presents
        # as a board that never answers.
        time.sleep(3.0)
        rr = run([str(PROG), "-c", "port=SWD", "mode=HOTPLUG",
                  "-r32", hex(RESULT_ADDR), hex(nwords * 4)])
        # The dump prints "0xADDR : WORD WORD WORD WORD" and only the
        # address carries an 0x prefix, so the data must be taken from
        # the right-hand side of the colon rather than by a bare hex
        # match -- which would silently collect the addresses instead.
        # stdout and stderr both carry parts of the dump, so the
        # reader must see both -- parsing stdout alone silently finds
        # nothing and looks exactly like a hung board.
        vals = []
        for line in (rr.stdout + rr.stderr).splitlines():
            m = re.match(r"\s*0x[0-9A-Fa-f]+\s*:\s*(.+)$", line)
            if m:
                vals += [int(w, 16) for w in
                         re.findall(r"([0-9A-Fa-f]{8})", m.group(1))]
        if len(vals) >= 6 and vals[0] == MAGIC:
            stage, core_hz, n_runs = vals[1], vals[2], vals[3]
            ai_err = vals[4] - (1 << 32) if vals[4] >= (1 << 31) else vals[4]
            if stage in (6, 99):
                cyc = sorted(vals[6:6 + min(n_runs, n_runs_max)])
                if stage != 6 or not cyc:
                    return {"flashed": True, "ran": False, "stage": stage,
                            "ai_error": ai_err,
                            "reason": f"stage {stage}, ai_error {ai_err}"}
                med = cyc[len(cyc) // 2]
                p95 = cyc[min(len(cyc) - 1, int(round(0.95 * (len(cyc) - 1))))]
                hz = core_hz or 100_000_000
                return {"flashed": True, "ran": True, "stage": stage,
                        "ai_error": ai_err, "core_hz": hz, "n_runs": n_runs,
                        "cycles_median": med, "cycles_p95": p95,
                        "ms_median": med / hz * 1e3,
                        "ms_p95": p95 / hz * 1e3,
                        "rtf": (med / hz) / WINDOW_S}
    return {"flashed": True, "ran": False, "reason": "timeout"}


def export_int8_batch1(name, budget, src, coh, seed, path_f32, path_i8):
    """Deployment graph: batch pinned to 1, same calibration as scoring."""
    from onnxruntime.quantization import (quantize_static, QuantFormat,
                                          QuantType, CalibrationMethod)
    import warnings
    # The deployed graph must be the trained graph, so the width comes
    # from the run record rather than from a fresh budget fit.
    run = np.load(TRAIN / f"{name}__{budget}__{src}.npz")
    width = float(run["width"])
    # Aliased: this module defines its own build() for the firmware,
    # and the two silently collided.
    model = build_model(name, width)
    # Folded before the load: the checkpoint was written folded.
    if hasattr(model, "reparameterise"):
        model.reparameterise()
    assert n_params(model) == int(run["params"]), (
        f"{name}/{budget}: rebuilt model has {n_params(model)} "
        f"parameters, checkpoint has {int(run['params'])}")
    model.load_state_dict(torch.load(
        TRAIN / f"{name}__{budget}__{src}.pt", map_location="cpu"))
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
    return float(width)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ZOO))
    ap.add_argument("--budgets", nargs="+", default=list(BUDGETS))
    ap.add_argument("--fold", default="PTBXL",
                    help="which fold's weights to deploy (footprint and "
                         "latency are identical across folds)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-flash", action="store_true",
                    help="analyze/generate/build only, do not touch the board")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for exe in (STEDGEAI, PROG, GCC_BIN / "arm-none-eabi-gcc.exe", AI_LIB):
        if not Path(exe).exists():
            sys.exit(f"missing toolchain component: {exe}")
    OUT.mkdir(parents=True, exist_ok=True)
    coh = Cohort()
    rows = []

    for budget in args.budgets:
        for name in args.models:
            tag = f"{name}__{budget}"
            dst = OUT / f"{tag}.json"
            if dst.exists() and not args.force:
                rows.append(json.loads(dst.read_text()))
                continue
            if not (TRAIN / f"{name}__{budget}__{args.fold}.pt").exists():
                continue

            work = OUT / tag
            work.mkdir(parents=True, exist_ok=True)
            f32, i8 = work / "model_f32.onnx", work / "model_int8.onnx"
            width = export_int8_batch1(name, budget, args.fold, coh,
                                       args.seed, f32, i8)

            rec = {"model": name, "budget": budget, "width": width,
                   "fold": args.fold}
            rec |= analyze(i8, work)
            rec |= generate(i8, work / "gen")
            ok, err = (False, "generate failed")
            if rec.get("generate_ok"):
                ok, err = build(work / "gen", work / "fw")
            rec["build_ok"] = ok
            if not ok:
                rec["build_error"] = err
            if ok and not args.no_flash:
                rec |= flash_and_time(work / "fw")
            rec["deployed"] = bool(rec.get("ran"))

            fkb = (rec.get("total_flash_b") or 0) / 1024
            rkb = (rec.get("total_ram_b") or 0) / 1024
            rec["fits_flash"] = fkb < FLASH_KB
            rec["fits_sram"] = 0 < rkb < SRAM_KB
            dst.write_text(json.dumps(rec, indent=1))
            rows.append(rec)
            ms = rec.get("ms_median")
            print(f"[{tag:28s}] flash {fkb:6.1f} KB  sram {rkb:5.1f} KB  "
                  f"macc {(rec.get('macc') or 0) / 1e6:5.2f}M  "
                  + (f"{ms:7.1f} ms  RTF {rec['rtf']:.3f}" if ms
                     else "NOT DEPLOYED")
                  + ("" if rec["deployed"] else
                     f"   <- {rec.get('reason', err or 'see log')}"),
                  flush=True)

    # Rebuild the aggregate from every per-model record on disk, not
    # from this run's `rows`. A scoped run (--models ...) would otherwise
    # rewrite deploy.json with only the subset it touched and silently
    # discard the rest, which is a data-loss bug rather than a filter.
    allrec = []
    for f in sorted(OUT.glob("*__*.json")):
        try:
            allrec.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    (OUT / "deploy.json").write_text(json.dumps(allrec, indent=1))
    n_ok = sum(r.get("deployed", False) for r in rows)
    print(f"\n{n_ok}/{len(rows)} ran on the board\n-> {OUT}")


if __name__ == "__main__":
    main()
