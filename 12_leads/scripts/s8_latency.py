r"""Study III-C: on-target inference latency on the STM32F411.

    .\.venv\Scripts\python.exe 12_leads\scripts\s8_latency.py

For each encoder that fits: generate the int8 C model, build a bare-metal
firmware around it, flash the board over SWD, let it time N inferences
with the DWT cycle counter at 100 MHz, then read the result block back out
of SRAM.

Cycles are the measured quantity. Milliseconds are derived using the clock
the firmware itself configured, which it also writes into the result block,
so the host cannot convert with a frequency the device did not use.

A model that does not fit, does not link, or faults on the board is
recorded as such. "Did not fit" is a result.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from ecgmcu.data import FS_OUT                                 # noqa: E402
from ecgmcu.models import ZOO                                  # noqa: E402

PACK = Path("C:/Users/Alith/STM32Cube/Repository/Packs/STMicroelectronics"
            "/X-CUBE-AI/10.2.0")
STEDGEAI = PACK / "Utilities/windows/stedgeai.exe"
AI_INC = PACK / "Middlewares/ST/AI/Inc"
AI_LIB = PACK / "Middlewares/ST/AI/Lib/GCC/ARMCortexM4/NetworkRuntime1020_CM4_GCC.a"
GCC_BIN = Path("F:/ST/STM32CubeIDE_1.18.1/STM32CubeIDE/plugins"
               "/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32"
               ".13.3.rel1.win32_1.0.0.202411081344/tools/bin")
PROG = Path("F:/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin"
            "/STM32_Programmer_CLI.exe")

FW = HERE / "firmware"
TAG = "" if FS_OUT == 250 else f"_{FS_OUT}hz"
S3B = HERE / "results" / f"study3b{TAG}"
OUT = HERE / "results" / f"study3c{TAG}"
RESULT_ADDR = 0x2001FE00
MAGIC = 0xEC61BEEF          # must match RESULT_MAGIC in firmware/main.c
N_RUNS = 32
ORDER = ["tiny", "dscnn", "mobilenet", "mbconv", "tcn", "resnet"]


def run(cmd, **kw):
    env = dict(os.environ)
    env["PATH"] = str(GCC_BIN) + os.pathsep + env.get("PATH", "")
    return subprocess.run(cmd, capture_output=True, text=True, env=env, **kw)


def generate(onnx: Path, out: Path) -> dict:
    """Emit C sources and capture the real Flash/RAM including runtime."""
    out.mkdir(parents=True, exist_ok=True)
    r = run([str(STEDGEAI), "generate", "--model", str(onnx),
             "--target", "stm32", "--name", "network", "--c-api", "legacy",
             "--input-data-type", "int8", "--allocate-inputs",
             "--allocate-outputs", "--workspace", str(out / "ws"),
             "--output", str(out)])
    txt = r.stdout + r.stderr
    (out / "generate.log").write_text(txt, encoding="utf-8", errors="ignore")

    def grab(pat, idx=1, cast=int):
        m = re.search(pat, txt, re.I)
        return cast(m.group(idx).replace(",", "")) if m else None

    rt = re.search(r"RT total\s+([\d,]+)\s+[\d.]+%\s+([\d,]+)\s+[\d.]+%", txt)
    tot = re.search(r"TOTAL\s+([\d,]+)\s+([\d,]+)\s*$", txt, re.M)
    return {
        "ok": r.returncode == 0,
        "weights_b": grab(r"weights \(ro\)\s*:\s*([\d,]+)"),
        "activations_b": grab(r"activations \(rw\)\s*:\s*([\d,]+)"),
        "macc": grab(r"macc\s*:\s*([\d,]+)"),
        "rt_flash_b": int(rt.group(1).replace(",", "")) if rt else None,
        "rt_ram_b": int(rt.group(2).replace(",", "")) if rt else None,
        "total_flash_b": int(tot.group(1).replace(",", "")) if tot else None,
        "total_ram_b": int(tot.group(2).replace(",", "")) if tot else None,
    }


def build(src: Path, work: Path) -> tuple[bool, str]:
    work.mkdir(parents=True, exist_ok=True)
    elf, binf = work / "fw.elf", work / "fw.bin"
    cflags = [
        "-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16",
        "-mfloat-abi=hard", "-O2", "-ffunction-sections", "-fdata-sections",
        "-std=c11", "-Wall",
        f"-I{src}", f"-I{AI_INC}", f"-I{FW}",
    ]
    sources = [str(FW / "main.c"), str(FW / "startup.c"),
               str(src / "network.c"), str(src / "network_data.c"),
               str(src / "network_data_params.c")]
    cmd = ([str(GCC_BIN / "arm-none-eabi-gcc.exe")] + cflags + sources +
           [str(AI_LIB), "-T", str(FW / "stm32f411.ld"),
            "-Wl,--gc-sections", "-Wl,-Map," + str(work / "fw.map"),
            "--specs=nosys.specs", "-lm", "-o", str(elf)])
    r = run(cmd)
    if r.returncode != 0:
        (work / "build.log").write_text(r.stdout + r.stderr, encoding="utf-8",
                                        errors="ignore")
        return False, (r.stdout + r.stderr)[-1500:]
    run([str(GCC_BIN / "arm-none-eabi-objcopy.exe"), "-O", "binary",
         str(elf), str(binf)])
    size = run([str(GCC_BIN / "arm-none-eabi-size.exe"), str(elf)])
    (work / "size.txt").write_text(size.stdout, encoding="utf-8")
    return binf.exists(), size.stdout


def flash_and_read(binf: Path) -> dict:
    r = run([str(PROG), "-c", "port=SWD", "mode=UR", "-d", str(binf),
             "0x08000000", "-v", "-rst"])
    log = r.stdout + r.stderr
    if "Download verified successfully" not in log and r.returncode != 0:
        return {"flashed": False, "log": log[-1200:]}
    # Poll rather than guess a sleep: a slow encoder can take minutes for
    # 32 runs, and a fixed wait would silently truncate the series.
    words = 6 + N_RUNS
    deadline = time.time() + 600.0
    vals, out = [], ""
    while time.time() < deadline:
        time.sleep(3.0)
        rr = run([str(PROG), "-c", "port=SWD", "mode=HOTPLUG",
                  "-r32", hex(RESULT_ADDR), hex(words * 4)])
        out = rr.stdout + rr.stderr
        vals = []
        for line in out.splitlines():
            m = re.match(r"\s*0x[0-9A-Fa-f]{8}\s*:\s*(.+)$", line)
            if m:
                vals += [int(x, 16) for x in re.findall(r"[0-9A-Fa-f]{8}",
                                                        m.group(1))]
        if len(vals) >= 6 and vals[0] == MAGIC and vals[1] in (6, 99):
            break
    if len(vals) < 6 or vals[0] != MAGIC:
        return {"flashed": True, "ran": False,
                "magic": hex(vals[0]) if vals else None,
                "log": out[-1200:]}
    # stage says how far the firmware got, so a failure points at a line
    # rather than just reporting that nothing came back.
    stage, core_hz, n_runs = vals[1], vals[2], vals[3]
    ai_err = struct.unpack("<i", struct.pack("<I", vals[4]))[0]
    cycles = vals[6:6 + min(n_runs, N_RUNS)]
    return {"flashed": True, "ran": stage == 6, "stage": stage,
            "core_hz": core_hz, "ai_error": ai_err, "cycles": cycles}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ORDER)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "latency.json"
    rows = [] if args.force or not dst.exists() else json.loads(
        dst.read_text())
    done = {r["model"] for r in rows}

    for m in args.models:
        if m in done:
            print(f"[{m}] cached")
            continue
        onnx = S3B / f"deploy_{m}" / "b1_int8.onnx"
        if not onnx.exists():
            print(f"[{m}] no quantised model at {onnx}")
            continue
        work = OUT / m
        gen = generate(onnx, work / "gen")
        rec = {"model": m, "fs": FS_OUT, **gen}
        ok, info = build(work / "gen", work / "build")
        rec["built"] = ok
        if not ok:
            rec["build_error"] = info
            print(f"[{m}] BUILD FAILED\n{info[:400]}")
            rows.append(rec)
            dst.write_text(json.dumps(rows, indent=1))
            continue
        res = flash_and_read(work / "build" / "fw.bin")
        rec.update(res)
        if res.get("ran") and res.get("cycles"):
            c = sorted(res["cycles"])
            hz = res["core_hz"] or 100_000_000
            rec["cycles_median"] = c[len(c) // 2]
            rec["ms_median"] = 1000.0 * c[len(c) // 2] / hz
            rec["ms_min"] = 1000.0 * c[0] / hz
            rec["ms_max"] = 1000.0 * c[-1] / hz
            rec["rtf"] = rec["ms_median"] / (10.0 * 1000.0)
            print(f"[{m}] {rec['ms_median']:.2f} ms median "
                  f"({rec['ms_min']:.2f}-{rec['ms_max']:.2f}), "
                  f"RTF {rec['rtf']:.5f}")
        else:
            print(f"[{m}] did not run: {json.dumps(res)[:300]}")
        rows.append(rec)
        dst.write_text(json.dumps(rows, indent=1))
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
