r"""Study III (static half): what each encoder costs on the STM32F411.

    FLOAT32 as exported. Quantised int8 figures come after Study II,
    because --quantize needs a calibration set from trained weights.

    .\.venv\Scripts\python.exe 12_leads\scripts\s3_analyze.py [--rates 500 250 125]

Exports every architecture to ONNX at each candidate input rate and runs the
ST Edge AI analyser against target stm32. The analyser's numbers are
authoritative -- it performs the real memory allocation -- whereas a
parameter count or a largest-tensor estimate is only a proxy.

"Does not fit" is recorded as a result, not quietly dropped. A model that
exceeds SRAM at 500 Hz but fits at 125 Hz is the central trade-off this
study is supposed to expose, and deleting the failures would hide it.

The input buffer is counted explicitly. A 10 s 12-lead record at 500 Hz is
60,000 samples -- 58.6 KB as int8, 234 KB as float32 -- on a part with
128 KB of SRAM in total, so the representation decides feasibility before
any architecture does.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]   # 12_leads/
ROOT = HERE.parent                           # repo root: shared data
sys.path.insert(0, str(HERE / "src"))

from ecgmcu.models import ZOO, build, n_params              # noqa: E402

STEDGEAI = Path("C:/Users/Alith/STM32Cube/Repository/Packs/STMicroelectronics"
                "/X-CUBE-AI/10.2.0/Utilities/windows/stedgeai.exe")
from ecgmcu.data import FS_OUT                               # noqa: E402

OUT = HERE / "results" / ("study3" if FS_OUT == 250
                         else f"study3_{FS_OUT}hz")
SRAM_KB, FLASH_KB = 128, 512          # STM32F411xC/E
N_LEADS = 12


def export(name: str, length: int, path: Path) -> None:
    m = build(name).eval()
    torch.onnx.export(m, torch.zeros(1, N_LEADS, length), str(path),
                      input_names=["input"], output_names=["output"],
                      opset_version=13, dynamo=False)


def analyse(onnx: Path, workdir: Path) -> dict:
    """Run the ST analyser and pull out the three numbers that matter."""
    r = subprocess.run(
        [str(STEDGEAI), "analyze", "--model", str(onnx), "--target", "stm32",
         "--allocate-inputs", "--allocate-outputs", "--verbosity", "1",
         "--workspace", str(workdir / "ws"), "--output", str(workdir)],
        capture_output=True, text=True)
    txt = r.stdout + r.stderr

    def grab(pat, cast=int):
        m = re.search(pat, txt)
        return cast(m.group(1).replace(",", "")) if m else None

    return {
        "macc": grab(r"macc\s*:\s*([\d,]+)"),
        "weights_b": grab(r"weights \(ro\)\s*:\s*([\d,]+)\s*B"),
        "activations_b": grab(r"activations \(rw\)\s*:\s*([\d,]+)\s*B"),
        "ok": r.returncode == 0,
        "error": None if r.returncode == 0 else txt.strip()[-300:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=int, nargs="+", default=[500, 250, 125])
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    assert STEDGEAI.exists(), f"ST Edge AI not found at {STEDGEAI}"

    rows = []
    for fs in args.rates:
        length = int(fs * args.seconds)
        inp_kb = N_LEADS * length * 4 / 1024      # float32 input buffer
        for name in ZOO:
            tag = f"{name}_{fs}hz"
            onnx = OUT / f"{tag}.onnx"
            export(name, length, onnx)
            a = analyse(onnx, OUT / tag)
            if a["activations_b"] is None:
                rows.append({"model": name, "fs": fs, "length": length,
                             "params": n_params(build(name)),
                             "fits": False, "note": "analyser failed"})
                print(f"  {tag:<18} analyser failed")
                continue
            act_kb = a["activations_b"] / 1024
            flash_kb = a["weights_b"] / 1024
            total_kb = act_kb + inp_kb
            fits = total_kb <= SRAM_KB and flash_kb <= FLASH_KB
            rows.append({
                "model": name, "fs": fs, "length": length,
                "params": n_params(build(name)), "macc": a["macc"],
                "flash_kb": round(flash_kb, 1),
                "act_kb": round(act_kb, 1), "input_kb": round(inp_kb, 1),
                "total_sram_kb": round(total_kb, 1), "fits": bool(fits),
            })
            print(f"  {tag:<18} flash {flash_kb:6.1f} KB   act {act_kb:6.1f}"
                  f" + in {inp_kb:5.1f} = {total_kb:6.1f} KB"
                  f"   {'fits' if fits else 'EXCEEDS SRAM'}", flush=True)

    (OUT / "deployment.json").write_text(json.dumps(rows, indent=2),
                                         encoding="utf-8")

    hdr = (f"{'model':<11}{'fs':>5}{'params':>9}{'MACC':>12}{'flash':>9}"
           f"{'act':>8}{'input':>8}{'SRAM':>8}  fit")
    lines = ["STUDY III. STM32F411 STATIC DEPLOYMENT COST", "=" * 78, "",
             f"Target STM32F411xC/E: {SRAM_KB} KB SRAM, {FLASH_KB} KB Flash.",
             "FLOAT32 weights and activations as exported; KB = 1024 B.",
             "int8 figures require quantisation with a calibration set,",
             "which needs the trained models from Study II.",
             "Input buffer counted as 12 leads x 10 s float32.", "", hdr,
             "-" * len(hdr)]
    for r in rows:
        if "flash_kb" not in r:
            continue
        lines.append(
            f"{r['model']:<11}{r['fs']:>5}{r['params']:>9,}{r['macc']:>12,}"
            f"{r['flash_kb']:>9.1f}{r['act_kb']:>8.1f}{r['input_kb']:>8.1f}"
            f"{r['total_sram_kb']:>8.1f}  {'yes' if r['fits'] else 'NO'}")
    (OUT / "study3.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
