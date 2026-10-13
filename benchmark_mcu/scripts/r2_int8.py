r"""int8 post-training quantisation of the forty deployed configurations.

    .\.venv\Scripts\python.exe benchmark_mcu\scripts\r2_int8.py

The reference tier is skipped: it carries no device constraint, exists
to answer what the published designs are worth as designs, and is scored
in float32 only.

Export, calibration and scoring are imported from `b2_int8` rather than
restated, so the two benchmarks cannot drift apart on the one choice
that matters most here -- percentile calibration at 99.99 over 256
records drawn only from each fold's training sources.

One thing this stage checks that the earlier one could not: a dilated
convolution is rejected by the int8 path and ships as float32 at four
bytes per weight (see `mcubench.footprint`). TCN is the only family
affected, and the share of its weights that stay float is recorded per
run so the deployment numbers can be read with that in view.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "benchmark_mcu"
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(ROOT / "12_leads" / "src"))

from b2_int8 import (Reader, export_onnx, macro_auprc,          # noqa: E402
                     macro_auroc, quantize, run_onnx)
from ecgmcu.data import Cohort, SOURCES, FS_OUT                 # noqa: E402
from mcubench.footprint import float_fallback_frac              # noqa: E402
from mcubench.models import n_params                            # noqa: E402
from mcubench.reference import REFERENCE, build, fuse           # noqa: E402
from r1_train import CONFIGS, SLUG                              # noqa: E402

TRAIN = HERE / "results" / f"recon_{FS_OUT}hz"
OUT = HERE / "results" / f"recon_int8_{FS_OUT}hz"
DEPLOYED = [c for c in CONFIGS if c != "REF"]
CALIB_N = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(REFERENCE))
    ap.add_argument("--configs", nargs="+", default=DEPLOYED)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    coh = Cohort()
    rng = np.random.default_rng(args.seed)
    done = 0

    for config in args.configs:
        for name in args.models:
            for src in args.sources:
                tag = f"{SLUG[name]}__{config}__{src}"
                ck, run = TRAIN / f"{tag}.pt", TRAIN / f"{tag}.npz"
                if not (ck.exists() and run.exists()):
                    print(f"[{tag}] not trained, skipped", flush=True)
                    continue
                probp = OUT / f"{tag}_prob.npz"
                if probp.exists() and not args.force:
                    done += 1
                    continue

                d = np.load(run)
                f32p, i8p = OUT / f"{tag}_f32.onnx", OUT / f"{tag}_int8.onnx"

                # Rebuilt from the width and depth the training run
                # recorded, never by re-running the budget fit: any
                # later change to the search would otherwise surface as
                # a state_dict shape mismatch halfway through the job.
                w, dep = float(d["width"]), float(d["depth"])
                model = fuse(build(name, w, dep))
                assert n_params(model) == int(d["params"]), (
                    f"{tag}: rebuilt model has {n_params(model)} "
                    f"parameters, checkpoint has {int(d['params'])}")
                model.load_state_dict(torch.load(ck, map_location="cpu"))
                export_onnx(model, f32p)

                tr, _, te = coh.loso(src, seed=args.seed)
                mean, std = coh.stats(tr)
                calib = rng.choice(tr, min(CALIB_N, tr.size), replace=False)
                quantize(f32p, i8p, Reader(coh.sig_path, calib, mean, std))

                y = d["test_y"].astype(np.int8)
                p32 = run_onnx(f32p, coh.sig_path, te, mean, std)
                p8 = run_onnx(i8p, coh.sig_path, te, mean, std)
                np.savez_compressed(probp, y=y,
                                    prob_f32=p32.astype(np.float32),
                                    prob=p8.astype(np.float32))

                rec = {"model": name, "config": config, "source": src,
                       "width": w, "depth": dep,
                       "params": int(d["params"]),
                       "weight_bytes": int(d["weight_bytes"]),
                       "fp32_frac": float_fallback_frac(model),
                       "onnx_f32_kb": f32p.stat().st_size / 1024,
                       "onnx_int8_kb": i8p.stat().st_size / 1024,
                       "ap_f32": macro_auprc(y, p32),
                       "ap_int8": macro_auprc(y, p8),
                       "auroc_f32": macro_auroc(y, p32),
                       "auroc_int8": macro_auroc(y, p8)}
                rec["d_ap"] = rec["ap_int8"] - rec["ap_f32"]
                (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=1))
                done += 1
                print(f"[{tag}] f32 {rec['ap_f32']:.4f} -> int8 "
                      f"{rec['ap_int8']:.4f} ({rec['d_ap']:+.4f})  "
                      f"{rec['onnx_int8_kb']:.1f} KB", flush=True)

    # Rebuilt from every record on disk rather than from this run's
    # rows, so a scoped --models run cannot truncate the aggregate.
    allrec = []
    for f in sorted(OUT.glob("*__*__*.json")):
        try:
            allrec.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    (OUT / "int8.json").write_text(json.dumps(allrec, indent=1))
    print(f"\n-> {OUT}  ({len(allrec)} records, {done} this run)", flush=True)


if __name__ == "__main__":
    main()
