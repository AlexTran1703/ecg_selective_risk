"""Run every (representation, held-out source) pipeline, then evaluate each.

    python scripts/05_run_all.py

Resumable: a pipeline whose ``eval.json`` already exists is skipped, so the
sweep can be interrupted and restarted without losing finished runs.  Each
pipeline is launched as its own process so a failure in one cannot take the
whole sweep down, and so GPU memory is fully released between runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

REPRESENTATIONS = ("raw", "vcg", "fft")
TARGETS = ("PTBXL", "Georgia", "Chapman", "Ningbo")


def run(cmd: list[str]) -> int:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "artifacts" / "runs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--representations", nargs="*", default=list(REPRESENTATIONS))
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = [(r, t) for r in args.representations for t in args.targets]
    print(f"{len(jobs)} pipelines: {args.representations} x {args.targets}\n")

    t_start = time.time()
    failed = []
    for i, (rep, target) in enumerate(jobs, start=1):
        run_dir = args.runs / f"{rep}__{target}"
        elapsed = (time.time() - t_start) / 60
        print(f"[{i}/{len(jobs)}] {rep} -> {target}   (elapsed {elapsed:.0f} min)",
              flush=True)

        if (run_dir / "eval.json").exists() and not args.force:
            print("  already evaluated, skipping\n", flush=True)
            continue

        # One batch size for every representation: Raw, VCG and FFT share the
        # same backbone and optimisation settings, so only the input differs.
        bs = args.batch_size
        if not (run_dir / "model.pt").exists() or args.force:
            rc = run([PY, ROOT / "scripts" / "03_train.py",
                      "--representation", rep, "--target", target,
                      "--epochs", args.epochs, "--batch-size", bs,
                      "--workers", args.workers, "--runs", args.runs])
            if rc != 0:
                print(f"  TRAINING FAILED (exit {rc})\n", flush=True)
                failed.append((rep, target, "train"))
                continue
        else:
            print("  checkpoint exists, reusing", flush=True)

        rc = run([PY, ROOT / "scripts" / "04_calibrate_eval.py",
                  "--run", run_dir, "--workers", args.workers])
        if rc != 0:
            print(f"  EVALUATION FAILED (exit {rc})", flush=True)
            failed.append((rep, target, "eval"))
        print(flush=True)

    print(f"Sweep finished in {(time.time() - t_start) / 60:.0f} min.")
    if failed:
        print(f"{len(failed)} pipeline(s) failed:")
        for rep, target, stage in failed:
            print(f"  {rep} -> {target} ({stage})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
