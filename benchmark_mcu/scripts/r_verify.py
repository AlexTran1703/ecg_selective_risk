r"""Do the trained checkpoints still match the code that would rebuild them?

    .\.venv\Scripts\python.exe benchmark_mcu\scripts\r_verify.py

Every later stage -- quantisation, deployment, scoring -- rebuilds a
model from the width and depth stored in its run record and then loads a
checkpoint into it. That only works while the architecture code is
unchanged. Editing a module mid-run once put twelve folds out of step
with their own stored widths, because the live process kept the module
it had already imported and wrote pre-edit shapes into post-edit
records. The mismatch did not surface until a state_dict load failed
hours later.

Three checks, cheapest first:

  shape    rebuild at the stored (width, depth) and compare parameter
           counts against the record, then actually load the checkpoint
  fit      compare the stored (width, depth) against what the budget
           search returns today, which catches a changed search rather
           than a changed architecture
  budget   confirm the model still meets the budget it was fitted to

A fit difference is not automatically an error -- the search is allowed
to improve -- but every affected run has to be retrained before it can
be deployed alongside the others, so it is reported loudly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "benchmark_mcu"
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(ROOT / "12_leads" / "src"))

from ecgmcu.data import FS_OUT                                 # noqa: E402
from mcubench.footprint import weight_bytes                    # noqa: E402
from mcubench.models import n_params                           # noqa: E402
from mcubench.reference import build, fuse                     # noqa: E402
from mcubench.scaling import BUDGETS, fit                      # noqa: E402
from r1_train import CONFIGS, shape                            # noqa: E402

TRAIN = HERE / "results" / f"recon_{FS_OUT}hz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true",
                    help="also load each checkpoint, not just count shapes")
    args = ap.parse_args()

    files = sorted(TRAIN.glob("*.npz"))
    if not files:
        sys.exit(f"no runs in {TRAIN}")

    bad_shape, bad_fit, over = [], [], []
    cache = {}
    for f in files:
        rec = np.load(f)
        name = str(rec["model"])
        config = str(rec["config"])
        w, d = float(rec["width"]), float(rec["depth"])
        tag = f.stem

        model = fuse(build(name, w, d))
        if n_params(model) != int(rec["params"]):
            bad_shape.append((tag, int(rec["params"]), n_params(model)))
            continue
        if args.load:
            try:
                model.load_state_dict(
                    torch.load(f.with_suffix(".pt"), map_location="cpu"))
            except Exception as exc:                          # noqa: BLE001
                bad_shape.append((tag, "load failed", str(exc)[:60]))
                continue

        method, budget = CONFIGS[config]
        if method is None:
            continue
        # Through the same function training uses, so the two can
        # never disagree about which source of truth is in force.
        if (name, config) not in cache:
            cache[(name, config)] = shape(name, config)
        fw, fd = cache[(name, config)]
        if abs(fw - w) > 1e-9 or abs(fd - d) > 1e-9:
            bad_fit.append((tag, (w, d), (fw, fd)))
        if weight_bytes(model) > BUDGETS[budget]:
            over.append((tag, weight_bytes(model), BUDGETS[budget]))

    print(f"checked {len(files)} runs in {TRAIN.name}")
    print(f"  shape mismatches : {len(bad_shape)}")
    for row in bad_shape:
        print("     ", row)
    print(f"  fit drifted      : {len(bad_fit)}")
    for tag, had, now in bad_fit:
        print(f"      {tag}: stored w={had[0]:.4f} d={had[1]:.4f} "
              f"-> now w={now[0]:.4f} d={now[1]:.4f}")
    print(f"  over budget      : {len(over)}")
    for tag, got, cap in over:
        print(f"      {tag}: {got:,} B > {cap:,} B")

    if bad_shape or bad_fit or over:
        names = sorted({t.split('__')[0] + '__' + t.split('__')[1]
                        for t, *_ in bad_shape + bad_fit + over})
        print("\nretrain these configurations before deploying:")
        for n in names:
            print("   ", n)
        sys.exit(1)
    print("\nall runs consistent with the current code")


if __name__ == "__main__":
    main()
