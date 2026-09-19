"""Train one pipeline for one leave-one-source-out experiment.

    python scripts/03_train.py --representation raw --target PTBXL

Network weights see only the three non-target sources.  The target source is
split 20/80 into a calibration and a test subset, and neither is used here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgsr.datasets import (ALL_REPRESENTATIONS, SOURCES,            # noqa: E402
                            load_index, make_split)
from ecgsr.models import BACKBONES_1D, DEFAULT_BACKBONE_1D          # noqa: E402
from ecgsr.train import TrainConfig, train_pipeline                 # noqa: E402


def require_amplitude_qc(cache_dir: Path) -> None:
    """Refuse to train unless every source passed the physical-unit QC.

    A mis-scaled source trains perfectly happily and only shows up as
    inexplicably poor cross-source transfer, so the check is a gate rather than
    a warning.
    """
    qc_path = Path(cache_dir) / "amplitude_qc.json"
    if not qc_path.exists():
        raise SystemExit(
            f"missing {qc_path}. Re-run scripts/02_cache_signals.py: signals must "
            "pass source-wise amplitude QC before any model is trained.")
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    if not qc.get("all_passed"):
        bad = [s for s, e in qc["sources"].items() if not e["passed"]]
        raise SystemExit(f"amplitude QC failed for {', '.join(bad)}; refusing to "
                         "train on signals that are not in physical mV.")
    medians = ", ".join(f"{s} {e['median']:.2f}"
                        for s, e in sorted(qc["sources"].items()))
    print(f"  amplitude QC passed (median limb QRS p-p mV: {medians})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--representation", choices=list(ALL_REPRESENTATIONS),
                    required=True)
    ap.add_argument("--target", choices=list(SOURCES), required=True)
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "cache" / "signals.npy")
    ap.add_argument("--meta", type=Path, default=ROOT / "data" / "meta")
    ap.add_argument("--runs", type=Path, default=ROOT / "artifacts" / "runs")
    ap.add_argument("--backbone", choices=sorted(BACKBONES_1D),
                    default=DEFAULT_BACKBONE_1D,
                    help="1-D network for raw/vcg; ignored for stft")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-sources", nargs="*", default=None,
                    help="restrict training to these sources (inner score "
                         "selection); default is all three non-target sources")
    ap.add_argument("--run-name", default=None,
                    help="output directory name; default <representation>__<target>")
    ap.add_argument("--limit-train", type=int, default=0,
                    help="debug only: subsample the training split")
    args = ap.parse_args()

    require_amplitude_qc(args.cache.parent)

    index = load_index(args.meta)
    split = make_split(index, args.target, seed=args.seed,
                       train_sources=tuple(args.train_sources)
                       if args.train_sources else None)

    if args.limit_train:
        rng = np.random.default_rng(args.seed)
        split = type(split)(
            target=split.target,
            train=rng.choice(split.train, min(args.limit_train, len(split.train)), replace=False),
            dev=rng.choice(split.dev, min(args.limit_train // 5 or 1, len(split.dev)), replace=False),
            cal=split.cal, test=split.test)

    cfg = TrainConfig(representation=args.representation, backbone=args.backbone,
                      epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr,
                      num_workers=args.workers, seed=args.seed)

    run_dir = args.runs / (args.run_name or
                           f"{args.representation}__{args.target}")
    run_dir.mkdir(parents=True, exist_ok=True)

    model_name = "resnet2d18" if args.representation == "stft" else args.backbone
    print(f"=== {args.representation.upper()} | {model_name} | target = {args.target} ===")
    print(f"  classes : {index.n_classes}")
    train_src = (tuple(args.train_sources) if args.train_sources
                 else tuple(s for s in SOURCES if s != args.target))
    print(f"  train   : {len(split.train):6d}  (sources: {', '.join(train_src)})")
    print(f"  dev     : {len(split.dev):6d}")
    print(f"  cal     : {len(split.cal):6d}  (target, thresholds only)")
    print(f"  test    : {len(split.test):6d}  (target, touched once)")

    np.savez(run_dir / "split.npz", train=split.train, dev=split.dev,
             cal=split.cal, test=split.test, target=np.array(split.target))

    result = train_pipeline(args.cache, index, split, cfg, run_dir)
    (run_dir / "train_result.json").write_text(
        json.dumps({**result, "config": asdict(cfg), "target": args.target},
                   indent=2), encoding="utf-8")
    print(f"  wrote {run_dir}")


if __name__ == "__main__":
    main()
