r"""Re-score the cached LOSO checkpoints in true float32.

    .\.venv\Scripts\python.exe 12_leads\scripts\s2d_repredict_fp32.py

Training used bfloat16 autocast, and so did the prediction pass that
produced the cached probabilities. That is fine for training but it makes
Study II a bfloat16 evaluation, which then sits oddly beside Study III,
where the float32 reference for quantisation is computed at full
precision. The two differ by a few thousandths -- small, but the same
order as the int8 penalty being reported, so the comparison stops meaning
anything.

Nothing is retrained. The saved weights are re-run in float32 and only
`test_prob` is replaced; labels and row indices are carried through
untouched, and the previous array is kept as `test_prob_bf16` so the
change is auditable rather than silent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from ecgmcu.data import Cohort, Feeder, SOURCES, FS_OUT        # noqa: E402
from ecgmcu.models import ZOO, build                           # noqa: E402

S2 = HERE / "results" / ("study2" if FS_OUT == 250
                         else f"study2_{FS_OUT}hz")


def macro_auprc(y, p):
    from sklearn.metrics import average_precision_score
    v = [average_precision_score(y[:, c], p[:, c])
         for c in range(y.shape[1]) if 0 < y[:, c].sum() < len(y)]
    return float(np.mean(v)) if v else float("nan")


@torch.no_grad()
def predict_fp32(model, feeder, device):
    model.eval()
    out = []
    for x, _ in feeder:
        out.append(torch.sigmoid(model(x.float())).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 13))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ZOO))
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    coh = Cohort()
    print(f"{FS_OUT} Hz cohort, {coh.y.shape[0]:,} records, {device}")

    for name in args.models:
        for src in args.sources:
            npz = S2 / f"{name}__{src}.npz"
            ckpt = S2 / f"{name}__{src}.pt"
            if not (npz.exists() and ckpt.exists()):
                continue
            d = dict(np.load(npz, allow_pickle=True))
            if "test_prob_bf16" in d:
                print(f"[{name}__{src}] already float32")
                continue
            tr, _, te = coh.loso(src, seed=0)
            mean, std = coh.stats(tr)
            feeder = Feeder(coh.sig_path, coh.y, te, args.batch_size,
                            mean, std, device, shuffle=False)
            model = build(name).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            p32 = predict_fp32(model, feeder, device)

            old, y = d["test_prob"], d["test_y"]
            if p32.shape != old.shape:
                print(f"[{name}__{src}] shape mismatch, skipped")
                continue
            a_old, a_new = macro_auprc(y, old), macro_auprc(y, p32)
            d["test_prob_bf16"] = old
            d["test_prob"] = p32.astype(np.float32)
            np.savez_compressed(npz, **d)
            print(f"[{name}__{src}] bf16 {a_old:.4f} -> fp32 {a_new:.4f}"
                  f"  ({a_new - a_old:+.4f})", flush=True)


if __name__ == "__main__":
    main()
