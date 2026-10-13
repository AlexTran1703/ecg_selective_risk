"""Validate the analytic predictor against the 20 measured deployments."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import torch                                                  # noqa: E402
from mcubench.models import ZOO                               # noqa: E402
from mcubench.footprint import weight_bytes                   # noqa: E402

TRAIN = HERE / "results" / "train_100hz"
DEP = HERE / "results" / "deploy_100hz"

print("{:<14} {:<9} {:>10} {:>10} {:>8}".format(
    "model", "budget", "measured", "predicted", "err %"))
errs = []
for f in sorted(DEP.glob("*.json")):
    if f.stem == "deploy":
        continue
    rec = json.load(open(f))
    meas = rec.get("weights_b")
    if not meas:
        continue
    key, budget = f.stem.split("__")
    ck = sorted(TRAIN.glob(f"{key}__{budget}__*.pt"))
    if not ck:
        print(f"{key:<14} {budget:<9}  no checkpoint")
        continue
    st = torch.load(ck[0], map_location="cpu", weights_only=False)
    w = st.get("width", st.get("w"))
    model = ZOO[key](w) if w is not None else ZOO[key]()
    if hasattr(model, "reparameterise"):
        model.reparameterise()
    pred = weight_bytes(model)
    e = 100 * (pred - meas) / meas
    errs.append(abs(e))
    print("{:<14} {:<9} {:>10,} {:>10,} {:>+8.1f}".format(
        key, budget, meas, pred, e))
if errs:
    print("\nmax |err| {:.1f}%   mean |err| {:.1f}%".format(
        max(errs), sum(errs) / len(errs)))
