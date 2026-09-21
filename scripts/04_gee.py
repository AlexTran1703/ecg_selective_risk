r"""Modified-Poisson GEE for the disagreement-error association.

    .\.venv\Scripts\python.exe scripts\04_gee.py

The bootstrap in 03_analyse.py gives an unadjusted risk ratio.  This adds the
adjusted estimate the Letter reports: does disagreement still mark error once
the diagnosis is accounted for, or is the whole effect that disagreement and
error both concentrate in 1dAVb?

Poisson working model with a log link on a binary outcome, so the coefficient
is a risk ratio rather than an odds ratio -- with 27% error in the disputed
stratum an odds ratio would badly overstate the risk ratio.  Poisson variance
is wrong for a binary outcome, hence "modified": robust sandwich standard
errors clustered on ECG repair the inference, and the same clustering handles
the six correlated labels contributed by each tracing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecguq.data import LABELS                                     # noqa: E402
from ecguq.uncertainty import (ensemble_mean, predict,            # noqa: E402
                               reader_disagreement)


def main() -> None:
    e = np.load(ROOT / "data" / "ensembles" / "code15" / "ensemble.npz")
    m = np.load(ROOT / "data" / "prepared" / "code_test" / "meta.npz")
    pbar = ensemble_mean(e["test"])
    y = m["y"].astype(np.int8)
    err = (predict(pbar, e["thresholds"]) != y).astype(int)
    dis = reader_disagreement(m["reader1"], m["reader2"]).astype(int)
    n, C = y.shape

    df = pd.DataFrame({
        "ecg": np.repeat(np.arange(n), C),
        "dx": np.tile(np.asarray(LABELS), n),
        "error": err.ravel(),
        "disputed": dis.ravel(),
    })
    print(f"{len(df)} diagnosis-record rows, {df.ecg.nunique()} ECGs, "
          f"{int(df.disputed.sum())} disputed, {int(df.error.sum())} errors\n")

    # Reference diagnosis is the one with most disputes, so the intercept is
    # not defined on a class with almost no events.
    base = df.groupby("dx").disputed.sum().idxmax()
    dx = pd.Categorical(df.dx, categories=[base] +
                        [d for d in LABELS if d != base])
    X = pd.get_dummies(dx, prefix="dx", drop_first=True).astype(float)
    X.insert(0, "disputed", df.disputed.to_numpy(float))
    X = sm.add_constant(X)

    fits = {
        "unadjusted": sm.add_constant(
            pd.DataFrame({"disputed": df.disputed.to_numpy(float)})),
        f"adjusted for diagnosis (ref {base})": X,
    }
    for name, design in fits.items():
        res = sm.GEE(df.error.to_numpy(float), design.to_numpy(float),
                     groups=df.ecg.to_numpy(),
                     family=sm.families.Poisson(),
                     cov_struct=sm.cov_struct.Independence()).fit()
        i = list(design.columns).index("disputed")
        rr = float(np.exp(res.params[i]))
        lo, hi = np.exp(res.conf_int()[i])
        print(f"{name}")
        print(f"    risk ratio {rr:.2f}  (95% CI {lo:.2f} to {hi:.2f})  "
              f"p = {res.pvalues[i]:.2e}")
    print("\nPoisson working model, log link, robust sandwich SE clustered "
          "on ECG\n(modified-Poisson); risk ratios, not odds ratios.")


if __name__ == "__main__":
    main()
