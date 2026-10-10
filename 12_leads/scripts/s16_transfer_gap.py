r"""Is there a transfer penalty, or only heterogeneity?

    .\.venv\Scripts\python.exe 12_leads\scripts\s16_transfer_gap.py

RQ1 shows external performance differs sharply by held-out source. That
is heterogeneity, not a measured transfer gap: a source could score low
externally simply because its diagnoses are harder there, and a model
trained with that source represented would score just as low. Nothing
reported so far separates the two, which is why the paper says
"heterogeneity" throughout rather than "degradation".

This measures the contrast directly. For each source s:

    s is split patient-disjointly into s_dev and s_test

    external  : trained on the other three sources, scored on s_test.
                These are the cached LOSO predictions subset to s_test,
                so the arm costs nothing and is literally the same model
                every other result in the paper describes.

    reference : trained on the other three sources with s_dev swapped in,
                scored on the same s_test records.

    delta(s)  = AP(reference, s) - AP(external, s)

The training budget is matched, which is the part that makes the
contrast mean anything. The reference does not simply get more data:
|s_dev| rows are dropped from the other-three pool before s_dev is added,
so both arms train on the same number of records, for the same epochs,
with the same optimiser, schedule and seed. What differs is only whether
the target source is represented at all.

delta is a measured performance contrast under this protocol. It is not
a causal estimate of distribution shift, and a positive delta does not
say which of acquisition, population or labelling produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

from ecgmcu.data import Cohort, Feeder, SOURCES, FS_OUT, PREP    # noqa: E402
from ecgmcu.models import build                                  # noqa: E402
from s2_train import (macro_auprc, macro_auroc, pos_weight,      # noqa: E402
                      predict)

TAG = "" if FS_OUT == 250 else f"_{FS_OUT}hz"
S2D = HERE / "results" / f"study2{TAG}"
OUT = HERE / "results" / f"study16{TAG}"
ORDER = ["tiny", "dscnn", "mobilenet", "mbconv", "tcn", "resnet"]
TEST_FRAC = 0.5          # half the target held back, shared by both arms


def patient_ids(coh):
    """Patient label per prepared row, joined from the raw record table."""
    idx = pd.read_csv(PREP / "index.csv")
    meta = pd.read_csv(HERE.parent / "data" / "meta" / "records.csv",
                       usecols=["source", "record", "patient"])
    j = idx.merge(meta, on=["source", "record"], how="left")
    assert len(j) == len(idx), "join changed the row count"
    pid = j["patient"].astype("string")
    # A record with no patient id is treated as its own patient, which is
    # the conservative choice: it can only split rows apart, never merge
    # two patients into one.
    miss = pid.isna()
    pid[miss] = ("row:" + j.index.to_series().astype(str))[miss]
    return pid.to_numpy()


def split_target(coh, pid, src, seed=0):
    """Patient-disjoint dev/test split of one source."""
    rows = np.flatnonzero(coh.source == src)
    pats = np.unique(pid[rows])
    rng = np.random.default_rng(seed)
    pats = pats[rng.permutation(pats.size)]
    cut = int(round(TEST_FRAC * pats.size))
    test_p = set(pats[:cut].tolist())
    in_test = np.array([p in test_p for p in pid[rows]])
    return rows[~in_test], rows[in_test]          # dev, test


def train_reference(name, coh, tr, va, te, args, device):
    mean, std = coh.stats(tr)

    def feed(rows, shuffle, seed=0):
        return Feeder(coh.sig_path, coh.y, rows, args.batch_size, mean,
                      std, device, shuffle=shuffle, seed=seed)

    f_tr = feed(tr, True, args.seed)
    f_va = feed(va, False)
    f_te = feed(te, False)
    y_va, y_te = f_va.labels(), f_te.labels()

    torch.manual_seed(args.seed)
    model = build(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(f_tr)))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight(coh.y[tr]).to(device))

    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        for x, y in f_tr:
            with torch.autocast("cuda", torch.bfloat16,
                                enabled=device == "cuda"):
                loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        score = macro_auprc(y_va, predict(model, f_va, device))
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(model, f_te, device), y_te, best


def paired_boot(y, p_ext, p_ref, reps, seed=0):
    """Bootstrap the difference over the shared test records."""
    rng = np.random.default_rng(seed)
    d = np.empty(reps)
    n = y.shape[0]
    for b in range(reps):
        i = rng.integers(0, n, n)
        d[b] = macro_auprc(y[i], p_ref[i]) - macro_auprc(y[i], p_ext[i])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ORDER)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "transfer_gap.json"
    if dst.exists() and not args.force:
        print(f"cached: {dst}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    coh = Cohort()
    pid = patient_ids(coh)
    rows_out = []
    means = []

    for src in args.sources:
        # Per-source prediction store, so the mean gap over encoders can
        # be bootstrapped on the shared records rather than assembled
        # from six marginal intervals, which would not be a CI on a mean.
        store = {}
        dev, test = split_target(coh, pid, src, seed=args.seed)
        pool_all = np.flatnonzero(coh.source != src)
        print(f"\n=== {src}: dev {dev.size}, test {test.size}, "
              f"other-three pool {pool_all.size} ===", flush=True)

        for name in args.models:
            f = S2D / f"{name}__{src}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            pos = {int(r): i for i, r in enumerate(d["test_rows"])}
            take = np.array([pos[int(r)] for r in test])
            p_ext = d["test_prob"][take].astype(np.float64)
            y_te = d["test_y"][take].astype(np.int8)

            # Budget-matched reference pool: drop |dev| rows from the
            # other three, then add dev, so both arms see the same count.
            rng = np.random.default_rng(args.seed)
            keep = rng.permutation(pool_all.size)[:max(pool_all.size
                                                       - dev.size, 1)]
            pool = np.concatenate([pool_all[keep], dev])
            pool = pool[rng.permutation(pool.size)]
            cut = int(round(0.1 * pool.size))
            va, tr = pool[:cut], pool[cut:]
            assert not np.intersect1d(tr, test).size, "target test leaked"
            assert not np.intersect1d(va, test).size, "target test leaked"

            t0 = time.time()
            p_ref, y_ref, val = train_reference(name, coh, tr, va, test,
                                                args, device)
            assert np.array_equal(y_ref, y_te), "test labels disagree"
            p_ref = p_ref.astype(np.float64)
            ap_e = macro_auprc(y_te, p_ext)
            ap_r = macro_auprc(y_te, p_ref)
            lo, hi = paired_boot(y_te, p_ext, p_ref, args.reps, args.seed)
            rows_out.append({
                "source": src, "model": name,
                "n_test": int(test.size), "n_dev": int(dev.size),
                "n_train": int(tr.size),
                "ap_external": ap_e, "ap_reference": ap_r,
                "delta": ap_r - ap_e, "lo": lo, "hi": hi,
                "auroc_external": macro_auroc(y_te, p_ext),
                "auroc_reference": macro_auroc(y_te, p_ref),
                "val_auprc": val})
            store[name] = (y_te, p_ext, p_ref)
            np.savez_compressed(OUT / f"{name}__{src}_pred.npz",
                                y=y_te.astype(np.int8),
                                p_external=p_ext.astype(np.float32),
                                p_reference=p_ref.astype(np.float32))
            print(f"  [{name:10s}] external {ap_e:.4f}  reference "
                  f"{ap_r:.4f}  delta {ap_r - ap_e:+.4f} "
                  f"({lo:+.4f}, {hi:+.4f})  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)

        # CI on the mean gap across encoders: one resample of the shared
        # test records, every encoder re-scored on it, then averaged.
        if store:
            rng = np.random.default_rng(args.seed)
            names = list(store)
            n = store[names[0]][0].shape[0]
            dm = np.empty(args.reps)
            for b in range(args.reps):
                i = rng.integers(0, n, n)
                dm[b] = np.mean([
                    macro_auprc(store[k][0][i], store[k][2][i])
                    - macro_auprc(store[k][0][i], store[k][1][i])
                    for k in names])
            means.append({
                "source": src,
                "mean_delta": float(np.mean([r["delta"] for r in rows_out
                                             if r["source"] == src])),
                "mean_lo": float(np.percentile(dm, 2.5)),
                "mean_hi": float(np.percentile(dm, 97.5))})
            m = means[-1]
            print(f"  mean over encoders {m['mean_delta']:+.4f} "
                  f"({m['mean_lo']:+.4f}, {m['mean_hi']:+.4f})", flush=True)

    dst.write_text(json.dumps({"cells": rows_out, "means": means}, indent=1))
    t = pd.DataFrame(rows_out)
    print("\n=== transfer gap by source, mean over encoders ===")
    print(t.groupby("source")[["ap_external", "ap_reference", "delta"]]
          .mean().to_string(float_format=lambda v: f"{v:+.4f}"))
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
