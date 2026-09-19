"""Training loop for one (representation, model, held-out source) pipeline.

Identical optimisation settings across all pipelines -- the study is about
representations and selective risk, so nothing here is tuned per pipeline.
Multi-label BCE with per-class positive weighting; never softmax, since a
record may carry several simultaneous diagnoses.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import ECGCacheDataset, RecordIndex, Representation, Split
from .metrics import macro_auprc, macro_auroc
from .models import DEFAULT_BACKBONE_1D, build_model


@dataclass
class TrainConfig:
    representation: str = "raw"
    backbone: str = DEFAULT_BACKBONE_1D
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_frac: float = 0.05
    num_workers: int = 6
    amp: bool = True
    # bfloat16 by default: float16 autocast overflowed the deeper xresnet1d101
    # activations late in training (loss was still falling, then the forward pass
    # produced NaN).  bf16 has fp32's exponent range, so it cannot overflow, and
    # Ada-class GPUs run it at the same speed.  GradScaler is only needed for fp16.
    amp_dtype: str = "bfloat16"
    pos_weight_cap: float = 20.0
    time_shift: int = 0
    seed: int = 0
    grad_clip: float = 5.0


def amp_dtype_for(cfg: TrainConfig, device: torch.device) -> torch.dtype:
    """Resolve the autocast dtype, falling back to fp16 if bf16 is unsupported."""
    if cfg.amp_dtype == "bfloat16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float16


def positive_weights(y: np.ndarray, cap: float) -> torch.Tensor:
    """``w_k = n_neg / n_pos``, capped so the rarest classes cannot dominate."""
    pos = y.sum(axis=0).astype(np.float64)
    neg = len(y) - pos
    w = np.divide(neg, pos, out=np.ones_like(pos), where=pos > 0)
    return torch.tensor(np.clip(w, 1.0, cap), dtype=torch.float32)


def make_loaders(cache: Path, index: RecordIndex, split: Split, cfg: TrainConfig
                 ) -> dict[str, DataLoader]:
    common = dict(num_workers=cfg.num_workers, pin_memory=True,
                  persistent_workers=cfg.num_workers > 0)
    loaders = {}
    for name, rows, shuffle, shift in (
            ("train", split.train, True, cfg.time_shift),
            ("dev", split.dev, False, 0),
            ("cal", split.cal, False, 0),
            ("test", split.test, False, 0)):
        ds = ECGCacheDataset(cache, rows, index.y[rows], time_shift=shift,
                             rng_seed=cfg.seed)
        loaders[name] = DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                                   drop_last=shuffle and len(rows) > cfg.batch_size,
                                   **common)
    return loaders


@torch.no_grad()
def predict(model: nn.Module, rep: Representation, loader: DataLoader,
            device: torch.device, amp: bool = True,
            dtype: torch.dtype = torch.bfloat16) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probabilities, labels)`` for a whole loader."""
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype,
                            enabled=amp and device.type == "cuda"):
            logits = model(rep(x))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def train_pipeline(cache: Path, index: RecordIndex, split: Split,
                   cfg: TrainConfig, out_dir: Path,
                   device: torch.device | None = None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    loaders = make_loaders(cache, index, split, cfg)

    # Standardisation statistics come from the training split only.
    rep = Representation(cfg.representation).to(device)
    stats_loader = DataLoader(loaders["train"].dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers)
    rep.fit(stats_loader, device)
    del stats_loader

    model = build_model(cfg.representation, index.n_classes,
                        backbone=cfg.backbone).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weights(index.y[split.train], cfg.pos_weight_cap).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps = max(1, cfg.epochs * len(loaders["train"]))
    warmup = max(1, int(cfg.warmup_frac * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / warmup if s < warmup
        else 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, steps - warmup)))))
    autocast_dtype = amp_dtype_for(cfg, device)
    use_amp = cfg.amp and device.type == "cuda"
    # Gradient scaling exists to stop fp16 gradients underflowing; bf16 does not
    # need it and scaling there only adds a failure mode.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and autocast_dtype is torch.float16)
    print(f"  autocast: {'off' if not use_amp else str(autocast_dtype).split('.')[-1]}")

    history, best = [], {"dev_auroc": -np.inf, "epoch": -1}
    ckpt_path = out_dir / "model.pt"
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        running, seen, nonfinite = 0.0, 0, 0
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                loss = criterion(model(rep(x)), y)
            if not torch.isfinite(loss):
                # Skip rather than poison every weight with NaN; a persistent
                # problem shows up as a rising skip count, not a silent ruin.
                nonfinite += 1
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * len(x)
            seen += len(x)

        dev_probs, dev_y = predict(model, rep, loaders["dev"], device, cfg.amp,
                                   autocast_dtype)
        auroc = macro_auroc(dev_y, dev_probs)
        auprc = macro_auprc(dev_y, dev_probs)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1),
                        "dev_auroc": auroc, "dev_auprc": auprc,
                        "lr": sched.get_last_lr()[0],
                        "nonfinite_batches": nonfinite,
                        "minutes": (time.time() - t0) / 60})
        star = ""
        if auroc > best["dev_auroc"]:
            best = {"dev_auroc": auroc, "dev_auprc": auprc, "epoch": epoch}
            torch.save({"model": model.state_dict(),
                        "rep_mean": rep.mean.cpu(), "rep_std": rep.std.cpu(),
                        "config": asdict(cfg), "target": split.target,
                        "classes": index.classes}, ckpt_path)
            star = "  *"
        print(f"  epoch {epoch:2d}  loss {running / max(seen, 1):.4f}  "
              f"dev AUROC {auroc:.4f}  AUPRC {auprc:.4f}"
              f"  [{(time.time() - t0) / 60:.1f} min]{star}"
              + (f"  [{nonfinite} non-finite batches skipped]" if nonfinite else ""),
              flush=True)

    (out_dir / "history.json").write_text(
        json.dumps({"history": history, "best": best, "config": asdict(cfg),
                    "target": split.target,
                    "n": {k: int(len(v.dataset)) for k, v in loaders.items()}},
                   indent=2), encoding="utf-8")
    print(f"  best epoch {best['epoch']} (dev AUROC {best['dev_auroc']:.4f})")
    return {"best": best, "checkpoint": str(ckpt_path)}
