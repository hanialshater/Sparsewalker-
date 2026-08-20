#!/usr/bin/env python
"""Common-protocol Amazon quality rerun for corrected SparseWalker v1.1.

This is an internal apples-to-apples benchmark, not a paper-protocol reproduction.
Both models use the same temporal leave-two-out split, full-catalog evaluation,
seen-item masking, FullCE objective, max_len=50, BF16 training on supported CUDA,
and validation selection by NDCG@10.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full
from sparsewalker.models import SASRec, SparseWalker
from sparsewalker.training import train_epoch

AMAZON = ("beauty", "video_games", "sports", "toys")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_lr(opt, epoch, max_epochs, peak=1e-3, floor=1e-4, warmup=3):
    if epoch <= warmup:
        lr = peak * epoch / warmup
    else:
        p = (epoch - warmup) / max(1, max_epochs - warmup)
        lr = floor + .5 * (peak - floor) * (1 + math.cos(math.pi * p))
    for g in opt.param_groups: g["lr"] = lr
    return lr


def build(name, n_items, max_len):
    if name == "SASRec":
        return SASRec(n_items, max_len, d=64, layers=2, heads=2, inner=256, dropout=.2)
    if name == "SparseWalker":
        return SparseWalker(n_items, max_len, d=64, layers=2, side=256, h=16,
                            active=8, top_side=2, degree=4, fresh_weight=.25)
    raise KeyError(name)


def run_one(dataset, model_name, args, device):
    seed_all(args.seed)
    data = load_dataset(dataset, args.data_dir)
    split = split_data(data["sequences"])
    max_len = 50
    model = build(model_name, data["n_items"], max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ds = WindowDataset(split["train"], max_len, args.seed)

    out = Path(args.output_dir) / dataset / f"seed{args.seed}" / model_name
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0; best_epoch = 0; best_state = None; bad = 0; history = []

    for epoch in range(1, args.epochs + 1):
        lr = set_lr(opt, epoch, args.epochs, peak=args.lr)
        t0 = time.perf_counter()
        stats = train_epoch(
            model_name, model, ds, opt, device,
            batch_size=args.batch_size, epoch=epoch, loss_mode="full",
            bucket_by_length=True, use_bf16=True, return_stats=True,
        )
        sec = time.perf_counter() - t0
        if epoch == 1 or epoch % args.eval_every == 0:
            val = evaluate_full(model, split["val_prefix"], split["val_target"],
                                data["n_items"], max_len, device, topks=(10,),
                                batch_size=args.eval_batch_size)
            row = {"epoch": epoch, "lr": lr, "train_seconds": sec, **stats, **val}
            history.append(row); print("AMAZON_EVAL", dataset, model_name, row, flush=True)
            ndcg = float(val["NDCG@10"])
            if ndcg > best:
                best = ndcg; best_epoch = epoch; best_state = cpu_state_dict(model); bad = 0
                torch.save({"model": best_state, "epoch": epoch, "val": val}, out / "best.pt")
            else:
                bad += args.eval_every
            (out / "history.json").write_text(json.dumps(history, indent=2))
            if bad >= args.patience: break

    if best_state is None: raise RuntimeError(f"No checkpoint for {dataset}/{model_name}")
    model.load_state_dict(best_state); model.eval()
    test = evaluate_full(model, split["test_prefix"], split["test_target"],
                         data["n_items"], max_len, device, topks=(10,20,50),
                         batch_size=args.eval_batch_size)
    result = {
        "dataset": dataset, "model": model_name, "seed": args.seed,
        "max_len": max_len, "objective": "FullCE", "best_epoch": best_epoch,
        "best_val_NDCG@10": best, "params": sum(p.numel() for p in model.parameters()),
        **test,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("AMAZON_RESULT", json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(AMAZON))
    p.add_argument("--models", default="SASRec,SparseWalker")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_amazon_quality_v11")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.set_float32_matmul_precision("high")
    datasets = [x.strip() for x in a.datasets.split(",") if x.strip()]
    models = [x.strip() for x in a.models.split(",") if x.strip()]
    bad = [x for x in datasets if x not in AMAZON]
    if bad: raise ValueError(f"Unknown Amazon datasets: {bad}")
    rows = []
    for d in datasets:
        for m in models:
            rows.append(run_one(d, m, a, device))
    root = Path(a.output_dir); root.mkdir(parents=True, exist_ok=True)
    (root / f"summary_seed{a.seed}.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    print("AMAZON_SUMMARY", json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__": main()
