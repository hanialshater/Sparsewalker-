#!/usr/bin/env python
"""Canonical ML-1M control for SparseWalker v1.1 recurrence fixes.

Changes under test versus the previous Walker:
1. fresh item mass is injected exactly once per event, not once per graph hop;
2. duplicate concept IDs are coalesced (mass summed) before every TopK prune;
3. pursuit rewiring is disabled completely.

Everything else intentionally stays aligned with the existing canonical ML-1M
Walker run: same data, split, full-catalog evaluator, max_len=200, K=8,
degree=4, two graph hops, FullCE, AdamW, BF16 training, and length-bucketed batches.
"""
import argparse
import hashlib
import inspect
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full
from sparsewalker.models import SparseWalker
from sparsewalker.training import train_epoch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def protocol_manifest(max_len, n_items):
    source = inspect.getsource(split_data) + "\n" + inspect.getsource(evaluate_full)
    source_hash = hashlib.sha256(source.encode()).hexdigest()[:16]
    manifest = {
        "protocol_version": "EVAL_CANONICAL_v1_candidate",
        "dataset": "ml1m",
        "split": "per-user leave-two-out: train=s[:-2], val=s[-2], test=s[-1]",
        "catalog": "all mapped item ids 1..n_items",
        "seen_item_masking": True,
        "validation_selection": "best validation full-catalog NDCG@10",
        "metrics": ["HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50", "MRR@10"],
        "max_len": int(max_len),
        "n_items": int(n_items),
        "implementation_hash": source_hash,
    }
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
    return manifest


def set_lr(optimizer, epoch, max_epochs, peak=1e-3, min_lr=1e-4, warmup=3):
    if epoch <= warmup:
        lr = peak * epoch / warmup
    else:
        progress = (epoch - warmup) / max(1, max_epochs - warmup)
        lr = min_lr + .5 * (peak - min_lr) * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


@torch.inference_mode()
def state_diagnostic(model, prefixes, max_len, device, cap=256):
    chosen = [p for p in prefixes if min(len(p), max_len) > 100][:cap]
    if not chosen:
        chosen = list(prefixes[:cap])
    rows = [list(s)[-max_len:] for s in chosen]
    lens = torch.tensor([len(s) for s in rows], dtype=torch.long, device=device)
    L = int(lens.max().item())
    x = torch.zeros(len(rows), L, dtype=torch.long, device=device)
    for i, s in enumerate(rows):
        x[i, :len(s)] = torch.as_tensor(s, dtype=torch.long, device=device)
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        _, I, M = model.encode_with_states(x)
    r = torch.arange(x.size(0), device=device)
    last = (lens - 1).clamp_min(0)
    ids = I[r, last].cpu()
    mass = M[r, last].float().cpu()
    positive = mass > 1e-8
    unique_counts = []
    duplicate_rows = 0
    entropies = []
    for ii, mm, pp in zip(ids, mass, positive):
        kept = ii[pp].tolist()
        u = len(set(int(z) for z in kept))
        unique_counts.append(u)
        duplicate_rows += int(u < len(kept))
        p = mm[pp]
        if p.numel():
            p = p / p.sum().clamp_min(1e-8)
            entropies.append(float((-(p * p.clamp_min(1e-8).log())).sum()))
    return {
        "users": len(rows),
        "mean_positive_slots": float(positive.sum(-1).float().mean()),
        "mean_unique_positive_concepts": float(np.mean(unique_counts)),
        "rows_with_duplicate_positive_concepts": int(duplicate_rows),
        "duplicate_row_rate": float(duplicate_rows / max(1, len(rows))),
        "mean_mass_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_v11")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print("DEVICE", device, torch.cuda.get_device_name(0) if device.type == "cuda" else None,
          "bf16", torch.cuda.is_bf16_supported() if device.type == "cuda" else False, flush=True)

    data = load_dataset("ml1m", args.data_dir)
    split = split_data(data["sequences"])
    max_len = 200
    protocol = protocol_manifest(max_len, data["n_items"])
    print("PROTOCOL", protocol, flush=True)
    print("V11_CONFIG", {
        "fresh_injections_per_event": 1,
        "graph_hops_per_event": 2,
        "coalesce_duplicates_before_topk": True,
        "pursuit": False,
        "K": 8,
        "degree": 4,
        "fresh_weight": .25,
        "max_len": max_len,
        "objective": "FullCE",
        "warm_start": False,
        "training_precision": "BF16 autocast",
        "evaluation_precision": "canonical / no autocast override",
    }, flush=True)

    model = SparseWalker(
        data["n_items"], max_len, d=64, layers=2, side=256, h=16,
        active=8, top_side=2, degree=4, fresh_weight=.25,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = WindowDataset(split["train"], max_len, args.seed)

    out = Path(args.output_dir) / "ml1m" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True))

    best = -1.0
    best_epoch = 0
    best_state = None
    bad = 0
    history = []

    for epoch in range(1, args.max_epochs + 1):
        lr = set_lr(opt, epoch, args.max_epochs)
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        stats = train_epoch(
            "SparseWalker", model, ds, opt, device,
            batch_size=args.batch_size, epoch=epoch, loss_mode="full",
            bucket_by_length=True, use_bf16=True, return_stats=True,
        )
        if device.type == "cuda": torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "loss": float(stats["loss"]),
            "lr": lr,
            "seconds": secs,
            "positions_per_s": float(stats["positions"]) / max(secs, 1e-9),
            "padding_efficiency": float(stats["padding_efficiency"]),
            "pursued_rows": 0,
        }
        print("TRAIN", row, flush=True)

        if epoch == 1 or epoch % args.eval_every == 0:
            val = evaluate_full(
                model, split["val_prefix"], split["val_target"], data["n_items"],
                max_len, device, topks=(10,), batch_size=args.eval_batch_size,
            )
            diag = state_diagnostic(model, split["val_prefix"], max_len, device)
            erow = {**row, **val}
            history.append(erow)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            print("EVAL", {"epoch": epoch, "NDCG@10": float(val["NDCG@10"]),
                           "HR@10": float(val["HR@10"]), "MRR@10": float(val["MRR@10"])}, flush=True)
            print("STATE_DIAGNOSTIC", diag, flush=True)
            ndcg = float(val["NDCG@10"])
            if ndcg > best:
                best = ndcg
                best_epoch = epoch
                best_state = cpu_state_dict(model)
                bad = 0
                torch.save({"model": best_state, "epoch": epoch, "val": val, "protocol": protocol}, out / "best.pt")
            else:
                bad += args.eval_every
            torch.save({
                "model": cpu_state_dict(model), "optimizer": opt.state_dict(), "epoch": epoch,
                "best_ndcg": best, "best_epoch": best_epoch, "best_state": best_state,
                "bad_epochs": bad, "history": history, "protocol": protocol,
            }, out / "last.pt")
            if bad >= args.patience:
                print("EARLY_STOP", {"best_epoch": best_epoch, "best_val_NDCG@10": best}, flush=True)
                break

    if best_state is None:
        raise RuntimeError("v1.1 produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_val = evaluate_full(
        model, split["val_prefix"], split["val_target"], data["n_items"],
        max_len, device, topks=(10,), batch_size=args.eval_batch_size,
    )
    test = evaluate_full(
        model, split["test_prefix"], split["test_target"], data["n_items"],
        max_len, device, topks=(10,20,50), batch_size=args.eval_batch_size,
    )
    result = {
        "cell": "SparseWalker-v1.1-FullCE",
        "selected_epoch": best_epoch,
        "canonical_val_NDCG@10": float(final_val["NDCG@10"]),
        "protocol_fingerprint": protocol["fingerprint"],
        "fixes": {
            "fresh_injection_once_per_event": True,
            "coalesce_duplicates_before_topk": True,
            "pursuit_disabled": True,
        },
        **test,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("FINAL_RESULT", result, flush=True)


if __name__ == "__main__":
    main()
