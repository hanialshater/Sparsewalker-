#!/usr/bin/env python
"""ML-1M diagnostic: Sparse Walker + three temporal skip memories.

This is intentionally an architecture experiment, not a canonical baseline.
It warm-starts from the best plain SparseWalker checkpoint and asks one narrow
question: does sparse access to frozen-in-time concept landmarks recover quality
that the purely local recurrent walk loses on max_len=200 histories?
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
from sparsewalker.models import SparseWalker, SparseWalkerTemporalMemory
from sparsewalker.training import train_epoch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


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
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()[:20]
    return manifest


def set_lr(optimizer, epoch, max_epochs, peak=5e-4, min_lr=1e-4, warmup=2):
    if epoch <= warmup:
        lr = peak * epoch / warmup
    else:
        progress = (epoch - warmup) / max(1, max_epochs - warmup)
        lr = min_lr + .5 * (peak - min_lr) * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument("--base-root", default="/content/drive/MyDrive/sparsewalker_canonical_pair")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_temporal_skips")
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--pursuit-every", type=int, default=0,
                   help="0 isolates temporal memory by disabling new topology rewires")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device,
          "GPU", torch.cuda.get_device_name(0) if device.type == "cuda" else None,
          "bf16_supported", torch.cuda.is_bf16_supported() if device.type == "cuda" else False)

    max_len = 200
    data = load_dataset("ml1m", args.data_dir)
    split = split_data(data["sequences"])
    protocol = protocol_manifest(max_len, data["n_items"])
    print("PROTOCOL", json.dumps(protocol, indent=2))

    base_path = Path(args.base_root) / "ml1m" / f"seed{args.seed}" / "SparseWalker_FullCE" / "best.pt"
    if not base_path.exists():
        raise FileNotFoundError(
            f"Missing {base_path}. Finish the plain ML-1M Walker run first."
        )
    base_ckpt = torch.load(base_path, map_location="cpu")
    base_protocol = base_ckpt.get("protocol", {})
    if base_protocol.get("fingerprint") and base_protocol.get("fingerprint") != protocol["fingerprint"]:
        raise RuntimeError(
            f"Base checkpoint fingerprint {base_protocol.get('fingerprint')} != {protocol['fingerprint']}"
        )

    base = SparseWalker(
        data["n_items"], max_len, d=64, layers=2, side=256, h=16,
        active=8, top_side=2, degree=4, fresh_weight=.25,
    ).to(device)
    base.load_state_dict(base_ckpt["model"])
    base.eval()
    base_val = evaluate_full(
        base, split["val_prefix"], split["val_target"], data["n_items"],
        max_len, device, topks=(10,), batch_size=args.eval_batch_size,
    )
    base_test = evaluate_full(
        base, split["test_prefix"], split["test_target"], data["n_items"],
        max_len, device, topks=(10, 20, 50), batch_size=args.eval_batch_size,
    )
    print("BASE_WALKER", {
        "checkpoint_epoch": int(base_ckpt.get("epoch", -1)),
        "val_NDCG@10": float(base_val["NDCG@10"]),
        "test_NDCG@10": float(base_test["NDCG@10"]),
        "HR@10": float(base_test["HR@10"]),
        "MRR@10": float(base_test["MRR@10"]),
    })
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out = Path(args.output_dir) / "ml1m" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "protocol.json", protocol)
    last_path = out / "last.pt"
    best_path = out / "best.pt"
    done_path = out / "done.json"

    if done_path.exists() and not args.force:
        done = json.loads(done_path.read_text())
        if done.get("protocol_fingerprint") == protocol["fingerprint"]:
            print("SKIP completed temporal-memory experiment", done)
            return

    model = SparseWalkerTemporalMemory(
        data["n_items"], max_len, d=64, layers=2, side=256, h=16,
        active=8, top_side=2, degree=4, fresh_weight=.25,
        memory_periods=(16, 64, 256),
        memory_offsets=(0, 16, 64),
        initial_memory_share=.25,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    ds = WindowDataset(split["train"], max_len, args.seed)

    start_epoch = 1
    best_ndcg = -1.0
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history = []

    if last_path.exists() and not args.force:
        last = torch.load(last_path, map_location="cpu")
        if last.get("protocol", {}).get("fingerprint") == protocol["fingerprint"]:
            model.load_state_dict(last["model"])
            optimizer.load_state_dict(last["optimizer"])
            start_epoch = int(last["epoch"]) + 1
            best_ndcg = float(last["best_ndcg"])
            best_epoch = int(last["best_epoch"])
            best_state = last["best_state"]
            bad_epochs = int(last.get("bad_epochs", 0))
            history = list(last.get("history", []))
            print("RESUME TEMPORAL from epoch", start_epoch - 1, "best", best_epoch)
    else:
        missing, unexpected = model.load_state_dict(base_ckpt["model"], strict=False)
        expected_missing = {"memory_q.weight", "memory_bias", "memory_share_logit"}
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(f"Unexpected warm-start mismatch missing={missing} unexpected={unexpected}")
        print("WARMSTART temporal Walker from base epoch", int(base_ckpt.get("epoch", -1)))
        print("NEW_PARAMS", sorted(missing))

    print("TEMPORAL_CONFIG", {
        "K": 8,
        "layers": 2,
        "degree": 4,
        "memory_slots": 3,
        "memory_periods": [16, 64, 256],
        "memory_offsets": [0, 16, 64],
        "stored_concept_slots_per_user": 24,
        "pursuit_every": args.pursuit_every,
        "batch_size": args.batch_size,
        "bf16": True,
        "bucket_by_length": True,
    })

    for epoch in range(start_epoch, args.max_epochs + 1):
        lr = set_lr(optimizer, epoch, args.max_epochs)
        t0 = time.perf_counter()
        stats = train_epoch(
            "SparseWalkerTemporalMemory", model, ds, optimizer, device,
            batch_size=args.batch_size, epoch=epoch, loss_mode="full",
            bucket_by_length=True, use_bf16=True, return_stats=True,
        )
        seconds = time.perf_counter() - t0
        pursued = 0
        if args.pursuit_every > 0 and epoch % args.pursuit_every == 0:
            pursued = model.graph.pursue(optimizer, refresh=2)
        train_row = {
            "epoch": epoch,
            "loss": stats["loss"],
            "lr": lr,
            "seconds": seconds,
            "positions_per_s": stats["positions"] / max(seconds, 1e-9),
            "padding_efficiency": stats["padding_efficiency"],
            "memory_share": float(torch.sigmoid(model.memory_share_logit).detach().cpu()),
            "pursued_rows": pursued,
        }
        print("TEMPORAL TRAIN", {k: round(v, 6) if isinstance(v, float) else v for k, v in train_row.items()})

        do_eval = epoch == 1 or epoch % args.eval_every == 0
        if do_eval:
            val = evaluate_full(
                model, split["val_prefix"], split["val_target"], data["n_items"],
                max_len, device, topks=(10,), batch_size=args.eval_batch_size,
            )
            row = {**train_row, **val}
            history.append(row)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            ndcg = float(val["NDCG@10"])
            print("TEMPORAL EVAL", {
                "epoch": epoch,
                "HR@10": float(val["HR@10"]),
                "NDCG@10": ndcg,
                "MRR@10": float(val["MRR@10"]),
                "gain_vs_base_val_pct": 100.0 * (ndcg / float(base_val["NDCG@10"]) - 1.0),
            })
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_epoch = epoch
                best_state = cpu_state_dict(model)
                bad_epochs = 0
                torch.save({
                    "model": best_state,
                    "epoch": epoch,
                    "val": val,
                    "protocol": protocol,
                    "architecture": "SparseWalkerTemporalMemory",
                }, best_path)
            else:
                bad_epochs += args.eval_every

            torch.save({
                "model": cpu_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_ndcg": best_ndcg,
                "best_state": best_state,
                "bad_epochs": bad_epochs,
                "history": history,
                "protocol": protocol,
            }, last_path)

            if bad_epochs >= args.patience:
                print("EARLY STOP temporal Walker best epoch", best_epoch)
                break

    if best_state is None:
        raise RuntimeError("Temporal Walker produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_val = evaluate_full(
        model, split["val_prefix"], split["val_target"], data["n_items"],
        max_len, device, topks=(10,), batch_size=args.eval_batch_size,
    )
    test = evaluate_full(
        model, split["test_prefix"], split["test_target"], data["n_items"],
        max_len, device, topks=(10, 20, 50), batch_size=args.eval_batch_size,
    )
    result = {
        "cell": "SparseWalker+TemporalSkips",
        "architecture": "SparseWalkerTemporalMemory",
        "warm_started_from_base_epoch": int(base_ckpt.get("epoch", -1)),
        "selected_epoch": best_epoch,
        "canonical_val_NDCG@10": float(final_val["NDCG@10"]),
        "protocol_fingerprint": protocol["fingerprint"],
        "memory_slots": 3,
        "stored_concept_slots_per_user": 24,
        "memory_periods": [16, 64, 256],
        "memory_offsets": [0, 16, 64],
        "final_memory_share": float(torch.sigmoid(model.memory_share_logit).detach().cpu()),
        "base_test_NDCG@10": float(base_test["NDCG@10"]),
        "test_gain_vs_base_pct": 100.0 * (float(test["NDCG@10"]) / float(base_test["NDCG@10"]) - 1.0),
        **test,
    }
    save_json(done_path, result)
    print("TEMPORAL RESULT", result)


if __name__ == "__main__":
    main()
