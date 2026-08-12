#!/usr/bin/env python
"""ML-1M diagnostic: corrected SparseWalker + one causal self-attention block.

Goal:
  Test whether dense temporal access rescues the Walker representation.

Architecture:
  corrected SparseWalker v1.1 (K=8, degree=4, 2 graph hops, single fresh injection,
  duplicate coalescing, no pursuit)
    -> one causal pre-norm multi-head self-attention block
    -> residual
    -> FFN residual
    -> tied item scorer

The model is trained end-to-end from scratch with the same canonical ML-1M
split/evaluator and FullCE objective. Every validation checkpoint is evaluated
with the same trained weights in two modes:
  attention ON  : full hybrid model
  attention OFF : bypass the temporal attention/FFN block

If ON strongly beats OFF, temporal retrieval is causally useful. If both stay
near the Walker plateau, the bottleneck is earlier than temporal access.
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
import torch.nn as nn

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
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()[:20]
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


class CausalTemporalBlock(nn.Module):
    def __init__(self, d_model, max_len, heads=2, ff_mult=4, dropout=0.1):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        hidden = d_model * ff_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h, pad_mask):
        B, L, D = h.shape
        pos_ids = torch.arange(L, device=h.device)
        x = h + self.pos(pos_ids)[None, :, :]
        qkv = self.norm1(x)
        causal = torch.ones(L, L, dtype=torch.bool, device=h.device).triu(1)
        a, _ = self.attn(
            qkv, qkv, qkv,
            attn_mask=causal,
            key_padding_mask=pad_mask,
            need_weights=False,
        )
        z = h + a
        z = z + self.ffn(self.norm2(z))
        z = z.masked_fill(pad_mask[..., None], 0.0)
        return z


class SparseWalkerAttention(SparseWalker):
    def __init__(
        self,
        n_items,
        max_len,
        d=64,
        layers=2,
        side=256,
        h=16,
        active=8,
        top_side=2,
        degree=4,
        fresh_weight=.25,
        attn_heads=2,
        ff_mult=4,
        dropout=.1,
    ):
        super().__init__(
            n_items, max_len, d=d, layers=layers, side=side, h=h,
            active=active, top_side=top_side, degree=degree,
            fresh_weight=fresh_weight,
        )
        self.temporal = CausalTemporalBlock(
            d, max_len, heads=attn_heads, ff_mult=ff_mult, dropout=dropout
        )
        self.attention_enabled = True

    def encode(self, seq):
        h = super().encode(seq)
        if not self.attention_enabled:
            return h
        return self.temporal(h, seq.eq(0))


@torch.inference_mode()
def causal_leak_test(model, device):
    model.eval()
    a = torch.tensor([[11, 17, 23, 31, 41, 43, 47, 53]], device=device)
    b = torch.tensor([[11, 17, 23, 31, 61, 67, 71, 73]], device=device)
    ha = model.encode(a).float()
    hb = model.encode(b).float()
    diff = float((ha[:, :4] - hb[:, :4]).abs().max().cpu())
    if diff > 1e-5:
        raise RuntimeError(f"Causal leak detected: prefix max diff={diff}")
    print("CAUSAL_LEAK_TEST OK", {"prefix_max_abs_diff": diff}, flush=True)


def evaluate_mode(model, enabled, split_prefix, split_target, n_items, max_len, device, batch_size, topks=(10,)):
    old = model.attention_enabled
    model.attention_enabled = bool(enabled)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = evaluate_full(
        model, split_prefix, split_target, n_items, max_len, device,
        topks=topks, batch_size=batch_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    model.attention_enabled = old
    return out, secs


def history_buckets(prefixes, max_len):
    out = {"short_<=50": [], "medium_51_100": [], "long_101_200": []}
    for i, p in enumerate(prefixes):
        n = min(len(p), max_len)
        if n <= 50:
            out["short_<=50"].append(i)
        elif n <= 100:
            out["medium_51_100"].append(i)
        else:
            out["long_101_200"].append(i)
    return out


def bucket_eval(model, prefixes, targets, n_items, max_len, device, batch_size):
    buckets = history_buckets(prefixes, max_len)
    result = {}
    for name, ids in buckets.items():
        if not ids:
            continue
        pp = [prefixes[i] for i in ids]
        tt = [targets[i] for i in ids]
        on, _ = evaluate_mode(model, True, pp, tt, n_items, max_len, device, batch_size)
        off, _ = evaluate_mode(model, False, pp, tt, n_items, max_len, device, batch_size)
        result[name] = {
            "n": len(ids),
            "attention_on_NDCG@10": float(on["NDCG@10"]),
            "attention_off_NDCG@10": float(off["NDCG@10"]),
            "attention_contribution": float(on["NDCG@10"] - off["NDCG@10"]),
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_attention_control")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--attn-heads", type=int, default=2)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=.1)
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print("DEVICE", device,
          torch.cuda.get_device_name(0) if device.type == "cuda" else None,
          "bf16", torch.cuda.is_bf16_supported() if device.type == "cuda" else False,
          flush=True)

    data = load_dataset("ml1m", args.data_dir)
    split = split_data(data["sequences"])
    max_len = 200
    protocol = protocol_manifest(max_len, data["n_items"])
    print("PROTOCOL", protocol, flush=True)

    model = SparseWalkerAttention(
        data["n_items"], max_len,
        d=64, layers=2, side=256, h=16, active=8, top_side=2,
        degree=4, fresh_weight=.25,
        attn_heads=args.attn_heads, ff_mult=args.ff_mult, dropout=args.dropout,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    attention_params = sum(p.numel() for p in model.temporal.parameters())
    walker_params = total_params - attention_params
    print("ATTENTION_CONTROL_CONFIG", {
        "walker": "v1.1 corrected recurrence",
        "fresh_injections_per_event": 1,
        "graph_hops_per_event": 2,
        "coalesce_duplicates_before_topk": True,
        "pursuit": False,
        "K": 8,
        "degree": 4,
        "d_model": 64,
        "attention_layers": 1,
        "attention_heads": args.attn_heads,
        "ff_mult": args.ff_mult,
        "dropout": args.dropout,
        "positional_embedding": True,
        "train_from_scratch": True,
        "objective": "FullCE all autoregressive positions",
        "walker_params": walker_params,
        "attention_block_params": attention_params,
        "total_params": total_params,
    }, flush=True)

    causal_leak_test(model, device)
    model.train()

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
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        stats = train_epoch(
            "SparseWalkerAttention", model, ds, opt, device,
            batch_size=args.batch_size,
            epoch=epoch,
            loss_mode="full",
            bucket_by_length=True,
            use_bf16=True,
            return_stats=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_s = time.perf_counter() - t0
        train_row = {
            "epoch": epoch,
            "loss": float(stats["loss"]),
            "lr": lr,
            "train_seconds": train_s,
            "positions_per_s": float(stats["positions"]) / max(train_s, 1e-9),
            "padding_efficiency": float(stats["padding_efficiency"]),
            "pursued_rows": 0,
        }
        print("TRAIN", train_row, flush=True)

        if epoch == 1 or epoch % args.eval_every == 0:
            model.eval()
            on, on_s = evaluate_mode(
                model, True,
                split["val_prefix"], split["val_target"], data["n_items"],
                max_len, device, args.eval_batch_size,
            )
            off, off_s = evaluate_mode(
                model, False,
                split["val_prefix"], split["val_target"], data["n_items"],
                max_len, device, args.eval_batch_size,
            )
            ndcg = float(on["NDCG@10"])
            row = {
                **train_row,
                "val_on_HR@10": float(on["HR@10"]),
                "val_on_NDCG@10": ndcg,
                "val_on_MRR@10": float(on["MRR@10"]),
                "val_off_NDCG@10": float(off["NDCG@10"]),
                "attention_contribution": ndcg - float(off["NDCG@10"]),
                "eval_seconds_on": on_s,
                "eval_seconds_off": off_s,
                "eval_cost_ratio": on_s / max(off_s, 1e-9),
            }
            history.append(row)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            print("EVAL", row, flush=True)

            if ndcg > best:
                best = ndcg
                best_epoch = epoch
                best_state = cpu_state_dict(model)
                bad = 0
                torch.save({
                    "model": best_state,
                    "epoch": epoch,
                    "val_on": on,
                    "val_off": off,
                    "protocol": protocol,
                }, out / "best.pt")
            else:
                bad += args.eval_every

            torch.save({
                "model": cpu_state_dict(model),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "best_ndcg": best,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "bad_epochs": bad,
                "history": history,
                "protocol": protocol,
            }, out / "last.pt")
            model.train()

            if bad >= args.patience:
                print("EARLY_STOP", {
                    "best_epoch": best_epoch,
                    "best_val_on_NDCG@10": best,
                }, flush=True)
                break

    if best_state is None:
        raise RuntimeError("attention control produced no validation checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    val_on, val_on_s = evaluate_mode(
        model, True, split["val_prefix"], split["val_target"], data["n_items"],
        max_len, device, args.eval_batch_size,
    )
    val_off, val_off_s = evaluate_mode(
        model, False, split["val_prefix"], split["val_target"], data["n_items"],
        max_len, device, args.eval_batch_size,
    )
    test_on, test_on_s = evaluate_mode(
        model, True, split["test_prefix"], split["test_target"], data["n_items"],
        max_len, device, args.eval_batch_size, topks=(10, 20, 50),
    )
    test_off, test_off_s = evaluate_mode(
        model, False, split["test_prefix"], split["test_target"], data["n_items"],
        max_len, device, args.eval_batch_size, topks=(10, 20, 50),
    )
    by_len = bucket_eval(
        model, split["val_prefix"], split["val_target"], data["n_items"],
        max_len, device, args.eval_batch_size,
    )

    result = {
        "cell": "SparseWalker-v1.1+1CausalAttention",
        "selected_epoch": best_epoch,
        "protocol_fingerprint": protocol["fingerprint"],
        "params": {
            "walker": walker_params,
            "attention_block": attention_params,
            "total": total_params,
        },
        "validation": {
            "attention_on_NDCG@10": float(val_on["NDCG@10"]),
            "attention_off_NDCG@10": float(val_off["NDCG@10"]),
            "attention_contribution": float(val_on["NDCG@10"] - val_off["NDCG@10"]),
            "eval_cost_ratio": val_on_s / max(val_off_s, 1e-9),
        },
        "history_length": by_len,
        "test_attention_on": test_on,
        "test_attention_off": test_off,
        "test_attention_contribution_NDCG@10": float(test_on["NDCG@10"] - test_off["NDCG@10"]),
        "test_eval_cost_ratio": test_on_s / max(test_off_s, 1e-9),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("FINAL_RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
