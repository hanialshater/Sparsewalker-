#!/usr/bin/env python
"""ML-1M depth diagnostic: corrected SparseWalker + TWO causal temporal blocks.

Goal
----
Test whether temporal depth closes the remaining gap between the successful
Walker+1-attention control (~0.161 NDCG@10) and SASRec (~0.1864) without
changing width, Walker state size, loss, or evaluator.

Architecture
------------
  corrected SparseWalker v1.1
    -> causal temporal block #1 (MHA + residual + FFN)
    -> causal temporal block #2 (MHA + residual + FFN)
    -> tied item scorer

Both temporal blocks are identical to the successful one-layer control:
  d_model=64, heads=2, ff_mult=4, dropout=.1.

At every validation checkpoint we evaluate the SAME trained weights at depth:
  0 = Walker only
  1 = Walker + temporal block #1
  2 = Walker + temporal blocks #1 and #2

The same-model ablations show where the trained system places information.
The clean architectural comparison is independently trained 1-layer versus
this independently trained 2-layer run.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full
from sparsewalker.models import SparseWalker
from sparsewalker.training import train_epoch

from run_ml1m_walker_attention_control import (
    CausalTemporalBlock,
    causal_leak_test,
    cpu_state_dict,
    protocol_manifest,
    seed_all,
    set_lr,
)


class SparseWalkerTwoTemporal(SparseWalker):
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
        self.temporal1 = CausalTemporalBlock(
            d, max_len, heads=attn_heads, ff_mult=ff_mult, dropout=dropout
        )
        self.temporal2 = CausalTemporalBlock(
            d, max_len, heads=attn_heads, ff_mult=ff_mult, dropout=dropout
        )
        self.temporal_depth = 2

    def encode(self, seq):
        h = super().encode(seq)
        depth = int(self.temporal_depth)
        if depth >= 1:
            h = self.temporal1(h, seq.eq(0))
        if depth >= 2:
            h = self.temporal2(h, seq.eq(0))
        return h


@torch.inference_mode()
def evaluate_depth(
    model,
    depth,
    prefixes,
    targets,
    n_items,
    max_len,
    device,
    batch_size,
    topks=(10,),
):
    old = model.temporal_depth
    model.temporal_depth = int(depth)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = evaluate_full(
        model, prefixes, targets, n_items, max_len, device,
        topks=topks, batch_size=batch_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    model.temporal_depth = old
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
    result = {}
    for name, ids in history_buckets(prefixes, max_len).items():
        if not ids:
            continue
        pp = [prefixes[i] for i in ids]
        tt = [targets[i] for i in ids]
        result[name] = {"n": len(ids)}
        vals = {}
        for depth in (0, 1, 2):
            m, _ = evaluate_depth(
                model, depth, pp, tt, n_items, max_len, device, batch_size
            )
            vals[depth] = float(m["NDCG@10"])
            result[name][f"depth{depth}_NDCG@10"] = vals[depth]
        result[name]["layer1_contribution"] = vals[1] - vals[0]
        result[name]["layer2_contribution"] = vals[2] - vals[1]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/sparsewalker_two_temporal_layers",
    )
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
    print(
        "DEVICE", device,
        torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "bf16", torch.cuda.is_bf16_supported() if device.type == "cuda" else False,
        flush=True,
    )

    data = load_dataset("ml1m", args.data_dir)
    split = split_data(data["sequences"])
    max_len = 200
    protocol = protocol_manifest(max_len, data["n_items"])
    print("PROTOCOL", protocol, flush=True)

    model = SparseWalkerTwoTemporal(
        data["n_items"], max_len,
        d=64, layers=2, side=256, h=16, active=8, top_side=2,
        degree=4, fresh_weight=.25,
        attn_heads=args.attn_heads, ff_mult=args.ff_mult, dropout=args.dropout,
    ).to(device)

    total_params = sum(x.numel() for x in model.parameters())
    layer1_params = sum(x.numel() for x in model.temporal1.parameters())
    layer2_params = sum(x.numel() for x in model.temporal2.parameters())
    walker_params = total_params - layer1_params - layer2_params
    print("TWO_TEMPORAL_CONFIG", {
        "walker": "v1.1 corrected recurrence",
        "fresh_injections_per_event": 1,
        "graph_hops_per_event": 2,
        "coalesce_duplicates_before_topk": True,
        "pursuit": False,
        "K": 8,
        "degree": 4,
        "d_model": 64,
        "temporal_layers": 2,
        "attention_heads_per_layer": args.attn_heads,
        "ff_mult": args.ff_mult,
        "dropout": args.dropout,
        "train_from_scratch": True,
        "objective": "FullCE all autoregressive positions",
        "walker_params": walker_params,
        "temporal1_params": layer1_params,
        "temporal2_params": layer2_params,
        "total_params": total_params,
    }, flush=True)

    # Existing leak test only depends on encode() being causal.
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
        model.temporal_depth = 2
        lr = set_lr(opt, epoch, args.max_epochs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        stats = train_epoch(
            "SparseWalkerTwoTemporal", model, ds, opt, device,
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
            d0, s0 = evaluate_depth(
                model, 0, split["val_prefix"], split["val_target"],
                data["n_items"], max_len, device, args.eval_batch_size,
            )
            d1, s1 = evaluate_depth(
                model, 1, split["val_prefix"], split["val_target"],
                data["n_items"], max_len, device, args.eval_batch_size,
            )
            d2, s2 = evaluate_depth(
                model, 2, split["val_prefix"], split["val_target"],
                data["n_items"], max_len, device, args.eval_batch_size,
            )
            nd0 = float(d0["NDCG@10"])
            nd1 = float(d1["NDCG@10"])
            nd2 = float(d2["NDCG@10"])
            row = {
                **train_row,
                "val_depth0_NDCG@10": nd0,
                "val_depth1_NDCG@10": nd1,
                "val_depth2_HR@10": float(d2["HR@10"]),
                "val_depth2_NDCG@10": nd2,
                "val_depth2_MRR@10": float(d2["MRR@10"]),
                "layer1_contribution": nd1 - nd0,
                "layer2_contribution": nd2 - nd1,
                "total_temporal_contribution": nd2 - nd0,
                "eval_seconds_depth0": s0,
                "eval_seconds_depth1": s1,
                "eval_seconds_depth2": s2,
                "eval_cost_ratio_depth2_vs_depth0": s2 / max(s0, 1e-9),
            }
            history.append(row)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            print("EVAL", row, flush=True)

            if nd2 > best:
                best = nd2
                best_epoch = epoch
                best_state = cpu_state_dict(model)
                bad = 0
                torch.save({
                    "model": best_state,
                    "epoch": epoch,
                    "val_depth0": d0,
                    "val_depth1": d1,
                    "val_depth2": d2,
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
                    "best_val_depth2_NDCG@10": best,
                }, flush=True)
                break

    if best_state is None:
        raise RuntimeError("two-temporal-layer run produced no validation checkpoint")

    model.load_state_dict(best_state)
    model.eval()

    final = {}
    for depth in (0, 1, 2):
        val, val_s = evaluate_depth(
            model, depth,
            split["val_prefix"], split["val_target"], data["n_items"],
            max_len, device, args.eval_batch_size, topks=(10, 20, 50),
        )
        test, test_s = evaluate_depth(
            model, depth,
            split["test_prefix"], split["test_target"], data["n_items"],
            max_len, device, args.eval_batch_size, topks=(10, 20, 50),
        )
        final[f"depth{depth}"] = {
            "val": {k: float(v) for k, v in val.items()},
            "test": {k: float(v) for k, v in test.items()},
            "val_seconds": val_s,
            "test_seconds": test_s,
        }

    by_len = bucket_eval(
        model,
        split["val_prefix"], split["val_target"], data["n_items"],
        max_len, device, args.eval_batch_size,
    )

    result = {
        "cell": "SparseWalker-v1.1+2CausalTemporalBlocks",
        "selected_epoch": best_epoch,
        "protocol_fingerprint": protocol["fingerprint"],
        "params": {
            "walker": walker_params,
            "temporal1": layer1_params,
            "temporal2": layer2_params,
            "total": total_params,
        },
        "reference": {
            "walker_best_NDCG@10_approx": 0.1337641,
            "walker_plus_1_temporal_best_NDCG@10_approx": 0.1609769,
            "sasrec_fullce_NDCG@10_approx": 0.186428,
        },
        "final": final,
        "by_history_length": by_len,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("FINAL", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
