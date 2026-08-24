#!/usr/bin/env python
"""Experiment 43: backward-free Local-Contrastive SparseWalker on ML-1M.

Purpose: test whether the strongest backward-free learning rule from Experiment 39
(Beauty) transfers unchanged to long-history sequential recommendation.

Architecture and learning rule are intentionally unchanged:
- corrected SparseWalker v1.1 recurrence;
- d=64, 65,536 concepts, K=8 active, degree=4, two graph hops;
- fresh concepts injected once/event, duplicate coalescing, random fixed topology;
- local contrastive item updates;
- competitive router/key updates;
- predictive concept-value/context/message updates;
- NO optimizer, NO backward(), NO autograd gradients, NO warm start.

Only protocol change versus Experiment 39: ML-1M with max_len=200 and the
canonical leave-two-out/full-catalog evaluator used by the existing ML-1M runs.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_amazon_local_contrastive_walker as base
from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full
from sparsewalker.models import SparseWalker

MAX_LEN = 200


def build(n_items):
    return SparseWalker(
        n_items,
        MAX_LEN,
        d=64,
        layers=2,
        side=256,
        h=16,
        active=8,
        top_side=2,
        degree=4,
        fresh_weight=.25,
    )


@torch.no_grad()
def evaluate(model, split, n_items, device, batch):
    model.eval()
    val = evaluate_full(
        model,
        split["val_prefix"],
        split["val_target"],
        n_items,
        MAX_LEN,
        device,
        topks=(10,),
        batch_size=batch,
    )
    test = evaluate_full(
        model,
        split["test_prefix"],
        split["test_target"],
        n_items,
        MAX_LEN,
        device,
        topks=(10, 20, 50),
        batch_size=batch,
    )
    return val, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--negatives", type=int, default=32)
    p.add_argument("--temperature", type=float, default=.15)
    p.add_argument("--item-lr", type=float, default=.02)
    p.add_argument("--input-lr", type=float, default=.004)
    p.add_argument("--prototype-lr", type=float, default=.025)
    p.add_argument("--key-lr", type=float, default=.015)
    p.add_argument("--value-lr", type=float, default=.035)
    p.add_argument("--context-lr", type=float, default=.006)
    p.add_argument("--message-lr", type=float, default=.0008)
    p.add_argument("--message-gain", type=float, default=8.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_local_contrastive_ml1m")
    a = p.parse_args()

    assert torch.cuda.is_available(), "GPU runtime required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    base.seed_all(a.seed)

    data = load_dataset("ml1m", a.data_dir)
    split = split_data(data["sequences"])
    print(
        "ML1M_PROTOCOL",
        json.dumps(
            {
                "users": len(data["sequences"]),
                "n_items": data["n_items"],
                "split": "per-user leave-two-out",
                "max_len": MAX_LEN,
                "evaluation": "full catalog, seen-item masking",
                "architecture": "plain corrected SparseWalker v1.1",
                "backward": False,
                "optimizer": None,
            }
        ),
        flush=True,
    )

    model = build(data["n_items"]).to(device)
    base.init_model(model, a.seed, a.message_gain)

    init_val, init_test = evaluate(model, split, data["n_items"], device, a.eval_batch_size)
    print("ML1M_LC_INIT", json.dumps({"val": init_val, "test": init_test}), flush=True)

    ds = WindowDataset(split["train"], MAX_LEN, a.seed)
    out = Path(a.output_dir) / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)

    cfg = {
        "experiment": "ML1M-LocalContrastiveWalker-v1",
        "source_rule": "Experiment 39 exact local-contrastive learning rule",
        "dataset": "ml1m",
        "max_len": MAX_LEN,
        "warm_start": False,
        "pretrained_checkpoint": None,
        "autograd": False,
        "optimizer": None,
        "loss_backward_calls": 0,
        "graph_topology": "random fixed",
        "static_edge_logits": "zero",
        "architecture": {
            "d": 64,
            "concepts": 65536,
            "K": 8,
            "degree": 4,
            "graph_hops": 2,
            "fresh_weight": .25,
            "fresh_once": True,
            "duplicate_coalescing": True,
        },
        "local_rules": "contrastive items + competitive router/keys + predictive values/context/message",
        "args": vars(a),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    best = float(init_val["NDCG@10"])
    best_epoch = 0
    best_state = base.cpu_state(model)
    hist = []
    start = 1

    last_path = out / "last.pt"
    if a.resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        model.to(device)
        best = float(ck["best"])
        best_epoch = int(ck["best_epoch"])
        best_state = ck["best_state"]
        hist = ck.get("history", [])
        start = int(ck["epoch"]) + 1
        print(
            "ML1M_LC_RESUME",
            json.dumps({"from_epoch": start - 1, "best_epoch": best_epoch, "best": best}),
            flush=True,
        )

    for e in range(start, a.epochs + 1):
        stats = base.train_epoch(model, ds, device, a, e)
        row = dict(stats)

        if e == 1 or e % a.eval_every == 0:
            val = evaluate_full(
                model,
                split["val_prefix"],
                split["val_target"],
                data["n_items"],
                MAX_LEN,
                device,
                topks=(10,),
                batch_size=a.eval_batch_size,
            )
            row.update({f"val_{k}": float(v) for k, v in val.items()})
            hist.append(row)
            print("ML1M_LC_EPOCH", json.dumps(row), flush=True)
            (out / "history.json").write_text(json.dumps(hist, indent=2))

            ndcg = float(val["NDCG@10"])
            if ndcg > best:
                best = ndcg
                best_epoch = e
                best_state = base.cpu_state(model)
                torch.save(
                    {"model": best_state, "epoch": e, "val": val, "config": cfg},
                    out / "best.pt",
                )

        # Backward-free training has no optimizer state, so this is a complete crash-safe checkpoint.
        torch.save(
            {
                "model": base.cpu_state(model),
                "epoch": e,
                "best": best,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "history": hist,
                "config": cfg,
            },
            last_path,
        )

    model.load_state_dict(best_state)
    model.to(device)
    best_val, test = evaluate(model, split, data["n_items"], device, a.eval_batch_size)
    result = {
        "config": cfg,
        "initial": {"val": init_val, "test": init_test},
        "best_epoch": best_epoch,
        "best_backward_free": {"val": best_val, "test": test},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("ML1M_LC_RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
