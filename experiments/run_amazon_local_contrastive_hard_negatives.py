#!/usr/bin/env python
"""Experiment 40: hard-negative local-contrastive SparseWalker v1.1.

This is a surgical extension of Experiment 39. The architecture and all local
plasticity rules are unchanged. The only learning change is the negative set:
after a short random-negative warmup, each position gets a mixture of

  * model-mined hard negatives from a candidate pool,
  * catalog-random negatives,
  * popularity negatives.

The hard pool is intentionally bounded (in-batch targets + popular items + a
fresh random pool), so this stays much cheaper than full-catalog mining at every
training position. There is still no warm start, optimizer, backward(), or
autograd learning.

This runner also saves ``last.pt`` every epoch and supports ``--resume`` so a
Colab/runtime crash does not require retraining from scratch.
"""

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Reuse the exact Experiment-39 initialization, recurrence, and local updates.
from run_amazon_local_contrastive_walker import (
    AMAZON,
    build,
    contrastive,
    cpu_state,
    evaluate,
    init_model,
    loader,
    seed_all,
    update_context,
    update_current_items,
    update_message,
    update_router_and_keys,
    update_values,
)
from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full


def popular_items(train_sequences, n_items, topn):
    c = Counter()
    for seq in train_sequences:
        c.update(int(x) for x in seq if int(x) > 0)
    ids = [i for i, _ in c.most_common(int(topn))]
    if not ids:
        ids = list(range(1, min(int(n_items), int(topn)) + 1))
    return torch.tensor(ids, dtype=torch.long)


def _avoid_positive(ids, pos, n_items):
    """Deterministically repair accidental positive hits."""
    same = ids.eq(pos[:, None])
    if same.any():
        ids = ids.clone()
        ids[same] = (ids[same] % int(n_items)) + 1
    return ids


@torch.no_grad()
def mixed_negatives(model, h, pos_ids, popular, generator, args, epoch):
    """Mine semi-hard negatives and mix them with random/popularity negatives."""
    n = int(h.size(0))
    n_items = int(model.n_items)
    hard_n = int(args.hard_negatives) if epoch >= int(args.hard_start_epoch) else 0
    rand_n = int(args.random_negatives) + (int(args.hard_negatives) if hard_n == 0 else 0)
    pop_n = int(args.popular_negatives)

    hard_ids = h.new_empty((n, 0), dtype=torch.long)
    hard_sim = 0.0
    pool_size = 0

    if hard_n > 0:
        rand_pool = torch.randint(
            1, n_items + 1, (int(args.random_pool),), device=h.device, generator=generator
        )
        pop_pool = popular[: min(int(args.popular_pool), int(popular.numel()))].to(h.device)
        # In-batch true next items are often the most useful confusions.
        pool = torch.unique(torch.cat([pos_ids.detach(), pop_pool, rand_pool]))
        pool_size = int(pool.numel())
        hp = F.normalize(h.float(), dim=-1)
        pe = F.normalize(model.item.weight[pool].float(), dim=-1)
        score = hp @ pe.T
        score.masked_fill_(pool[None, :].eq(pos_ids[:, None]), -1e9)
        k = min(hard_n, max(1, int(pool.numel()) - 1))
        top = score.topk(k, dim=-1)
        hard_ids = pool[top.indices]
        hard_sim = float(top.values.mean().item())
        if k < hard_n:
            filler = torch.randint(
                1, n_items + 1, (n, hard_n - k), device=h.device, generator=generator
            )
            filler = _avoid_positive(filler, pos_ids, n_items)
            hard_ids = torch.cat([hard_ids, filler], dim=-1)

    if rand_n > 0:
        rand_ids = torch.randint(
            1, n_items + 1, (n, rand_n), device=h.device, generator=generator
        )
        rand_ids = _avoid_positive(rand_ids, pos_ids, n_items)
    else:
        rand_ids = h.new_empty((n, 0), dtype=torch.long)

    if pop_n > 0 and popular.numel() > 0:
        p = popular.to(h.device)
        ix = torch.randint(0, int(p.numel()), (n, pop_n), device=h.device, generator=generator)
        pop_ids = p[ix]
        pop_ids = _avoid_positive(pop_ids, pos_ids, n_items)
    else:
        pop_ids = h.new_empty((n, 0), dtype=torch.long)

    neg = torch.cat([hard_ids, rand_ids, pop_ids], dim=-1)
    # Diagnostic: how much harder are the mined negatives than random ones?
    random_sim = 0.0
    if rand_ids.numel():
        hn = F.normalize(h.float(), dim=-1)
        re = F.normalize(model.item.weight[rand_ids].float(), dim=-1)
        random_sim = float((re * hn[:, None, :]).sum(-1).mean().item())

    return neg, {
        "hard_similarity": hard_sim,
        "random_similarity": random_sim,
        "candidate_pool_size": pool_size,
        "hard_active": int(hard_n > 0),
    }


@torch.no_grad()
def train_epoch_hard(model, ds, device, args, epoch, popular):
    model.eval()
    ngen = torch.Generator(device=device)
    ngen.manual_seed(args.seed * 100003 + epoch)
    total = steps = batches = 0
    sm = sp = sn = rc = vc = cc = 0.0
    hs = rs = pool = hard_active = 0.0
    t0 = time.perf_counter()

    for tokens, lengths in loader(ds, args.batch_size, epoch):
        tokens = tokens.to(device, non_blocking=True)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        B, L = x.shape
        ids = torch.zeros(B, model.active, device=device, dtype=torch.long)
        mass = torch.zeros(B, model.active, device=device)

        for t in range(L):
            act = x[:, t].ne(0)
            valid = act & y[:, t].ne(0)
            if not act.any():
                continue

            cur_ids = x[:, t]
            cur = model.item(cur_ids).float()
            context = cur * math.sqrt(model.d_model)
            fi, fm = model.router(context, model.space)
            af = act.float()[:, None]
            xids, xmass = model._merge(ids, mass * af, fi, fm * af)
            for _ in range(model.layers_n):
                xids, xmass = model.graph(
                    xids, xmass, context, model.space, track_touched=False
                )
            ids = torch.where(act[:, None], xids, ids)
            mass = torch.where(act[:, None], xmass, mass)

            if valid.any():
                curv = cur[valid]
                target_ids = y[valid, t]
                target_pre = F.normalize(model.item.weight[target_ids].float(), dim=-1)

                rc += update_router_and_keys(
                    model, curv, fi[valid], fm[valid].float(), args.prototype_lr, args.key_lr
                )
                target_context = target_pre * math.sqrt(model.d_model)
                tr, tm = model.router(target_context, model.space)
                cc += update_context(model, curv, tr, tm.float(), args.context_lr)

                msg = (model.space.value(ids[valid]) * mass[valid, :, None]).sum(1).float()
                h = model.norm(context[valid] + model.message_proj(msg)).float()

                neg, md = mixed_negatives(model, h, target_ids, popular, ngen, args, epoch)
                signal, d = contrastive(
                    model, h, target_ids, neg, args.item_lr, args.temperature
                )
                update_current_items(model, cur_ids[valid], signal, args.input_lr)
                target_now = F.normalize(model.item.weight[target_ids].float(), dim=-1)
                vc += update_values(
                    model, ids[valid], mass[valid].float(), target_now, args.value_lr
                )
                update_message(model, msg, signal, args.message_lr)

                sm += d["margin"]
                sp += d["pp"]
                sn += d["pn"]
                hs += md["hard_similarity"]
                rs += md["random_similarity"]
                pool += md["candidate_pool_size"]
                hard_active += md["hard_active"]
                steps += 1
                total += int(valid.sum())
        batches += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    grads = [n for n, p in model.named_parameters() if p.grad is not None]
    if grads:
        raise AssertionError(f"grad tensors created: {grads[:8]}")
    z = max(1, steps)
    return {
        "epoch": epoch,
        "positions": total,
        "seconds": sec,
        "positions_per_s": total / max(sec, 1e-9),
        "batches": batches,
        "mean_contrastive_margin": sm / z,
        "mean_positive_prob": sp / z,
        "mean_negative_prob": sn / z,
        "mean_router_confidence": rc / z,
        "mean_value_update": vc / z,
        "mean_context_error": cc / z,
        "mean_hard_similarity": hs / z,
        "mean_random_similarity": rs / z,
        "mean_candidate_pool_size": pool / z,
        "hard_mining_active_fraction": hard_active / z,
        "autograd_grad_tensors": 0,
        "loss_backward_calls": 0,
        "optimizer": None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=AMAZON, default="beauty")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=1)

    # Same local learning rates as Experiment 39.
    p.add_argument("--temperature", type=float, default=.15)
    p.add_argument("--item-lr", type=float, default=.02)
    p.add_argument("--input-lr", type=float, default=.004)
    p.add_argument("--prototype-lr", type=float, default=.025)
    p.add_argument("--key-lr", type=float, default=.015)
    p.add_argument("--value-lr", type=float, default=.035)
    p.add_argument("--context-lr", type=float, default=.006)
    p.add_argument("--message-lr", type=float, default=.0008)
    p.add_argument("--message-gain", type=float, default=8.0)

    # 32 total negatives after warmup: 8 hard + 16 random + 8 popularity.
    p.add_argument("--hard-start-epoch", type=int, default=8)
    p.add_argument("--hard-negatives", type=int, default=8)
    p.add_argument("--random-negatives", type=int, default=16)
    p.add_argument("--popular-negatives", type=int, default=8)
    p.add_argument("--random-pool", type=int, default=256)
    p.add_argument("--popular-pool", type=int, default=256)
    p.add_argument("--popular-table", type=int, default=512)

    p.add_argument("--resume", action="store_true")
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/sparsewalker_local_contrastive_hard",
    )
    a = p.parse_args()

    assert torch.cuda.is_available(), "GPU runtime required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed_all(a.seed)

    data = load_dataset(a.dataset, a.data_dir)
    split = split_data(data["sequences"])
    model = build(data["n_items"]).to(device)
    init_model(model, a.seed, a.message_gain)

    out = Path(a.output_dir) / a.dataset / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    last_path = out / "last.pt"
    best_path = out / "best.pt"

    pop = popular_items(split["train"], data["n_items"], a.popular_table).to(device)
    init_val, init_test = evaluate(model, split, data["n_items"], device, a.eval_batch_size)
    print("LCH_INIT", json.dumps({"val": init_val, "test": init_test}), flush=True)

    cfg = {
        "experiment": "LocalContrastiveHardNegativeWalker-v2",
        "base_experiment": "LocalContrastiveWalker-v1",
        "base_architecture": "corrected SparseWalker v1.1 Amazon winner",
        "warm_start": False,
        "pretrained_checkpoint": None,
        "autograd": False,
        "optimizer": None,
        "loss_backward_calls": 0,
        "graph_topology": "random fixed",
        "static_edge_logits": "zero",
        "negative_mining": "in-batch + popular + random candidate pool; top model confusions",
        "references": {
            "LC_v1_best_val_NDCG@10": 0.040515991131704246,
            "SASRec_val_NDCG@10": 0.04296780165590764,
            "SparseWalker_v11_val_NDCG@10": 0.056502000722332545,
        },
        "args": vars(a),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    ds = WindowDataset(split["train"], 50, a.seed)
    best = float(init_val["NDCG@10"])
    best_epoch = 0
    hist = []
    start_epoch = 1

    if a.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        hist = ck.get("history", [])
        best = float(ck.get("best", best))
        best_epoch = int(ck.get("best_epoch", 0))
        start_epoch = int(ck["epoch"]) + 1
        print(
            "LCH_RESUME",
            json.dumps({"from_epoch": int(ck["epoch"]), "best_epoch": best_epoch, "best": best}),
            flush=True,
        )

    for e in range(start_epoch, a.epochs + 1):
        s = train_epoch_hard(model, ds, device, a, e, pop)
        if e == 1 or e % a.eval_every == 0:
            val = evaluate_full(
                model,
                split["val_prefix"],
                split["val_target"],
                data["n_items"],
                50,
                device,
                topks=(10,),
                batch_size=a.eval_batch_size,
            )
            row = {**s, **{f"val_{k}": float(v) for k, v in val.items()}}
            hist.append(row)
            print("LCH_EPOCH", json.dumps(row), flush=True)
            (out / "history.json").write_text(json.dumps(hist, indent=2))
            nd = float(val["NDCG@10"])
            if nd > best:
                best = nd
                best_epoch = e
                torch.save(
                    {"model": cpu_state(model), "epoch": e, "val": val, "config": cfg},
                    best_path,
                )

            # Crash-safe checkpoint every evaluated epoch. No optimizer state exists.
            torch.save(
                {
                    "model": cpu_state(model),
                    "epoch": e,
                    "history": hist,
                    "best": best,
                    "best_epoch": best_epoch,
                    "config": cfg,
                },
                last_path,
            )

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was written")
    best_ck = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ck["model"])
    bv, bt = evaluate(model, split, data["n_items"], device, a.eval_batch_size)
    result = {
        "config": cfg,
        "initial": {"val": init_val, "test": init_test},
        "best_epoch": int(best_ck["epoch"]),
        "best_hard_negative": {"val": bv, "test": bt},
        "vs_lc_v1_val_ratio": float(bv["NDCG@10"] / 0.040515991131704246),
        "vs_sasrec_val_ratio": float(bv["NDCG@10"] / 0.04296780165590764),
        "vs_sasrec_test_ratio": float(bt["NDCG@10"] / 0.031195719394901355),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("LCH_RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
