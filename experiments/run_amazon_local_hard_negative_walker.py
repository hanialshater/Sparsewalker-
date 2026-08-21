#!/usr/bin/env python
"""Experiment 40: hard-negative local-contrastive SparseWalker v1.1.

This is a surgical extension of Experiment 39. The architecture and all local
plasticity rules stay the same; only negative selection changes after a short
warmup. Training remains from scratch with no pretrained checkpoint, no optimizer,
no backward(), and no autograd gradient tensors.

Negative mixture after warmup:
- hard negatives: highest-scoring mistakes from a shared candidate pool;
- popularity negatives: sampled from frequent training items;
- random negatives: preserve exploration and broad catalog pressure.

The script also saves last.pt every epoch and supports --resume / --test-only so
Colab disconnects do not require retraining.
"""
import argparse, json, math, time
from pathlib import Path
import torch
import torch.nn.functional as F

from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full
from run_amazon_local_contrastive_walker import (
    AMAZON, seed_all, cpu_state, build, init_model, loader,
    update_router_and_keys, update_context, contrastive,
    update_current_items, update_values, update_message, evaluate,
)


def popularity_order(train_sequences, n_items, device):
    counts = torch.zeros(n_items + 1, dtype=torch.long)
    for seq in train_sequences:
        if len(seq):
            ids = torch.as_tensor(seq, dtype=torch.long)
            counts += torch.bincount(ids, minlength=n_items + 1)[: n_items + 1]
    counts[0] = -1
    return torch.argsort(counts, descending=True).to(device)


@torch.no_grad()
def mine_negatives(model, h, pos_ids, popular, args, generator, epoch):
    """Mine a mixed [M,N] negative set from the current forward representation."""
    M = h.size(0)
    if epoch <= args.hard_warmup_epochs or args.hard_negatives <= 0:
        neg = torch.randint(
            1, model.n_items + 1, (M, args.total_negatives),
            device=h.device, generator=generator,
        )
        for _ in range(3):
            bad = neg.eq(pos_ids[:, None])
            if not bad.any():
                break
            neg[bad] = torch.randint(
                1, model.n_items + 1, (int(bad.sum()),),
                device=h.device, generator=generator,
            )
        return neg, {"hard_score": 0.0, "hard_fraction": 0.0, "mode": "warmup_random"}

    hard_k = int(args.hard_negatives)
    pop_k = int(args.popularity_negatives)
    rand_k = int(args.total_negatives - hard_k - pop_k)
    if rand_k < 0:
        raise ValueError("hard_negatives + popularity_negatives must be <= total_negatives")

    # Shared candidate pool keeps mining inexpensive while using the model's own
    # current geometry. Include popular items so frequent confusions are mineable.
    n_pop_pool = min(int(args.hard_pool_size // 4), int(popular.numel()))
    pop_pool = popular[:n_pop_pool]
    n_rand_pool = max(1, int(args.hard_pool_size) - n_pop_pool)
    rand_pool = torch.randint(
        1, model.n_items + 1, (n_rand_pool,), device=h.device, generator=generator
    )
    pool = torch.unique(torch.cat([pop_pool, rand_pool]))

    hn = F.normalize(h.float(), dim=-1)
    pe = F.normalize(model.item.weight[pool].float(), dim=-1)
    score = hn @ pe.T
    score.masked_fill_(pool[None, :].eq(pos_ids[:, None]), -float("inf"))
    hk = min(hard_k, int(pool.numel()))
    hi = torch.topk(score, hk, dim=-1, largest=True).indices
    hard = pool[hi]
    hard_score = float(score.gather(1, hi).mean().item()) if hk else 0.0

    if hk < hard_k:
        extra = torch.randint(
            1, model.n_items + 1, (M, hard_k - hk),
            device=h.device, generator=generator,
        )
        hard = torch.cat([hard, extra], dim=1)

    if pop_k:
        pop_window = min(int(args.popularity_pool), int(popular.numel()))
        pi = torch.randint(0, pop_window, (M, pop_k), device=h.device, generator=generator)
        pop = popular[pi]
    else:
        pop = torch.empty(M, 0, device=h.device, dtype=torch.long)

    if rand_k:
        rnd = torch.randint(
            1, model.n_items + 1, (M, rand_k), device=h.device, generator=generator
        )
    else:
        rnd = torch.empty(M, 0, device=h.device, dtype=torch.long)

    neg = torch.cat([hard, pop, rnd], dim=1)
    # Accidental positives are replaced, preserving a true negative teaching set.
    for _ in range(4):
        bad = neg.eq(pos_ids[:, None])
        if not bad.any():
            break
        neg[bad] = torch.randint(
            1, model.n_items + 1, (int(bad.sum()),),
            device=h.device, generator=generator,
        )
    return neg, {
        "hard_score": hard_score,
        "hard_fraction": float(hard_k / max(1, args.total_negatives)),
        "mode": "mixed_hard",
    }


@torch.no_grad()
def train_epoch_hard(model, ds, device, args, epoch, popular):
    model.eval()
    ngen = torch.Generator(device=device)
    ngen.manual_seed(args.seed * 100003 + epoch)
    total = steps = batches = 0
    sm = sp = sn = rc = vc = cc = hs = hf = 0.0
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
                    model, curv, fi[valid], fm[valid].float(),
                    args.prototype_lr, args.key_lr,
                )
                target_context = target_pre * math.sqrt(model.d_model)
                tr, tm = model.router(target_context, model.space)
                cc += update_context(model, curv, tr, tm.float(), args.context_lr)

                msg = (model.space.value(ids[valid]) * mass[valid, :, None]).sum(1).float()
                h = model.norm(context[valid] + model.message_proj(msg)).float()
                neg, md = mine_negatives(model, h, target_ids, popular, args, ngen, epoch)
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
                hs += md["hard_score"]
                hf += md["hard_fraction"]
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
        "mean_hard_negative_score": hs / z,
        "hard_negative_fraction": hf / z,
        "negative_mode": "warmup_random" if epoch <= args.hard_warmup_epochs else "mixed_hard",
        "mean_router_confidence": rc / z,
        "mean_value_update": vc / z,
        "mean_context_error": cc / z,
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
    p.add_argument("--total-negatives", type=int, default=32)
    p.add_argument("--hard-negatives", type=int, default=8)
    p.add_argument("--popularity-negatives", type=int, default=8)
    p.add_argument("--hard-pool-size", type=int, default=512)
    p.add_argument("--popularity-pool", type=int, default=256)
    p.add_argument("--hard-warmup-epochs", type=int, default=5)
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
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/sparsewalker_local_hard_negative",
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
    popular = popularity_order(split["train"], data["n_items"], device)

    out = Path(a.output_dir) / a.dataset / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "best.pt"
    last_path = out / "last.pt"
    history_path = out / "history.json"

    cfg = {
        "experiment": "LocalHardNegativeWalker-v2",
        "base_architecture": "corrected SparseWalker v1.1 Amazon winner",
        "base_learning": "LocalContrastiveWalker-v1",
        "warm_start": False,
        "pretrained_checkpoint": None,
        "autograd": False,
        "optimizer": None,
        "loss_backward_calls": 0,
        "negative_schedule": {
            "warmup_epochs": a.hard_warmup_epochs,
            "hard": a.hard_negatives,
            "popularity": a.popularity_negatives,
            "random": a.total_negatives - a.hard_negatives - a.popularity_negatives,
            "hard_pool_size": a.hard_pool_size,
        },
        "references": {
            "LocalContrastive_best_val_NDCG@10": 0.040515991131704246,
            "SASRec_val_NDCG@10": 0.04296780165590764,
            "SASRec_test_NDCG@10": 0.031195719394901355,
            "SparseWalker_v11_val_NDCG@10": 0.056502000722332545,
            "SparseWalker_v11_test_NDCG@10": 0.044882819399656555,
        },
        "args": vars(a),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    hist = json.loads(history_path.read_text()) if history_path.exists() else []
    start_epoch = 1
    if a.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        start_epoch = int(ck["epoch"]) + 1
        print("HARD_RESUME", json.dumps({"from_epoch": int(ck["epoch"]), "next_epoch": start_epoch}), flush=True)

    if a.test_only:
        if not best_path.exists():
            raise FileNotFoundError(f"Missing best checkpoint: {best_path}")
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck["model"])
        bv, bt = evaluate(model, split, data["n_items"], device, a.eval_batch_size)
        print("HARD_TEST_ONLY", json.dumps({"epoch": int(ck["epoch"]), "val": bv, "test": bt}), flush=True)
        return

    if best_path.exists() and a.resume:
        bck = torch.load(best_path, map_location="cpu")
        best = float(bck["val"]["NDCG@10"])
        best_epoch = int(bck["epoch"])
    else:
        init_val, init_test = evaluate(model, split, data["n_items"], device, a.eval_batch_size)
        print("HARD_INIT", json.dumps({"val": init_val, "test": init_test}), flush=True)
        best = float(init_val["NDCG@10"])
        best_epoch = 0

    ds = WindowDataset(split["train"], 50, a.seed)
    for e in range(start_epoch, a.epochs + 1):
        s = train_epoch_hard(model, ds, device, a, e, popular)
        val = evaluate_full(
            model, split["val_prefix"], split["val_target"], data["n_items"], 50,
            device, topks=(10,), batch_size=a.eval_batch_size,
        )
        row = {**s, **{f"val_{k}": float(v) for k, v in val.items()}}
        hist.append(row)
        print("HARD_EPOCH", json.dumps(row), flush=True)
        history_path.write_text(json.dumps(hist, indent=2))

        state = cpu_state(model)
        torch.save({"model": state, "epoch": e, "val": val, "config": cfg}, last_path)
        nd = float(val["NDCG@10"])
        if nd > best:
            best = nd
            best_epoch = e
            torch.save({"model": state, "epoch": e, "val": val, "config": cfg}, best_path)

    if not best_path.exists():
        raise RuntimeError("No best checkpoint saved")
    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck["model"])
    bv, bt = evaluate(model, split, data["n_items"], device, a.eval_batch_size)
    result = {
        "config": cfg,
        "best_epoch": int(ck["epoch"]),
        "best_hard_negative": {"val": bv, "test": bt},
        "vs_sasrec_val_ratio": float(bv["NDCG@10"] / 0.04296780165590764),
        "vs_sasrec_test_ratio": float(bt["NDCG@10"] / 0.031195719394901355),
        "vs_backprop_walker_test_ratio": float(bt["NDCG@10"] / 0.044882819399656555),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("HARD_RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
