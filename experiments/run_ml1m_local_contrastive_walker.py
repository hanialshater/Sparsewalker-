#!/usr/bin/env python
"""Experiment 43 v2: backward-free Local-Contrastive SparseWalker on ML-1M.

Same learning rule as the successful Beauty Experiment 39, but with an ML-1M
execution path that removes padding work, buckets similar sequence lengths, and
uses the Beauty batch size (512). No optimizer, backward(), or autograd grads.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_amazon_local_contrastive_walker as base
from sparsewalker.data import load_dataset, split_data, WindowDataset, collate_windows
from sparsewalker.evaluation import evaluate_full
from sparsewalker.models import SparseWalker

MAX_LEN = 200


def build(n_items):
    return SparseWalker(
        n_items, MAX_LEN, d=64, layers=2, side=256, h=16,
        active=8, top_side=2, degree=4, fresh_weight=.25,
    )


@torch.no_grad()
def evaluate(model, split, n_items, device, batch):
    model.eval()
    val = evaluate_full(
        model, split["val_prefix"], split["val_target"], n_items,
        MAX_LEN, device, topks=(10,), batch_size=batch,
    )
    test = evaluate_full(
        model, split["test_prefix"], split["test_target"], n_items,
        MAX_LEN, device, topks=(10, 20, 50), batch_size=batch,
    )
    return val, test


def _loader(ds, batch_size, epoch, bucket_by_length=True):
    """Deterministic epoch shuffle with optional length bucketing.

    Bucketing changes batch composition only. WindowDataset still samples the
    same per-user window for a given epoch/seed.
    """
    ds.set_epoch(epoch)
    g = torch.Generator()
    g.manual_seed(ds.seed + epoch)
    if not bucket_by_length:
        return DataLoader(
            ds, batch_size=batch_size, shuffle=True, generator=g,
            collate_fn=collate_windows, pin_memory=True,
        )

    indices = torch.randperm(len(ds), generator=g).tolist()
    # Python sort is stable, so equal-length users retain randomized order.
    indices.sort(key=lambda i: min(len(ds.seqs[i]), ds.max_len + 1))
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    if len(batches) > 1:
        order = torch.randperm(len(batches), generator=g).tolist()
        batches = [batches[i] for i in order]
    return DataLoader(
        ds, batch_sampler=batches, collate_fn=collate_windows, pin_memory=True,
    )


@torch.no_grad()
def train_epoch_fast(model, ds, device, args, epoch):
    """Experiment-39 local rules, but compute only rows active at each time step."""
    model.eval()
    ngen = torch.Generator(device=device)
    ngen.manual_seed(args.seed * 100003 + epoch)
    loader = _loader(ds, args.batch_size, epoch, not args.no_length_bucketing)

    total = steps = batches = 0
    actual_positions = padded_positions = 0
    sm = sp = sn = rc = vc = cc = 0.0
    max_lens = []
    t0 = time.perf_counter()
    n_batches = len(loader)

    for tokens, lengths in loader:
        actual_positions += int((lengths - 1).clamp_min(0).sum().item())
        padded_positions += int(tokens.size(0) * max(0, tokens.size(1) - 1))
        max_lens.append(max(0, int(tokens.size(1)) - 1))
        tokens = tokens.to(device, non_blocking=True)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        B, L = x.shape

        ids = torch.zeros(B, model.active, device=device, dtype=torch.long)
        mass = torch.zeros(B, model.active, device=device)

        for t in range(L):
            # Critical ML-1M speed fix: do not run router/graph for padded rows.
            active_rows = x[:, t].ne(0).nonzero(as_tuple=False).squeeze(-1)
            if active_rows.numel() == 0:
                continue

            cur_ids = x[active_rows, t]
            cur = model.item(cur_ids).float()
            context = cur * math.sqrt(model.d_model)
            fi, fm = model.router(context, model.space)

            xids, xmass = model._merge(ids[active_rows], mass[active_rows], fi, fm)
            for _ in range(model.layers_n):
                xids, xmass = model.graph(
                    xids, xmass, context, model.space, track_touched=False
                )
            ids.index_copy_(0, active_rows, xids)
            mass.index_copy_(0, active_rows, xmass)

            valid_local = y[active_rows, t].ne(0)
            if not valid_local.any():
                continue

            rows = active_rows[valid_local]
            curv = cur[valid_local]
            target_ids = y[rows, t]
            target_pre = torch.nn.functional.normalize(
                model.item.weight[target_ids].float(), dim=-1
            )

            rc += base.update_router_and_keys(
                model, curv, fi[valid_local], fm[valid_local].float(),
                args.prototype_lr, args.key_lr,
            )
            target_context = target_pre * math.sqrt(model.d_model)
            tr, tm = model.router(target_context, model.space)
            cc += base.update_context(model, curv, tr, tm.float(), args.context_lr)

            msg = (
                model.space.value(ids[rows]) * mass[rows, :, None]
            ).sum(1).float()
            h = model.norm(
                context[valid_local] + model.message_proj(msg)
            ).float()

            neg = torch.randint(
                1, model.n_items + 1,
                (int(rows.numel()), args.negatives),
                device=device, generator=ngen,
            )
            signal, d = base.contrastive(
                model, h, target_ids, neg, args.item_lr, args.temperature
            )
            base.update_current_items(
                model, cur_ids[valid_local], signal, args.input_lr
            )
            target_now = torch.nn.functional.normalize(
                model.item.weight[target_ids].float(), dim=-1
            )
            vc += base.update_values(
                model, ids[rows], mass[rows].float(), target_now, args.value_lr
            )
            base.update_message(model, msg, signal, args.message_lr)

            sm += d["margin"]
            sp += d["pp"]
            sn += d["pn"]
            steps += 1
            total += int(rows.numel())

        batches += 1
        if args.progress_every > 0 and (
            batches == 1 or batches % args.progress_every == 0 or batches == n_batches
        ):
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print(
                "ML1M_LC_PROGRESS",
                json.dumps({
                    "epoch": epoch,
                    "batch": batches,
                    "batches": n_batches,
                    "batch_max_len": max_lens[-1],
                    "learned_positions": total,
                    "elapsed_s": round(elapsed, 2),
                    "positions_per_s": round(total / max(elapsed, 1e-9), 1),
                }),
                flush=True,
            )

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
        "mean_batch_max_len": sum(max_lens) / max(1, len(max_lens)),
        "padding_efficiency": actual_positions / max(1, padded_positions),
        "length_bucketing": not args.no_length_bucketing,
        "mean_contrastive_margin": sm / z,
        "mean_positive_prob": sp / z,
        "mean_negative_prob": sn / z,
        "mean_router_confidence": rc / z,
        "mean_value_update": vc / z,
        "mean_context_error": cc / z,
        "autograd_grad_tensors": 0,
        "loss_backward_calls": 0,
        "optimizer": None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=70)
    # 512 is the successful Experiment-39 batch size; 128 was an unnecessary
    # carry-over from the gradient ML-1M baseline and caused huge Python overhead.
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=2)
    p.add_argument("--no-length-bucketing", action="store_true")
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
    p.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/sparsewalker_local_contrastive_ml1m",
    )
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
        json.dumps({
            "users": len(data["sequences"]),
            "n_items": data["n_items"],
            "split": "per-user leave-two-out",
            "max_len": MAX_LEN,
            "evaluation": "full catalog, seen-item masking",
            "architecture": "plain corrected SparseWalker v1.1",
            "backward": False,
            "optimizer": None,
            "batch_size": a.batch_size,
            "length_bucketing": not a.no_length_bucketing,
            "active_row_compaction": True,
        }),
        flush=True,
    )

    model = build(data["n_items"]).to(device)
    base.init_model(model, a.seed, a.message_gain)

    init_val, init_test = evaluate(
        model, split, data["n_items"], device, a.eval_batch_size
    )
    print("ML1M_LC_INIT", json.dumps({"val": init_val, "test": init_test}), flush=True)

    ds = WindowDataset(split["train"], MAX_LEN, a.seed)
    out = Path(a.output_dir) / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)

    cfg = {
        "experiment": "ML1M-LocalContrastiveWalker-v2-fast",
        "source_rule": "Experiment 39 local-contrastive learning rule",
        "dataset": "ml1m",
        "max_len": MAX_LEN,
        "warm_start": False,
        "pretrained_checkpoint": None,
        "autograd": False,
        "optimizer": None,
        "loss_backward_calls": 0,
        "graph_topology": "random fixed",
        "static_edge_logits": "zero",
        "execution": {
            "active_row_compaction": True,
            "length_bucketing": not a.no_length_bucketing,
            "batch_size": a.batch_size,
        },
        "architecture": {
            "d": 64, "concepts": 65536, "K": 8, "degree": 4,
            "graph_hops": 2, "fresh_weight": .25,
            "fresh_once": True, "duplicate_coalescing": True,
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
        stats = train_epoch_fast(model, ds, device, a, e)
        row = dict(stats)

        if e == 1 or e % a.eval_every == 0:
            val = evaluate_full(
                model, split["val_prefix"], split["val_target"], data["n_items"],
                MAX_LEN, device, topks=(10,), batch_size=a.eval_batch_size,
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
