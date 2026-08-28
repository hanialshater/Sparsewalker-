#!/usr/bin/env python
"""Experiment 45: where does SparseWalker need gradient credit?

This is a diagnostic ladder, not a new architecture. Every arm uses the same
SparseWalker v1.1 initialization, FullCE next-item objective, AdamW optimizer,
data windows, and validation protocol. The only variable is how far gradients
may travel through recurrent time.

Arms
----
readout_only (radius 0)
    Walker routing/state transition is computed under no_grad. The current
    readout (item/value/message/norm path) receives the exact FullCE gradient.
event_local (radius 1)
    Previous recurrent mass is detached before every event. FullCE can
    differentiate through the current router + hop1 + hop2 + readout only.
tbptt2 / tbptt4 / tbptt8
    Exact gradients through windows of 2/4/8 events; state is detached at
    window boundaries. Gradients are accumulated and AdamW steps once/batch.
full_bptt
    No temporal detach inside the sampled training window (up to 200 events).

The concept IDs selected by top-k remain discrete in every arm, exactly as in
ordinary SparseWalker training; gradients flow through the selected routing
scores/masses/values.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_amazon_local_contrastive_walker as local_base
import run_ml1m_local_contrastive_walker as fast
from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full

MAX_LEN = 200
ARM_RADIUS = {
    "readout_only": 0,
    "event_local": 1,
    "tbptt2": 2,
    "tbptt4": 4,
    "tbptt8": 8,
    "full_bptt": MAX_LEN,
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_lr(opt, epoch, max_epochs, peak=1e-3, min_lr=1e-4, warmup=3):
    if epoch <= warmup:
        lr = peak * epoch / warmup
    else:
        p = (epoch - warmup) / max(1, max_epochs - warmup)
        lr = min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * p))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


def grad_groups(model):
    groups = {
        "item": [model.item.weight],
        "router": [model.router.left_q.weight, model.router.right_q.weight, model.router.scale],
        "concept_router": [model.space.left_router, model.space.right_router],
        "concept_key": [model.space.left_key, model.space.right_key],
        "concept_value": [
            model.space.left_value.weight,
            model.space.right_value.weight,
            model.space.value_proj.weight,
            model.space.value_proj.bias,
        ],
        "graph": [model.graph.edge_logits.weight, model.graph.context_q.weight, model.graph.scale],
        "readout": [model.message_proj.weight, model.norm.weight, model.norm.bias],
    }
    out = {}
    for name, ps in groups.items():
        sq = 0.0
        for p in ps:
            if p.grad is not None:
                sq += float(p.grad.detach().float().square().sum())
        out[name] = math.sqrt(max(0.0, sq))
    return out


def step_with_grad(model, ids, mass, item_ids, active):
    """One corrected v1.1 event, differentiable through the current walk."""
    af = active.to(model.item.weight.dtype)[:, None]
    context = model.item(item_ids) * math.sqrt(model.d_model)
    fi, fm = model.router(context, model.space)
    xids, xmass = model._merge(ids, mass * af, fi, fm * af)
    for _ in range(model.layers_n):
        xids, xmass = model.graph(
            xids, xmass, context, model.space, track_touched=False
        )
    ids = torch.where(active[:, None], xids, ids)
    mass = torch.where(active[:, None], xmass, mass)
    msg = (model.space.value(ids) * mass[:, :, None]).sum(1)
    h = model.norm(context + model.message_proj(msg)) * af
    return ids, mass, h


@torch.no_grad()
def transition_no_grad(model, ids, mass, item_ids, active):
    af = active.to(model.item.weight.dtype)[:, None]
    context = model.item(item_ids) * math.sqrt(model.d_model)
    fi, fm = model.router(context, model.space)
    xids, xmass = model._merge(ids, mass * af, fi, fm * af)
    for _ in range(model.layers_n):
        xids, xmass = model.graph(
            xids, xmass, context, model.space, track_touched=False
        )
    ids = torch.where(active[:, None], xids, ids)
    mass = torch.where(active[:, None], xmass, mass)
    return ids, mass


def readout_from_detached_state(model, item_ids, ids, mass, active):
    af = active.to(model.item.weight.dtype)[:, None]
    context = model.item(item_ids) * math.sqrt(model.d_model)
    msg = (model.space.value(ids) * mass[:, :, None]).sum(1)
    return model.norm(context + model.message_proj(msg)) * af


def train_epoch_radius(model, ds, opt, device, args, epoch, arm):
    radius = ARM_RADIUS[arm]
    batch_size = args.batch_size
    loader = fast._loader(ds, batch_size, epoch, True)
    model.train()
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    positions = 0
    padded_positions = 0
    backward_calls = 0
    loss_sum = 0.0
    grad_acc = {k: 0.0 for k in [
        "item", "router", "concept_router", "concept_key",
        "concept_value", "graph", "readout"
    ]}
    grad_batches = 0
    t0 = time.perf_counter()

    for bi, (tokens, lengths) in enumerate(loader, start=1):
        tokens = tokens.to(device, non_blocking=True)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        B, L = x.shape
        total_valid = int(y.ne(0).sum().item())
        if total_valid == 0:
            continue
        positions += total_valid
        padded_positions += int(B * L)

        ids = torch.zeros(B, model.active, dtype=torch.long, device=device)
        mass = torch.zeros(B, model.active, dtype=torch.float32, device=device)
        opt.zero_grad(set_to_none=True)

        chunk_loss = None
        chunk_steps = 0
        batch_loss_value = 0.0

        for t in range(L):
            active = x[:, t].ne(0)
            valid = active & y[:, t].ne(0)

            if radius == 0:
                ids, mass = transition_no_grad(model, ids, mass, x[:, t], active)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    h = readout_from_detached_state(
                        model, x[:, t], ids.detach(), mass.detach(), active
                    )
                    if valid.any():
                        ce_sum = F.cross_entropy(
                            model.score_hidden(h[valid]), y[valid, t], reduction="sum"
                        )
                        loss = ce_sum / total_valid
                    else:
                        loss = None
                if loss is not None:
                    batch_loss_value += float(loss.detach())
                    loss.backward()
                    backward_calls += 1
                mass = mass.detach()
                continue

            # Radius >= 1: previous state is detached only at the requested
            # temporal boundary. The entire current router/hops/readout remains
            # differentiable.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                ids, mass, h = step_with_grad(model, ids, mass, x[:, t], active)
                if valid.any():
                    ce_sum = F.cross_entropy(
                        model.score_hidden(h[valid]), y[valid, t], reduction="sum"
                    )
                    piece = ce_sum / total_valid
                    chunk_loss = piece if chunk_loss is None else chunk_loss + piece
                    batch_loss_value += float(piece.detach())
            chunk_steps += 1

            boundary = (chunk_steps >= radius) or (t == L - 1)
            if boundary:
                if chunk_loss is not None:
                    chunk_loss.backward()
                    backward_calls += 1
                # Break recurrent credit, but keep accumulated parameter grads.
                mass = mass.detach()
                chunk_loss = None
                chunk_steps = 0

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        gg = grad_groups(model)
        for k, v in gg.items():
            grad_acc[k] += v
        grad_batches += 1
        opt.step()
        loss_sum += batch_loss_value

        if args.progress_every > 0 and (
            bi == 1 or bi % args.progress_every == 0 or bi == len(loader)
        ):
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print(
                "CREDIT_PROGRESS",
                json.dumps({
                    "arm": arm,
                    "epoch": epoch,
                    "batch": bi,
                    "batches": len(loader),
                    "positions": positions,
                    "elapsed_s": round(elapsed, 2),
                    "positions_per_s": round(positions / max(elapsed, 1e-9), 1),
                }),
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    z = max(1, grad_batches)
    return {
        "epoch": epoch,
        "arm": arm,
        "credit_radius": radius,
        "loss": loss_sum / max(1, grad_batches),
        "positions": positions,
        "seconds": sec,
        "positions_per_s": positions / max(sec, 1e-9),
        "padding_efficiency": positions / max(1, padded_positions),
        "backward_calls": backward_calls,
        **{f"grad_{k}": grad_acc[k] / z for k in grad_acc},
    }


@torch.inference_mode()
def val_eval(model, split, n_items, device, batch):
    model.eval()
    return evaluate_full(
        model,
        split["val_prefix"],
        split["val_target"],
        n_items,
        MAX_LEN,
        device,
        topks=(10,),
        batch_size=batch,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs-per-arm", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--peak-lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--message-gain", type=float, default=8.0)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--arms", nargs="+", choices=list(ARM_RADIUS), default=list(ARM_RADIUS))
    p.add_argument("--data-dir", default="/content/drive/MyDrive/sparsewalker_data")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_credit_radius")
    a = p.parse_args()

    assert torch.cuda.is_available(), "GPU runtime required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed_all(a.seed)

    print("CREDIT_DATA_START", flush=True)
    t0 = time.perf_counter()
    data = load_dataset("ml1m", a.data_dir)
    split = split_data(data["sequences"])
    ds = WindowDataset(split["train"], MAX_LEN, a.seed)
    print(
        "CREDIT_DATA_DONE",
        json.dumps({
            "seconds": round(time.perf_counter() - t0, 2),
            "users": len(data["sequences"]),
            "n_items": data["n_items"],
        }),
        flush=True,
    )

    # Use exactly the initialization that produced the strong analytic local
    # learner, so the ladder isolates the teaching signal rather than init.
    template = fast.build(data["n_items"]).to(device)
    local_base.init_model(template, a.seed, a.message_gain)
    init_state = cpu_state(template)
    del template
    torch.cuda.empty_cache()

    out = Path(a.output_dir) / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "experiment": "ML1M-CreditRadiusLadder-v1",
        "arms": a.arms,
        "radii": {k: ARM_RADIUS[k] for k in a.arms},
        "same_initialization": True,
        "initialization": "Experiment-43 local-contrastive init",
        "objective": "FullCE",
        "optimizer": "AdamW",
        "test_evaluation_during_screen": False,
        "protocol": "ML1M leave-two-out, max_len=200, full-catalog val, seen masking",
        "args": vars(a),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    ranking = []
    for arm in a.arms:
        print("\nCREDIT_ARM_START", json.dumps({"arm": arm, "radius": ARM_RADIUS[arm]}), flush=True)
        seed_all(a.seed)
        model = fast.build(data["n_items"]).to(device)
        model.load_state_dict(init_state)
        for p0 in model.parameters():
            p0.requires_grad_(True)
            p0.grad = None
        opt = torch.optim.AdamW(
            model.parameters(), lr=a.peak_lr, weight_decay=a.weight_decay
        )

        hist = []
        best = -1.0
        best_epoch = 0
        best_state = None
        for epoch in range(1, a.epochs_per_arm + 1):
            lr = set_lr(opt, epoch, a.epochs_per_arm, a.peak_lr, a.min_lr)
            stats = train_epoch_radius(model, ds, opt, device, a, epoch, arm)
            vt = time.perf_counter()
            val = val_eval(model, split, data["n_items"], device, a.eval_batch_size)
            row = {
                **stats,
                "lr": lr,
                **{f"val_{k}": float(v) for k, v in val.items()},
                "val_seconds": time.perf_counter() - vt,
            }
            hist.append(row)
            print("CREDIT_EPOCH", json.dumps(row), flush=True)
            nd = float(val["NDCG@10"])
            if nd > best:
                best = nd
                best_epoch = epoch
                best_state = cpu_state(model)

        torch.save(
            {
                "model": best_state,
                "arm": arm,
                "credit_radius": ARM_RADIUS[arm],
                "epoch": best_epoch,
                "best_val_NDCG@10": best,
            },
            out / f"best_{arm}.pt",
        )
        (out / f"history_{arm}.json").write_text(json.dumps(hist, indent=2))
        result = {
            "arm": arm,
            "credit_radius": ARM_RADIUS[arm],
            "best_epoch": best_epoch,
            "best_val_NDCG@10": best,
            "final_val_NDCG@10": float(hist[-1]["val_NDCG@10"]),
            "mean_positions_per_s": float(np.mean([r["positions_per_s"] for r in hist])),
            "mean_backward_calls": float(np.mean([r["backward_calls"] for r in hist])),
            "final_grad_router": float(hist[-1]["grad_router"]),
            "final_grad_graph": float(hist[-1]["grad_graph"]),
            "final_grad_concept_key": float(hist[-1]["grad_concept_key"]),
        }
        ranking.append(result)
        print("CREDIT_ARM_DONE", json.dumps(result), flush=True)
        del model, opt
        torch.cuda.empty_cache()

    ranking.sort(key=lambda r: r["best_val_NDCG@10"], reverse=True)
    summary = {
        "config": cfg,
        "ranking": ranking,
        "references": {
            "analytic_backward_free_best_val_NDCG@10": 0.09352186187904432,
            "SASRec_public_protocol_reference_NDCG@10": 0.15965,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("\nCREDIT_RESULT", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
