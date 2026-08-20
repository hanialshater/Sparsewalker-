#!/usr/bin/env python
"""Backward-free local graph learning for corrected SparseWalker v1.1.

This is intentionally the first, narrow backward-free experiment rather than a
claim that the whole representation is learned without gradients.

Protocol
--------
1. Load a trained SparseWalker v1.1 checkpoint to provide item embeddings,
   factorized router, concept keys/values, context projection and readout.
2. Freeze *every* parameter (requires_grad=False).
3. Optionally reset the graph's learned static edge logits to zero.
4. Relearn only those sparse edge biases using a forward-only local rule:

       delta(edge) ~ source_mass * p(edge|context) * (reward(edge)-E_p[reward])

   where reward(edge) is cosine compatibility between the destination concept's
   projected value and the actual next-item embedding.
5. No loss.backward(), no optimizer and no autograd graph are used during the
   learning epochs.

Because degree=4, all outgoing actions are enumerated exactly. The update is the
local expected-reward / replicator-gradient rule for the four-edge policy, but
is applied analytically rather than through backpropagation.

The experiment reports:
- pretrained checkpoint quality;
- quality immediately after resetting edge logits;
- validation recovery during backward-free learning;
- final test quality at the best backward-free validation checkpoint;
- positions/sec and an explicit assertion that no gradients were created.

This tests whether SparseWalker's *graph dynamics* can be learned locally. A
positive result would justify progressively replacing gradient learning in the
router/concept geometry in later experiments.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sparsewalker.data import load_dataset, split_data, WindowDataset, collate_windows
from sparsewalker.evaluation import evaluate_full
from sparsewalker.models import SparseWalker


AMAZON = ("beauty", "video_games", "sports", "toys")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def build_model(n_items, max_len=50):
    return SparseWalker(
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
    )


def evaluate(model, split, n_items, device, max_len=50, batch_size=1024):
    model.eval()
    val = evaluate_full(
        model,
        split["val_prefix"],
        split["val_target"],
        n_items,
        max_len,
        device,
        topks=(10,),
        batch_size=batch_size,
    )
    test = evaluate_full(
        model,
        split["test_prefix"],
        split["test_target"],
        n_items,
        max_len,
        device,
        topks=(10, 20, 50),
        batch_size=batch_size,
    )
    return val, test


def _loader(dataset, batch_size, epoch):
    dataset.set_epoch(epoch)
    g = torch.Generator()
    g.manual_seed(dataset.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        collate_fn=collate_windows,
        pin_memory=True,
    )


@torch.no_grad()
def backward_free_epoch(model, dataset, device, *, batch_size, epoch, local_lr):
    """One completely autograd-free graph-learning epoch.

    The learned state is only graph.edge_logits. All representation parameters
    are frozen and destination topology is held fixed. Updates are averaged per
    visited (source concept, edge slot) within each batch before being applied.
    """
    model.eval()
    degree = int(model.degree)
    n_concepts = int(model.n_concepts)
    edge_weight = model.graph.edge_logits.weight
    flat_size = n_concepts * degree

    total_positions = 0
    reward_sum = 0.0
    reward_count = 0
    update_l1_sum = 0.0
    batches = 0
    t0 = time.perf_counter()

    for tokens, lengths in _loader(dataset, batch_size, epoch):
        tokens = tokens.to(device, non_blocking=True)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        B, L = x.shape
        valid = y.ne(0)
        valid_positions = int(valid.sum().cpu())
        if valid_positions == 0:
            continue

        # Route all current items once; recurrence itself remains chronological.
        item_state = model.item(x) * math.sqrt(model.d_model)
        fi, fm = model.router(item_state.reshape(B * L, model.d_model), model.space)
        fi = fi.view(B, L, -1)
        fm = fm.view(B, L, -1)

        ids = torch.zeros(B, model.active, dtype=torch.long, device=device)
        mass = torch.zeros(B, model.active, dtype=item_state.dtype, device=device)

        # Aggregate local plasticity for this batch. 65,536*4 is only 262k
        # scalar slots, so dense scratch is cheap and avoids write races.
        update = torch.zeros(flat_size, device=device, dtype=torch.float32)
        visits = torch.zeros(flat_size, device=device, dtype=torch.float32)

        for t in range(L):
            act = x[:, t].ne(0)
            af = act.to(item_state.dtype)[:, None]
            yt = y[:, t]
            target_valid = yt.ne(0) & act

            xids, xmass = model._merge(
                ids,
                mass * af,
                fi[:, t],
                fm[:, t] * af,
            )

            # Learn at each sparse graph hop using the actual next item as a
            # local teaching signal. No gradient is propagated through time or
            # through any representation parameter.
            for _ in range(model.layers_n):
                src = xids
                sm = xmass.float()
                dest = model.graph.destination[src].long()  # [B,K,degree]
                static = model.graph.edge_logits(src).float()

                q = F.normalize(model.graph.context_q(item_state[:, t]), dim=-1)
                key = model.space.key(dest).float()
                contextual = torch.exp(model.graph.scale.float()) * (
                    key * q[:, None, None, :].float()
                ).sum(-1)
                prob = torch.softmax(static + contextual, dim=-1)

                # Concept message compatibility with the actual next item. This
                # is the same representation family used by Walker readout and
                # tied item scoring, but converted into a local scalar reward.
                dv = model.space.value(dest)
                projected = F.normalize(model.message_proj(dv).float(), dim=-1)
                target = F.normalize(model.item.weight[yt.clamp_min(0)].float(), dim=-1)
                reward = (projected * target[:, None, None, :]).sum(-1)
                reward = reward * target_valid[:, None, None].float()

                baseline = (prob * reward).sum(-1, keepdim=True)
                advantage = reward - baseline
                delta = (
                    float(local_lr)
                    * sm[:, :, None]
                    * prob
                    * advantage
                    * target_valid[:, None, None].float()
                )

                slots = torch.arange(degree, device=device)[None, None, :]
                flat = (src[:, :, None] * degree + slots).expand_as(dest).reshape(-1)
                update.scatter_add_(0, flat, delta.reshape(-1))
                visits.scatter_add_(
                    0,
                    flat,
                    (
                        sm[:, :, None]
                        * target_valid[:, None, None].float()
                    ).expand_as(delta).reshape(-1),
                )

                if target_valid.any():
                    reward_sum += float(
                        (baseline.squeeze(-1) * sm * target_valid[:, None].float()).sum().cpu()
                    )
                    reward_count += int(
                        (target_valid[:, None].expand_as(sm)).sum().cpu()
                    )

                # Forward state transition is unchanged SparseWalker v1.1.
                xids, xmass = model.graph(
                    xids,
                    xmass,
                    item_state[:, t],
                    model.space,
                    track_touched=False,
                )

            ids = torch.where(act[:, None], xids, ids)
            mass = torch.where(act[:, None], xmass, mass)

        mask = visits > 0
        batch_update = torch.zeros_like(update)
        batch_update[mask] = update[mask] / visits[mask].clamp_min(1e-8)
        edge_weight.add_(batch_update.view_as(edge_weight))
        edge_weight.clamp_(-8.0, 8.0)
        update_l1_sum += float(batch_update.abs().sum().cpu())
        total_positions += valid_positions
        batches += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    # Hard safety gate: backward-free means *no* gradient tensors were created.
    grads = [name for name, p in model.named_parameters() if p.grad is not None]
    if grads:
        raise AssertionError(f"Backward-free epoch unexpectedly created gradients: {grads[:8]}")

    return {
        "epoch": int(epoch),
        "positions": int(total_positions),
        "seconds": float(seconds),
        "positions_per_s": float(total_positions / max(seconds, 1e-9)),
        "batches": int(batches),
        "mean_local_reward": float(reward_sum / max(1, reward_count)),
        "edge_update_l1": float(update_l1_sum),
        "autograd_grad_tensors": 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=AMAZON, default="beauty")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--local-lr", type=float, default=.5)
    p.add_argument("--reset-edge-logits", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument(
        "--checkpoint-root",
        default="/content/drive/MyDrive/sparsewalker_amazon_quality_v11",
    )
    p.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/sparsewalker_backward_free_graph",
    )
    args = p.parse_args()

    assert torch.cuda.is_available(), "GPU runtime required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed_all(args.seed)

    data = load_dataset(args.dataset, args.data_dir)
    split = split_data(data["sequences"])
    model = build_model(data["n_items"], 50).to(device)

    ckpt_path = (
        Path(args.checkpoint_root)
        / args.dataset
        / f"seed{args.seed}"
        / "SparseWalker"
        / "best.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing {ckpt_path}. Run notebook 33's Amazon quality section first."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Freeze all representation learning. The loop below is pure torch.no_grad().
    for p0 in model.parameters():
        p0.requires_grad_(False)
        p0.grad = None

    pretrained_val, pretrained_test = evaluate(
        model, split, data["n_items"], device, batch_size=args.eval_batch_size
    )
    print(
        "BF_PRETRAINED",
        json.dumps({"val": pretrained_val, "test": pretrained_test}),
        flush=True,
    )

    if args.reset_edge_logits:
        with torch.no_grad():
            model.graph.edge_logits.weight.zero_()
    reset_val, reset_test = evaluate(
        model, split, data["n_items"], device, batch_size=args.eval_batch_size
    )
    print(
        "BF_RESET",
        json.dumps({"val": reset_val, "test": reset_test}),
        flush=True,
    )

    ds = WindowDataset(split["train"], 50, args.seed)
    best = float(reset_val["NDCG@10"])
    best_epoch = 0
    best_state = cpu_state_dict(model)
    history = []

    out = Path(args.output_dir) / args.dataset / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "BackwardFreeGraph-v1",
        "dataset": args.dataset,
        "seed": args.seed,
        "warm_start": str(ckpt_path),
        "representation_parameters": "frozen pretrained SparseWalker v1.1",
        "learned_without_backward": "graph.edge_logits only",
        "destination_topology": "fixed checkpoint topology",
        "reset_edge_logits": bool(args.reset_edge_logits),
        "local_rule": "source_mass * p(edge|context) * (reward-E[reward])",
        "reward": "cosine(projected destination concept value, next-item embedding)",
        "local_lr": args.local_lr,
        "optimizer": None,
        "loss_backward_calls": 0,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    for epoch in range(1, args.epochs + 1):
        stats = backward_free_epoch(
            model,
            ds,
            device,
            batch_size=args.batch_size,
            epoch=epoch,
            local_lr=args.local_lr,
        )
        val = evaluate_full(
            model,
            split["val_prefix"],
            split["val_target"],
            data["n_items"],
            50,
            device,
            topks=(10,),
            batch_size=args.eval_batch_size,
        )
        row = {**stats, **{f"val_{k}": float(v) for k, v in val.items()}}
        history.append(row)
        print("BF_EPOCH", json.dumps(row), flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))

        ndcg = float(val["NDCG@10"])
        if ndcg > best:
            best = ndcg
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            torch.save(
                {
                    "model": best_state,
                    "epoch": epoch,
                    "val": val,
                    "config": config,
                },
                out / "best.pt",
            )

    model.load_state_dict(best_state)
    model.eval()
    best_val, best_test = evaluate(
        model, split, data["n_items"], device, batch_size=args.eval_batch_size
    )
    result = {
        "config": config,
        "pretrained": {"val": pretrained_val, "test": pretrained_test},
        "after_edge_reset": {"val": reset_val, "test": reset_test},
        "best_backward_free_epoch": int(best_epoch),
        "best_backward_free": {"val": best_val, "test": best_test},
        "recovery_fraction_val": (
            (float(best_val["NDCG@10"]) - float(reset_val["NDCG@10"]))
            / max(
                1e-12,
                float(pretrained_val["NDCG@10"]) - float(reset_val["NDCG@10"]),
            )
        ),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("BF_RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
