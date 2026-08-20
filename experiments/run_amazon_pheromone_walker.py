#!/usr/bin/env python
"""From-scratch pheromone/ACO learning on corrected SparseWalker v1.1.

This experiment keeps the successful Amazon SparseWalker recurrence exactly:
- K=8 active concepts
- 65,536 factorized concepts (256x256)
- fresh concepts injected once per event
- duplicate masses coalesced before top-k
- degree=4 graph, 2 graph hops
- fresh_weight=0.25
- same tied item scoring / full-catalog evaluator

There is NO pretrained checkpoint, NO optimizer, NO backward(), and all
parameters have requires_grad=False during behavioral learning.

To isolate whether ACO can learn the sequential structure, item identity is
encoded deterministically into the existing factorized concept space at
initialization. This is not behavior-trained or warm-started: each catalog item
gets a unique concept address through a bijection over the 65,536 concepts.
Behavioral structure is learned only by:
  * positive pheromone deposition on useful existing graph edges,
  * pheromone evaporation,
  * once-per-epoch evidence-accumulated sparse rewiring.

The graph score used by the unchanged SparseWalker recurrence is:
    beta * log(tau_edge) + exp(scale) * <q(context), key(destination)>

Pheromone reward is shaped by agreement with the actual next item's unique
concept address (exact match plus left/right factor matches).
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
    return SparseWalker(n_items, max_len, d=64, layers=2, side=256, h=16,
                        active=8, top_side=2, degree=4, fresh_weight=.25)


def concept_for_item(item_ids, n_concepts):
    """Bijection on 2^16 concepts for item IDs <= 65,535."""
    x = item_ids.long()
    c = (x * 40503 + 7919) % int(n_concepts)
    return torch.where(x.ne(0), c, torch.zeros_like(c))


@torch.no_grad()
def initialize_identity_geometry(model, seed=42, message_gain=16.0):
    """From-scratch stationary item/concept addressing, with no behavior learning."""
    device = model.item.weight.device
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) + 2027)

    left = F.normalize(torch.randn(model.side, model.h, device=device, generator=g), dim=-1)
    right = F.normalize(torch.randn(model.side, model.h, device=device, generator=g), dim=-1)
    model.space.left_router.copy_(left)
    model.space.right_router.copy_(right)
    model.space.left_key.copy_(left)
    model.space.right_key.copy_(right)

    model.router.left_q.weight.zero_()
    model.router.right_q.weight.zero_()
    eye_h = torch.eye(model.h, device=device)
    model.router.left_q.weight[:, : model.h].copy_(eye_h)
    model.router.right_q.weight[:, model.h : 2 * model.h].copy_(eye_h)
    model.router.scale.fill_(math.log(12.0))

    model.graph.context_q.weight.zero_()
    model.graph.context_q.weight[:, : model.h].copy_(eye_h)
    model.graph.context_q.weight[:, model.h : 2 * model.h].add_(eye_h)
    model.graph.scale.fill_(math.log(1.0))

    ids = torch.arange(model.n_items + 1, device=device)
    cid = concept_for_item(ids, model.n_concepts)
    li = cid // model.side
    ri = cid % model.side
    item = torch.zeros(model.n_items + 1, model.d_model, device=device)
    item[:, : model.h] = left[li] / math.sqrt(2.0)
    item[:, model.h : 2 * model.h] = right[ri] / math.sqrt(2.0)
    item[0].zero_()
    model.item.weight.copy_(item)

    model.space.left_value.weight.zero_()
    model.space.right_value.weight.zero_()
    model.space.left_value.weight[:, : model.h].copy_(left)
    model.space.right_value.weight[:, model.h : 2 * model.h].copy_(right)
    model.space.value_proj.weight.zero_()
    model.space.value_proj.bias.zero_()
    eye_d = torch.eye(model.d_model, device=device)
    model.space.value_proj.weight[:, : model.d_model].copy_(eye_d / math.sqrt(2.0))
    model.space.value_proj.weight[:, model.d_model :].add_(eye_d / math.sqrt(2.0))

    model.graph.edge_logits.weight.zero_()
    model.message_proj.weight.copy_(float(message_gain) * eye_d)
    model.norm.weight.fill_(1.0)
    model.norm.bias.zero_()

    dest = torch.randint(0, model.n_concepts, (model.n_concepts, model.degree),
                         device=device, dtype=torch.int32, generator=g)
    dest[:, 0] = torch.arange(model.n_concepts, device=device, dtype=torch.int32)
    model.graph.destination.copy_(dest)

    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None


@torch.no_grad()
def sync_pheromone_logits(model, tau, beta):
    model.graph.edge_logits.weight.copy_(float(beta) * torch.log(tau.clamp_min(1e-8)))


@torch.no_grad()
def router_identity_hit_rate(model):
    ids = torch.arange(1, model.n_items + 1, device=model.item.weight.device)
    e = model.item(ids) * math.sqrt(model.d_model)
    routed, _ = model.router(e, model.space)
    target = concept_for_item(ids, model.n_concepts)
    return float(routed.eq(target[:, None]).any(-1).float().mean().item())


def _loader(dataset, batch_size, epoch):
    dataset.set_epoch(epoch)
    g = torch.Generator()
    g.manual_seed(dataset.seed + epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g,
                      collate_fn=collate_windows, pin_memory=True)


@torch.no_grad()
def pheromone_step(model, ids, mass, context, tau, beta):
    dest = model.graph.destination[ids].long()
    static = float(beta) * torch.log(tau[ids].clamp_min(1e-8))
    q = F.normalize(model.graph.context_q(context), dim=-1)
    key = model.space.key(dest)
    contextual = torch.exp(model.graph.scale) * (key * q[:, None, None, :]).sum(-1)
    prob = F.softmax(static + contextual, dim=-1)
    B = ids.size(0)
    out_ids, out_mass = model.graph.topk(
        dest.reshape(B, -1), (mass.unsqueeze(-1) * prob).reshape(B, -1)
    )
    return out_ids, out_mass, dest, prob


@torch.no_grad()
def edge_reward(dest, target, side):
    dl = dest // int(side)
    dr = dest % int(side)
    tl = (target // int(side))[:, None, None]
    tr = (target % int(side))[:, None, None]
    exact = dest.eq(target[:, None, None]).float()
    factor = 0.2 * dl.eq(tl).float() + 0.2 * dr.eq(tr).float()
    return factor + 0.6 * exact


@torch.no_grad()
def accumulate_proposals(src, src_mass, target, n_concepts, proposal_target, proposal_score):
    K = src.size(1)
    s = src.reshape(-1)
    t = target[:, None].expand(-1, K).reshape(-1)
    w = src_mass.reshape(-1).float()
    valid = w > 0
    if not valid.any():
        return
    s, t, w = s[valid], t[valid], w[valid]
    pair = s * int(n_concepts) + t
    up, inv = torch.unique(pair, return_inverse=True)
    agg = torch.zeros(up.numel(), device=w.device, dtype=torch.float32)
    agg.scatter_add_(0, inv, w)
    us = up // int(n_concepts)
    ut = up % int(n_concepts)

    best = torch.full((int(n_concepts),), -float("inf"), device=w.device,
                      dtype=torch.float32)
    best.scatter_reduce_(0, us, agg, reduce="amax", include_self=True)
    is_best = agg >= best[us]
    sentinel = int(n_concepts)
    bt = torch.full((sentinel,), sentinel, device=w.device, dtype=torch.long)
    bt.scatter_reduce_(0, us[is_best], ut[is_best], reduce="amin", include_self=True)
    rows = (bt < sentinel).nonzero(as_tuple=False).squeeze(-1)
    if rows.numel() == 0:
        return

    key = us * int(n_concepts) + ut
    selected_key = rows * int(n_concepts) + bt[rows]
    order = torch.argsort(key)
    key_sorted = key[order]
    agg_sorted = agg[order]
    pos = torch.searchsorted(key_sorted, selected_key)
    bs = agg_sorted[pos.clamp_max(max(0, agg_sorted.numel() - 1))]

    same = proposal_target[rows].eq(bt[rows])
    proposal_score[rows[same]] += bs[same]
    other_rows = rows[~same]
    other_score = bs[~same]
    replace = other_score > proposal_score[other_rows]
    rr = other_rows[replace]
    if rr.numel():
        proposal_target[rr] = bt[rr]
        proposal_score[rr] = other_score[replace]
    lose = other_rows[~replace]
    if lose.numel():
        proposal_score[lose] *= 0.98


@torch.no_grad()
def rewire_epoch(model, tau, proposal_target, proposal_score, fraction, init_tau):
    valid = proposal_target.ge(0) & proposal_score.gt(0)
    rows = valid.nonzero(as_tuple=False).squeeze(-1)
    if rows.numel() == 0 or fraction <= 0:
        return 0
    n_take = max(1, int(math.ceil(rows.numel() * float(fraction))))
    scores = proposal_score[rows]
    take = rows[torch.topk(scores, min(n_take, rows.numel()), largest=True).indices]
    tgt = proposal_target[take]
    current = model.graph.destination[take].long()
    missing = ~current.eq(tgt[:, None]).any(-1)
    take = take[missing]
    tgt = tgt[missing]
    if take.numel() == 0:
        return 0
    if model.degree <= 1:
        slot = torch.zeros_like(take)
    else:
        slot = 1 + tau[take, 1:].argmin(-1)
    model.graph.destination[take, slot] = tgt.to(torch.int32)
    tau[take, slot] = float(init_tau)
    return int(take.numel())


@torch.no_grad()
def aco_epoch(model, dataset, device, args, epoch, tau):
    model.eval()
    total_positions = 0
    batches = 0
    total_reward = 0.0
    reward_weight = 0.0
    total_deposit = 0.0
    proposal_target = torch.full((model.n_concepts,), -1, device=device, dtype=torch.long)
    proposal_score = torch.zeros(model.n_concepts, device=device, dtype=torch.float32)
    t0 = time.perf_counter()

    for tokens, lengths in _loader(dataset, args.batch_size, epoch):
        tokens = tokens.to(device, non_blocking=True)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        valid_all = x.ne(0) & y.ne(0)
        npos = int(valid_all.sum().item())
        if npos == 0:
            continue

        item_state = model.item(x) * math.sqrt(model.d_model)
        fi, fm = model.router(item_state.reshape(-1, model.d_model), model.space)
        B, L = x.shape
        fi = fi.view(B, L, -1)
        fm = fm.view(B, L, -1)
        ids = torch.zeros(B, model.active, device=device, dtype=torch.long)
        mass = torch.zeros(B, model.active, device=device, dtype=item_state.dtype)
        deposit = torch.zeros_like(tau)
        visit = torch.zeros_like(tau)

        for t in range(L):
            act = x[:, t].ne(0)
            if not act.any():
                continue
            af = act.to(item_state.dtype)[:, None]
            xids, xmass = model._merge(ids, mass * af, fi[:, t], fm[:, t] * af)
            valid = valid_all[:, t]
            target = concept_for_item(y[valid, t], model.n_concepts) if valid.any() else None

            for _ in range(model.layers_n):
                if valid.any():
                    accumulate_proposals(xids[valid], xmass[valid].float(), target,
                                         model.n_concepts, proposal_target, proposal_score)

                out_ids, out_mass, dest, prob = pheromone_step(
                    model, xids, xmass, item_state[:, t], tau, args.pheromone_beta
                )

                if valid.any():
                    src = xids[valid]
                    sm = xmass[valid].float()
                    d = dest[valid]
                    p = prob[valid].float()
                    r = edge_reward(d, target, model.side)
                    weighted = sm[:, :, None] * p
                    dep = weighted * r
                    slots = torch.arange(model.degree, device=device)[None, None, :]
                    flat = (src[:, :, None] * model.degree + slots).expand_as(d).reshape(-1)
                    deposit.view(-1).scatter_add_(0, flat, dep.reshape(-1))
                    visit.view(-1).scatter_add_(0, flat, weighted.reshape(-1))
                    total_reward += float(dep.sum().item())
                    reward_weight += float(weighted.sum().item())

                xids, xmass = out_ids, out_mass

            ids = torch.where(act[:, None], xids, ids)
            mass = torch.where(act[:, None], xmass, mass)

        used = visit > 0
        avg = torch.zeros_like(deposit)
        avg[used] = deposit[used] / visit[used].clamp_min(1e-8)
        tau.add_(float(args.deposit_lr) * avg)
        tau.clamp_(float(args.tau_min), float(args.tau_max))
        total_deposit += float(avg.sum().item())
        total_positions += npos
        batches += 1

    tau.sub_(float(args.tau_min)).mul_(1.0 - float(args.evaporation)).add_(float(args.tau_min))
    tau.clamp_(float(args.tau_min), float(args.tau_max))
    rewired = rewire_epoch(model, tau, proposal_target, proposal_score,
                           args.rewire_fraction, args.new_edge_tau)
    sync_pheromone_logits(model, tau, args.pheromone_beta)

    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    grads = [name for name, p in model.named_parameters() if p.grad is not None]
    if grads:
        raise AssertionError(f"ACO run created gradients: {grads[:8]}")
    return {
        "epoch": int(epoch), "positions": int(total_positions), "seconds": float(sec),
        "positions_per_s": float(total_positions / max(sec, 1e-9)), "batches": int(batches),
        "mean_edge_reward": float(total_reward / max(reward_weight, 1e-9)),
        "deposit_sum": float(total_deposit), "rewired_edges": int(rewired),
        "tau_mean": float(tau.mean().item()), "tau_max": float(tau.max().item()),
        "autograd_grad_tensors": 0, "loss_backward_calls": 0, "optimizer": None,
    }


def evaluate(model, split, n_items, device, batch_size=1024):
    model.eval()
    val = evaluate_full(model, split["val_prefix"], split["val_target"], n_items, 50,
                        device, topks=(10,), batch_size=batch_size)
    test = evaluate_full(model, split["test_prefix"], split["test_target"], n_items, 50,
                         device, topks=(10, 20, 50), batch_size=batch_size)
    return val, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=AMAZON, default="beauty")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--pheromone-beta", type=float, default=2.0)
    p.add_argument("--deposit-lr", type=float, default=0.5)
    p.add_argument("--evaporation", type=float, default=0.10)
    p.add_argument("--tau-min", type=float, default=0.10)
    p.add_argument("--tau-max", type=float, default=20.0)
    p.add_argument("--new-edge-tau", type=float, default=2.0)
    p.add_argument("--rewire-fraction", type=float, default=0.05)
    p.add_argument("--message-gain", type=float, default=16.0)
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_pheromone_from_scratch")
    args = p.parse_args()

    assert torch.cuda.is_available(), "GPU runtime required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed_all(args.seed)

    data = load_dataset(args.dataset, args.data_dir)
    split = split_data(data["sequences"])
    model = build_model(data["n_items"], 50).to(device)
    initialize_identity_geometry(model, args.seed, args.message_gain)
    tau = torch.ones(model.n_concepts, model.degree, device=device, dtype=torch.float32)
    sync_pheromone_logits(model, tau, args.pheromone_beta)

    all_items = torch.arange(1, data["n_items"] + 1, device=device)
    mapped = concept_for_item(all_items, model.n_concepts)
    unique_addresses = int(torch.unique(mapped).numel())
    router_hit = router_identity_hit_rate(model)
    init_val, init_test = evaluate(model, split, data["n_items"], device, args.eval_batch_size)
    print("PHERO_INIT", json.dumps({
        "unique_item_concept_addresses": unique_addresses,
        "n_items": int(data["n_items"]), "router_target_hit_rate": router_hit,
        "val": init_val, "test": init_test,
    }), flush=True)
    if unique_addresses != int(data["n_items"]):
        raise AssertionError(f"Item concept addresses collided: {unique_addresses}/{data['n_items']}")
    if router_hit < 0.99:
        raise AssertionError(f"Identity router initialization failed: hit_rate={router_hit}")

    ds = WindowDataset(split["train"], 50, args.seed)
    out = Path(args.output_dir) / args.dataset / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "PheromoneWalker-v1-from-scratch",
        "base_architecture": "corrected SparseWalker v1.1 Amazon winner",
        "dataset": args.dataset, "seed": args.seed, "warm_start": False,
        "pretrained_checkpoint": None, "autograd": False, "optimizer": None,
        "loss_backward_calls": 0,
        "behavior_learning": "ACO pheromone + evaporation + epoch-level sparse rewiring",
        "identity_initialization": "unique bijective item->concept address; no sequence data",
        "architecture": {"d": 64, "side": 256, "n_concepts": 65536, "active": 8,
                         "degree": 4, "graph_hops": 2, "fresh_weight": 0.25, "max_len": 50},
        "aco": {"pheromone_beta": args.pheromone_beta, "deposit_lr": args.deposit_lr,
                "evaporation": args.evaporation, "tau_min": args.tau_min,
                "tau_max": args.tau_max, "new_edge_tau": args.new_edge_tau,
                "rewire_fraction": args.rewire_fraction},
        "references": {"SASRec_test_NDCG@10": 0.031195719394901355,
                       "SparseWalker_v11_test_NDCG@10": 0.044882819399656555},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    best = float(init_val["NDCG@10"])
    best_epoch = 0
    best_state = cpu_state_dict(model)
    best_tau = tau.detach().cpu().clone()
    history = []
    for epoch in range(1, args.epochs + 1):
        stats = aco_epoch(model, ds, device, args, epoch, tau)
        if epoch == 1 or epoch % args.eval_every == 0:
            val = evaluate_full(model, split["val_prefix"], split["val_target"],
                                data["n_items"], 50, device, topks=(10,),
                                batch_size=args.eval_batch_size)
            row = {**stats, **{f"val_{k}": float(v) for k, v in val.items()}}
            history.append(row)
            print("PHERO_EPOCH", json.dumps(row), flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=2))
            ndcg = float(val["NDCG@10"])
            if ndcg > best:
                best = ndcg
                best_epoch = epoch
                best_state = cpu_state_dict(model)
                best_tau = tau.detach().cpu().clone()
                torch.save({"model": best_state, "tau": best_tau, "epoch": epoch,
                            "val": val, "config": config}, out / "best.pt")

    model.load_state_dict(best_state)
    tau.copy_(best_tau.to(device))
    sync_pheromone_logits(model, tau, args.pheromone_beta)
    best_val, best_test = evaluate(model, split, data["n_items"], device, args.eval_batch_size)
    result = {
        "config": config, "initial": {"val": init_val, "test": init_test},
        "best_epoch": int(best_epoch), "best_pheromone": {"val": best_val, "test": best_test},
        "vs_sasrec_test_ndcg_ratio": float(best_test["NDCG@10"] / 0.031195719394901355),
        "vs_backprop_walker_test_ndcg_ratio": float(best_test["NDCG@10"] / 0.044882819399656555),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("PHERO_RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
