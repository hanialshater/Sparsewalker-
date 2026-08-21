#!/usr/bin/env python
"""Contextual multi-timescale PheromoneWalker v2.

From-scratch experiment built on corrected SparseWalker v1.1 recurrence:
- K=8 active concepts, 65,536 concepts, degree=4, two graph hops
- fresh concepts injected once/event, duplicate masses coalesced before top-k
- no warm start, no pretrained checkpoint, no optimizer, no backward(), no autograd

Compared with PheromoneWalker v1 this adds:
1) slow global pheromone;
2) fast per-session pheromone carried inside the forward pass;
3) per-edge context prototypes learned from successful usage;
4) trajectory / eligibility credit to recent successful trails;
5) reward tied directly to next-item ranking against deterministic negatives.

Item->concept addressing is unique and behavior-free, exactly as in v1.
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

from run_amazon_pheromone_walker import (
    AMAZON,
    seed_all,
    cpu_state_dict,
    concept_for_item,
    initialize_identity_geometry,
    router_identity_hit_rate,
    accumulate_proposals,
    rewire_epoch,
)

REFERENCE_SASREC = 0.031195719394901355
REFERENCE_WALKER = 0.044882819399656555


def deterministic_negatives(target, n_items, n_negs):
    """Deterministic catalog negatives for a stationary local ranking reward."""
    target = target.long()
    j = torch.arange(1, int(n_negs) + 1, device=target.device, dtype=torch.long)
    neg = ((target[:, None] * 1103515245 + j[None, :] * 2654435761 + 1013904223) % int(n_items)) + 1
    same = neg.eq(target[:, None])
    neg = torch.where(same, (neg % int(n_items)) + 1, neg)
    return neg


@torch.no_grad()
def pairwise_rank_reward(model, hidden, target, n_negs=64, temperature=1.0):
    """Centered pairwise next-item reward in [-1, 1]. 0 ~= chance."""
    if hidden.numel() == 0:
        return hidden.new_zeros((0,), dtype=torch.float32)
    target = target.long()
    pos = (hidden.float() * model.item.weight[target].float()).sum(-1)
    neg = deterministic_negatives(target, model.n_items, n_negs)
    neg_score = (
        hidden[:, None, :].float() * model.item.weight[neg].float()
    ).sum(-1)
    win = torch.sigmoid((pos[:, None] - neg_score) / float(temperature)).mean(-1)
    return (2.0 * win - 1.0).clamp(-1.0, 1.0)


class ContextualPheromoneWalker(SparseWalker):
    """SparseWalker v1.1 with mutable slow/context pheromone buffers.

    Fast pheromone is session-local and therefore is not stored globally.
    """

    def __init__(
        self,
        n_items,
        max_len,
        *,
        slow_beta=1.5,
        fast_beta=1.0,
        context_gamma=1.0,
        fast_capacity=32,
        fast_decay=0.90,
        fast_add=4,
        reward_negs=64,
        reward_temperature=1.0,
    ):
        super().__init__(
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
        self.slow_beta = float(slow_beta)
        self.fast_beta = float(fast_beta)
        self.context_gamma = float(context_gamma)
        self.fast_capacity = int(fast_capacity)
        self.fast_decay = float(fast_decay)
        self.fast_add = int(fast_add)
        self.reward_negs = int(reward_negs)
        self.reward_temperature = float(reward_temperature)

        self.register_buffer(
            "slow_tau",
            torch.ones(self.n_concepts, self.degree, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "context_proto",
            torch.zeros(self.n_concepts, self.degree, self.h, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "context_strength",
            torch.zeros(self.n_concepts, self.degree, dtype=torch.float32),
            persistent=True,
        )

    @torch.no_grad()
    def reset_pheromone(self):
        self.slow_tau.fill_(1.0)
        self.context_proto.zero_()
        self.context_strength.zero_()
        self.graph.edge_logits.weight.zero_()

    @torch.no_grad()
    def _fast_bonus(self, flat_edges, trail_ids, trail_values):
        # flat_edges [B,K,D], trail_* [B,C]
        if trail_ids.numel() == 0:
            return flat_edges.new_zeros(flat_edges.shape, dtype=torch.float32)
        same = flat_edges[..., None].eq(trail_ids[:, None, None, :])
        return (same.float() * trail_values[:, None, None, :]).sum(-1)

    @torch.no_grad()
    def pheromone_step(self, ids, mass, context, trail_ids, trail_values):
        dest = self.graph.destination[ids].long()
        q = F.normalize(self.graph.context_q(context), dim=-1).float()
        key = self.space.key(dest).float()
        geometry = torch.exp(self.graph.scale.float()) * (
            key * q[:, None, None, :]
        ).sum(-1)

        slow = self.slow_beta * torch.log(self.slow_tau[ids].clamp_min(1e-8))
        slots = torch.arange(self.degree, device=ids.device)[None, None, :]
        flat = (ids[:, :, None] * self.degree + slots).expand_as(dest)

        fast = self.fast_beta * self._fast_bonus(flat, trail_ids, trail_values)

        proto = self.context_proto.view(-1, self.h)[flat.reshape(-1)].view(
            *flat.shape, self.h
        )
        proto = F.normalize(proto, dim=-1)
        ctx = self.context_gamma * (
            proto * q[:, None, None, :]
        ).sum(-1)
        strength = self.context_strength.view(-1)[flat.reshape(-1)].view_as(flat).float()
        ctx = ctx * strength.clamp(0.0, 1.0)

        prob = F.softmax(geometry + slow + fast + ctx, dim=-1)
        B = ids.size(0)
        usage = mass.unsqueeze(-1).float() * prob.float()
        out_ids, out_mass = self.graph.topk(
            dest.reshape(B, -1),
            usage.reshape(B, -1).to(mass.dtype),
        )
        return out_ids, out_mass, flat, usage, q

    @torch.no_grad()
    def _update_fast_trail(self, trail_ids, trail_values, traces, reward):
        """Immediate session memory update after the next event is observed."""
        trail_values.mul_(self.fast_decay)
        if not traces or reward.numel() == 0:
            return
        ids = torch.cat([x[0].reshape(x[0].size(0), -1) for x in traces], dim=1)
        use = torch.cat([x[1].reshape(x[1].size(0), -1) for x in traces], dim=1)
        positive = reward.clamp_min(0.0)[:, None]
        score = use * positive
        k = min(self.fast_add, score.size(1))
        if k <= 0:
            return
        topv, topi = torch.topk(score, k=k, dim=1)
        chosen = ids.gather(1, topi)
        rows_all = torch.arange(ids.size(0), device=ids.device)

        for j in range(k):
            edge = chosen[:, j]
            val = topv[:, j]
            valid = val > 0
            if not valid.any():
                continue
            eq = trail_ids.eq(edge[:, None])
            exists = eq.any(-1)
            existing_slot = eq.float().argmax(-1)
            weakest_slot = trail_values.argmin(-1)
            slot = torch.where(exists, existing_slot, weakest_slot)
            rr = rows_all[valid]
            ss = slot[valid]
            old = trail_values[rr, ss]
            trail_ids[rr, ss] = edge[valid]
            trail_values[rr, ss] = (old + val[valid]).clamp_max(4.0)

    @torch.no_grad()
    def encode(self, seq):
        """Evaluation/serving forward pass with online fast pheromone only.

        Global slow/context pheromone is read-only. Within a known prefix, each
        observed transition can update session-local fast pheromone before later
        events are processed. The held-out target is never used.
        """
        B, L = seq.shape
        valid = seq.ne(0)
        item_state = self.item(seq) * math.sqrt(self.d_model)
        fi, fm = self.router(item_state.reshape(B * L, self.d_model), self.space)
        fi = fi.view(B, L, -1)
        fm = fm.view(B, L, -1)

        ids = torch.zeros(B, self.active, dtype=torch.long, device=seq.device)
        mass = torch.zeros(B, self.active, dtype=item_state.dtype, device=seq.device)
        trail_ids = torch.full(
            (B, self.fast_capacity), -1, dtype=torch.long, device=seq.device
        )
        trail_values = torch.zeros(
            B, self.fast_capacity, dtype=torch.float32, device=seq.device
        )
        outs = []

        for t in range(L):
            act = valid[:, t]
            af = act.to(item_state.dtype)[:, None]
            xids, xmass = self._merge(
                ids,
                mass * af,
                fi[:, t],
                fm[:, t] * af,
            )
            traces = []
            for _ in range(self.layers_n):
                xids, xmass, flat, usage, q = self.pheromone_step(
                    xids, xmass, item_state[:, t], trail_ids, trail_values
                )
                traces.append((flat, usage, q))

            ids = torch.where(act[:, None], xids, ids)
            mass = torch.where(act[:, None], xmass, mass)
            msg = (self.space.value(ids) * mass[:, :, None]).sum(1)
            h = self.norm(item_state[:, t] + self.message_proj(msg)) * af
            outs.append(h)

            # The next event inside the prefix is already observed history, so
            # it may update session-local forward memory. Never mutate globals.
            if t + 1 < L:
                nxt = seq[:, t + 1]
                known = act & nxt.ne(0)
                if known.any():
                    r = pairwise_rank_reward(
                        self,
                        h[known],
                        nxt[known],
                        self.reward_negs,
                        self.reward_temperature,
                    )
                    full_r = torch.zeros(B, device=seq.device, dtype=torch.float32)
                    full_r[known] = r
                    self._update_fast_trail(trail_ids, trail_values, traces, full_r)
                else:
                    trail_values.mul_(self.fast_decay)

        return torch.stack(outs, 1)


def build_model(n_items, max_len, args):
    return ContextualPheromoneWalker(
        n_items,
        max_len,
        slow_beta=args.slow_beta,
        fast_beta=args.fast_beta,
        context_gamma=args.context_gamma,
        fast_capacity=args.fast_capacity,
        fast_decay=args.fast_decay,
        fast_add=args.fast_add,
        reward_negs=args.reward_negs,
        reward_temperature=args.reward_temperature,
    )


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
def _accumulate_context(ctx_sum, ctx_weight, flat, usage, q, positive_reward):
    w = usage * positive_reward[:, None, None]
    idx = flat.reshape(-1)
    ww = w.reshape(-1)
    qv = q[:, None, None, :].expand(
        -1, flat.size(1), flat.size(2), -1
    ).reshape(-1, q.size(-1))
    ctx_sum.index_add_(0, idx, qv * ww[:, None])
    ctx_weight.index_add_(0, idx, ww)


@torch.no_grad()
def _accumulate_edge_credit(deposit, visit, flat, usage, reward, decay=1.0):
    idx = flat.reshape(-1)
    use = usage.reshape(-1)
    rr = reward[:, None, None].expand_as(usage).reshape(-1)
    deposit.view(-1).scatter_add_(0, idx, float(decay) * use * rr)
    visit.view(-1).scatter_add_(0, idx, float(decay) * use.abs())


@torch.no_grad()
def _accumulate_trail_credit(deposit, visit, trail_ids, trail_values, reward, weight):
    valid = trail_ids.ge(0) & trail_values.gt(0)
    if not valid.any():
        return
    idx = trail_ids[valid]
    rr = reward[:, None].expand_as(trail_values)[valid]
    use = trail_values[valid]
    deposit.view(-1).scatter_add_(0, idx, float(weight) * use * rr)
    visit.view(-1).scatter_add_(0, idx, float(weight) * use.abs())


@torch.no_grad()
def _apply_context_update(model, ctx_sum, ctx_weight, lr):
    used = ctx_weight > 0
    rows = used.nonzero(as_tuple=False).squeeze(-1)
    if rows.numel() == 0:
        return 0
    target = F.normalize(ctx_sum[rows] / ctx_weight[rows, None].clamp_min(1e-8), dim=-1)
    proto = model.context_proto.view(-1, model.h)
    strength = model.context_strength.view(-1)
    old = proto[rows]
    new = F.normalize((1.0 - float(lr)) * old + float(lr) * target, dim=-1)
    proto[rows] = new
    strength[rows] = (strength[rows] + float(lr)).clamp_max(1.0)
    return int(rows.numel())


@torch.no_grad()
def contextual_aco_epoch(model, dataset, device, args, epoch):
    model.eval()
    total_positions = 0
    batches = 0
    reward_sum = 0.0
    reward_count = 0
    positive_count = 0
    total_credit = 0.0
    context_updates = 0

    proposal_target = torch.full(
        (model.n_concepts,), -1, device=device, dtype=torch.long
    )
    proposal_score = torch.zeros(
        model.n_concepts, device=device, dtype=torch.float32
    )
    t0 = time.perf_counter()

    for tokens, lengths in _loader(dataset, args.batch_size, epoch):
        tokens = tokens.to(device, non_blocking=True)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        valid_all = x.ne(0) & y.ne(0)
        npos = int(valid_all.sum().item())
        if npos == 0:
            continue

        B, L = x.shape
        item_state = model.item(x) * math.sqrt(model.d_model)
        fi, fm = model.router(item_state.reshape(B * L, model.d_model), model.space)
        fi = fi.view(B, L, -1)
        fm = fm.view(B, L, -1)

        ids = torch.zeros(B, model.active, dtype=torch.long, device=device)
        mass = torch.zeros(B, model.active, dtype=item_state.dtype, device=device)
        trail_ids = torch.full(
            (B, model.fast_capacity), -1, dtype=torch.long, device=device
        )
        trail_values = torch.zeros(
            B, model.fast_capacity, dtype=torch.float32, device=device
        )

        deposit = torch.zeros_like(model.slow_tau)
        visit = torch.zeros_like(model.slow_tau)
        n_edges = model.n_concepts * model.degree
        ctx_sum = torch.zeros(n_edges, model.h, device=device, dtype=torch.float32)
        ctx_weight = torch.zeros(n_edges, device=device, dtype=torch.float32)

        for t in range(L):
            act = x[:, t].ne(0)
            if not act.any():
                continue
            af = act.to(item_state.dtype)[:, None]
            xids, xmass = model._merge(
                ids, mass * af, fi[:, t], fm[:, t] * af
            )
            traces = []

            for _ in range(model.layers_n):
                out_ids, out_mass, flat, usage, q = model.pheromone_step(
                    xids, xmass, item_state[:, t], trail_ids, trail_values
                )
                traces.append((flat, usage, q))
                xids, xmass = out_ids, out_mass

            ids = torch.where(act[:, None], xids, ids)
            mass = torch.where(act[:, None], xmass, mass)

            msg = (model.space.value(ids) * mass[:, :, None]).sum(1)
            h = model.norm(item_state[:, t] + model.message_proj(msg)) * af

            valid = valid_all[:, t]
            if valid.any():
                reward = pairwise_rank_reward(
                    model,
                    h[valid],
                    y[valid, t],
                    args.reward_negs,
                    args.reward_temperature,
                )
                full_reward = torch.zeros(B, device=device, dtype=torch.float32)
                full_reward[valid] = reward

                reward_sum += float(reward.sum().item())
                reward_count += int(reward.numel())
                positive_count += int((reward > 0).sum().item())

                for hop, (flat, usage, q) in enumerate(traces):
                    hop_decay = args.trace_decay ** (len(traces) - 1 - hop)
                    _accumulate_edge_credit(
                        deposit, visit, flat, usage, full_reward, hop_decay
                    )
                    _accumulate_context(
                        ctx_sum,
                        ctx_weight,
                        flat,
                        usage,
                        q,
                        full_reward.clamp_min(0.0),
                    )

                _accumulate_trail_credit(
                    deposit,
                    visit,
                    trail_ids,
                    trail_values,
                    full_reward,
                    args.trail_credit,
                )

                target_concept = concept_for_item(y[valid, t], model.n_concepts)
                need = ((1.0 - reward.clamp(-1.0, 1.0)) * 0.5).float()
                weighted_mass = xmass[valid].float() * need[:, None]
                accumulate_proposals(
                    xids[valid],
                    weighted_mass,
                    target_concept,
                    model.n_concepts,
                    proposal_target,
                    proposal_score,
                )

                model._update_fast_trail(
                    trail_ids, trail_values, traces, full_reward
                )
            else:
                trail_values.mul_(model.fast_decay)

        used = visit > 0
        avg_credit = torch.zeros_like(deposit)
        avg_credit[used] = deposit[used] / visit[used].clamp_min(1e-8)
        model.slow_tau.add_(float(args.deposit_lr) * avg_credit)
        model.slow_tau.clamp_(float(args.tau_min), float(args.tau_max))
        total_credit += float(avg_credit.abs().sum().item())

        context_updates += _apply_context_update(
            model, ctx_sum, ctx_weight, args.context_lr
        )
        total_positions += npos
        batches += 1

    model.slow_tau.sub_(float(args.tau_min)).mul_(
        1.0 - float(args.evaporation)
    ).add_(float(args.tau_min))
    model.slow_tau.clamp_(float(args.tau_min), float(args.tau_max))
    model.context_strength.mul_(1.0 - float(args.context_evaporation))

    rewired = rewire_epoch(
        model,
        model.slow_tau,
        proposal_target,
        proposal_score,
        args.rewire_fraction,
        args.new_edge_tau,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0

    grads = [name for name, p in model.named_parameters() if p.grad is not None]
    if grads:
        raise AssertionError(f"Contextual pheromone run created gradients: {grads[:8]}")

    return {
        "epoch": int(epoch),
        "positions": int(total_positions),
        "seconds": float(sec),
        "positions_per_s": float(total_positions / max(sec, 1e-9)),
        "batches": int(batches),
        "mean_rank_reward": float(reward_sum / max(1, reward_count)),
        "positive_reward_fraction": float(
            positive_count / max(1, reward_count)
        ),
        "credit_l1": float(total_credit),
        "rewired_edges": int(rewired),
        "context_edges_updated": int(context_updates),
        "slow_tau_mean": float(model.slow_tau.mean().item()),
        "slow_tau_max": float(model.slow_tau.max().item()),
        "context_strength_mean": float(model.context_strength.mean().item()),
        "autograd_grad_tensors": 0,
        "loss_backward_calls": 0,
        "optimizer": None,
    }


def evaluate(model, split, n_items, device, batch_size=1024):
    model.eval()
    val = evaluate_full(
        model,
        split["val_prefix"],
        split["val_target"],
        n_items,
        50,
        device,
        topks=(10,),
        batch_size=batch_size,
    )
    test = evaluate_full(
        model,
        split["test_prefix"],
        split["test_target"],
        n_items,
        50,
        device,
        topks=(10, 20, 50),
        batch_size=batch_size,
    )
    return val, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=AMAZON, default="beauty")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=1)

    p.add_argument("--slow-beta", type=float, default=1.5)
    p.add_argument("--fast-beta", type=float, default=1.0)
    p.add_argument("--context-gamma", type=float, default=1.0)
    p.add_argument("--fast-capacity", type=int, default=32)
    p.add_argument("--fast-decay", type=float, default=0.90)
    p.add_argument("--fast-add", type=int, default=4)

    p.add_argument("--reward-negs", type=int, default=64)
    p.add_argument("--reward-temperature", type=float, default=1.0)
    p.add_argument("--trace-decay", type=float, default=0.7)
    p.add_argument("--trail-credit", type=float, default=0.25)

    p.add_argument("--deposit-lr", type=float, default=0.5)
    p.add_argument("--evaporation", type=float, default=0.08)
    p.add_argument("--tau-min", type=float, default=0.10)
    p.add_argument("--tau-max", type=float, default=20.0)
    p.add_argument("--new-edge-tau", type=float, default=1.5)
    p.add_argument("--rewire-fraction", type=float, default=0.02)

    p.add_argument("--context-lr", type=float, default=0.10)
    p.add_argument("--context-evaporation", type=float, default=0.02)
    p.add_argument("--message-gain", type=float, default=16.0)

    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/sparsewalker_contextual_pheromone_v2",
    )
    args = p.parse_args()

    assert torch.cuda.is_available(), "GPU runtime required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed_all(args.seed)

    data = load_dataset(args.dataset, args.data_dir)
    split = split_data(data["sequences"])
    model = build_model(data["n_items"], 50, args).to(device)
    initialize_identity_geometry(model, args.seed, args.message_gain)
    model.reset_pheromone()

    all_items = torch.arange(1, data["n_items"] + 1, device=device)
    mapped = concept_for_item(all_items, model.n_concepts)
    unique_addresses = int(torch.unique(mapped).numel())
    router_hit = router_identity_hit_rate(model)

    init_val, init_test = evaluate(
        model, split, data["n_items"], device, args.eval_batch_size
    )
    init = {
        "unique_item_concept_addresses": unique_addresses,
        "n_items": int(data["n_items"]),
        "router_target_hit_rate": router_hit,
        "val": init_val,
        "test": init_test,
    }
    print("CPHERO_INIT", json.dumps(init), flush=True)

    if unique_addresses != int(data["n_items"]):
        raise AssertionError(
            f"Item concept addresses collided: {unique_addresses}/{data['n_items']}"
        )
    if router_hit < 0.99:
        raise AssertionError(
            f"Identity router initialization failed: hit_rate={router_hit}"
        )

    ds = WindowDataset(split["train"], 50, args.seed)
    out = Path(args.output_dir) / args.dataset / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "ContextualPheromoneWalker-v2-from-scratch",
        "base_architecture": "corrected SparseWalker v1.1 Amazon winner",
        "warm_start": False,
        "pretrained_checkpoint": None,
        "autograd": False,
        "optimizer": None,
        "loss_backward_calls": 0,
        "behavior_learning": {
            "slow_global_pheromone": True,
            "fast_session_pheromone": True,
            "per_edge_context_prototype": True,
            "eligibility_trail_credit": True,
            "reward": "centered pairwise next-item ranking vs deterministic negatives",
            "rewiring": "slow epoch-level failure-weighted proposals",
        },
        "architecture": {
            "d": 64,
            "side": 256,
            "n_concepts": 65536,
            "active": 8,
            "degree": 4,
            "graph_hops": 2,
            "fresh_weight": 0.25,
            "max_len": 50,
        },
        "rates": vars(args),
        "references": {
            "SASRec_test_NDCG@10": REFERENCE_SASREC,
            "SparseWalker_v11_test_NDCG@10": REFERENCE_WALKER,
        },
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    best = float(init_val["NDCG@10"])
    best_epoch = 0
    best_state = cpu_state_dict(model)
    history = []

    for epoch in range(1, args.epochs + 1):
        stats = contextual_aco_epoch(model, ds, device, args, epoch)
        if epoch == 1 or epoch % args.eval_every == 0:
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
            print("CPHERO_EPOCH", json.dumps(row), flush=True)
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
    best_val, best_test = evaluate(
        model, split, data["n_items"], device, args.eval_batch_size
    )
    result = {
        "config": config,
        "initial": {"val": init_val, "test": init_test},
        "best_epoch": int(best_epoch),
        "best_contextual_pheromone": {
            "val": best_val,
            "test": best_test,
        },
        "vs_sasrec_test_ndcg_ratio": float(
            best_test["NDCG@10"] / REFERENCE_SASREC
        ),
        "vs_backprop_walker_test_ndcg_ratio": float(
            best_test["NDCG@10"] / REFERENCE_WALKER
        ),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("CPHERO_RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
