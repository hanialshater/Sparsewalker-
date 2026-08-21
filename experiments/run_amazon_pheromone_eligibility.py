#!/usr/bin/env python
"""Experiment 38: PheromoneWalker v1 + delayed eligibility traces.

This is deliberately a surgical extension of the from-scratch PheromoneWalker
that learned on Amazon Beauty.  The successful v1 ingredients remain unchanged:

* corrected SparseWalker v1.1 recurrence (K=8, 65,536 concepts, degree 4, 2 hops)
* unique item -> concept identity geometry initialized without sequence data
* direct target-concept pheromone reward
* evaporation and once-per-epoch evidence-accumulated sparse rewiring
* no pretrained checkpoint, optimizer, backward(), or autograd learning

The only new mechanism is a bounded per-user eligibility trail.  When the next
item is observed, recent graph edges receive additional discounted pheromone if
the concepts they reached are compatible with the current target concept:

    credit(lag) = eligibility_gain * eligibility_decay**lag * reward

This asks whether forward-only delayed credit improves the v1 result without
changing its reward, geometry, or structural-plasticity mechanism.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full

# Reuse the exact v1 implementation so the intervention stays isolated.
from run_amazon_pheromone_walker import (
    AMAZON,
    seed_all,
    cpu_state_dict,
    build_model,
    concept_for_item,
    initialize_identity_geometry,
    sync_pheromone_logits,
    router_identity_hit_rate,
    _loader,
    pheromone_step,
    edge_reward,
    accumulate_proposals,
    rewire_epoch,
    evaluate,
)


def _scatter_edge_credit(src, weighted, reward, scale, degree, deposit, visit):
    """Accumulate reward and normalized exposure into [concept, edge-slot] tables."""
    if src.numel() == 0:
        return 0.0, 0.0
    dep = weighted.float() * reward.float() * float(scale)
    exp = weighted.float() * float(scale)
    slots = torch.arange(degree, device=src.device)[None, None, :]
    flat = (src[:, :, None] * degree + slots).expand_as(weighted).reshape(-1)
    deposit.view(-1).scatter_add_(0, flat, dep.reshape(-1))
    visit.view(-1).scatter_add_(0, flat, exp.reshape(-1))
    return float(dep.sum().item()), float(exp.sum().item())


@torch.no_grad()
def aco_eligibility_epoch(model, dataset, device, args, epoch, tau):
    model.eval()
    total_positions = 0
    batches = 0
    direct_reward_sum = 0.0
    direct_exposure = 0.0
    trace_reward_sum = 0.0
    trace_exposure = 0.0
    total_avg_deposit = 0.0
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

        # Each entry is one prior event and contains its two sparse graph hops.
        # Tensors stay batched by user so a future outcome only credits that
        # same user's recent path, never another user's edges.
        trail = []

        for t in range(L):
            act = x[:, t].ne(0)
            if not act.any():
                continue
            af = act.to(item_state.dtype)[:, None]
            xids, xmass = model._merge(ids, mass * af, fi[:, t], fm[:, t] * af)
            valid = valid_all[:, t]
            target_all = concept_for_item(y[:, t], model.n_concepts)
            target = target_all[valid] if valid.any() else None
            event_records = []

            for _hop in range(model.layers_n):
                if valid.any():
                    accumulate_proposals(
                        xids[valid], xmass[valid].float(), target,
                        model.n_concepts, proposal_target, proposal_score,
                    )

                out_ids, out_mass, dest, prob = pheromone_step(
                    model, xids, xmass, item_state[:, t], tau, args.pheromone_beta
                )
                weighted_all = xmass.float()[:, :, None] * prob.float()

                # Exact v1 immediate reward remains intact.
                if valid.any():
                    src = xids[valid]
                    weighted = weighted_all[valid]
                    d = dest[valid]
                    reward = edge_reward(d, target, model.side)
                    rs, ex = _scatter_edge_credit(
                        src, weighted, reward, 1.0, model.degree, deposit, visit
                    )
                    direct_reward_sum += rs
                    direct_exposure += ex

                # Store this hop before state changes. Rewiring only happens at
                # epoch end, so edge slots remain stable throughout the trail.
                event_records.append({
                    "src": xids.detach().clone(),
                    "dest": dest.detach().clone(),
                    "weighted": weighted_all.detach().clone(),
                    "act": act.detach().clone(),
                })
                xids, xmass = out_ids, out_mass

            # Delayed credit: current next-item outcome reinforces edges used in
            # recent prior events of the same user. Lag 1 is the immediately
            # preceding event; older traces decay geometrically.
            if valid.any() and args.eligibility_steps > 0 and args.eligibility_gain > 0:
                for lag, prior_event in enumerate(reversed(trail), start=1):
                    if lag > int(args.eligibility_steps):
                        break
                    scale = float(args.eligibility_gain) * (float(args.eligibility_decay) ** lag)
                    if scale <= 0:
                        continue
                    for rec in prior_event:
                        eligible = valid & rec["act"]
                        if not eligible.any():
                            continue
                        src = rec["src"][eligible]
                        weighted = rec["weighted"][eligible]
                        d = rec["dest"][eligible]
                        delayed_target = target_all[eligible]
                        reward = edge_reward(d, delayed_target, model.side)
                        rs, ex = _scatter_edge_credit(
                            src, weighted, reward, scale,
                            model.degree, deposit, visit,
                        )
                        trace_reward_sum += rs
                        trace_exposure += ex

            ids = torch.where(act[:, None], xids, ids)
            mass = torch.where(act[:, None], xmass, mass)
            trail.append(event_records)
            if len(trail) > int(args.eligibility_steps):
                trail.pop(0)

        used = visit > 0
        avg = torch.zeros_like(deposit)
        avg[used] = deposit[used] / visit[used].clamp_min(1e-8)
        tau.add_(float(args.deposit_lr) * avg)
        tau.clamp_(float(args.tau_min), float(args.tau_max))
        total_avg_deposit += float(avg.sum().item())
        total_positions += npos
        batches += 1

    # Same v1 global plasticity: evaporation, then slow evidence-based rewiring.
    tau.sub_(float(args.tau_min)).mul_(1.0 - float(args.evaporation)).add_(float(args.tau_min))
    tau.clamp_(float(args.tau_min), float(args.tau_max))
    rewired = rewire_epoch(
        model, tau, proposal_target, proposal_score,
        args.rewire_fraction, args.new_edge_tau,
    )
    sync_pheromone_logits(model, tau, args.pheromone_beta)

    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    grads = [name for name, p in model.named_parameters() if p.grad is not None]
    if grads:
        raise AssertionError(f"Eligibility ACO run created gradients: {grads[:8]}")

    direct_mean = direct_reward_sum / max(direct_exposure, 1e-9)
    trace_mean = trace_reward_sum / max(trace_exposure, 1e-9)
    return {
        "epoch": int(epoch),
        "positions": int(total_positions),
        "seconds": float(sec),
        "positions_per_s": float(total_positions / max(sec, 1e-9)),
        "batches": int(batches),
        "mean_edge_reward": float(direct_mean),
        "mean_trace_reward": float(trace_mean),
        "trace_exposure": float(trace_exposure),
        "trace_credit_fraction": float(trace_reward_sum / max(direct_reward_sum + trace_reward_sum, 1e-9)),
        "deposit_sum": float(total_avg_deposit),
        "rewired_edges": int(rewired),
        "tau_mean": float(tau.mean().item()),
        "tau_max": float(tau.max().item()),
        "eligibility_steps": int(args.eligibility_steps),
        "eligibility_decay": float(args.eligibility_decay),
        "eligibility_gain": float(args.eligibility_gain),
        "autograd_grad_tensors": 0,
        "loss_backward_calls": 0,
        "optimizer": None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=AMAZON, default="beauty")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=1)

    # Exact v1 ACO defaults.
    p.add_argument("--pheromone-beta", type=float, default=2.0)
    p.add_argument("--deposit-lr", type=float, default=0.5)
    p.add_argument("--evaporation", type=float, default=0.10)
    p.add_argument("--tau-min", type=float, default=0.10)
    p.add_argument("--tau-max", type=float, default=20.0)
    p.add_argument("--new-edge-tau", type=float, default=2.0)
    p.add_argument("--rewire-fraction", type=float, default=0.05)
    p.add_argument("--message-gain", type=float, default=16.0)

    # Only new degrees of freedom.
    p.add_argument("--eligibility-steps", type=int, default=4)
    p.add_argument("--eligibility-decay", type=float, default=0.60)
    p.add_argument("--eligibility-gain", type=float, default=0.50)

    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/sparsewalker_pheromone_eligibility_v3",
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
    initialize_identity_geometry(model, args.seed, args.message_gain)
    tau = torch.ones(model.n_concepts, model.degree, device=device, dtype=torch.float32)
    sync_pheromone_logits(model, tau, args.pheromone_beta)

    all_items = torch.arange(1, data["n_items"] + 1, device=device)
    mapped = concept_for_item(all_items, model.n_concepts)
    unique_addresses = int(torch.unique(mapped).numel())
    router_hit = router_identity_hit_rate(model)
    init_val, init_test = evaluate(model, split, data["n_items"], device, args.eval_batch_size)
    print("ELIG_INIT", json.dumps({
        "unique_item_concept_addresses": unique_addresses,
        "n_items": int(data["n_items"]),
        "router_target_hit_rate": router_hit,
        "val": init_val,
        "test": init_test,
    }), flush=True)
    if unique_addresses != int(data["n_items"]):
        raise AssertionError(f"Item concept addresses collided: {unique_addresses}/{data['n_items']}")
    if router_hit < 0.99:
        raise AssertionError(f"Identity router initialization failed: hit_rate={router_hit}")

    ds = WindowDataset(split["train"], 50, args.seed)
    out = Path(args.output_dir) / args.dataset / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "PheromoneWalker-v3-eligibility",
        "base": "PheromoneWalker-v1-from-scratch",
        "base_architecture": "corrected SparseWalker v1.1 Amazon winner",
        "dataset": args.dataset,
        "seed": args.seed,
        "warm_start": False,
        "pretrained_checkpoint": None,
        "autograd": False,
        "optimizer": None,
        "loss_backward_calls": 0,
        "intervention": "v1 target-concept reward + bounded per-user eligibility traces",
        "eligibility": {
            "steps": args.eligibility_steps,
            "decay": args.eligibility_decay,
            "gain": args.eligibility_gain,
        },
        "aco": {
            "pheromone_beta": args.pheromone_beta,
            "deposit_lr": args.deposit_lr,
            "evaporation": args.evaporation,
            "tau_min": args.tau_min,
            "tau_max": args.tau_max,
            "new_edge_tau": args.new_edge_tau,
            "rewire_fraction": args.rewire_fraction,
        },
        "references": {
            "PheromoneWalker_v1_best_val_NDCG@10": 0.0023507147682577624,
            "SASRec_test_NDCG@10": 0.031195719394901355,
            "SparseWalker_v11_test_NDCG@10": 0.044882819399656555,
        },
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    best = float(init_val["NDCG@10"])
    best_epoch = 0
    best_state = cpu_state_dict(model)
    best_tau = tau.detach().cpu().clone()
    history = []

    for epoch in range(1, args.epochs + 1):
        stats = aco_eligibility_epoch(model, ds, device, args, epoch, tau)
        if epoch == 1 or epoch % args.eval_every == 0:
            val = evaluate_full(
                model, split["val_prefix"], split["val_target"],
                data["n_items"], 50, device, topks=(10,),
                batch_size=args.eval_batch_size,
            )
            row = {**stats, **{f"val_{k}": float(v) for k, v in val.items()}}
            history.append(row)
            print("ELIG_EPOCH", json.dumps(row), flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=2))
            ndcg = float(val["NDCG@10"])
            if ndcg > best:
                best = ndcg
                best_epoch = epoch
                best_state = cpu_state_dict(model)
                best_tau = tau.detach().cpu().clone()
                torch.save({
                    "model": best_state,
                    "tau": best_tau,
                    "epoch": epoch,
                    "val": val,
                    "config": config,
                }, out / "best.pt")

    model.load_state_dict(best_state)
    tau.copy_(best_tau.to(device))
    sync_pheromone_logits(model, tau, args.pheromone_beta)
    best_val, best_test = evaluate(model, split, data["n_items"], device, args.eval_batch_size)
    result = {
        "config": config,
        "initial": {"val": init_val, "test": init_test},
        "best_epoch": int(best_epoch),
        "best_eligibility": {"val": best_val, "test": best_test},
        "vs_v1_best_val_ratio": float(best_val["NDCG@10"] / 0.0023507147682577624),
        "vs_sasrec_test_ndcg_ratio": float(best_test["NDCG@10"] / 0.031195719394901355),
        "vs_backprop_walker_test_ndcg_ratio": float(best_test["NDCG@10"] / 0.044882819399656555),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("ELIG_RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
