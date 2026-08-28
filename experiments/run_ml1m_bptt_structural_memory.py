#!/usr/bin/env python
"""Experiment 48: BPTT + allocate-on-arrival structural SparseWalker on ML-1M.

Question
--------
Can discrete structural plasticity make the gradient problem easier?

We separate responsibilities cleanly:
* Adam/BPTT learns continuous parameters (router, keys, graph probabilities,
  values, readout, item embeddings).
* Allocate-on-arrival owns a non-differentiable concept->item memory.
* Structural rewiring owns the discrete graph destination table.

A training miss may bind the observed next item to an empty terminal concept
that the current walk already reached. Once that happens, an auxiliary memory
loss can immediately differentiate through the *mass* that reached that node:

    L_mem = -log sum_{c: owner(c)=target} mass(c)

Thus the label is not required to have a globally pre-assigned concept address.
The structure places a useful address in the current neighborhood; gradients
learn to put more probability mass on it.

Arms
----
canonical_full
    Exact canonical SparseWalker FullCE/BPTT training. No structural memory.
full_alloc
    Full BPTT + allocate-on-arrival + memory-mass auxiliary loss.
full_alloc_rewire
    Same, plus local discrete shortcuts toward an existing target alias when a
    target is absent and no local allocation can be made.
event_alloc_rewire
    Same structural mechanism, but recurrent mass is detached before each
    event. Gradients therefore cover only the current router + two graph hops +
    readout; there is no temporal BPTT.

All gradient arms use canonical default initialization, FullCE, AdamW, BF16,
one backward/optimizer step per batch, and the canonical 50-epoch LR schedule.
Rewires are proposed during the forward pass but applied only *after* backward
and optimizer.step(), avoiding in-place mutation of tensors needed by autograd.
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

from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.models import SparseWalker
from sparsewalker.training import train_epoch
from sparsewalker.evaluation.metrics import make_eval_batch

import run_ml1m_local_contrastive_walker as fast
import run_ml1m_walker_v11 as canon
import run_ml1m_allocate_on_arrival as alloc

MAX_LEN = 200
ARMS = (
    "canonical_full",
    "full_alloc",
    "full_alloc_rewire",
    "event_alloc_rewire",
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


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


def one_event(model, ids, mass, item_ids, active, detach_previous=False):
    """Canonical v1.1 event; also retain last-hop sources for rewire proposals."""
    if detach_previous:
        mass = mass.detach()
    af = active.to(model.item.weight.dtype)[:, None]
    context = model.item(item_ids) * math.sqrt(model.d_model)
    fi, fm = model.router(context, model.space)
    xids, xmass = model._merge(ids, mass * af, fi, fm * af)
    last_trace = None
    for hop in range(model.layers_n):
        if hop == model.layers_n - 1:
            last_trace = {
                "src_ids": xids.detach(),
                "src_mass": xmass.detach().float(),
            }
        xids, xmass = model.graph(
            xids, xmass, context, model.space, track_touched=False
        )
    ids = torch.where(active[:, None], xids, ids)
    mass = torch.where(active[:, None], xmass, mass)
    msg = (model.space.value(ids) * mass[:, :, None]).sum(1)
    h = model.norm(context + model.message_proj(msg)) * af
    return ids, mass, h, last_trace


@torch.no_grad()
def propose_rewires(memory, trace_last, unresolved_rows, target):
    """Return local shortcut proposals without mutating model parameters."""
    if unresolved_rows.numel() == 0:
        z = torch.empty(0, dtype=torch.long, device=target.device)
        f = torch.empty(0, dtype=torch.float32, device=target.device)
        return z, z, f
    tgt = target[unresolved_rows]
    anchor = memory.anchor[tgt]
    has_anchor = anchor.ge(0)
    if not has_anchor.any():
        z = torch.empty(0, dtype=torch.long, device=target.device)
        f = torch.empty(0, dtype=torch.float32, device=target.device)
        return z, z, f
    rows = unresolved_rows[has_anchor]
    anchors = anchor[has_anchor]
    src_mass = trace_last["src_mass"][rows]
    src_slot = src_mass.argmax(-1)
    rr = torch.arange(rows.numel(), device=rows.device)
    src = trace_last["src_ids"][rows, src_slot]
    priority = src_mass[rr, src_slot]
    return src.long(), anchors.long(), priority.float()


@torch.no_grad()
def apply_rewire_proposals(model, opt, proposals, max_rewires, rewire_logit):
    """Apply strongest proposal/source after optimizer step and reset stale Adam state."""
    if not proposals:
        return 0
    src = torch.cat([p[0] for p in proposals if p[0].numel()], 0) if any(
        p[0].numel() for p in proposals
    ) else torch.empty(0, dtype=torch.long, device=model.graph.destination.device)
    if src.numel() == 0:
        return 0
    anchor = torch.cat([p[1] for p in proposals if p[0].numel()], 0)
    priority = torch.cat([p[2] for p in proposals if p[0].numel()], 0)

    keep = alloc._amax_keep(src, priority, model.n_concepts)
    src, anchor, priority = src[keep], anchor[keep], priority[keep]
    if src.numel() == 0:
        return 0

    already = model.graph.destination[src].long().eq(anchor[:, None]).any(-1)
    src, anchor, priority = src[~already], anchor[~already], priority[~already]
    if src.numel() == 0:
        return 0

    if max_rewires > 0 and src.numel() > max_rewires:
        take = priority.topk(max_rewires).indices
        src, anchor = src[take], anchor[take]

    logits = model.graph.edge_logits.weight[src]
    if model.degree > 1:
        slot = logits[:, 1:].argmin(-1) + 1  # preserve self edge 0
    else:
        slot = torch.zeros(src.numel(), dtype=torch.long, device=src.device)

    model.graph.destination[src, slot] = anchor.to(model.graph.destination.dtype)
    model.graph.edge_logits.weight[src, slot] = float(rewire_logit)

    # The slot now refers to a different destination; old Adam momentum is stale.
    state = opt.state.get(model.graph.edge_logits.weight, {})
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        buf = state.get(name)
        if buf is not None:
            buf[src, slot] = 0
    return int(src.numel())


@torch.no_grad()
def memory_update_after_walk(memory, ids, mass, target, args):
    """Allocate misses, reinforce hits, and return post-allocation target mask."""
    pre_mask = memory.owner[ids].eq(target[:, None])
    pre_hit = pre_mask.any(-1)
    allocations = 0

    miss = ~pre_hit
    if miss.any():
        miss_rows = miss.nonzero(as_tuple=False).squeeze(-1)
        ar_local, anode = alloc.allocate_from_reached(
            memory,
            ids[miss_rows],
            mass[miss_rows],
            target[miss_rows],
            args.max_aliases,
            args.alloc_strength,
        )
        allocations = int(anode.numel())

    post_mask = memory.owner[ids].eq(target[:, None])
    covered = post_mask.any(-1)
    if covered.any():
        score = mass.masked_fill(~post_mask, -1.0)
        slot = score.argmax(-1)
        node = ids.gather(1, slot[:, None]).squeeze(1)
        alloc.reinforce_association(memory, node[covered], args.assoc_lr)

    unresolved = (~covered).nonzero(as_tuple=False).squeeze(-1)
    return pre_hit, post_mask, unresolved, allocations


def train_epoch_hybrid(model, memory, ds, opt, device, args, epoch, arm):
    loader = fast._loader(ds, args.batch_size, epoch, True)
    model.train()
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    detach_previous = arm == "event_alloc_rewire"
    allow_rewire = arm in ("full_alloc_rewire", "event_alloc_rewire")

    positions = padded = batches = backward_calls = 0
    pre_hits = mem_supervised = allocations = rewires = unresolved = 0
    loss_total = ce_total = mem_total = 0.0
    t0 = time.perf_counter()

    for bi, (tokens, lengths) in enumerate(loader, start=1):
        positions += int((lengths - 1).clamp_min(0).sum().item())
        padded += int(tokens.size(0) * max(0, tokens.size(1) - 1))
        tokens = tokens.to(device, non_blocking=True)
        x, y = tokens[:, :-1], tokens[:, 1:]
        B, L = x.shape
        total_valid = int(y.ne(0).sum().item())
        if total_valid == 0:
            continue

        ids = torch.zeros(B, model.active, dtype=torch.long, device=device)
        mass = torch.zeros(B, model.active, dtype=torch.float32, device=device)
        outs = []
        mem_loss_sum = None
        proposals = []
        opt.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            for t in range(L):
                active = x[:, t].ne(0)
                ids, mass, h, trace_last = one_event(
                    model, ids, mass, x[:, t], active,
                    detach_previous=detach_previous,
                )
                outs.append(h)
                valid = active & y[:, t].ne(0)
                if not valid.any():
                    continue

                target = y[valid, t]
                sid = ids[valid].detach()
                smass_det = mass[valid].detach().float()

                with torch.no_grad():
                    pre_hit, post_mask, unr, nalloc = memory_update_after_walk(
                        memory, sid, smass_det, target, args
                    )
                    pre_hits += int(pre_hit.sum())
                    allocations += nalloc
                    covered = post_mask.any(-1)
                    mem_supervised += int(covered.sum())
                    unresolved += int(unr.numel())
                    if allow_rewire and unr.numel():
                        trv = {
                            "src_ids": trace_last["src_ids"][valid],
                            "src_mass": trace_last["src_mass"][valid],
                        }
                        psrc, panchor, ppriority = propose_rewires(
                            memory, trv, unr, target
                        )
                        if psrc.numel():
                            proposals.append((psrc, panchor, ppriority))

                # owner/mask is discrete, but terminal mass remains differentiable.
                if covered.any():
                    smass = mass[valid][covered]
                    mask = post_mask[covered].to(smass.dtype)
                    target_mass = (smass * mask).sum(-1).clamp_min(args.memory_eps)
                    piece = -target_mass.log().sum()
                    mem_loss_sum = piece if mem_loss_sum is None else mem_loss_sum + piece

            H = torch.stack(outs, 1)
            valid_all = y.ne(0)
            ce = F.cross_entropy(model.score_hidden(H[valid_all]), y[valid_all])
            if mem_loss_sum is None:
                mem_loss = ce.new_zeros(())
            else:
                mem_loss = mem_loss_sum / total_valid
            loss = ce + float(args.memory_lambda) * mem_loss

        loss.backward()
        backward_calls += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if allow_rewire:
            rewires += apply_rewire_proposals(
                model, opt, proposals, args.max_rewires_per_batch, args.rewire_logit
            )

        loss_total += float(loss.detach())
        ce_total += float(ce.detach())
        mem_total += float(mem_loss.detach())
        batches += 1

        if args.progress_every > 0 and (
            bi == 1 or bi % args.progress_every == 0 or bi == len(loader)
        ):
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print("HYBRID_PROGRESS", json.dumps({
                "arm": arm,
                "epoch": epoch,
                "batch": bi,
                "batches": len(loader),
                "positions": positions,
                "allocations": allocations,
                "rewires": rewires,
                "elapsed_s": round(elapsed, 2),
            }), flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    occupied = int(memory.owner.ne(0).sum())
    covered_items = int(memory.alias_count.gt(0).sum())
    return {
        "loss": loss_total / max(1, batches),
        "dense_ce": ce_total / max(1, batches),
        "memory_loss": mem_total / max(1, batches),
        "batches": batches,
        "positions": positions,
        "seconds": sec,
        "positions_per_s": positions / max(sec, 1e-9),
        "padding_efficiency": positions / max(1, padded),
        "backward_calls": backward_calls,
        "preallocation_memory_hit_rate": pre_hits / max(1, positions),
        "memory_objective_coverage": mem_supervised / max(1, positions),
        "allocations": allocations,
        "rewires": rewires,
        "unresolved_misses": unresolved,
        "occupied_concepts": occupied,
        "occupied_fraction": occupied / model.n_concepts,
        "covered_items": covered_items,
        "item_coverage": covered_items / model.n_items,
        "mean_aliases_per_covered_item": occupied / max(1, covered_items),
    }


def _metric_from_top(top, target):
    match = top.eq(target[:, None])
    hit = match.any(-1)
    rank0 = match.float().argmax(-1)
    rank = rank0 + 1
    h = float(hit.float().sum())
    nd = float((hit.float() / torch.log2(rank.float() + 1.0)).sum())
    mr = float((hit.float() / rank.float()).sum())
    return h, nd, mr


@torch.inference_mode()
def evaluate_views(model, memory, prefixes, targets, n_items, device, batch_size, hybrid_bonus):
    model.eval()
    total = len(targets)
    sums = {
        "dense_HR@10": 0.0, "dense_NDCG@10": 0.0, "dense_MRR@10": 0.0,
        "sparse_HR@10": 0.0, "sparse_NDCG@10": 0.0, "sparse_MRR@10": 0.0,
        "hybrid_HR@10": 0.0, "hybrid_NDCG@10": 0.0, "hybrid_MRR@10": 0.0,
    }
    candidate_hits = candidate_sum = 0.0

    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        idxs = list(range(start, end))
        seq, lens = make_eval_batch(prefixes, idxs, MAX_LEN)
        seq = seq.to(device)
        lens = lens.to(device)
        tgt = torch.as_tensor(targets[start:end], dtype=torch.long, device=device)

        H, I, M = model.encode_with_states(seq)
        row = torch.arange(seq.size(0), device=device)
        last = (lens - 1).clamp_min(0)
        h = H[row, last]
        ids = I[row, last]
        mass = M[row, last].float()
        dense = model.score_hidden(h).float()

        mem_scores = torch.zeros(seq.size(0), n_items + 1, device=device)
        if memory is not None:
            labels = memory.owner[ids]
            contrib = mass * memory.strength[ids]
            contrib = contrib * labels.ne(0).float()
            mem_scores.scatter_add_(1, labels.long(), contrib)
            mem_scores[:, 0] = 0.0
        hybrid = dense + float(hybrid_bonus) * mem_scores
        sparse = mem_scores.masked_fill(mem_scores.le(0), -1e20)
        sparse[:, 0] = -1e20

        for r, i in enumerate(idxs):
            truth = int(tgt[r])
            seen = set(prefixes[i])
            seen.discard(truth)
            if seen:
                si = torch.as_tensor(list(seen), dtype=torch.long, device=device)
                dense[r, si] = -1e20
                hybrid[r, si] = -1e20
                sparse[r, si] = -1e20
            if memory is not None:
                positive = mem_scores[r].gt(0)
                if seen:
                    positive[si] = False
                candidate_sum += float(positive.sum())
                candidate_hits += float(positive[truth])

        for name, scores in (("dense", dense), ("sparse", sparse), ("hybrid", hybrid)):
            top = scores.topk(10, dim=-1).indices
            hh, nn, mm = _metric_from_top(top, tgt)
            sums[f"{name}_HR@10"] += hh
            sums[f"{name}_NDCG@10"] += nn
            sums[f"{name}_MRR@10"] += mm

    out = {k: v / total for k, v in sums.items()}
    out["target_in_sparse_candidates"] = candidate_hits / total
    out["mean_sparse_candidates"] = candidate_sum / total
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--schedule-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--peak-lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--memory-lambda", type=float, default=.5)
    p.add_argument("--memory-eps", type=float, default=1e-5)
    p.add_argument("--max-aliases", type=int, default=16)
    p.add_argument("--alloc-strength", type=float, default=.55)
    p.add_argument("--assoc-lr", type=float, default=.08)
    p.add_argument("--rewire-logit", type=float, default=2.0)
    p.add_argument("--max-rewires-per-batch", type=int, default=1024)
    p.add_argument("--hybrid-bonus", type=float, default=1.0)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--data-dir", default="/content/drive/MyDrive/sparsewalker_data")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_bptt_structural")
    args = p.parse_args()

    assert torch.cuda.is_available(), "GPU runtime required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed_all(args.seed)

    print("HYBRID_DATA_START", flush=True)
    t0 = time.perf_counter()
    data = load_dataset("ml1m", args.data_dir)
    split = split_data(data["sequences"])
    ds = WindowDataset(split["train"], MAX_LEN, args.seed)
    print("HYBRID_DATA_DONE", json.dumps({
        "seconds": round(time.perf_counter() - t0, 2),
        "users": len(data["sequences"]),
        "n_items": data["n_items"],
        "concepts": 65536,
    }), flush=True)

    # Canonical default initialization shared by all arms.
    seed_all(args.seed)
    template = build(data["n_items"]).to(device)
    init_state = cpu_state(template)
    del template
    torch.cuda.empty_cache()

    out = Path(args.output_dir) / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "experiment": "ML1M-BPTT-StructuralMemory-v1",
        "arms": args.arms,
        "same_canonical_initialization": True,
        "dense_objective": "FullCE",
        "memory_objective": "-log terminal mass on any target-owned reached concept",
        "optimizer": "AdamW continuous parameters only",
        "discrete_structural_updates": "allocate concept owner; optional graph destination rewire",
        "one_backward_and_step_per_batch": True,
        "schedule_epochs": args.schedule_epochs,
        "test_evaluation_during_screen": False,
        "args": vars(args),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    ranking = []
    for arm in args.arms:
        print("\nHYBRID_ARM_START", json.dumps({"arm": arm}), flush=True)
        seed_all(args.seed)
        model = build(data["n_items"]).to(device)
        model.load_state_dict(init_state)
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay
        )
        memory = None if arm == "canonical_full" else alloc.AssocMemory(
            model.n_concepts, data["n_items"], device
        )
        hist = []
        best_dense = best_hybrid = best_sparse_recall = -1.0
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            lr = canon.set_lr(
                opt, epoch, args.schedule_epochs,
                peak=args.peak_lr, min_lr=args.min_lr, warmup=3,
            )
            if arm == "canonical_full":
                if device.type == "cuda":
                    torch.cuda.synchronize()
                st = time.perf_counter()
                s = train_epoch(
                    "SparseWalker", model, ds, opt, device,
                    batch_size=args.batch_size, epoch=epoch, loss_mode="full",
                    bucket_by_length=True, use_bf16=True, return_stats=True,
                    grad_clip=args.grad_clip,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                sec = time.perf_counter() - st
                stats = {
                    "loss": float(s["loss"]),
                    "dense_ce": float(s["loss"]),
                    "memory_loss": 0.0,
                    "batches": int(s["batches"]),
                    "positions": int(s["positions"]),
                    "seconds": sec,
                    "positions_per_s": float(s["positions"]) / max(sec, 1e-9),
                    "padding_efficiency": float(s["padding_efficiency"]),
                    "backward_calls": int(s["batches"]),
                    "preallocation_memory_hit_rate": 0.0,
                    "memory_objective_coverage": 0.0,
                    "allocations": 0,
                    "rewires": 0,
                    "unresolved_misses": 0,
                    "occupied_concepts": 0,
                    "occupied_fraction": 0.0,
                    "covered_items": 0,
                    "item_coverage": 0.0,
                    "mean_aliases_per_covered_item": 0.0,
                }
            else:
                stats = train_epoch_hybrid(
                    model, memory, ds, opt, device, args, epoch, arm
                )

            row = {"arm": arm, "epoch": epoch, "lr": lr, **stats}
            if epoch == 1 or epoch % args.eval_every == 0:
                vt = time.perf_counter()
                val = evaluate_views(
                    model, memory,
                    split["val_prefix"], split["val_target"],
                    data["n_items"], device, args.eval_batch_size,
                    args.hybrid_bonus,
                )
                row.update({f"val_{k}": float(v) for k, v in val.items()})
                row["val_seconds"] = time.perf_counter() - vt
                dense_nd = float(val["dense_NDCG@10"])
                hybrid_nd = float(val["hybrid_NDCG@10"])
                sparse_recall = float(val["target_in_sparse_candidates"])
                best_dense = max(best_dense, dense_nd)
                best_hybrid = max(best_hybrid, hybrid_nd)
                best_sparse_recall = max(best_sparse_recall, sparse_recall)
                if hybrid_nd >= best_hybrid - 1e-15:
                    best_epoch = epoch
                    payload = {
                        "model": cpu_state(model),
                        "arm": arm,
                        "epoch": epoch,
                        "val": val,
                        "config": cfg,
                    }
                    if memory is not None:
                        payload["memory"] = memory.state_cpu()
                    torch.save(payload, out / f"best_{arm}.pt")

            hist.append(row)
            print("HYBRID_EPOCH", json.dumps(row), flush=True)
            (out / f"history_{arm}.json").write_text(json.dumps(hist, indent=2))

        result = {
            "arm": arm,
            "temporal_credit": "full" if arm != "event_alloc_rewire" else "current_event_only",
            "structural_allocation": arm != "canonical_full",
            "structural_rewire": arm in ("full_alloc_rewire", "event_alloc_rewire"),
            "best_epoch_by_hybrid": best_epoch,
            "best_dense_NDCG@10": best_dense,
            "best_hybrid_NDCG@10": best_hybrid,
            "best_sparse_candidate_recall": best_sparse_recall,
            "final_dense_NDCG@10": float(hist[-1].get("val_dense_NDCG@10", float("nan"))),
            "final_hybrid_NDCG@10": float(hist[-1].get("val_hybrid_NDCG@10", float("nan"))),
            "final_sparse_candidate_recall": float(hist[-1].get("val_target_in_sparse_candidates", 0.0)),
            "mean_positions_per_s": float(np.mean([r["positions_per_s"] for r in hist])),
            "final_occupied_concepts": int(hist[-1]["occupied_concepts"]),
            "final_item_coverage": float(hist[-1]["item_coverage"]),
        }
        ranking.append(result)
        print("HYBRID_ARM_DONE", json.dumps(result), flush=True)
        del model, opt, memory
        torch.cuda.empty_cache()

    ranking.sort(key=lambda r: r["best_hybrid_NDCG@10"], reverse=True)
    control = next((r for r in ranking if r["arm"] == "canonical_full"), None)
    summary = {
        "config": cfg,
        "ranking": ranking,
        "control_best_dense_NDCG@10": None if control is None else control["best_dense_NDCG@10"],
        "references": {
            "analytic_backward_free_best_val_NDCG@10": 0.09352186187904432,
            "SASRec_public_protocol_reference_NDCG@10": 0.15965,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("\nHYBRID_RESULT", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
