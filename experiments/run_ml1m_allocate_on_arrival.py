#!/usr/bin/env python
"""Experiment 47: allocate-on-arrival associative SparseWalker on ML-1M.

Core hypothesis
---------------
Do not force the sparse walk to find a globally pre-assigned concept address for
an item. Let the walk arrive in a local region; if a reached concept is empty,
bind the observed next item to that concept. On later visits the concept acts as
a sparse associative-memory address.

Each concept owns at most one item label. An item may have a small number of
aliases (default 4) in different regions. Inference never searches a global
item->concept index: it reads only the labels attached to the K active terminal
concepts. The optional label->anchor index is used only during supervised
training by the structural-rewire arm.

Arms
----
allocate_only
    Bind a target to the highest-mass empty terminal concept on a miss.
allocate_path
    Same, plus reinforce only the two graph transitions that reached a correct
    or newly allocated terminal concept.
allocate_rewire
    Same as allocate_path. If the local region cannot allocate a new alias,
    rewire one last-hop edge toward an already allocated alias of the target.

All arms are strict forward/local learning: no optimizer, no backward(), and no
autograd gradient tensors. Item geometry and fresh routing retain the successful
Experiment-39 local contrastive/competitive rules; prediction itself is purely
through the allocated sparse associative memory.
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

import run_amazon_local_contrastive_walker as base
import run_ml1m_local_contrastive_walker as fast
from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation.metrics import make_eval_batch

MAX_LEN = 200
ARMS = ("allocate_only", "allocate_path", "allocate_rewire")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class AssocMemory:
    def __init__(self, n_concepts, n_items, device):
        self.owner = torch.zeros(n_concepts, dtype=torch.long, device=device)
        self.strength = torch.zeros(n_concepts, dtype=torch.float32, device=device)
        self.alias_count = torch.zeros(n_items + 1, dtype=torch.long, device=device)
        self.anchor = torch.full((n_items + 1,), -1, dtype=torch.long, device=device)

    def state_cpu(self):
        return {
            "owner": self.owner.detach().cpu().clone(),
            "strength": self.strength.detach().cpu().clone(),
            "alias_count": self.alias_count.detach().cpu().clone(),
            "anchor": self.anchor.detach().cpu().clone(),
        }

    def load_cpu(self, state):
        self.owner.copy_(state["owner"].to(self.owner.device))
        self.strength.copy_(state["strength"].to(self.strength.device))
        self.alias_count.copy_(state["alias_count"].to(self.alias_count.device))
        self.anchor.copy_(state["anchor"].to(self.anchor.device))


def graph_step_trace(model, ids, mass, context):
    """CompactGraph.forward with the tiny touched subgraph retained for credit."""
    dest = model.graph.destination[ids].long()
    static = model.graph.edge_logits(ids)
    q = F.normalize(model.graph.context_q(context), dim=-1)
    key = model.space.key(dest)
    score = static + torch.exp(model.graph.scale) * (key * q[:, None, None, :]).sum(-1)
    prob = F.softmax(score, dim=-1)
    B = ids.size(0)
    out_ids, out_mass = model.graph.topk(
        dest.reshape(B, -1), (mass.unsqueeze(-1) * prob).reshape(B, -1)
    )
    trace = {
        "src_ids": ids,
        "src_mass": mass,
        "dest": dest,
        "prob": prob,
    }
    return out_ids, out_mass, trace


@torch.no_grad()
def walk_event(model, ids, mass, current_ids):
    cur = model.item(current_ids).float()
    context = cur * math.sqrt(model.d_model)
    fi, fm = model.router(context, model.space)
    ids, mass = model._merge(ids, mass, fi, fm)
    traces = []
    for _ in range(model.layers_n):
        ids, mass, tr = graph_step_trace(model, ids, mass, context)
        traces.append(tr)
    return ids, mass.float(), traces, fi, fm.float(), cur


@torch.no_grad()
def semantic_local_update(model, cur_ids, cur, fi, fm, target_ids, gen, args):
    """Keep a useful local item/router geometry without assigning labels to addresses."""
    base.update_router_and_keys(
        model, cur, fi, fm, args.prototype_lr, args.key_lr
    )
    h = F.normalize(cur.float(), dim=-1)
    neg = torch.randint(
        1, model.n_items + 1,
        (cur_ids.numel(), args.negatives),
        device=cur_ids.device,
        generator=gen,
    )
    signal, diag = base.contrastive(
        model, h, target_ids, neg, args.item_lr, args.temperature
    )
    base.update_current_items(model, cur_ids, signal, args.input_lr)
    return float(diag["margin"])


def _amax_keep(keys, scores, size):
    """Keep proposals attaining the maximum score for each integer key."""
    if keys.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=keys.device)
    best = torch.full((size,), -1e30, device=scores.device, dtype=scores.dtype)
    best.scatter_reduce_(0, keys.long(), scores, reduce="amax", include_self=True)
    return scores >= best[keys.long()] - 1e-12


@torch.no_grad()
def allocate_from_reached(memory, ids, mass, target, max_aliases, init_strength):
    """At most one new alias per target in this event; resolve node collisions by mass."""
    empty = memory.owner[ids].eq(0)
    has = empty.any(-1)
    masked = mass.masked_fill(~empty, -1.0)
    slot = masked.argmax(-1)
    node = ids.gather(1, slot[:, None]).squeeze(1)
    score = mass.gather(1, slot[:, None]).squeeze(1)
    room = memory.alias_count[target] < int(max_aliases)
    proposal = has & room
    if not proposal.any():
        z = torch.empty(0, dtype=torch.long, device=ids.device)
        return z, z

    rows = proposal.nonzero(as_tuple=False).squeeze(-1)
    pn = node[rows]
    pt = target[rows]
    ps = score[rows]

    keep_t = _amax_keep(pt, ps, memory.alias_count.numel())
    rows, pn, pt, ps = rows[keep_t], pn[keep_t], pt[keep_t], ps[keep_t]
    keep_n = _amax_keep(pn, ps, memory.owner.numel())
    rows, pn, pt = rows[keep_n], pn[keep_n], pt[keep_n]
    free = memory.owner[pn].eq(0)
    rows, pn, pt = rows[free], pn[free], pt[free]
    if pn.numel() == 0:
        return rows, pn

    memory.owner.index_copy_(0, pn, pt)
    memory.strength[pn] = float(init_strength)
    memory.alias_count.index_add_(0, pt, torch.ones_like(pt))
    no_anchor = memory.anchor[pt].lt(0)
    if no_anchor.any():
        memory.anchor[pt[no_anchor]] = pn[no_anchor]
    return rows, pn


@torch.no_grad()
def reinforce_association(memory, nodes, lr):
    if nodes.numel() == 0:
        return
    u, counts = torch.unique(nodes.long(), return_counts=True)
    old = memory.strength[u]
    # Repeated local evidence saturates toward confidence 1 rather than growing unbounded.
    remain = torch.pow(torch.full_like(old, 1.0 - float(lr)), counts.to(old.dtype))
    memory.strength[u] = 1.0 - (1.0 - old) * remain


@torch.no_grad()
def choose_best_hit(memory, ids, mass, target):
    labels = memory.owner[ids]
    hit = labels.eq(target[:, None])
    score = mass * memory.strength[ids]
    masked = score.masked_fill(~hit, -1.0)
    has = hit.any(-1)
    slot = masked.argmax(-1)
    node = ids.gather(1, slot[:, None]).squeeze(1)
    return has, node


@torch.no_grad()
def reinforce_one_hop(model, trace, target_node, reward, lr):
    """Reinforce the highest-contribution touched edge reaching target_node."""
    if target_node.numel() == 0:
        z = torch.empty(0, dtype=torch.long, device=target_node.device)
        return z
    dest = trace["dest"]
    prob = trace["prob"]
    src_mass = trace["src_mass"]
    match = dest.eq(target_node[:, None, None])
    contrib = src_mass[:, :, None] * prob
    flat = contrib.masked_fill(~match, -1.0).reshape(target_node.size(0), -1)
    ok = match.reshape(target_node.size(0), -1).any(-1)
    best = flat.argmax(-1)
    D = dest.size(-1)
    src_slot = torch.div(best, D, rounding_mode="floor")
    edge_slot = best.remainder(D)
    row = torch.arange(target_node.size(0), device=target_node.device)
    src_node = trace["src_ids"][row, src_slot]
    prow = prob[row, src_slot]

    if ok.any():
        rr = reward.float()[:, None]
        delta = -float(lr) * rr * prow
        delta[row, edge_slot] += float(lr) * reward.float()
        model.graph.edge_logits.weight.index_add_(0, src_node[ok], delta[ok].to(model.graph.edge_logits.weight.dtype))
    return src_node


@torch.no_grad()
def reinforce_path(model, traces, success_node, cur, args):
    if success_node.numel() == 0:
        return
    reward = torch.ones(success_node.size(0), device=success_node.device)
    target = success_node
    for tr in reversed(traces):
        target = reinforce_one_hop(model, tr, target, reward, args.path_lr)

    # Teach the local navigation query to point toward the successful terminal memory.
    key = model.space.key(success_node)
    q = F.normalize(model.graph.context_q(cur), dim=-1)
    err = key - q
    model.graph.context_q.weight.add_(
        float(args.path_context_lr) * (err.T @ cur.float()) / max(1, cur.size(0))
    )


@torch.no_grad()
def structural_rewire(model, memory, trace_last, unresolved_rows, target, args):
    """Create one local shortcut to an existing target alias; target index is training-only."""
    if unresolved_rows.numel() == 0:
        return 0
    tgt = target[unresolved_rows]
    anchor = memory.anchor[tgt]
    has_anchor = anchor.ge(0)
    if not has_anchor.any():
        return 0
    rows = unresolved_rows[has_anchor]
    anchors = anchor[has_anchor]

    src_mass = trace_last["src_mass"][rows]
    src_slot = src_mass.argmax(-1)
    rr = torch.arange(rows.numel(), device=rows.device)
    src = trace_last["src_ids"][rows, src_slot]
    priority = src_mass[rr, src_slot]

    # One rewire per source node in this local event; strongest proposal wins.
    keep = _amax_keep(src, priority, model.n_concepts)
    src, anchors = src[keep], anchors[keep]
    if src.numel() == 0:
        return 0
    logits = model.graph.edge_logits.weight[src]
    if model.degree > 1:
        slot = logits[:, 1:].argmin(-1) + 1  # preserve self edge in slot 0
    else:
        slot = torch.zeros(src.numel(), dtype=torch.long, device=src.device)
    model.graph.destination[src, slot] = anchors.to(model.graph.destination.dtype)
    model.graph.edge_logits.weight[src, slot] = float(args.rewire_logit)
    return int(src.numel())


@torch.inference_mode()
def evaluate_memory(model, memory, prefixes, targets, n_items, device, batch_size=1024):
    model.eval()
    hr = ndcg = mrr = 0.0
    candidate_sum = 0.0
    recall_candidate = 0.0
    total = len(targets)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        idxs = list(range(start, end))
        seq, lens = make_eval_batch(prefixes, idxs, MAX_LEN)
        seq = seq.to(device)
        lens = lens.to(device)
        _, I, M = model.encode_with_states(seq)
        row = torch.arange(seq.size(0), device=device)
        last = (lens - 1).clamp_min(0)
        ids = I[row, last]
        mass = M[row, last].float()
        labels = memory.owner[ids]
        contrib = mass * memory.strength[ids]
        contrib = contrib * labels.ne(0).float()
        scores = torch.zeros(seq.size(0), n_items + 1, device=device)
        scores.scatter_add_(1, labels.long(), contrib)
        scores[:, 0] = 0.0

        for r, i in enumerate(idxs):
            truth = int(targets[i])
            seen = set(prefixes[i])
            seen.discard(truth)
            if seen:
                scores[r, torch.as_tensor(list(seen), device=device, dtype=torch.long)] = -1e20
            positive = scores[r].gt(0)
            candidate_sum += float(positive.sum())
            ts = float(scores[r, truth])
            if ts <= 0:
                continue
            recall_candidate += 1.0
            better = int((scores[r] > scores[r, truth]).sum())
            # deterministic tie-break by lower item id
            ties_lower = int(((scores[r] == scores[r, truth]) & (torch.arange(n_items + 1, device=device) < truth)).sum())
            rank = 1 + better + ties_lower
            if rank <= 10:
                hr += 1.0
                ndcg += 1.0 / math.log2(rank + 1)
                mrr += 1.0 / rank
    return {
        "HR@10": hr / total,
        "NDCG@10": ndcg / total,
        "MRR@10": mrr / total,
        "target_in_sparse_candidates": recall_candidate / total,
        "mean_positive_candidates": candidate_sum / total,
    }


@torch.no_grad()
def train_epoch(model, memory, ds, device, args, epoch, arm):
    model.eval()
    loader = fast._loader(ds, args.batch_size, epoch, True)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed * 100003 + epoch)
    positions = hits = allocations = rewires = no_space = 0
    semantic_margin = 0.0
    local_steps = 0
    t0 = time.perf_counter()

    for bi, (tokens, lengths) in enumerate(loader, start=1):
        tokens = tokens.to(device, non_blocking=True)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        B, L = x.shape
        ids = torch.zeros(B, model.active, dtype=torch.long, device=device)
        mass = torch.zeros(B, model.active, dtype=torch.float32, device=device)

        for t in range(L):
            active_rows = x[:, t].ne(0).nonzero(as_tuple=False).squeeze(-1)
            if active_rows.numel() == 0:
                continue
            cur_ids = x[active_rows, t]
            xids, xmass, traces, fi, fm, cur = walk_event(
                model, ids[active_rows], mass[active_rows], cur_ids
            )
            ids.index_copy_(0, active_rows, xids)
            mass.index_copy_(0, active_rows, xmass)

            valid_local = y[active_rows, t].ne(0)
            if not valid_local.any():
                continue
            rows = active_rows[valid_local]
            target = y[rows, t]
            sid = ids[rows]
            smass = mass[rows]
            curv = cur[valid_local]
            fiv = fi[valid_local]
            fmv = fm[valid_local]
            positions += int(rows.numel())

            semantic_margin += semantic_local_update(
                model, cur_ids[valid_local], curv, fiv, fmv, target, gen, args
            )
            local_steps += 1

            has_hit, hit_node = choose_best_hit(memory, sid, smass, target)
            hits += int(has_hit.sum())
            if has_hit.any():
                reinforce_association(memory, hit_node[has_hit], args.assoc_lr)
                if arm != "allocate_only":
                    trv = [
                        {k: v[valid_local][has_hit] for k, v in tr.items()}
                        for tr in traces
                    ]
                    reinforce_path(model, trv, hit_node[has_hit], curv[has_hit], args)

            miss = ~has_hit
            allocated_global_rows = torch.empty(0, dtype=torch.long, device=device)
            if miss.any():
                miss_rows = miss.nonzero(as_tuple=False).squeeze(-1)
                ar_local, anode = allocate_from_reached(
                    memory,
                    sid[miss_rows],
                    smass[miss_rows],
                    target[miss_rows],
                    args.max_aliases,
                    args.alloc_strength,
                )
                if ar_local.numel():
                    allocated_global_rows = miss_rows[ar_local]
                    allocations += int(anode.numel())
                    if arm != "allocate_only":
                        trv = [
                            {k: v[valid_local][allocated_global_rows] for k, v in tr.items()}
                            for tr in traces
                        ]
                        reinforce_path(
                            model,
                            trv,
                            anode,
                            curv[allocated_global_rows],
                            args,
                        )

                unresolved = miss.clone()
                if allocated_global_rows.numel():
                    unresolved[allocated_global_rows] = False
                unr = unresolved.nonzero(as_tuple=False).squeeze(-1)
                if unr.numel():
                    no_space += int(unr.numel())
                    if arm == "allocate_rewire":
                        tr_last = {k: v[valid_local] for k, v in traces[-1].items()}
                        rewires += structural_rewire(
                            model, memory, tr_last, unr, target, args
                        )

        if args.progress_every > 0 and (
            bi == 1 or bi % args.progress_every == 0 or bi == len(loader)
        ):
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print(
                "ALLOC_PROGRESS",
                json.dumps({
                    "arm": arm,
                    "epoch": epoch,
                    "batch": bi,
                    "batches": len(loader),
                    "positions": positions,
                    "hit_rate": hits / max(1, positions),
                    "allocations": allocations,
                    "rewires": rewires,
                    "occupied": int(memory.owner.ne(0).sum()),
                    "elapsed_s": round(elapsed, 2),
                }),
                flush=True,
            )

    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    grads = [n for n, p in model.named_parameters() if p.grad is not None]
    if grads:
        raise AssertionError(f"gradient tensors created: {grads[:8]}")
    occupied = int(memory.owner.ne(0).sum())
    covered_items = int(memory.alias_count.gt(0).sum())
    return {
        "epoch": epoch,
        "positions": positions,
        "seconds": sec,
        "positions_per_s": positions / max(sec, 1e-9),
        "train_memory_hit_rate": hits / max(1, positions),
        "allocations": allocations,
        "rewires": rewires,
        "unresolved_misses": no_space,
        "occupied_concepts": occupied,
        "occupied_fraction": occupied / model.n_concepts,
        "covered_items": covered_items,
        "item_coverage": covered_items / model.n_items,
        "mean_aliases_per_covered_item": occupied / max(1, covered_items),
        "mean_semantic_margin": semantic_margin / max(1, local_steps),
        "autograd_grad_tensors": 0,
        "loss_backward_calls": 0,
        "optimizer": None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs-per-arm", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--data-dir", default="/content/drive/MyDrive/sparsewalker_data")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_allocate_on_arrival")
    p.add_argument("--progress-every", type=int, default=0)

    # Semantic local learner (same family as Experiment 39).
    p.add_argument("--negatives", type=int, default=32)
    p.add_argument("--temperature", type=float, default=.15)
    p.add_argument("--item-lr", type=float, default=.02)
    p.add_argument("--input-lr", type=float, default=.004)
    p.add_argument("--prototype-lr", type=float, default=.025)
    p.add_argument("--key-lr", type=float, default=.015)
    p.add_argument("--message-gain", type=float, default=8.0)

    # Allocated memory / sparse path plasticity.
    p.add_argument("--max-aliases", type=int, default=4)
    p.add_argument("--alloc-strength", type=float, default=.55)
    p.add_argument("--assoc-lr", type=float, default=.08)
    p.add_argument("--path-lr", type=float, default=.08)
    p.add_argument("--path-context-lr", type=float, default=.002)
    p.add_argument("--rewire-logit", type=float, default=2.0)
    a = p.parse_args()

    assert torch.cuda.is_available(), "GPU runtime required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed_all(a.seed)

    print("ALLOC_DATA_START", flush=True)
    t0 = time.perf_counter()
    data = load_dataset("ml1m", a.data_dir)
    split = split_data(data["sequences"])
    ds = WindowDataset(split["train"], MAX_LEN, a.seed)
    print("ALLOC_DATA_DONE", json.dumps({
        "seconds": round(time.perf_counter() - t0, 2),
        "users": len(data["sequences"]),
        "n_items": data["n_items"],
        "concepts": 65536,
    }), flush=True)

    template = fast.build(data["n_items"]).to(device)
    base.init_model(template, a.seed, a.message_gain)
    init_state = cpu_state(template)
    del template
    torch.cuda.empty_cache()

    out = Path(a.output_dir) / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "experiment": "ML1M-AllocateOnArrival-v1",
        "hypothesis": "write label where sparse walk arrives instead of finding a preassigned label address",
        "arms": a.arms,
        "strict_forward_only": True,
        "global_target_address_at_inference": False,
        "single_label_per_concept": True,
        "max_aliases_per_item": a.max_aliases,
        "protocol": "ML1M leave-two-out, max_len=200, seen masking; sparse associative candidate retrieval",
        "args": vars(a),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    ranking = []
    for arm in a.arms:
        print("\nALLOC_ARM_START", json.dumps({"arm": arm}), flush=True)
        seed_all(a.seed)
        model = fast.build(data["n_items"]).to(device)
        model.load_state_dict(init_state)
        for p0 in model.parameters():
            p0.requires_grad_(False)
            p0.grad = None
        memory = AssocMemory(model.n_concepts, model.n_items, device)
        hist = []
        best = -1.0
        best_epoch = 0
        best_model = None
        best_memory = None

        for epoch in range(1, a.epochs_per_arm + 1):
            stats = train_epoch(model, memory, ds, device, a, epoch, arm)
            vt = time.perf_counter()
            val = evaluate_memory(
                model, memory, split["val_prefix"], split["val_target"],
                data["n_items"], device, a.eval_batch_size,
            )
            row = {**stats, **{f"val_{k}": float(v) for k, v in val.items()},
                   "val_seconds": time.perf_counter() - vt}
            hist.append(row)
            print("ALLOC_EPOCH", json.dumps({"arm": arm, **row}), flush=True)
            nd = float(val["NDCG@10"])
            if nd > best:
                best = nd
                best_epoch = epoch
                best_model = cpu_state(model)
                best_memory = memory.state_cpu()

        torch.save({
            "model": best_model,
            "memory": best_memory,
            "arm": arm,
            "epoch": best_epoch,
            "best_val_NDCG@10": best,
            "config": cfg,
        }, out / f"best_{arm}.pt")
        (out / f"history_{arm}.json").write_text(json.dumps(hist, indent=2))
        result = {
            "arm": arm,
            "best_epoch": best_epoch,
            "best_val_NDCG@10": best,
            "final_val_NDCG@10": float(hist[-1]["val_NDCG@10"]),
            "final_target_in_sparse_candidates": float(hist[-1]["val_target_in_sparse_candidates"]),
            "final_mean_positive_candidates": float(hist[-1]["val_mean_positive_candidates"]),
            "final_occupied_concepts": int(hist[-1]["occupied_concepts"]),
            "final_item_coverage": float(hist[-1]["item_coverage"]),
            "mean_positions_per_s": float(np.mean([r["positions_per_s"] for r in hist])),
        }
        ranking.append(result)
        print("ALLOC_ARM_DONE", json.dumps(result), flush=True)
        del model, memory
        torch.cuda.empty_cache()

    ranking.sort(key=lambda r: r["best_val_NDCG@10"], reverse=True)
    summary = {
        "config": cfg,
        "ranking": ranking,
        "references": {
            "analytic_local_contrastive_val_NDCG@10": 0.09352186187904432,
            "SASRec_public_protocol_reference_NDCG@10": 0.15965,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("\nALLOC_RESULT", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
