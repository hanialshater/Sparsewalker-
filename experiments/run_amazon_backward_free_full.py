#!/usr/bin/env python
"""Full from-scratch backward-free SparseWalker experiment.

Goal
----
Train a useful SparseWalker on Amazon Beauty with:
  * no autograd graph during learning,
  * zero loss.backward() calls,
  * no optimizer,
  * no pretrained neural parameters.

The model keeps fixed random item codes as stable sensory / target identifiers.
Every adaptive component is updated with local forward-only rules:
  1. predictive router: current item -> sparse target concept code;
  2. concept values: active/teacher concepts -> next-item code;
  3. graph context projection: current context -> rewarding destination keys;
  4. concept keys: rewarding destinations <-> local context query;
  5. graph topology: Hebbian rewiring from active source concepts to target concepts;
  6. message projection: local delta rule mapping sparse concept message to next-item code.

The target concept code is a deterministic sparse hash of the next item ID. This
provides a stationary teaching signal without a learned teacher network.

This is an exploratory architecture experiment, not an optimized trainer.
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


@torch.no_grad()
def initialize_backward_free(model, seed=42):
    """Start from scratch but create stable local-learning geometry."""
    g = torch.Generator(device=model.item.weight.device)
    g.manual_seed(int(seed) + 1009)

    # Stable random item codes. Padding stays zero. These are identifiers, not a
    # learned pretrained representation.
    w = torch.randn(
        model.n_items + 1,
        model.d_model,
        device=model.item.weight.device,
        dtype=model.item.weight.dtype,
        generator=g,
    )
    w = F.normalize(w, dim=-1)
    w[0].zero_()
    model.item.weight.copy_(w)

    # Local concept values are an additive factorization. This makes left/right
    # prototypes directly plastic without requiring error backprop through an
    # extra random matrix.
    model.space.left_value.weight.normal_(0, 0.05, generator=g)
    model.space.right_value.weight.normal_(0, 0.05, generator=g)
    model.space.value_proj.weight.zero_()
    eye = torch.eye(model.d_model, device=w.device, dtype=w.dtype)
    model.space.value_proj.weight[:, : model.d_model].copy_(0.5 * eye)
    model.space.value_proj.weight[:, model.d_model :].copy_(0.5 * eye)
    model.space.value_proj.bias.zero_()

    # Random factorized router/key geometry and random current->concept maps.
    model.space.left_router.copy_(F.normalize(torch.randn_like(model.space.left_router, generator=g), dim=-1))
    model.space.right_router.copy_(F.normalize(torch.randn_like(model.space.right_router, generator=g), dim=-1))
    model.space.left_key.copy_(F.normalize(torch.randn_like(model.space.left_key, generator=g), dim=-1))
    model.space.right_key.copy_(F.normalize(torch.randn_like(model.space.right_key, generator=g), dim=-1))

    model.router.left_q.weight.normal_(0, 1 / math.sqrt(model.d_model), generator=g)
    model.router.right_q.weight.normal_(0, 1 / math.sqrt(model.d_model), generator=g)
    model.graph.context_q.weight.normal_(0, 1 / math.sqrt(model.d_model), generator=g)

    # Static edge preference is deliberately removed: graph behavior comes from
    # geometry + local structural rewiring.
    model.graph.edge_logits.weight.zero_()

    # Start message readout close to identity; it is then updated locally.
    model.message_proj.weight.copy_(eye)
    model.norm.weight.fill_(1.0)
    model.norm.bias.zero_()

    # Fixed random initial topology, including self edge in slot 0.
    n = model.n_concepts
    dest = torch.randint(0, n, (n, model.degree), device=w.device, dtype=torch.int32, generator=g)
    dest[:, 0] = torch.arange(n, device=w.device, dtype=torch.int32)
    model.graph.destination.copy_(dest)

    # No parameter participates in autograd.
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None


def _hash_teacher(item_ids, side):
    """Deterministic 4-concept target code, [N] -> [N,4].

    Two independent left hashes x two right hashes produce the same cardinality
    as the normal top_side=2 router.
    """
    x = item_ids.long()
    mask = x.ne(0)
    # Prime-ish odd multipliers; all arithmetic is deterministic integer math.
    l1 = (x * 131 + 17) % side
    l2 = (x * 197 + 73) % side
    r1 = (x * 151 + 29) % side
    r2 = (x * 211 + 101) % side
    left = torch.stack([l1, l2], -1)
    right = torch.stack([r1, r2], -1)
    ids = (left[:, :, None] * side + right[:, None, :]).reshape(x.numel(), 4)
    ids = torch.where(mask[:, None], ids, torch.zeros_like(ids))
    return ids


def _scatter_mean(index, value, size, weight=None):
    """Return weighted mean per integer index."""
    if weight is None:
        weight = torch.ones(index.numel(), device=value.device, dtype=value.dtype)
    weight = weight.to(value.dtype)
    out = torch.zeros(size, value.size(-1), device=value.device, dtype=value.dtype)
    den = torch.zeros(size, device=value.device, dtype=value.dtype)
    out.index_add_(0, index, value * weight[:, None])
    den.index_add_(0, index, weight)
    used = den > 0
    out[used] /= den[used, None]
    return out, den


@torch.no_grad()
def _update_teacher_values(model, teacher_ids, target_vec, lr):
    """Pull factorized concept value prototypes toward next-item target code."""
    if teacher_ids.numel() == 0:
        return 0.0
    l, r = model.space.split(teacher_ids)
    # Each teacher concept contributes equally. Updating both factors toward the
    # target makes their 0.5*(left+right) value converge to the target.
    tgt = target_vec[:, None, :].expand(-1, teacher_ids.size(1), -1).reshape(-1, model.d_model)
    li = l.reshape(-1)
    ri = r.reshape(-1)
    lmean, lden = _scatter_mean(li, tgt, model.side)
    rmean, rden = _scatter_mean(ri, tgt, model.side)
    lused = lden > 0
    rused = rden > 0
    lrows = lused.nonzero(as_tuple=False).squeeze(-1)
    rrows = rused.nonzero(as_tuple=False).squeeze(-1)
    old_l = model.space.left_value.weight[lrows]
    old_r = model.space.right_value.weight[rrows]
    dl = float(lr) * (lmean[lrows] - old_l)
    dr = float(lr) * (rmean[rrows] - old_r)
    model.space.left_value.weight.index_copy_(0, lrows, old_l + dl)
    model.space.right_value.weight.index_copy_(0, rrows, old_r + dr)
    return float(dl.abs().mean().item() + dr.abs().mean().item())


@torch.no_grad()
def _update_router_predictive(model, current_vec, teacher_ids, lr_q, lr_proto):
    """Direct local predictive rule: current item query -> next-item concept code."""
    if current_vec.numel() == 0:
        return 0.0
    l, r = model.space.split(teacher_ids)
    # Average the 4 teacher codes down to two left / two right targets.
    luniq = l[:, [0, 2]] if l.size(1) >= 3 else l[:, :2]
    runiq = r[:, [0, 1]] if r.size(1) >= 2 else r[:, :2]
    lp = F.normalize(model.space.left_router[luniq], dim=-1).mean(1)
    rp = F.normalize(model.space.right_router[runiq], dim=-1).mean(1)
    lp = F.normalize(lp, dim=-1)
    rp = F.normalize(rp, dim=-1)

    ql = F.normalize(F.linear(current_vec, model.router.left_q.weight), dim=-1)
    qr = F.normalize(F.linear(current_vec, model.router.right_q.weight), dim=-1)
    el = lp - ql
    er = rp - qr
    # Delta rule is local to each projection: output error x local input.
    model.router.left_q.weight.add_(float(lr_q) * (el.T @ current_vec) / max(1, current_vec.size(0)))
    model.router.right_q.weight.add_(float(lr_q) * (er.T @ current_vec) / max(1, current_vec.size(0)))

    # Symmetric competitive prototype adaptation on the teacher rows.
    for ids2, proto, q, eta in (
        (luniq, model.space.left_router, ql, lr_proto),
        (runiq, model.space.right_router, qr, lr_proto),
    ):
        idx = ids2.reshape(-1)
        val = q[:, None, :].expand(-1, ids2.size(1), -1).reshape(-1, model.h)
        mean, den = _scatter_mean(idx, val, model.side)
        used = den > 0
        rows = used.nonzero(as_tuple=False).squeeze(-1)
        old = proto[rows]
        tgt = F.normalize(mean[rows], dim=-1)
        new = F.normalize((1.0 - float(eta)) * old + float(eta) * tgt, dim=-1)
        proto.index_copy_(0, rows, new)
    return float((el.abs().mean() + er.abs().mean()).item())


@torch.no_grad()
def _update_context_and_keys(model, context_vec, teacher_ids, lr_q, lr_key):
    """Teach graph query geometry using only local next-concept keys."""
    if context_vec.numel() == 0:
        return 0.0
    tl, tr = model.space.split(teacher_ids)
    tkey = F.normalize(
        model.space.left_key[tl] + model.space.right_key[tr],
        dim=-1,
    ).mean(1)
    tkey = F.normalize(tkey, dim=-1)
    q = F.normalize(F.linear(context_vec, model.graph.context_q.weight), dim=-1)
    err = tkey - q
    model.graph.context_q.weight.add_(float(lr_q) * (err.T @ context_vec) / max(1, context_vec.size(0)))

    # Hebbian key adaptation for the teacher destination concepts.
    for idx2, key_table in ((tl, model.space.left_key), (tr, model.space.right_key)):
        idx = idx2.reshape(-1)
        val = q[:, None, :].expand(-1, idx2.size(1), -1).reshape(-1, model.h)
        mean, den = _scatter_mean(idx, val, model.side)
        used = den > 0
        rows = used.nonzero(as_tuple=False).squeeze(-1)
        old = key_table[rows]
        tgt = F.normalize(mean[rows], dim=-1)
        new = F.normalize((1.0 - float(lr_key)) * old + float(lr_key) * tgt, dim=-1)
        key_table.index_copy_(0, rows, new)
    return float(err.abs().mean().item())


@torch.no_grad()
def _hebbian_rewire(model, source_ids, source_mass, teacher_ids, teacher_weight, epoch, rate):
    """Locally rewire a fraction of source concepts toward observed target concepts.

    Each touched source proposes the strongest target concept observed in the
    current batch. The slot rotates across non-self edges, preserving slot 0.
    """
    if source_ids.numel() == 0 or rate <= 0:
        return 0
    B, K = source_ids.shape
    T = teacher_ids.size(1)

    # Pair each source with every target teacher; score by source mass * teacher mass.
    src = source_ids[:, :, None].expand(B, K, T).reshape(-1)
    tgt = teacher_ids[:, None, :].expand(B, K, T).reshape(-1)
    score = (source_mass[:, :, None] * teacher_weight[:, None, :]).reshape(-1).float()
    valid = (src > 0) & (tgt > 0) & (score > 0)
    if not valid.any():
        return 0
    src = src[valid]
    tgt = tgt[valid]
    score = score[valid]

    pair = src * model.n_concepts + tgt
    up, inv = torch.unique(pair, return_inverse=True)
    agg = torch.zeros(up.numel(), device=score.device, dtype=torch.float32)
    agg.scatter_add_(0, inv, score)
    usrc = up // model.n_concepts
    utgt = up % model.n_concepts

    # Best target per source without Python/GPU synchronization.
    best_score = torch.full(
        (model.n_concepts,),
        -float("inf"),
        device=score.device,
        dtype=torch.float32,
    )
    best_score.scatter_reduce_(0, usrc, agg, reduce="amax", include_self=True)
    is_best = agg >= best_score[usrc]
    sentinel = int(model.n_concepts)
    best_target = torch.full(
        (model.n_concepts,),
        sentinel,
        device=score.device,
        dtype=torch.long,
    )
    best_target.scatter_reduce_(
        0,
        usrc[is_best],
        utgt[is_best],
        reduce="amin",
        include_self=True,
    )
    usrc = (best_target < sentinel).nonzero(as_tuple=False).squeeze(-1)
    if usrc.numel() == 0:
        return 0
    utgt = best_target[usrc]

    # Deterministic subsampling by source ID so rate is reproducible.
    if rate < 1.0:
        gate = ((usrc * 1103515245 + int(epoch) * 12345) % 10000).float() < float(rate) * 10000
        usrc = usrc[gate]
        utgt = utgt[gate]
    if usrc.numel() == 0:
        return 0

    if model.degree <= 1:
        slot = torch.zeros_like(usrc)
    else:
        slot = 1 + ((usrc + int(epoch)) % (model.degree - 1))
    model.graph.destination[usrc, slot] = utgt.to(torch.int32)
    return int(usrc.numel())


@torch.no_grad()
def _update_message_projection(model, msg, current_vec, target_vec, lr):
    """Local predictive delta rule for sparse message -> next-item code."""
    if msg.numel() == 0:
        return 0.0
    raw = current_vec + F.linear(msg, model.message_proj.weight)
    # Normalize target scale to roughly match residual pre-LN scale.
    tgt = target_vec * math.sqrt(model.d_model)
    err = (tgt - raw).clamp(-4.0, 4.0)
    dW = (err.T @ msg) / max(1, msg.size(0))
    model.message_proj.weight.add_(float(lr) * dW)
    model.message_proj.weight.clamp_(-4.0, 4.0)
    return float(err.abs().mean().item())


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
def backward_free_full_epoch(model, dataset, device, args, epoch):
    """One complete no-autograd, no-optimizer learning epoch."""
    model.eval()
    total_positions = 0
    batches = 0
    rewired = 0
    diag = dict(router=0.0, value=0.0, context=0.0, message=0.0)
    diag_n = 0
    t0 = time.perf_counter()

    for tokens, lengths in _loader(dataset, args.batch_size, epoch):
        tokens = tokens.to(device, non_blocking=True)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        B, L = x.shape
        valid_all = y.ne(0) & x.ne(0)
        npos = int(valid_all.sum().item())
        if npos == 0:
            continue

        # Fixed item codes are both sensory input and stationary prediction targets.
        xcode = model.item(x)
        ycode = model.item(y.clamp_min(0))
        item_state = xcode * math.sqrt(model.d_model)

        # Current learned router; target teacher codes are deterministic hashes.
        fi, fm = model.router(item_state.reshape(B * L, model.d_model), model.space)
        fi = fi.view(B, L, -1)
        fm = fm.view(B, L, -1)

        ids = torch.zeros(B, model.active, dtype=torch.long, device=device)
        mass = torch.zeros(B, model.active, dtype=item_state.dtype, device=device)

        for t in range(L):
            valid = valid_all[:, t]
            act = x[:, t].ne(0)
            if not act.any():
                continue
            af = act.to(item_state.dtype)[:, None]

            xids, xmass = model._merge(
                ids,
                mass * af,
                fi[:, t],
                fm[:, t] * af,
            )

            if valid.any():
                cv = xcode[valid, t].float()
                tv = ycode[valid, t].float()
                teacher = _hash_teacher(y[valid, t], model.side)
                diag["router"] += _update_router_predictive(
                    model, cv, teacher, args.router_lr, args.prototype_lr
                )
                diag["value"] += _update_teacher_values(
                    model, teacher, tv, args.value_lr
                )
                diag["context"] += _update_context_and_keys(
                    model, cv, teacher, args.context_lr, args.key_lr
                )

            # Two sparse graph hops; topology can adapt after observing y.
            for hop in range(model.layers_n):
                if valid.any():
                    teacher = _hash_teacher(y[valid, t], model.side)
                    tw = torch.full(
                        teacher.shape,
                        1.0 / teacher.size(1),
                        device=device,
                        dtype=torch.float32,
                    )
                    rewired += _hebbian_rewire(
                        model,
                        xids[valid],
                        xmass[valid].float(),
                        teacher,
                        tw,
                        epoch * model.layers_n + hop,
                        args.rewire_rate,
                    )
                xids, xmass = model.graph(
                    xids,
                    xmass,
                    item_state[:, t],
                    model.space,
                    track_touched=False,
                )

            ids = torch.where(act[:, None], xids, ids)
            mass = torch.where(act[:, None], xmass, mass)

            if valid.any():
                msg = (
                    model.space.value(ids[valid]) * mass[valid, :, None]
                ).sum(1).float()
                diag["message"] += _update_message_projection(
                    model,
                    msg,
                    item_state[valid, t].float(),
                    ycode[valid, t].float(),
                    args.message_lr,
                )
                diag_n += 1

        total_positions += npos
        batches += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    grads = [name for name, p in model.named_parameters() if p.grad is not None]
    if grads:
        raise AssertionError(f"Backward-free run created gradient tensors: {grads[:8]}")

    out = {
        "epoch": int(epoch),
        "positions": int(total_positions),
        "seconds": float(sec),
        "positions_per_s": float(total_positions / max(sec, 1e-9)),
        "batches": int(batches),
        "rewired_edges": int(rewired),
        "autograd_grad_tensors": 0,
        "loss_backward_calls": 0,
        "optimizer": None,
    }
    for k, v in diag.items():
        out[f"mean_{k}_signal"] = float(v / max(1, diag_n))
    return out


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=AMAZON, default="beauty")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=1)

    # Independent local plasticity rates; intentionally conservative defaults.
    p.add_argument("--router-lr", type=float, default=0.01)
    p.add_argument("--prototype-lr", type=float, default=0.02)
    p.add_argument("--value-lr", type=float, default=0.05)
    p.add_argument("--context-lr", type=float, default=0.01)
    p.add_argument("--key-lr", type=float, default=0.02)
    p.add_argument("--message-lr", type=float, default=0.002)
    p.add_argument("--rewire-rate", type=float, default=0.25)

    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/sparsewalker_backward_free_full",
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
    initialize_backward_free(model, args.seed)

    init_val, init_test = evaluate(
        model, split, data["n_items"], device, batch_size=args.eval_batch_size
    )
    print("BFFULL_INIT", json.dumps({"val": init_val, "test": init_test}), flush=True)

    ds = WindowDataset(split["train"], 50, args.seed)
    out = Path(args.output_dir) / args.dataset / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "BackwardFreeFull-v1",
        "dataset": args.dataset,
        "seed": args.seed,
        "pretrained_parameters": False,
        "autograd": False,
        "optimizer": None,
        "loss_backward_calls": 0,
        "item_codes": "fixed random unit vectors",
        "target_concepts": "deterministic 4-way sparse hash of next item",
        "router_rule": "local predictive delta + prototype Hebbian update",
        "concept_value_rule": "local EMA toward next-item code",
        "graph_context_rule": "local delta toward target concept key",
        "concept_key_rule": "local Hebbian alignment with context query",
        "graph_structure_rule": "Hebbian rewiring source concept -> next-item target concept",
        "message_rule": "local delta mapping sparse message -> next-item code",
        "rates": {
            "router_lr": args.router_lr,
            "prototype_lr": args.prototype_lr,
            "value_lr": args.value_lr,
            "context_lr": args.context_lr,
            "key_lr": args.key_lr,
            "message_lr": args.message_lr,
            "rewire_rate": args.rewire_rate,
        },
        "reference_test_NDCG@10": {
            "SASRec_FullCE_seed42": 0.031195719394901355,
            "SparseWalker_backprop_seed42": 0.044882819399656555,
        },
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    best = float(init_val["NDCG@10"])
    best_epoch = 0
    best_state = cpu_state_dict(model)
    history = []

    for epoch in range(1, args.epochs + 1):
        stats = backward_free_full_epoch(model, ds, device, args, epoch)
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
            print("BFFULL_EPOCH", json.dumps(row), flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=2))

            ndcg = float(val["NDCG@10"])
            if ndcg > best:
                best = ndcg
                best_epoch = epoch
                best_state = cpu_state_dict(model)
                torch.save(
                    {"model": best_state, "epoch": epoch, "val": val, "config": config},
                    out / "best.pt",
                )

    model.load_state_dict(best_state)
    best_val, best_test = evaluate(
        model, split, data["n_items"], device, batch_size=args.eval_batch_size
    )
    result = {
        "config": config,
        "initial": {"val": init_val, "test": init_test},
        "best_epoch": int(best_epoch),
        "best_backward_free": {"val": best_val, "test": best_test},
        "vs_sasrec_test_ndcg_ratio": float(
            best_test["NDCG@10"] / 0.031195719394901355
        ),
        "vs_backprop_walker_test_ndcg_ratio": float(
            best_test["NDCG@10"] / 0.044882819399656555
        ),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("BFFULL_RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
