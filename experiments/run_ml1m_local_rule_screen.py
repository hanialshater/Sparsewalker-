#!/usr/bin/env python
"""Experiment 44: diverse local-credit screening tournament on ML-1M.

The architecture, initialization, data split, negative stream, and evaluation are
held fixed. Only the local teaching/update rule changes. This is intentionally a
short-horizon family screen, not a tuned benchmark.

Arms:
- lc_sigmoid: Experiment-43 analytic local contrastive baseline.
- ff_npair: N-pair / Distance-Forward-style local metric objective.
- target_route_prop: target-propagation-inspired next-concept teaching signal.
- target_broadcast: NoProp-inspired direct next-item representation broadcast.
- forward_gradient: forward finite-difference estimate of the hidden teaching signal.
- eligibility_trace: e-prop-inspired decayed pre/message traces with vector errors.

All arms remain free of end-to-end backpropagation. The forward-gradient arm uses
only forward loss evaluations; none of the arms calls loss.backward().
"""
import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_amazon_local_contrastive_walker as base
import run_ml1m_local_contrastive_walker as mlbase
from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full

MAX_LEN = 200
DEFAULT_ARMS = [
    "lc_sigmoid",
    "ff_npair",
    "target_route_prop",
    "target_broadcast",
    "forward_gradient",
    "eligibility_trace",
]


def _stats_from_logits(logits):
    p = F.softmax(logits, dim=-1)
    pos = logits[:, 0]
    neg = logits[:, 1:].mean(-1)
    return {
        "margin": float((pos - neg).mean()),
        "pp": float(p[:, 0].mean()),
        "pn": float(p[:, 1:].mean()),
    }


@torch.no_grad()
def npair_signal(model, h, pos_ids, neg_ids, lr, temp):
    """Exact analytic local gradient of sampled N-pair metric loss."""
    h = F.normalize(h.float(), dim=-1)
    pos = F.normalize(model.item.weight[pos_ids].float(), dim=-1)
    neg = F.normalize(model.item.weight[neg_ids].float(), dim=-1)
    logits = torch.cat([
        (h * pos).sum(-1, keepdim=True),
        (neg * h[:, None, :]).sum(-1),
    ], dim=1) / float(temp)
    p = F.softmax(logits, dim=-1)
    signal = (
        (1.0 - p[:, 0])[:, None] * pos
        - (p[:, 1:, None] * neg).sum(1)
    ) / float(temp)

    pos_delta = float(lr) * (1.0 - p[:, 0])[:, None] * h / float(temp)
    neg_delta = -float(lr) * p[:, 1:, None] * h[:, None, :] / float(temp)
    ids = torch.cat([pos_ids, neg_ids.reshape(-1)])
    delta = torch.cat([pos_delta, neg_delta.reshape(-1, model.d_model)]).to(model.item.weight.dtype)
    model.item.weight.index_add_(0, ids, delta)
    base.normalize_rows(model.item.weight, ids)
    model.item.weight[0].zero_()
    return signal, _stats_from_logits(logits)


@torch.no_grad()
def target_broadcast_signal(model, h, pos_ids, lr):
    """Direct target-vector broadcast: every local module sees the next-item vector."""
    h = F.normalize(h.float(), dim=-1)
    target = F.normalize(model.item.weight[pos_ids].float(), dim=-1)
    signal = target - h
    model.item.weight.index_add_(
        0, pos_ids, (float(lr) * (h - target)).to(model.item.weight.dtype)
    )
    base.normalize_rows(model.item.weight, pos_ids)
    model.item.weight[0].zero_()
    sim = (h * target).sum(-1)
    return signal, {
        "margin": float(sim.mean()),
        "pp": float(((sim + 1.0) * 0.5).mean()),
        "pn": 0.0,
    }


@torch.no_grad()
def forward_gradient_signal(model, h, pos_ids, neg_ids, lr, temp, generator, eps, dirs):
    """Forward finite-difference hidden gradient; output items still get exact local N-pair updates."""
    h0 = F.normalize(h.float(), dim=-1)
    pos = F.normalize(model.item.weight[pos_ids].float(), dim=-1)
    neg = F.normalize(model.item.weight[neg_ids].float(), dim=-1)

    logits0 = torch.cat([
        (h0 * pos).sum(-1, keepdim=True),
        (neg * h0[:, None, :]).sum(-1),
    ], dim=1) / float(temp)
    p = F.softmax(logits0, dim=-1)

    # Keep the output embedding update exact and local; only the hidden teaching
    # signal is estimated with forward directional derivatives.
    pos_delta = float(lr) * (1.0 - p[:, 0])[:, None] * h0 / float(temp)
    neg_delta = -float(lr) * p[:, 1:, None] * h0[:, None, :] / float(temp)
    all_ids = torch.cat([pos_ids, neg_ids.reshape(-1)])
    all_delta = torch.cat([pos_delta, neg_delta.reshape(-1, model.d_model)]).to(model.item.weight.dtype)
    model.item.weight.index_add_(0, all_ids, all_delta)
    base.normalize_rows(model.item.weight, all_ids)
    model.item.weight[0].zero_()

    estimate = torch.zeros_like(h0)
    for _ in range(max(1, int(dirs))):
        v = torch.randint(
            0, 2, h0.shape, device=h0.device, generator=generator, dtype=torch.int64
        ).float().mul_(2.0).sub_(1.0)
        hp = F.normalize(h0 + float(eps) * v, dim=-1)
        hm = F.normalize(h0 - float(eps) * v, dim=-1)
        lp = torch.cat([
            (hp * pos).sum(-1, keepdim=True),
            (neg * hp[:, None, :]).sum(-1),
        ], dim=1) / float(temp)
        lm = torch.cat([
            (hm * pos).sum(-1, keepdim=True),
            (neg * hm[:, None, :]).sum(-1),
        ], dim=1) / float(temp)
        loss_p = -F.log_softmax(lp, dim=-1)[:, 0]
        loss_m = -F.log_softmax(lm, dim=-1)[:, 0]
        directional = (loss_p - loss_m) / (2.0 * float(eps))
        estimate.add_(-directional[:, None] * v)
    estimate.div_(max(1, int(dirs)))
    # One-direction forward gradients are high variance. Limit pathological steps
    # without changing their direction.
    n = estimate.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    estimate = estimate * (8.0 / n).clamp(max=1.0)
    return estimate, _stats_from_logits(logits0)


@torch.no_grad()
def update_target_router(model, current, target_route, target_mass, lr, proto_lr, key_lr):
    """Teach the current router to point toward the next item's routed concept."""
    l, r = model.space.split(target_route)
    w = target_mass.float()
    tl = F.normalize((model.space.left_router[l] * w[:, :, None]).sum(1), dim=-1)
    tr = F.normalize((model.space.right_router[r] * w[:, :, None]).sum(1), dim=-1)
    ql = F.normalize(model.router.left_q(current), dim=-1)
    qr = F.normalize(model.router.right_q(current), dim=-1)
    el = tl - ql
    er = tr - qr
    b = max(1, current.size(0))
    model.router.left_q.weight.add_(float(lr) * (el.T @ current) / b)
    model.router.right_q.weight.add_(float(lr) * (er.T @ current) / b)
    # Move the target's selected prototypes/keys toward the current query as a
    # local target-propagation analogue.
    conf = base.update_router_and_keys(
        model, current, target_route, target_mass, proto_lr, key_lr
    )
    return conf


@torch.no_grad()
def update_context_with_trace(model, current_trace, current, target_route, target_mass, lr):
    key = model.space.key(target_route)
    target = F.normalize((key * target_mass[:, :, None]).sum(1), dim=-1)
    q = F.normalize(model.graph.context_q(current), dim=-1)
    err = target - q
    model.graph.context_q.weight.add_(
        float(lr) * (err.T @ current_trace) / max(1, current.size(0))
    )
    return float(err.abs().mean())


@torch.no_grad()
def train_epoch_rule(model, ds, device, args, epoch, rule):
    model.eval()
    ngen = torch.Generator(device=device)
    ngen.manual_seed(args.seed * 100003 + epoch)
    fgen = torch.Generator(device=device)
    fgen.manual_seed(args.seed * 700001 + epoch)
    loader = mlbase._loader(ds, args.batch_size, epoch, True)

    total = steps = batches = 0
    sm = sp = sn = rc = vc = cc = sig_norm = 0.0
    t0 = time.perf_counter()
    n_batches = len(loader)

    for tokens, lengths in loader:
        tokens = tokens.to(device, non_blocking=True)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        B, L = x.shape
        ids = torch.zeros(B, model.active, device=device, dtype=torch.long)
        mass = torch.zeros(B, model.active, device=device)
        pre_trace = torch.zeros(B, model.d_model, device=device)
        msg_trace = torch.zeros(B, model.d_model, device=device)

        for t in range(L):
            active_rows = x[:, t].ne(0).nonzero(as_tuple=False).squeeze(-1)
            if active_rows.numel() == 0:
                continue
            cur_ids = x[active_rows, t]
            cur = model.item(cur_ids).float()
            context = cur * math.sqrt(model.d_model)
            fi, fm = model.router(context, model.space)
            xids, xmass = model._merge(ids[active_rows], mass[active_rows], fi, fm)
            for _ in range(model.layers_n):
                xids, xmass = model.graph(xids, xmass, context, model.space, track_touched=False)
            ids.index_copy_(0, active_rows, xids)
            mass.index_copy_(0, active_rows, xmass)

            pre_trace[active_rows].mul_(args.trace_decay).add_(cur)
            valid_local = y[active_rows, t].ne(0)
            if not valid_local.any():
                continue
            rows = active_rows[valid_local]
            curv = cur[valid_local]
            target_ids = y[rows, t]
            target_pre = F.normalize(model.item.weight[target_ids].float(), dim=-1)
            target_context = target_pre * math.sqrt(model.d_model)
            target_route, target_mass = model.router(target_context, model.space)

            if rule == "target_route_prop":
                rc += update_target_router(
                    model, curv, target_route, target_mass.float(),
                    args.route_target_lr, args.prototype_lr, args.key_lr,
                )
            else:
                rc += base.update_router_and_keys(
                    model, curv, fi[valid_local], fm[valid_local].float(),
                    args.prototype_lr, args.key_lr,
                )

            if rule == "eligibility_trace":
                cc += update_context_with_trace(
                    model, pre_trace[rows], curv, target_route,
                    target_mass.float(), args.context_lr,
                )
            else:
                cc += base.update_context(
                    model, curv, target_route, target_mass.float(), args.context_lr
                )

            msg = (model.space.value(ids[rows]) * mass[rows, :, None]).sum(1).float()
            msg_trace[rows].mul_(args.trace_decay).add_(msg)
            h = model.norm(context[valid_local] + model.message_proj(msg)).float()
            neg = torch.randint(
                1, model.n_items + 1,
                (int(rows.numel()), args.negatives),
                device=device, generator=ngen,
            )

            if rule == "lc_sigmoid":
                signal, d = base.contrastive(
                    model, h, target_ids, neg, args.item_lr, args.temperature
                )
            elif rule in ("ff_npair", "target_route_prop", "eligibility_trace"):
                signal, d = npair_signal(
                    model, h, target_ids, neg, args.item_lr, args.temperature
                )
            elif rule == "target_broadcast":
                signal, d = target_broadcast_signal(
                    model, h, target_ids, args.item_lr
                )
            elif rule == "forward_gradient":
                signal, d = forward_gradient_signal(
                    model, h, target_ids, neg, args.item_lr, args.temperature,
                    fgen, args.fg_eps, args.fg_dirs,
                )
            else:
                raise ValueError(rule)

            base.update_current_items(model, cur_ids[valid_local], signal, args.input_lr)
            target_now = F.normalize(model.item.weight[target_ids].float(), dim=-1)
            vc += base.update_values(
                model, ids[rows], mass[rows].float(), target_now, args.value_lr
            )
            if rule == "eligibility_trace":
                base.update_message(model, msg_trace[rows], signal, args.message_lr)
            else:
                base.update_message(model, msg, signal, args.message_lr)

            sm += d["margin"]
            sp += d["pp"]
            sn += d["pn"]
            sig_norm += float(signal.norm(dim=-1).mean())
            steps += 1
            total += int(rows.numel())

        batches += 1
        if args.progress_every > 0 and (
            batches == 1 or batches % args.progress_every == 0 or batches == n_batches
        ):
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print("RULE_PROGRESS", json.dumps({
                "rule": rule, "epoch": epoch, "batch": batches,
                "batches": n_batches, "positions": total,
                "elapsed_s": round(elapsed, 2),
                "positions_per_s": round(total / max(elapsed, 1e-9), 1),
            }), flush=True)

    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    grads = [n for n, p in model.named_parameters() if p.grad is not None]
    if grads:
        raise AssertionError(f"grad tensors created: {grads[:8]}")
    z = max(1, steps)
    return {
        "rule": rule,
        "epoch": epoch,
        "positions": total,
        "seconds": sec,
        "positions_per_s": total / max(sec, 1e-9),
        "mean_margin": sm / z,
        "mean_positive_prob": sp / z,
        "mean_negative_prob": sn / z,
        "mean_router_confidence": rc / z,
        "mean_value_update": vc / z,
        "mean_context_error": cc / z,
        "mean_signal_norm": sig_norm / z,
        "autograd_grad_tensors": 0,
        "loss_backward_calls": 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs-per-arm", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--progress-every", type=int, default=0)
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
    p.add_argument("--route-target-lr", type=float, default=.006)
    p.add_argument("--trace-decay", type=float, default=.8)
    p.add_argument("--fg-eps", type=float, default=.01)
    p.add_argument("--fg-dirs", type=int, default=4)
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    p.add_argument("--data-dir", default="/content/drive/MyDrive/sparsewalker_data")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_local_rule_screen_ml1m")
    a = p.parse_args()

    unknown = [x for x in a.arms if x not in DEFAULT_ARMS]
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}; choices={DEFAULT_ARMS}")
    assert torch.cuda.is_available(), "GPU runtime required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    base.seed_all(a.seed)

    print("LOCAL_RULE_SCREEN_LOAD_START", flush=True)
    data = load_dataset("ml1m", a.data_dir)
    split = split_data(data["sequences"])
    ds = WindowDataset(split["train"], MAX_LEN, a.seed)
    print("LOCAL_RULE_SCREEN_LOAD_DONE", json.dumps({
        "users": len(data["sequences"]), "n_items": data["n_items"],
        "arms": a.arms, "epochs_per_arm": a.epochs_per_arm,
    }), flush=True)

    out = Path(a.output_dir) / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    all_results = []

    for arm_i, rule in enumerate(a.arms):
        print("RULE_START", json.dumps({
            "rule": rule, "index": arm_i + 1, "total": len(a.arms)
        }), flush=True)
        model = mlbase.build(data["n_items"]).to(device)
        base.init_model(model, a.seed, a.message_gain)
        arm_dir = out / rule
        arm_dir.mkdir(parents=True, exist_ok=True)
        history = []
        best = -1.0
        best_epoch = 0
        best_state = None
        arm_t0 = time.perf_counter()
        error = None
        try:
            for epoch in range(1, a.epochs_per_arm + 1):
                stats = train_epoch_rule(model, ds, device, a, epoch, rule)
                print("RULE_VAL_START", json.dumps({"rule": rule, "epoch": epoch}), flush=True)
                v0 = time.perf_counter()
                val = evaluate_full(
                    model, split["val_prefix"], split["val_target"], data["n_items"],
                    MAX_LEN, device, topks=(10,), batch_size=a.eval_batch_size,
                )
                row = {**stats, **{f"val_{k}": float(v) for k, v in val.items()}}
                history.append(row)
                print("RULE_EPOCH", json.dumps({
                    "rule": rule, "epoch": epoch,
                    "train_s": round(stats["seconds"], 2),
                    "val_s": round(time.perf_counter() - v0, 2),
                    "NDCG@10": float(val["NDCG@10"]),
                    "HR@10": float(val["HR@10"]),
                    "MRR@10": float(val["MRR@10"]),
                    "signal_norm": round(stats["mean_signal_norm"], 4),
                }), flush=True)
                (arm_dir / "history.json").write_text(json.dumps(history, indent=2))
                nd = float(val["NDCG@10"])
                if nd > best:
                    best = nd
                    best_epoch = epoch
                    best_state = base.cpu_state(model)
                    torch.save({
                        "model": best_state, "rule": rule, "epoch": epoch,
                        "val": val, "args": vars(a),
                    }, arm_dir / "best.pt")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print("RULE_ERROR", json.dumps({"rule": rule, "error": error}), flush=True)
            traceback.print_exc()

        result = {
            "rule": rule,
            "best_val_NDCG@10": best if best >= 0 else None,
            "best_epoch": best_epoch,
            "seconds": time.perf_counter() - arm_t0,
            "error": error,
            "history": history,
        }
        all_results.append(result)
        print("RULE_DONE", json.dumps({
            "rule": rule, "best_val_NDCG@10": result["best_val_NDCG@10"],
            "best_epoch": best_epoch, "seconds": round(result["seconds"], 1),
            "error": error,
        }), flush=True)
        (out / "partial_results.json").write_text(json.dumps(all_results, indent=2))

    leaderboard = sorted(
        [r for r in all_results if r["best_val_NDCG@10"] is not None],
        key=lambda r: r["best_val_NDCG@10"], reverse=True,
    )
    final = {
        "experiment": "ML1M-local-rule-diverse-screen-v1",
        "protocol": {
            "same_architecture": "corrected SparseWalker v1.1",
            "same_initialization_seed": a.seed,
            "max_len": MAX_LEN,
            "full_catalog_validation": True,
            "global_backprop": False,
        },
        "args": vars(a),
        "leaderboard": [{
            "rank": i + 1,
            "rule": r["rule"],
            "best_val_NDCG@10": r["best_val_NDCG@10"],
            "best_epoch": r["best_epoch"],
            "seconds": r["seconds"],
        } for i, r in enumerate(leaderboard)],
        "results": all_results,
    }
    (out / "result.json").write_text(json.dumps(final, indent=2))
    print("LOCAL_RULE_SCREEN_RESULT", json.dumps(final["leaderboard"], indent=2), flush=True)


if __name__ == "__main__":
    main()
