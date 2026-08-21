#!/usr/bin/env python
"""Experiment 42: three-factor backward-free SparseWalker.

Forks the successful Experiment 39 local-contrastive Walker. Item contrastive
learning and router/key competitive learning stay unchanged. Direct predictive
updates for concept values, graph context projection, and message projection
are replaced by a shared three-factor rule:

    local pre activity x local target/post activity x (reward - reward_baseline)

A short eligibility trace lets the current prediction error modulate recently
active states. There is no warm start, optimizer, backward(), or autograd
learning.
"""
import argparse, json, math, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_amazon_local_contrastive_walker as base
from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full

AMAZON = base.AMAZON


@torch.no_grad()
def contrastive_with_reward(model, h, pos_ids, neg_ids, lr, temp):
    """Experiment-39 item update plus a per-example local success signal."""
    h = F.normalize(h.float(), dim=-1)
    pos = F.normalize(model.item.weight[pos_ids].float(), dim=-1)
    neg = F.normalize(model.item.weight[neg_ids].float(), dim=-1)
    sp = (h * pos).sum(-1) / temp
    sn = (neg * h[:, None, :]).sum(-1) / temp
    pp = torch.sigmoid(sp)
    pn = torch.sigmoid(sn)
    signal = ((1 - pp)[:, None] * pos - (pn[:, :, None] * neg).mean(1)) / temp

    # Keep the exact analytic item-table update from Experiment 39.
    pd = float(lr) * (1 - pp)[:, None] * h / temp
    nd = -float(lr) * pn[:, :, None] * h[:, None, :] / (temp * neg_ids.size(1))
    touched = torch.cat([pos_ids, neg_ids.reshape(-1)])
    delta = torch.cat([pd, nd.reshape(-1, model.d_model)]).to(model.item.weight.dtype)
    model.item.weight.index_add_(0, touched, delta)
    base.normalize_rows(model.item.weight, touched)
    model.item.weight[0].zero_()

    # Bounded, local prediction success. High when positive is accepted and
    # negatives are rejected; no ranking metric or future information is used.
    reward = pp - pn.mean(-1)
    stats = {
        "margin": float((sp - sn.mean(-1)).mean()),
        "pp": float(pp.mean()),
        "pn": float(pn.mean()),
        "reward": float(reward.mean()),
    }
    return signal, reward, stats


@torch.no_grad()
def target_route_key(model, target):
    context = target * math.sqrt(model.d_model)
    ids, mass = model.router(context, model.space)
    key = model.space.key(ids)
    return F.normalize((key * mass[:, :, None].float()).sum(1), dim=-1)


@torch.no_grad()
def three_factor_values(model, ids, mass, target, mod, lr):
    """Reward-modulated Hebbian/predictive update for active concept values."""
    if ids.numel() == 0:
        return 0.0
    l, r = model.space.split(ids)
    total_change = 0.0
    for idx, table in ((l, model.space.left_value.weight), (r, model.space.right_value.weight)):
        flat = idx.reshape(-1).long()
        tgt = target[:, None, :].expand(-1, idx.size(1), -1).reshape(-1, model.d_model)
        w = (mass.float() * mod[:, None]).reshape(-1)
        num = torch.zeros(model.side, model.d_model, device=table.device, dtype=torch.float32)
        den = torch.zeros(model.side, device=table.device, dtype=torch.float32)
        num.index_add_(0, flat, tgt.float() * w[:, None])
        den.index_add_(0, flat, w.abs())
        rows = (den > 0).nonzero().squeeze(-1)
        if rows.numel():
            direction = num[rows] / den[rows, None].clamp_min(1e-8)
            old = table[rows].float()
            new = F.normalize(old + float(lr) * direction, dim=-1).to(table.dtype)
            total_change += float((new.float() - old).abs().mean())
            table.index_copy_(0, rows, new)
    return total_change


@torch.no_grad()
def three_factor_linear(weight, pre, post, mod, lr, clamp=None):
    """Delta W = eta * sum_i mod_i * post_i outer pre_i."""
    if pre.numel() == 0:
        return 0.0
    pre = F.normalize(pre.float(), dim=-1)
    post = F.normalize(post.float(), dim=-1)
    denom = mod.abs().sum().clamp_min(1.0)
    upd = ((post * mod[:, None]).T @ pre) / denom
    weight.add_(float(lr) * upd.to(weight.dtype))
    if clamp is not None:
        weight.clamp_(-float(clamp), float(clamp))
    return float(upd.abs().mean())


@torch.no_grad()
def make_modulator(reward, state, beta, gain):
    rmean = float(reward.mean())
    if not state["initialized"]:
        state["baseline"] = rmean
        state["initialized"] = True
    raw = reward - float(state["baseline"])
    mod = torch.clamp(float(gain) * raw, -1.0, 1.0)
    state["baseline"] = float(beta) * float(state["baseline"]) + (1.0 - float(beta)) * rmean
    return mod, raw


@torch.no_grad()
def train_epoch(model, ds, device, args, epoch, mod_state):
    model.eval()
    ngen = torch.Generator(device=device)
    ngen.manual_seed(args.seed * 100003 + epoch)

    total = steps = batches = 0
    sm = sp = sn = sr = sabs = smod = spos = sneg = 0.0
    rc = vc = cc = mc = 0.0
    credit_total = credit_delayed = 0
    t0 = time.perf_counter()

    for tokens, lengths in base.loader(ds, args.batch_size, epoch):
        tokens = tokens.to(device, non_blocking=True)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        B, L = x.shape
        ids = torch.zeros(B, model.active, device=device, dtype=torch.long)
        mass = torch.zeros(B, model.active, device=device)
        trace = []

        for t in range(L):
            act = x[:, t].ne(0)
            valid = act & y[:, t].ne(0)
            if not act.any():
                continue

            cur_ids = x[:, t]
            cur = model.item(cur_ids).float()
            context = cur * math.sqrt(model.d_model)
            fi, fm = model.router(context, model.space)
            af = act.float()[:, None]
            xids, xmass = model._merge(ids, mass * af, fi, fm * af)
            for _ in range(model.layers_n):
                xids, xmass = model.graph(xids, xmass, context, model.space, track_touched=False)
            ids = torch.where(act[:, None], xids, ids)
            mass = torch.where(act[:, None], xmass, mass)
            msg_full = (model.space.value(ids) * mass[:, :, None]).sum(1).float()

            # Eligibility contains only local forward activity from this event.
            trace.append({
                "ids": ids.detach().clone(),
                "mass": mass.detach().clone(),
                "cur": cur.detach().clone(),
                "msg": msg_full.detach().clone(),
                "act": act.detach().clone(),
            })
            if len(trace) > args.trace_len:
                trace.pop(0)

            if not valid.any():
                continue

            curv = cur[valid]
            target_ids = y[valid, t]
            target = F.normalize(model.item.weight[target_ids].float(), dim=-1)

            # Keep self-organizing router/key learning exactly as in Experiment 39.
            rc += base.update_router_and_keys(
                model, curv, fi[valid], fm[valid].float(), args.prototype_lr, args.key_lr
            )

            h = model.norm(context[valid] + model.message_proj(msg_full[valid])).float()
            neg = torch.randint(
                1, model.n_items + 1, (int(valid.sum()), args.negatives),
                device=device, generator=ngen,
            )
            signal, reward, d = contrastive_with_reward(
                model, h, target_ids, neg, args.item_lr, args.temperature
            )
            base.update_current_items(model, cur_ids[valid], signal, args.input_lr)

            # Third factor: surprise relative to a slow running reward expectation.
            mod, raw = make_modulator(reward, mod_state, args.reward_beta, args.mod_gain)
            tkey = target_route_key(model, target)

            mod_full = torch.zeros(B, device=device)
            target_full = torch.zeros(B, model.d_model, device=device)
            tkey_full = torch.zeros(B, model.h, device=device)
            mod_full[valid] = mod
            target_full[valid] = target
            tkey_full[valid] = tkey

            # Current outcome modulates the last few locally active states.
            for lag, ent in enumerate(reversed(trace)):
                decay = float(args.trace_decay) ** lag
                use = valid & ent["act"]
                if not use.any():
                    continue
                m = mod_full[use] * decay
                if float(m.abs().sum()) == 0.0:
                    continue
                vc += three_factor_values(
                    model, ent["ids"][use], ent["mass"][use], target_full[use], m, args.value_lr
                )
                cc += three_factor_linear(
                    model.graph.context_q.weight, ent["cur"][use], tkey_full[use], m,
                    args.context_lr, clamp=8.0,
                )
                mc += three_factor_linear(
                    model.message_proj.weight, ent["msg"][use], target_full[use], m,
                    args.message_lr, clamp=12.0,
                )
                credit_total += int(use.sum())
                if lag > 0:
                    credit_delayed += int(use.sum())

            sm += d["margin"]
            sp += d["pp"]
            sn += d["pn"]
            sr += d["reward"]
            sabs += float(raw.abs().mean())
            smod += float(mod.abs().mean())
            spos += float((mod > 0).float().mean())
            sneg += float((mod < 0).float().mean())
            steps += 1
            total += int(valid.sum())
        batches += 1

    if device.type == "cuda":
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
        "mean_contrastive_margin": sm / z,
        "mean_positive_prob": sp / z,
        "mean_negative_prob": sn / z,
        "mean_reward": sr / z,
        "reward_baseline": float(mod_state["baseline"]),
        "mean_abs_prediction_error": sabs / z,
        "mean_abs_modulator": smod / z,
        "positive_modulator_fraction": spos / z,
        "negative_modulator_fraction": sneg / z,
        "mean_router_confidence": rc / z,
        "mean_three_factor_value_change": vc / z,
        "mean_three_factor_context_update": cc / z,
        "mean_three_factor_message_update": mc / z,
        "eligibility_delayed_credit_fraction": credit_delayed / max(1, credit_total),
        "trace_len": int(args.trace_len),
        "trace_decay": float(args.trace_decay),
        "autograd_grad_tensors": 0,
        "loss_backward_calls": 0,
        "optimizer": None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=AMAZON, default="beauty")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
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
    p.add_argument("--reward-beta", type=float, default=.99)
    p.add_argument("--mod-gain", type=float, default=4.0)
    p.add_argument("--trace-len", type=int, default=4)
    p.add_argument("--trace-decay", type=float, default=.6)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_three_factor")
    a = p.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    base.seed_all(a.seed)

    data = load_dataset(a.dataset, a.data_dir)
    split = split_data(data["sequences"])
    model = base.build(data["n_items"]).to(device)
    base.init_model(model, a.seed, a.message_gain)
    init_val, init_test = base.evaluate(model, split, data["n_items"], device, a.eval_batch_size)
    print("TF_INIT", json.dumps({"val": init_val, "test": init_test}), flush=True)

    ds = WindowDataset(split["train"], 50, a.seed)
    out = Path(a.output_dir) / a.dataset / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "experiment": "ThreeFactorLocalContrastiveWalker-v1",
        "base": "Experiment39 exact item contrastive + router/key competitive learning",
        "three_factor_modules": ["concept_values", "graph_context_q", "message_proj"],
        "rule": "pre x target/post x centered local prediction reward",
        "eligibility": {"length": a.trace_len, "decay": a.trace_decay},
        "warm_start": False,
        "autograd": False,
        "optimizer": None,
        "loss_backward_calls": 0,
        "reference_val_NDCG@10": {"LC_v1": 0.040515991131704246, "SASRec": 0.04296780165590764},
        "args": vars(a),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    best = float(init_val["NDCG@10"])
    best_epoch = 0
    best_state = base.cpu_state(model)
    hist = []
    start = 1
    mod_state = {"baseline": 0.0, "initialized": False}
    last = out / "last.pt"

    if a.resume and last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"])
        best = float(ck["best"])
        best_epoch = int(ck["best_epoch"])
        best_state = ck["best_state"]
        hist = ck.get("history", [])
        mod_state = ck.get("mod_state", mod_state)
        start = int(ck["epoch"]) + 1
        print("TF_RESUME", json.dumps({"from_epoch": start - 1, "best_epoch": best_epoch, "best": best}), flush=True)

    for e in range(start, a.epochs + 1):
        s = train_epoch(model, ds, device, a, e, mod_state)
        val = evaluate_full(
            model, split["val_prefix"], split["val_target"], data["n_items"], 50,
            device, topks=(10,), batch_size=a.eval_batch_size,
        )
        row = {**s, **{f"val_{k}": float(v) for k, v in val.items()}}
        hist.append(row)
        print("TF_EPOCH", json.dumps(row), flush=True)
        (out / "history.json").write_text(json.dumps(hist, indent=2))
        nd = float(val["NDCG@10"])
        if nd > best:
            best = nd
            best_epoch = e
            best_state = base.cpu_state(model)
            torch.save({"model": best_state, "epoch": e, "val": val, "config": cfg, "mod_state": mod_state}, out / "best.pt")
        torch.save({
            "model": base.cpu_state(model), "epoch": e, "best": best, "best_epoch": best_epoch,
            "best_state": best_state, "history": hist, "config": cfg, "mod_state": mod_state,
        }, last)

    model.load_state_dict(best_state)
    bv, bt = base.evaluate(model, split, data["n_items"], device, a.eval_batch_size)
    result = {
        "config": cfg,
        "initial": {"val": init_val, "test": init_test},
        "best_epoch": best_epoch,
        "best_three_factor": {"val": bv, "test": bt},
        "vs_lc_v1_val_ratio": float(bv["NDCG@10"] / 0.040515991131704246),
        "vs_sasrec_val_ratio": float(bv["NDCG@10"] / 0.04296780165590764),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("TF_RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
