#!/usr/bin/env python
"""Experiment 44: ML-1M local-credit tournament.

Screen several genuinely different local-learning families on the same
SparseWalker initialization and the same ML-1M protocol. This is deliberately a
short-horizon exploration, not a tuned benchmark.

Arms
----
analytic_lc
    Experiment-43 strict backward-free local-contrastive rule.
local_ce
    Exact LOCAL sampled-CE gradient with the recurrent state detached. No
    gradient crosses time or a graph hop; reverse AD exists only inside the
    current readout block.
ff_goodness
    Forward-Forward-inspired positive/negative goodness critic. Recurrent state
    is detached; the local block and critic receive only a local gradient.
target_denoise
    NoProp-inspired target broadcast: a noisy frozen next-item target embedding
    is locally denoised from the current state. No cross-block gradient.
forward_gradient
    Strict forward-only activation directional derivative of the actual sampled
    ranking objective. No autograd/backward; updates are analytic from the
    estimated dL/dh.
vector_eligibility
    e-prop-inspired vector learning signal times a decaying local message
    eligibility trace. No autograd/backward.

All non-baseline arms retain the same simple competitive router/key and local
context update so that the tournament mainly tests the predictive teaching
signal. Validation is full-catalog with seen-item masking; test is intentionally
not touched during screening.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_amazon_local_contrastive_walker as base
import run_ml1m_local_contrastive_walker as fast
from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full

MAX_LEN = 200
ARMS = (
    "analytic_lc",
    "local_ce",
    "ff_goodness",
    "target_denoise",
    "forward_gradient",
    "vector_eligibility",
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


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


@torch.no_grad()
def advance_state(model, ids, mass, current_ids):
    """One corrected SparseWalker event with state transition detached."""
    cur = model.item(current_ids).float()
    context = cur * math.sqrt(model.d_model)
    fi, fm = model.router(context, model.space)
    xids, xmass = model._merge(ids, mass, fi, fm)
    for _ in range(model.layers_n):
        xids, xmass = model.graph(
            xids, xmass, context, model.space, track_touched=False
        )
    return xids, xmass, fi, fm, cur, context


@torch.no_grad()
def shared_router_teacher(model, cur, fi, fm, target_ids, args):
    """Common local router/key/context teacher used by screening arms."""
    rc = base.update_router_and_keys(
        model, cur, fi, fm.float(), args.prototype_lr, args.key_lr
    )
    target_pre = F.normalize(model.item.weight[target_ids].float(), dim=-1)
    tr, tm = model.router(target_pre * math.sqrt(model.d_model), model.space)
    cc = base.update_context(model, cur, tr, tm.float(), args.context_lr)
    return rc, cc, target_pre


def predictive_params(model):
    return [
        model.item.weight,
        model.space.left_value.weight,
        model.space.right_value.weight,
        model.space.value_proj.weight,
        model.space.value_proj.bias,
        model.message_proj.weight,
        model.norm.weight,
        model.norm.bias,
    ]


def set_requires(params, flag):
    for p in params:
        p.requires_grad_(flag)
        p.grad = None


def local_hidden(model, cur_ids, state_ids, state_mass):
    cur = model.item(cur_ids).float()
    context = cur * math.sqrt(model.d_model)
    msg = (model.space.value(state_ids) * state_mass[:, :, None]).sum(1).float()
    h = model.norm(context + model.message_proj(msg)).float()
    return h, msg, cur


def sample_negs(model, n, k, device, gen):
    return torch.randint(1, model.n_items + 1, (n, k), device=device, generator=gen)


def candidate_ce_loss(model, h, target_ids, neg_ids, reduction="mean"):
    pos = (h * model.item.weight[target_ids].float()).sum(-1, keepdim=True)
    neg = model.item.weight[neg_ids].float()
    nlog = torch.bmm(neg, h.unsqueeze(-1)).squeeze(-1)
    logits = torch.cat([pos, nlog], dim=1)
    labels = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
    return F.cross_entropy(logits, labels, reduction=reduction)


class FFCritic(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(2 * d, d, bias=False)

    def goodness(self, h, item):
        z = F.relu(self.proj(torch.cat([h, item], dim=-1)))
        return z.square().mean(-1)


class TargetDenoiser(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

    def forward(self, h, noisy_target):
        return self.net(torch.cat([h, noisy_target], dim=-1))


@torch.no_grad()
def sgd_apply(params, grads, lr, clip=5.0):
    sq = 0.0
    for g in grads:
        if g is not None:
            sq += float(g.float().square().sum())
    norm = math.sqrt(max(sq, 1e-20))
    scale = min(1.0, float(clip) / max(norm, 1e-12))
    for p, g in zip(params, grads):
        if g is not None:
            p.add_(g, alpha=-float(lr) * scale)
    return norm


@torch.no_grad()
def normalize_touched_items(model, cur_ids, target_ids, neg_ids=None):
    pieces = [cur_ids.reshape(-1), target_ids.reshape(-1)]
    if neg_ids is not None:
        pieces.append(neg_ids.reshape(-1))
    rows = torch.unique(torch.cat(pieces))
    base.normalize_rows(model.item.weight, rows)
    model.item.weight[0].zero_()


def local_grad_step(model, method, aux, target_table, cur_ids, state_ids,
                    state_mass, target_ids, neg_ids, args):
    params = predictive_params(model)
    aux_params = list(aux.parameters()) if aux is not None else []
    set_requires(params, True)
    set_requires(aux_params, True)

    h, msg, cur = local_hidden(model, cur_ids, state_ids, state_mass)
    if method == "local_ce":
        loss = candidate_ce_loss(model, h, target_ids, neg_ids)
    elif method == "ff_goodness":
        pos = model.item.weight[target_ids].float()
        neg = model.item.weight[neg_ids].float()
        gp = aux.goodness(h, pos)
        hh = h[:, None, :].expand(-1, neg.size(1), -1).reshape(-1, h.size(-1))
        gn = aux.goodness(hh, neg.reshape(-1, neg.size(-1))).view(h.size(0), -1)
        # Positive goodness high, negative goodness low. Logistic threshold 1.
        loss = F.softplus(1.0 - gp).mean() + F.softplus(gn - 1.0).mean()
    elif method == "target_denoise":
        target = F.normalize(target_table[target_ids].float(), dim=-1)
        noise = torch.randn_like(target)
        alpha = float(args.denoise_alpha)
        noisy = math.sqrt(alpha) * target + math.sqrt(1.0 - alpha) * noise
        pred = F.normalize(aux(h, noisy), dim=-1)
        # The direct term prevents the denoiser from simply ignoring h.
        loss = (1.0 - (pred * target).sum(-1)).mean()
        loss = loss + float(args.denoise_direct) * (
            1.0 - (F.normalize(h, dim=-1) * target).sum(-1)
        ).mean()
    else:
        raise ValueError(method)

    all_params = params + aux_params
    grads = torch.autograd.grad(loss, all_params, allow_unused=True)
    grad_norm = sgd_apply(all_params, grads, args.local_grad_lr, args.local_grad_clip)
    with torch.no_grad():
        normalize_touched_items(model, cur_ids, target_ids, neg_ids)
    set_requires(params, False)
    set_requires(aux_params, False)
    return float(loss.detach()), grad_norm


@torch.no_grad()
def forward_gradient_step(model, h, msg, cur_ids, state_ids, state_mass,
                          target_ids, neg_ids, gen, args):
    """Forward directional derivative in activation space; no reverse AD."""
    g = torch.zeros_like(h)
    eps = float(args.fg_eps)
    dirs = int(args.fg_dirs)
    pos = model.item.weight[target_ids].float()
    neg = model.item.weight[neg_ids].float()
    labels = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

    def per_example_loss(hh):
        lp = (hh * pos).sum(-1, keepdim=True)
        ln = torch.bmm(neg, hh.unsqueeze(-1)).squeeze(-1)
        return F.cross_entropy(torch.cat([lp, ln], 1), labels, reduction="none")

    for _ in range(dirs):
        v = torch.randint(0, 2, h.shape, device=h.device, generator=gen).float()
        v = v.mul_(2).sub_(1)
        deriv = (per_example_loss(h + eps * v) - per_example_loss(h - eps * v)) / (2 * eps)
        g.add_(deriv[:, None] * v, alpha=1.0 / dirs)

    # h = norm(context + W msg). We deliberately use a simple local straight-
    # through Jacobian here; the purpose is to test whether true-objective
    # forward derivatives are a better teacher than the handcrafted signal.
    model.message_proj.weight.add_(
        g.T @ msg.float() / max(1, h.size(0)), alpha=-float(args.fg_message_lr)
    )
    model.message_proj.weight.clamp_(-12, 12)
    model.item.weight.index_add_(
        0, cur_ids, (-float(args.fg_input_lr) * g).to(model.item.weight.dtype)
    )
    normalize_touched_items(model, cur_ids, target_ids)

    target_now = F.normalize(model.item.weight[target_ids].float(), dim=-1)
    base.update_values(model, state_ids, state_mass.float(), target_now, args.value_lr)
    return float(g.abs().mean())


@torch.no_grad()
def vector_eligibility_step(model, msg, trace, rows, signal, args):
    tr = trace[rows]
    tr.mul_(float(args.elig_decay)).add_(msg)
    trace.index_copy_(0, rows, tr)
    model.message_proj.weight.add_(
        signal.T @ tr.float() / max(1, signal.size(0)), alpha=float(args.message_lr)
    )
    model.message_proj.weight.clamp_(-12, 12)



def train_epoch_arm(model, arm, aux, target_table, ds, device, args, epoch):
    if arm == "analytic_lc":
        return fast.train_epoch_fast(model, ds, device, args, epoch)

    model.eval()
    loader = fast._loader(ds, args.batch_size, epoch, True)
    ngen = torch.Generator(device=device)
    ngen.manual_seed(args.seed * 100003 + epoch)
    total = 0
    local_steps = 0
    local_metric = 0.0
    router_metric = 0.0
    context_metric = 0.0
    t0 = time.perf_counter()

    for bi, (tokens, lengths) in enumerate(loader, start=1):
        tokens = tokens.to(device, non_blocking=True)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        B, L = x.shape
        ids = torch.zeros(B, model.active, device=device, dtype=torch.long)
        mass = torch.zeros(B, model.active, device=device)
        trace = torch.zeros(B, model.d_model, device=device) if arm == "vector_eligibility" else None

        for t in range(L):
            rows = x[:, t].ne(0).nonzero(as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                continue
            cur_ids = x[rows, t]
            xids, xmass, fi, fm, cur0, context0 = advance_state(
                model, ids[rows], mass[rows], cur_ids
            )
            ids.index_copy_(0, rows, xids)
            mass.index_copy_(0, rows, xmass)
            valid = y[rows, t].ne(0)
            if not valid.any():
                continue

            vr = rows[valid]
            ci = cur_ids[valid]
            target_ids = y[vr, t]
            sid = ids[vr]
            smass = mass[vr].float()
            fi_v = fi[valid]
            fm_v = fm[valid]
            cur = cur0[valid]

            rc, cc, target_pre = shared_router_teacher(
                model, cur, fi_v, fm_v, target_ids, args
            )
            router_metric += rc
            context_metric += cc

            neg = sample_negs(model, int(vr.numel()), args.negatives, device, ngen)

            if arm in ("local_ce", "ff_goodness", "target_denoise"):
                lm, _ = local_grad_step(
                    model, arm, aux, target_table, ci, sid, smass,
                    target_ids, neg, args
                )
                local_metric += lm

            elif arm == "forward_gradient":
                with torch.no_grad():
                    h, msg, _ = local_hidden(model, ci, sid, smass)
                    lm = forward_gradient_step(
                        model, h, msg, ci, sid, smass, target_ids,
                        neg, ngen, args
                    )
                local_metric += lm

            elif arm == "vector_eligibility":
                with torch.no_grad():
                    h, msg, _ = local_hidden(model, ci, sid, smass)
                    signal, diag = base.contrastive(
                        model, h, target_ids, neg, args.item_lr, args.temperature
                    )
                    base.update_current_items(model, ci, signal, args.input_lr)
                    target_now = F.normalize(model.item.weight[target_ids].float(), dim=-1)
                    base.update_values(model, sid, smass, target_now, args.value_lr)
                    vector_eligibility_step(model, msg, trace, vr, signal, args)
                    lm = diag["margin"]
                local_metric += lm
            else:
                raise ValueError(arm)

            total += int(vr.numel())
            local_steps += 1

        if bi == 1 or bi == len(loader):
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(
                "TOURNAMENT_PROGRESS",
                json.dumps({
                    "arm": arm,
                    "epoch": epoch,
                    "batch": bi,
                    "batches": len(loader),
                    "positions": total,
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                }),
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    z = max(1, local_steps)
    return {
        "epoch": epoch,
        "positions": total,
        "seconds": sec,
        "positions_per_s": total / max(sec, 1e-9),
        "local_metric": local_metric / z,
        "router_metric": router_metric / z,
        "context_metric": context_metric / z,
        "global_backward_calls": 0,
        "local_autograd": arm in ("local_ce", "ff_goodness", "target_denoise"),
        "strict_forward_only": arm in ("forward_gradient", "vector_eligibility"),
    }


def make_aux(arm, d, device, seed):
    seed_all(seed)
    if arm == "ff_goodness":
        aux = FFCritic(d).to(device)
    elif arm == "target_denoise":
        aux = TargetDenoiser(d).to(device)
    else:
        return None
    set_requires(list(aux.parameters()), False)
    return aux


def make_fast_args(a):
    # fast.train_epoch_fast expects the Experiment-43 argument names.
    return SimpleNamespace(**vars(a))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs-per-arm", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--negatives", type=int, default=32)
    p.add_argument("--temperature", type=float, default=.15)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--data-dir", default="/content/drive/MyDrive/sparsewalker_data")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_local_credit_tournament")

    # Shared Experiment-39 local rules.
    p.add_argument("--item-lr", type=float, default=.02)
    p.add_argument("--input-lr", type=float, default=.004)
    p.add_argument("--prototype-lr", type=float, default=.025)
    p.add_argument("--key-lr", type=float, default=.015)
    p.add_argument("--value-lr", type=float, default=.035)
    p.add_argument("--context-lr", type=float, default=.006)
    p.add_argument("--message-lr", type=float, default=.0008)
    p.add_argument("--message-gain", type=float, default=8.0)

    # Local-gradient controls.
    p.add_argument("--local-grad-lr", type=float, default=.003)
    p.add_argument("--local-grad-clip", type=float, default=5.0)
    p.add_argument("--denoise-alpha", type=float, default=.15)
    p.add_argument("--denoise-direct", type=float, default=.25)

    # Activation forward-gradient.
    p.add_argument("--fg-eps", type=float, default=.03)
    p.add_argument("--fg-dirs", type=int, default=4)
    p.add_argument("--fg-message-lr", type=float, default=.0005)
    p.add_argument("--fg-input-lr", type=float, default=.001)

    # Eligibility trace.
    p.add_argument("--elig-decay", type=float, default=.7)

    # Compatibility fields consumed by fast.train_epoch_fast.
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--no-length-bucketing", action="store_true")
    p.add_argument("--eval-every", type=int, default=1)
    a = p.parse_args()

    assert torch.cuda.is_available(), "GPU runtime required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed_all(a.seed)

    print("TOURNAMENT_DATA_START", flush=True)
    t = time.perf_counter()
    data = load_dataset("ml1m", a.data_dir)
    split = split_data(data["sequences"])
    ds = WindowDataset(split["train"], MAX_LEN, a.seed)
    print(
        "TOURNAMENT_DATA_DONE",
        json.dumps({
            "seconds": round(time.perf_counter() - t, 2),
            "users": len(data["sequences"]),
            "n_items": data["n_items"],
        }),
        flush=True,
    )

    template = fast.build(data["n_items"]).to(device)
    base.init_model(template, a.seed, a.message_gain)
    init_state = cpu_state(template)
    frozen_target_table = init_state["item.weight"].to(device)
    del template

    out = Path(a.output_dir) / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "experiment": "ML1M-LocalCreditTournament-v1",
        "arms": a.arms,
        "epochs_per_arm": a.epochs_per_arm,
        "same_model_initialization": True,
        "test_evaluation_during_screen": False,
        "protocol": "ML1M leave-two-out, max_len=200, full-catalog val, seen masking",
        "args": vars(a),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    results = []
    for ai, arm in enumerate(a.arms):
        print("\nTOURNAMENT_ARM_START", json.dumps({"arm": arm}), flush=True)
        seed_all(a.seed)
        model = fast.build(data["n_items"]).to(device)
        model.load_state_dict(init_state)
        model.to(device)
        for p0 in model.parameters():
            p0.requires_grad_(False)
            p0.grad = None
        aux = make_aux(arm, model.d_model, device, a.seed + 1000 + ai)
        arm_hist = []
        best = -1.0
        best_epoch = 0
        best_state = None

        for epoch in range(1, a.epochs_per_arm + 1):
            stats = train_epoch_arm(
                model, arm, aux, frozen_target_table, ds, device,
                make_fast_args(a), epoch
            )
            vt = time.perf_counter()
            val = val_eval(model, split, data["n_items"], device, a.eval_batch_size)
            row = {
                "arm": arm,
                **stats,
                **{f"val_{k}": float(v) for k, v in val.items()},
                "val_seconds": time.perf_counter() - vt,
            }
            arm_hist.append(row)
            print("TOURNAMENT_EPOCH", json.dumps(row), flush=True)
            if float(val["NDCG@10"]) > best:
                best = float(val["NDCG@10"])
                best_epoch = epoch
                best_state = cpu_state(model)

        torch.save(
            {"model": best_state, "arm": arm, "epoch": best_epoch, "best_val_NDCG@10": best},
            out / f"best_{arm}.pt",
        )
        (out / f"history_{arm}.json").write_text(json.dumps(arm_hist, indent=2))
        result = {
            "arm": arm,
            "best_epoch": best_epoch,
            "best_val_NDCG@10": best,
            "final_val_NDCG@10": float(arm_hist[-1]["val_NDCG@10"]),
            "local_autograd": bool(arm_hist[-1].get("local_autograd", False)),
            "strict_forward_only": bool(arm_hist[-1].get("strict_forward_only", arm == "analytic_lc")),
            "mean_positions_per_s": float(np.mean([r["positions_per_s"] for r in arm_hist])),
        }
        results.append(result)
        print("TOURNAMENT_ARM_DONE", json.dumps(result), flush=True)
        del model, aux
        torch.cuda.empty_cache()

    results.sort(key=lambda r: r["best_val_NDCG@10"], reverse=True)
    summary = {
        "config": cfg,
        "ranking": results,
        "reference": {
            "experiment43_best_seen_val_NDCG@10": 0.09352186187904432,
            "SASRec_public_protocol_reference_NDCG@10": 0.15965,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("\nTOURNAMENT_RESULT", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
