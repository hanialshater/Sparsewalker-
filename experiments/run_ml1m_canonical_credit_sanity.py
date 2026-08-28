#!/usr/bin/env python
"""Experiment 46: canonical SparseWalker credit-radius sanity ladder.

This repairs the confounds in Experiment 45. Every arm uses the canonical
SparseWalker v1.1 default initialization, FullCE, AdamW, BF16, batch=128,
length-bucketed windows, and the canonical 50-epoch LR schedule. The diagnostic
run can stop earlier (default 15 epochs) without prematurely decaying the LR.

Arms:
  canonical_full : exact existing sparsewalker.training.train_epoch path.
  event_local    : detach recurrent mass after every event; gradient still flows
                   through current router + two graph hops + readout.
  tbptt4         : detach recurrent mass every four events.

All arms take exactly one backward() and one optimizer.step() per batch. Test is
not touched; selection uses full-catalog validation NDCG@10 with seen masking.
"""
import argparse, json, math, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full
from sparsewalker.models import SparseWalker
from sparsewalker.training import train_epoch

import run_ml1m_local_contrastive_walker as fast
import run_ml1m_walker_v11 as canon

MAX_LEN = 200
ARMS = ("canonical_full", "event_local", "tbptt4")
RADIUS = {"event_local": 1, "tbptt4": 4}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def build(n_items):
    return SparseWalker(
        n_items, MAX_LEN, d=64, layers=2, side=256, h=16,
        active=8, top_side=2, degree=4, fresh_weight=.25,
    )


def one_event(model, ids, mass, item_ids, active):
    af = active.to(model.item.weight.dtype)[:, None]
    context = model.item(item_ids) * math.sqrt(model.d_model)
    fi, fm = model.router(context, model.space)
    xids, xmass = model._merge(ids, mass * af, fi, fm * af)
    for _ in range(model.layers_n):
        xids, xmass = model.graph(xids, xmass, context, model.space, track_touched=False)
    ids = torch.where(active[:, None], xids, ids)
    mass = torch.where(active[:, None], xmass, mass)
    msg = (model.space.value(ids) * mass[:, :, None]).sum(1)
    h = model.norm(context + model.message_proj(msg)) * af
    return ids, mass, h


def train_epoch_detached(model, ds, opt, device, batch_size, epoch, radius, grad_clip=5.0):
    """Canonical FullCE, but detach recurrent mass at fixed temporal radius.

    Losses from all positions are summed into one scalar, so there is exactly one
    backward call and one optimizer step per batch, matching canonical training.
    """
    loader = fast._loader(ds, batch_size, epoch, True)
    model.train()
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    positions = padded = batches = backward_calls = 0
    loss_total = 0.0
    t0 = time.perf_counter()

    for tokens, lengths in loader:
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
        opt.zero_grad(set_to_none=True)
        batch_loss = None
        since_detach = 0

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            for t in range(L):
                active = x[:, t].ne(0)
                ids, mass, h = one_event(model, ids, mass, x[:, t], active)
                valid = active & y[:, t].ne(0)
                if valid.any():
                    piece = F.cross_entropy(
                        model.score_hidden(h[valid]), y[valid, t], reduction="sum"
                    ) / total_valid
                    batch_loss = piece if batch_loss is None else batch_loss + piece
                since_detach += 1
                if since_detach >= radius:
                    mass = mass.detach()
                    since_detach = 0

        if batch_loss is None:
            continue
        batch_loss.backward()
        backward_calls += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        loss_total += float(batch_loss.detach())
        batches += 1

    if device.type == "cuda": torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    return {
        "loss": loss_total / max(1, batches),
        "batches": batches,
        "positions": positions,
        "seconds": sec,
        "positions_per_s": positions / max(sec, 1e-9),
        "padding_efficiency": positions / max(1, padded),
        "backward_calls": backward_calls,
    }


@torch.inference_mode()
def val_eval(model, split, n_items, device, batch):
    model.eval()
    return evaluate_full(
        model, split["val_prefix"], split["val_target"], n_items,
        MAX_LEN, device, topks=(10,), batch_size=batch,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--schedule-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--peak-lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--data-dir", default="/content/drive/MyDrive/sparsewalker_data")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_canonical_credit_sanity")
    a = p.parse_args()

    assert torch.cuda.is_available(), "GPU runtime required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed_all(a.seed)

    print("SANITY_DATA_START", flush=True)
    t = time.perf_counter()
    data = load_dataset("ml1m", a.data_dir)
    split = split_data(data["sequences"])
    ds = WindowDataset(split["train"], MAX_LEN, a.seed)
    print("SANITY_DATA_DONE", json.dumps({
        "seconds": round(time.perf_counter()-t, 2),
        "users": len(data["sequences"]), "n_items": data["n_items"]
    }), flush=True)

    # Canonical default initialization: no Experiment-43 init_model override.
    seed_all(a.seed)
    template = build(data["n_items"]).to(device)
    init_state = cpu_state(template)
    del template
    torch.cuda.empty_cache()

    out = Path(a.output_dir) / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "experiment": "ML1M-CanonicalCreditSanity-v1",
        "arms": a.arms,
        "default_canonical_initialization": True,
        "objective": "FullCE",
        "optimizer": "AdamW",
        "bf16": True,
        "one_backward_and_step_per_batch": True,
        "schedule_epochs": a.schedule_epochs,
        "diagnostic_epochs": a.epochs,
        "test_evaluation_during_screen": False,
        "args": vars(a),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    ranking = []
    for arm in a.arms:
        print("\nSANITY_ARM_START", json.dumps({"arm": arm}), flush=True)
        seed_all(a.seed)
        model = build(data["n_items"]).to(device)
        model.load_state_dict(init_state)
        opt = torch.optim.AdamW(model.parameters(), lr=a.peak_lr, weight_decay=a.weight_decay)
        hist, best, best_epoch = [], -1.0, 0

        for epoch in range(1, a.epochs + 1):
            lr = canon.set_lr(
                opt, epoch, a.schedule_epochs,
                peak=a.peak_lr, min_lr=a.min_lr, warmup=3,
            )
            if arm == "canonical_full":
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                s = train_epoch(
                    "SparseWalker", model, ds, opt, device,
                    batch_size=a.batch_size, epoch=epoch, loss_mode="full",
                    bucket_by_length=True, use_bf16=True, return_stats=True,
                    grad_clip=a.grad_clip,
                )
                if device.type == "cuda": torch.cuda.synchronize()
                sec = time.perf_counter() - t0
                stats = {
                    "loss": float(s["loss"]), "batches": int(s["batches"]),
                    "positions": int(s["positions"]), "seconds": sec,
                    "positions_per_s": float(s["positions"]) / max(sec, 1e-9),
                    "padding_efficiency": float(s["padding_efficiency"]),
                    "backward_calls": int(s["batches"]),
                }
            else:
                stats = train_epoch_detached(
                    model, ds, opt, device, a.batch_size, epoch,
                    RADIUS[arm], a.grad_clip,
                )

            row = {"arm": arm, "epoch": epoch, "lr": lr, **stats}
            if epoch == 1 or epoch % a.eval_every == 0:
                vt = time.perf_counter()
                val = val_eval(model, split, data["n_items"], device, a.eval_batch_size)
                row.update({f"val_{k}": float(v) for k, v in val.items()})
                row["val_seconds"] = time.perf_counter() - vt
                nd = float(val["NDCG@10"])
                if nd > best:
                    best, best_epoch = nd, epoch
                    torch.save({
                        "model": cpu_state(model), "arm": arm,
                        "epoch": epoch, "val": val, "config": cfg,
                    }, out / f"best_{arm}.pt")
            hist.append(row)
            print("SANITY_EPOCH", json.dumps(row), flush=True)
            (out / f"history_{arm}.json").write_text(json.dumps(hist, indent=2))

        result = {
            "arm": arm,
            "credit_radius": 200 if arm == "canonical_full" else RADIUS[arm],
            "best_epoch": best_epoch,
            "best_val_NDCG@10": best,
            "final_val_NDCG@10": float(hist[-1].get("val_NDCG@10", float("nan"))),
            "mean_positions_per_s": float(np.mean([r["positions_per_s"] for r in hist])),
            "mean_backward_calls": float(np.mean([r["backward_calls"] for r in hist])),
        }
        ranking.append(result)
        print("SANITY_ARM_DONE", json.dumps(result), flush=True)
        del model, opt
        torch.cuda.empty_cache()

    ranking.sort(key=lambda r: r["best_val_NDCG@10"], reverse=True)
    full = next((r for r in ranking if r["arm"] == "canonical_full"), None)
    control_valid = None if full is None else bool(full["best_val_NDCG@10"] >= 0.08)
    summary = {
        "config": cfg,
        "control_valid_min_0.08": control_valid,
        "ranking": ranking,
        "references": {
            "analytic_backward_free_best_val_NDCG@10": 0.09352186187904432,
            "SASRec_public_protocol_reference_NDCG@10": 0.15965,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("\nSANITY_RESULT", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
