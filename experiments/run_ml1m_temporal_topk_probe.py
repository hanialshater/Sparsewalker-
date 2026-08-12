#!/usr/bin/env python
"""Probe how sparse the successful Walker+Attention temporal read already is.

Loads the best checkpoint from the dense-attention ML-1M control and evaluates the
*same weights* with the attention read restricted to exact top-K historical keys.

This is deliberately an oracle sparsity test, not yet a speedup: QK scores are
still computed densely so we can isolate whether only a small number of past
states are needed. If top-K preserves quality, a later SWG/HNSW navigation layer
has a concrete sparse target to approximate without changing the representation.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_ml1m_walker_attention_control import (
    SparseWalkerAttention,
    protocol_manifest,
    seed_all,
)
from sparsewalker.data import load_dataset, split_data
from sparsewalker.evaluation import evaluate_full


def _manual_attention(block, h, pad_mask, topk=None, collect_stats=False):
    """Reproduce block.attn at eval, optionally retaining only exact top-K keys.

    Uses the trained nn.MultiheadAttention projection weights directly so dense
    mode should match the original block to numerical tolerance.
    """
    B, L, D = h.shape
    H = block.attn.num_heads
    HD = D // H

    pos_ids = torch.arange(L, device=h.device)
    x = h + block.pos(pos_ids)[None, :, :]
    qkv_in = block.norm1(x)

    proj = F.linear(qkv_in, block.attn.in_proj_weight, block.attn.in_proj_bias)
    q, k, v = proj.chunk(3, dim=-1)
    q = q.view(B, L, H, HD).transpose(1, 2)  # B,H,L,HD
    k = k.view(B, L, H, HD).transpose(1, 2)
    v = v.view(B, L, H, HD).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HD)

    # causal: key j must satisfy j <= query i
    causal = torch.ones(L, L, dtype=torch.bool, device=h.device).triu(1)
    scores = scores.masked_fill(causal[None, None, :, :], -torch.inf)
    scores = scores.masked_fill(pad_mask[:, None, None, :], -torch.inf)

    valid = torch.isfinite(scores)
    # Padded query rows will have no semantic output anyway; avoid NaNs.
    padded_q = pad_mask[:, None, :, None]

    dense_prob = torch.softmax(scores.masked_fill(~valid, -1e30), dim=-1)
    dense_prob = dense_prob.masked_fill(~valid, 0.0)
    dense_prob = dense_prob.masked_fill(padded_q, 0.0)

    retained_mass = None
    if topk is None or int(topk) >= L:
        prob = dense_prob
        if collect_stats:
            retained_mass = torch.ones(B, H, L, device=h.device)
    else:
        kk = min(int(topk), L)
        topv, topi = scores.topk(kk, dim=-1)
        top_valid = torch.isfinite(topv)
        top_prob = torch.softmax(topv.masked_fill(~top_valid, -1e30), dim=-1)
        top_prob = top_prob.masked_fill(~top_valid, 0.0)
        top_prob = top_prob.masked_fill(pad_mask[:, None, :, None], 0.0)

        prob = torch.zeros_like(scores)
        prob.scatter_(-1, topi, top_prob)

        if collect_stats:
            retained_mass = dense_prob.gather(-1, topi).sum(-1)

    ctx = torch.matmul(prob, v)
    ctx = ctx.transpose(1, 2).contiguous().view(B, L, D)
    a = F.linear(ctx, block.attn.out_proj.weight, block.attn.out_proj.bias)

    z = h + a
    z = z + block.ffn(block.norm2(z))
    z = z.masked_fill(pad_mask[..., None], 0.0)

    stats = None
    if collect_stats:
        valid_q = ~pad_mask[:, None, :]
        mass = retained_mass[valid_q.expand_as(retained_mass)]
        # Attention entropy in the original dense distribution.
        p = dense_prob.clamp_min(1e-12)
        entropy = -(p * p.log()).sum(-1)
        ent = entropy[valid_q.expand_as(entropy)]
        stats = {
            "mean_dense_entropy": float(ent.mean().detach().cpu()) if ent.numel() else 0.0,
            "mean_retained_dense_mass": float(mass.mean().detach().cpu()) if mass.numel() else 0.0,
            "p10_retained_dense_mass": float(torch.quantile(mass.float(), .10).detach().cpu()) if mass.numel() else 0.0,
            "p50_retained_dense_mass": float(torch.quantile(mass.float(), .50).detach().cpu()) if mass.numel() else 0.0,
            "p90_retained_dense_mass": float(torch.quantile(mass.float(), .90).detach().cpu()) if mass.numel() else 0.0,
        }
    return z, stats


class TopKMode:
    def __init__(self, model):
        self.model = model
        self.topk = None
        self.orig_encode = model.encode

    def install(self, topk):
        self.topk = topk
        model = self.model

        def encode(seq):
            h = super(type(model), model).encode(seq)
            if not model.attention_enabled:
                return h
            z, _ = _manual_attention(model.temporal, h, seq.eq(0), topk=self.topk)
            return z

        model.encode = encode

    def restore(self):
        self.model.encode = self.orig_encode


@torch.inference_mode()
def manual_dense_equivalence(model, device):
    model.eval()
    x = torch.tensor([
        [11, 17, 23, 31, 41, 43, 47, 53],
        [7, 13, 19, 29, 37, 0, 0, 0],
    ], dtype=torch.long, device=device)
    h = super(SparseWalkerAttention, model).encode(x)
    ref = model.temporal(h, x.eq(0)).float()
    got, _ = _manual_attention(model.temporal, h, x.eq(0), topk=None)
    diff = float((ref - got.float()).abs().max().cpu())
    print("MANUAL_DENSE_EQUIVALENCE", {"max_abs_diff": diff}, flush=True)
    if diff > 5e-4:
        raise RuntimeError(f"manual MHA path mismatch: {diff}")


@torch.inference_mode()
def attention_mass_probe(model, prefixes, max_len, device, ks, cap=512):
    # Prefer long histories because that is where sparsity matters.
    idx = [i for i, p in enumerate(prefixes) if min(len(p), max_len) > 100][:cap]
    if not idx:
        idx = list(range(min(cap, len(prefixes))))

    aggregate = {int(k): [] for k in ks}
    entropies = []
    batch = 128
    for st in range(0, len(idx), batch):
        ids = idx[st:st + batch]
        rows = [list(prefixes[i])[-max_len:] for i in ids]
        lens = [len(s) for s in rows]
        L = max(lens)
        x = torch.zeros(len(rows), L, dtype=torch.long, device=device)
        for r, s in enumerate(rows):
            x[r, :len(s)] = torch.as_tensor(s, device=device)
        h = super(SparseWalkerAttention, model).encode(x)
        for k in ks:
            _, stats = _manual_attention(model.temporal, h, x.eq(0), topk=int(k), collect_stats=True)
            aggregate[int(k)].append(stats)
            if int(k) == int(ks[0]):
                entropies.append(stats["mean_dense_entropy"])

    out = {}
    for k, rows in aggregate.items():
        out[str(k)] = {
            name: float(np.mean([r[name] for r in rows]))
            for name in rows[0]
            if name != "mean_dense_entropy"
        }
    out["mean_dense_entropy"] = float(np.mean(entropies)) if entropies else 0.0
    return out


def evaluate_topk(model, split, n_items, max_len, device, batch_size, topk):
    mode = TopKMode(model)
    mode.install(topk)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    val = evaluate_full(
        model,
        split["val_prefix"], split["val_target"], n_items, max_len, device,
        topks=(10,), batch_size=batch_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    mode.restore()
    return val, secs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument(
        "--checkpoint",
        default="/content/drive/MyDrive/sparsewalker_attention_control/ml1m/seed42/best.pt",
    )
    p.add_argument(
        "--output",
        default="/content/drive/MyDrive/sparsewalker_temporal_topk/result.json",
    )
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print("DEVICE", device,
          torch.cuda.get_device_name(0) if device.type == "cuda" else None,
          "bf16", torch.cuda.is_bf16_supported() if device.type == "cuda" else False,
          flush=True)

    data = load_dataset("ml1m", args.data_dir)
    split = split_data(data["sequences"])
    max_len = 200
    protocol = protocol_manifest(max_len, data["n_items"])
    print("PROTOCOL", protocol, flush=True)

    ck_path = Path(args.checkpoint)
    if not ck_path.exists():
        raise FileNotFoundError(
            f"Missing dense-attention checkpoint {ck_path}. The current Walker+Attention run "
            "writes best.pt at every improving validation checkpoint."
        )
    ck = torch.load(ck_path, map_location="cpu")

    model = SparseWalkerAttention(
        data["n_items"], max_len,
        d=64, layers=2, side=256, h=16, active=8, top_side=2,
        degree=4, fresh_weight=.25, attn_heads=2, ff_mult=4, dropout=.1,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print("CHECKPOINT", {
        "epoch": int(ck.get("epoch", -1)),
        "saved_val_NDCG@10": ck.get("val_on", {}).get("NDCG@10"),
    }, flush=True)

    manual_dense_equivalence(model, device)

    # Native dense reference.
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    dense = evaluate_full(
        model, split["val_prefix"], split["val_target"], data["n_items"],
        max_len, device, topks=(10,), batch_size=args.eval_batch_size,
    )
    if device.type == "cuda": torch.cuda.synchronize()
    dense_s = time.perf_counter() - t0
    print("DENSE_REFERENCE", {
        "NDCG@10": float(dense["NDCG@10"]),
        "HR@10": float(dense["HR@10"]),
        "seconds": dense_s,
    }, flush=True)

    mass = attention_mass_probe(model, split["val_prefix"], max_len, device, args.ks)
    print("ATTENTION_MASS", json.dumps(mass, indent=2), flush=True)

    rows = []
    for k in args.ks:
        val, secs = evaluate_topk(
            model, split, data["n_items"], max_len, device,
            args.eval_batch_size, int(k),
        )
        row = {
            "topk": int(k),
            "NDCG@10": float(val["NDCG@10"]),
            "HR@10": float(val["HR@10"]),
            "MRR@10": float(val["MRR@10"]),
            "delta_vs_dense": float(val["NDCG@10"] - dense["NDCG@10"]),
            "retained_quality_pct": float(val["NDCG@10"] / max(float(dense["NDCG@10"]), 1e-12) * 100.0),
            "eval_seconds": secs,
            "oracle_compute_note": "still computes dense QK; this probes sparsifiability, not speed",
        }
        rows.append(row)
        print("TOPK_EVAL", row, flush=True)

    # Smallest K within 0.003 absolute of dense is the first navigation target.
    acceptable = [r for r in rows if r["NDCG@10"] >= float(dense["NDCG@10"]) - .003]
    target_k = min([r["topk"] for r in acceptable], default=None)
    decision = {
        "dense_NDCG@10": float(dense["NDCG@10"]),
        "smallest_K_within_0.003": target_k,
        "sparse_temporal_attention_supported": target_k is not None and target_k <= 32,
    }
    print("DECISION", decision, flush=True)

    result = {
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "protocol_fingerprint": protocol["fingerprint"],
        "dense": {**{k: float(v) for k, v in dense.items()}, "seconds": dense_s},
        "attention_mass": mass,
        "topk": rows,
        "decision": decision,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
