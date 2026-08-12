#!/usr/bin/env python
"""Correctness-first audit of SASRec serving latency.

This answers three questions before we use a SASRec latency number:

1. Does token-by-token KV-cache inference reproduce the ACTUAL trained SASRec
   checkpoint and canonical ML-1M NDCG while the prefix grows inside max_len=200?
2. Is the same cache still exact after the 200-token window slides?  The trained
   model uses learned absolute positions, so cached old tokens may become stale.
3. Are precision and training conditions being compared fairly?  Report FP32 and
   BF16 quality/latency separately and print the training-precision provenance.

Dense catalog scoring is used only as a quality oracle. It is not a Walker
serving-path benchmark.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsewalker.data import load_dataset, split_data
from sparsewalker.evaluation import evaluate_full, make_eval_batch
from sparsewalker.models import SASRec


class ExactIncrementalSASRec(nn.Module):
    """Exact token-by-token evaluation of a SASRec prefix (no sliding window).

    For a prefix of length <= max_len this caches each layer's projected K/V at
    the same learned absolute position used by full-prefix SASRec. At eval time
    this should reproduce the last hidden state up to numerical precision.
    """

    def __init__(self, base: SASRec):
        super().__init__()
        self.base = base
        self.n_items = base.n_items
        self.max_len = base.max_len
        self.d_model = base.d_model

    @property
    def item_weight(self):
        return self.base.item_weight

    def _step_block(self, x, block, kc, vc, t):
        B = x.size(0)
        H = block.attn.num_heads
        HD = self.d_model // H
        z = block.n1(x)
        qkv = F.linear(z, block.attn.in_proj_weight, block.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, H, HD)
        k = k.view(B, H, HD)
        v = v.view(B, H, HD)
        kc[:, :, t, :].copy_(k)
        vc[:, :, t, :].copy_(v)
        score = torch.einsum("bhd,bhld->bhl", q, kc[:, :, : t + 1]) / math.sqrt(HD)
        # FP32 softmax accumulation is stable and mirrors common fused attention kernels.
        prob = torch.softmax(score.float(), dim=-1).to(x.dtype)
        ctx = torch.einsum("bhl,bhld->bhd", prob, vc[:, :, : t + 1]).reshape(B, self.d_model)
        a = F.linear(ctx, block.attn.out_proj.weight, block.attn.out_proj.bias)
        x = x + a
        x = x + block.ffn(block.n2(x))
        return x

    def last_hidden(self, seq, lengths, return_caches=False):
        B, L = seq.shape
        dtype = self.base.item.weight.dtype
        device = seq.device
        caches = []
        for block in self.base.blocks:
            H = block.attn.num_heads
            HD = self.d_model // H
            caches.append([
                torch.empty(B, H, L, HD, device=device, dtype=dtype),
                torch.empty(B, H, L, HD, device=device, dtype=dtype),
            ])
        last = torch.zeros(B, self.d_model, device=device, dtype=dtype)
        for t in range(L):
            pos = torch.full((B,), t, device=device, dtype=torch.long)
            x = self.base.item(seq[:, t]) + self.base.pos(pos)
            for li, block in enumerate(self.base.blocks):
                x = self._step_block(x, block, caches[li][0], caches[li][1], t)
            x = self.base.norm(x)
            is_last = (lengths - 1 == t)[:, None]
            last = torch.where(is_last, x, last)
        return (last, caches) if return_caches else last

    def full_scores(self, seq, lengths):
        h = self.last_hidden(seq, lengths)
        z = h @ self.item_weight[: self.n_items + 1].T
        z[..., 0] = -1e9
        return z


def stale_sliding_step(base: SASRec, previous_window: torch.Tensor, new_item: torch.Tensor):
    """Append one token after dropping the oldest KV WITHOUT re-encoding survivors.

    previous_window is [B,200] encoded at learned positions 0..199. After the
    window slides, its surviving tokens SHOULD move to positions 0..198, but a
    naive KV cache retains their old position-dependent K/V. This function
    intentionally models that stale-cache serving shortcut and compares it to
    canonical recomputation of the new 200-token window.
    """
    B, L = previous_window.shape
    assert L == base.max_len
    inc = ExactIncrementalSASRec(base)
    lens = torch.full((B,), L, device=previous_window.device, dtype=torch.long)
    _, caches = inc.last_hidden(previous_window, lens, return_caches=True)

    pos = torch.full((B,), L - 1, device=previous_window.device, dtype=torch.long)
    x = base.item(new_item) + base.pos(pos)
    for li, block in enumerate(base.blocks):
        H = block.attn.num_heads
        HD = base.d_model // H
        z = block.n1(x)
        qkv = F.linear(z, block.attn.in_proj_weight, block.attn.in_proj_bias)
        q, knew, vnew = qkv.chunk(3, -1)
        q = q.view(B, H, HD)
        knew = knew.view(B, H, 1, HD)
        vnew = vnew.view(B, H, 1, HD)
        # Drop oldest cache slot; surviving slots keep their OLD position encodings.
        kall = torch.cat([caches[li][0][:, :, 1:, :], knew], dim=2)
        vall = torch.cat([caches[li][1][:, :, 1:, :], vnew], dim=2)
        score = torch.einsum("bhd,bhld->bhl", q, kall) / math.sqrt(HD)
        prob = torch.softmax(score.float(), dim=-1).to(x.dtype)
        ctx = torch.einsum("bhl,bhld->bhd", prob, vall).reshape(B, base.d_model)
        a = F.linear(ctx, block.attn.out_proj.weight, block.attn.out_proj.bias)
        x = x + a
        x = x + block.ffn(block.n2(x))
    return base.norm(x)


def mask_seen_and_ndcg(scores, prefixes, targets, topk=10):
    scores = scores.float().clone()
    for r, p in enumerate(prefixes):
        seen = set(p)
        seen.discard(int(targets[r]))
        if seen:
            ids = torch.tensor(list(seen), device=scores.device, dtype=torch.long)
            scores[r, ids] = -1e20
    top = scores.topk(topk, dim=-1).indices.cpu().numpy()
    ndcg = hr = mrr = 0.0
    for r, truth in enumerate(targets):
        pos = np.where(top[r] == int(truth))[0]
        if len(pos):
            rank = int(pos[0]) + 1
            hr += 1.0
            ndcg += 1.0 / math.log2(rank + 1)
            mrr += 1.0 / rank
    n = max(1, len(targets))
    return {"HR@10": hr / n, "NDCG@10": ndcg / n, "MRR@10": mrr / n}


@torch.inference_mode()
def incremental_equivalence_probe(base, prefixes, targets, n_items, device, cap=1024, batch=128):
    pp = list(prefixes[:cap])
    tt = list(targets[:cap])
    inc = ExactIncrementalSASRec(base).to(device).eval()
    full_metric = evaluate_full(base, pp, tt, n_items, base.max_len, device, topks=(10,), batch_size=batch)
    inc_metric = evaluate_full(inc, pp, tt, n_items, base.max_len, device, topks=(10,), batch_size=batch)

    max_hidden = 0.0
    max_score = 0.0
    # Direct numerical comparison on a few representative batches.
    for st in range(0, min(cap, 256), batch):
        ids = list(range(st, min(min(cap, 256), st + batch)))
        seq, lens = make_eval_batch(pp, ids, base.max_len)
        seq, lens = seq.to(device), lens.to(device)
        hf = base.last_hidden(seq, lens)
        hi = inc.last_hidden(seq, lens)
        max_hidden = max(max_hidden, float((hf.float() - hi.float()).abs().max().cpu()))
        sf = base.score_hidden(hf)
        si = inc.full_scores(seq, lens)
        max_score = max(max_score, float((sf.float() - si.float()).abs().max().cpu()))
    return {
        "users": len(pp),
        "full_prefix": {k: float(v) for k, v in full_metric.items()},
        "exact_incremental": {k: float(v) for k, v in inc_metric.items()},
        "ndcg_delta": float(inc_metric["NDCG@10"] - full_metric["NDCG@10"]),
        "max_abs_last_hidden": max_hidden,
        "max_abs_score": max_score,
    }


@torch.inference_mode()
def sliding_window_probe(base, prefixes, targets, device, cap=512, batch=64):
    ids = [i for i, p in enumerate(prefixes) if len(p) > base.max_len][:cap]
    if not ids:
        return {"users": 0, "note": "no prefixes longer than max_len"}
    all_full, all_stale, all_prefix, all_target = [], [], [], []
    max_hidden = 0.0
    for st in range(0, len(ids), batch):
        chunk = ids[st: st + batch]
        prev = [list(prefixes[i])[-(base.max_len + 1):-1] for i in chunk]
        cur = [list(prefixes[i])[-base.max_len:] for i in chunk]
        new = [int(prefixes[i][-1]) for i in chunk]
        assert all(len(x) == base.max_len for x in prev)
        prev_t = torch.tensor(prev, device=device, dtype=torch.long)
        cur_t = torch.tensor(cur, device=device, dtype=torch.long)
        new_t = torch.tensor(new, device=device, dtype=torch.long)
        lens = torch.full((len(chunk),), base.max_len, device=device, dtype=torch.long)
        hf = base.last_hidden(cur_t, lens)
        hs = stale_sliding_step(base, prev_t, new_t)
        max_hidden = max(max_hidden, float((hf.float() - hs.float()).abs().max().cpu()))
        all_full.append(base.score_hidden(hf))
        all_stale.append(base.score_hidden(hs))
        all_prefix.extend([prefixes[i] for i in chunk])
        all_target.extend([targets[i] for i in chunk])
    sf = torch.cat(all_full, 0)
    ss = torch.cat(all_stale, 0)
    mf = mask_seen_and_ndcg(sf, all_prefix, all_target)
    ms = mask_seen_and_ndcg(ss, all_prefix, all_target)
    return {
        "users": len(ids),
        "canonical_recompute": mf,
        "stale_sliding_kv": ms,
        "ndcg_delta": float(ms["NDCG@10"] - mf["NDCG@10"]),
        "max_abs_last_hidden": max_hidden,
        "interpretation": "nonzero delta means persistent KV after window saturation is not quality-equivalent to trained absolute-position SASRec",
    }


def latency_samples(fn, warm=30, n=200):
    with torch.inference_mode():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        vals = []
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        for _ in range(n):
            a.record(); fn(); b.record(); b.synchronize()
            vals.append(a.elapsed_time(b) * 1000.0)
    x = np.asarray(vals)
    return {"mean_us": float(x.mean()), "p50_us": float(np.percentile(x, 50)),
            "p95_us": float(np.percentile(x, 95)), "p99_us": float(np.percentile(x, 99))}


class CachedAppend(nn.Module):
    """Quality-equivalent append only while a prefix grows and does NOT slide."""
    def __init__(self, base, prefix199):
        super().__init__()
        self.base = base
        self.inc = ExactIncrementalSASRec(base)
        lens = torch.tensor([prefix199.size(1)], device=prefix199.device)
        with torch.inference_mode():
            _, caches = self.inc.last_hidden(prefix199, lens, return_caches=True)
        self.caches = caches
        self.t = prefix199.size(1)

    def forward(self, item):
        B = item.size(0)
        pos = torch.full((B,), self.t, device=item.device, dtype=torch.long)
        x = self.base.item(item) + self.base.pos(pos)
        for li, block in enumerate(self.base.blocks):
            H = block.attn.num_heads; HD = self.base.d_model // H
            z = block.n1(x)
            qkv = F.linear(z, block.attn.in_proj_weight, block.attn.in_proj_bias)
            q, k, v = qkv.chunk(3, -1)
            q = q.view(B, H, HD); k = k.view(B, H, 1, HD); v = v.view(B, H, 1, HD)
            kall = torch.cat([self.caches[li][0], k], 2)
            vall = torch.cat([self.caches[li][1], v], 2)
            score = torch.einsum("bhd,bhld->bhl", q, kall) / math.sqrt(HD)
            prob = torch.softmax(score.float(), -1).to(x.dtype)
            ctx = torch.einsum("bhl,bhld->bhd", prob, vall).reshape(B, self.base.d_model)
            x = x + F.linear(ctx, block.attn.out_proj.weight, block.attn.out_proj.bias)
            x = x + block.ffn(block.n2(x))
        return self.base.norm(x)


@torch.inference_mode()
def latency_probe(base, device):
    L = base.max_len
    seq = torch.randint(1, base.n_items + 1, (1, L), device=device)
    lens = torch.tensor([L], device=device)
    prefix199 = seq[:, :-1].contiguous()
    new_item = seq[:, -1].contiguous()
    cached = CachedAppend(base, prefix199).to(device).eval()

    def full_window():
        return base.last_hidden(seq, lens)
    def append():
        return cached(new_item)

    out = {
        "full_window_recompute": latency_samples(full_window, n=120),
        "cached_append_before_window_slides": latency_samples(append, n=200),
        "cached_append_quality_equivalent_after_saturation": False,
    }
    # Optional compile of the quality-equivalent full-window path.
    try:
        compiled = torch.compile(full_window, mode="reduce-overhead", fullgraph=False)
        compiled(); torch.cuda.synchronize()
        out["full_window_recompute_torch_compile"] = latency_samples(compiled, n=120)
    except Exception as e:
        out["full_window_recompute_torch_compile_error"] = repr(e)
    return out


def build_sasrec(n_items, device):
    return SASRec(n_items, 200, d=64, layers=2, heads=1, inner=256, dropout=.1).to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument("--sasrec-checkpoint", default="/content/drive/MyDrive/sparsewalker_esasrec_2x2/ml1m/seed42/SASRec_FullCE/best.pt")
    p.add_argument("--walker-checkpoint", default="/content/drive/MyDrive/sparsewalker_two_temporal_layers/ml1m/seed42/best.pt")
    p.add_argument("--incremental-users", type=int, default=1024)
    p.add_argument("--sliding-users", type=int, default=512)
    p.add_argument("--output", default="/content/drive/MyDrive/sparsewalker_speed/sasrec_serving_correctness_audit.json")
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    torch.manual_seed(42)
    data = load_dataset("ml1m", args.data_dir)
    split = split_data(data["sequences"])
    ck = torch.load(args.sasrec_checkpoint, map_location="cpu")
    sas = build_sasrec(data["n_items"], device)
    sas.load_state_dict(ck["model"])
    sas.eval()

    print("DEVICE", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported(), flush=True)
    fairness = {
        "sasrec_geometry": {"d": 64, "layers": 2, "heads": 1, "max_len": 200, "dropout_train": .1},
        "sasrec_checkpoint_epoch": int(ck.get("epoch", -1)),
        "sasrec_training_precision": "FP32 (baseline runner calls train_epoch with use_bf16 default False)",
        "walker_training_precision": "BF16 autocast (two-temporal runner calls train_epoch(use_bf16=True))",
        "strict_training_precision_match": False,
        "inference_precision_policy": "report FP32 and BF16 separately; never compare FP32 one side to BF16 the other",
        "warning": "strict training-precision apples-to-apples requires retraining one side; latency fairness is handled here by matched inference precision",
    }
    print("FAIRNESS", fairness, flush=True)

    # Canonical full validation NDCG in both inference precisions.
    full_fp32 = evaluate_full(sas, split["val_prefix"], split["val_target"], data["n_items"], 200, device,
                               topks=(10,), batch_size=1024)
    full_bf16 = evaluate_full(sas, split["val_prefix"], split["val_target"], data["n_items"], 200, device,
                               topks=(10,), batch_size=1024, autocast_dtype=torch.bfloat16)
    print("SASREC_NDCG", {"FP32": full_fp32, "BF16_autocast": full_bf16,
                           "delta_NDCG": float(full_bf16["NDCG@10"] - full_fp32["NDCG@10"])}, flush=True)

    eq_fp32 = incremental_equivalence_probe(sas, split["val_prefix"], split["val_target"], data["n_items"],
                                             device, cap=args.incremental_users)
    print("INCREMENTAL_EQUIVALENCE_FP32", json.dumps(eq_fp32, indent=2), flush=True)

    slide_fp32 = sliding_window_probe(sas, split["val_prefix"], split["val_target"], device,
                                      cap=args.sliding_users)
    print("SLIDING_WINDOW_FP32", json.dumps(slide_fp32, indent=2), flush=True)

    lat_fp32 = latency_probe(sas, device)
    print("LATENCY_FP32", json.dumps(lat_fp32, indent=2), flush=True)

    # Matched BF16 inference: cast the same trained checkpoint to BF16.
    sas16 = copy.deepcopy(sas).to(dtype=torch.bfloat16).eval()
    eq_bf16 = incremental_equivalence_probe(sas16, split["val_prefix"], split["val_target"], data["n_items"],
                                             device, cap=min(args.incremental_users, 512))
    slide_bf16 = sliding_window_probe(sas16, split["val_prefix"], split["val_target"], device,
                                      cap=min(args.sliding_users, 256))
    lat_bf16 = latency_probe(sas16, device)
    print("INCREMENTAL_EQUIVALENCE_BF16", json.dumps(eq_bf16, indent=2), flush=True)
    print("SLIDING_WINDOW_BF16", json.dumps(slide_bf16, indent=2), flush=True)
    print("LATENCY_BF16", json.dumps(lat_bf16, indent=2), flush=True)

    walker_quality = None
    wp = Path(args.walker_checkpoint)
    if wp.exists():
        try:
            import sys
            exp = "/content/Sparsewalker/experiments"
            if exp not in sys.path: sys.path.insert(0, exp)
            from run_ml1m_walker_two_temporal_layers import SparseWalkerTwoTemporal
            wck = torch.load(wp, map_location="cpu")
            walker = SparseWalkerTwoTemporal(data["n_items"], 200, d=64, layers=2, side=256, h=16,
                active=8, top_side=2, degree=4, fresh_weight=.25, attn_heads=2, ff_mult=4, dropout=.1).to(device).eval()
            walker.load_state_dict(wck["model"])
            w32 = evaluate_full(walker, split["val_prefix"], split["val_target"], data["n_items"], 200, device,
                                topks=(10,), batch_size=1024)
            w16 = evaluate_full(walker, split["val_prefix"], split["val_target"], data["n_items"], 200, device,
                                topks=(10,), batch_size=1024, autocast_dtype=torch.bfloat16)
            walker_quality = {"epoch": int(wck.get("epoch", -1)), "FP32": w32, "BF16_autocast": w16,
                              "delta_NDCG": float(w16["NDCG@10"] - w32["NDCG@10"]),
                              "note": "dense catalog used only as quality oracle; not a timed Walker serving path"}
            print("WALKER_NDCG", walker_quality, flush=True)
        except Exception as e:
            walker_quality = {"error": repr(e)}
            print("WALKER_NDCG_ERROR", repr(e), flush=True)

    decision = {
        "fast_kv_valid_before_window_saturation": abs(eq_fp32["ndcg_delta"]) < 1e-4,
        "fast_kv_valid_after_window_saturation": abs(slide_fp32.get("ndcg_delta", 1.0)) < 1e-4,
        "quality_equivalent_serving_latency_should_use": "cached append" if abs(slide_fp32.get("ndcg_delta", 1.0)) < 1e-4 else "200-token window recompute (or change positional scheme/retrain for cacheable serving)",
        "training_precision_match": False,
    }
    print("DECISION", json.dumps(decision, indent=2), flush=True)

    result = {"fairness": fairness, "sasrec_full_quality": {"FP32": full_fp32, "BF16": full_bf16},
              "incremental_fp32": eq_fp32, "sliding_fp32": slide_fp32, "latency_fp32": lat_fp32,
              "incremental_bf16": eq_bf16, "sliding_bf16": slide_bf16, "latency_bf16": lat_bf16,
              "walker_quality": walker_quality, "decision": decision}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
