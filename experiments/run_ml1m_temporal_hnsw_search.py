#!/usr/bin/env python
"""Temporal SWG/HNSW search probe for Walker+Attention on ML-1M.

Loads the latest best dense Walker+Attention checkpoint and asks the next
question after the exact Top-K sparsity result:

  Can approximate small-world navigation FIND the exact temporal Top-16 keys
  without scanning every historical state, while preserving recommendation
  quality?

Important: this is a search-quality diagnostic. It builds a tiny per-user,
per-head HNSW index over the trained temporal attention K-vectors. In a streaming
system these indexes would be maintained incrementally as events arrive; here we
rebuild them during evaluation for simplicity and report build time separately.
The Python/CPU implementation is not intended as serving latency code.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_ml1m_walker_attention_control import SparseWalkerAttention, protocol_manifest, seed_all
from run_ml1m_temporal_topk_probe import manual_dense_equivalence, evaluate_topk
from sparsewalker.data import load_dataset, split_data
from sparsewalker.evaluation import evaluate_full


def _softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x, dtype=np.float32)
    return e / max(float(e.sum()), 1e-12)


class TemporalHNSWScorer(nn.Module):
    """Wrap a trained model and replace only the LAST temporal attention read.

    Evaluation only needs the last hidden state. We therefore keep the corrected
    Walker exactly as trained, construct q/k/v from the trained MHA projections,
    retrieve K temporal states independently for each head with HNSW, recompute
    exact attention logits inside that retrieved set, then apply the original
    out-projection, residual and FFN.
    """

    def __init__(self, model, topk=16, hnsw_m=8, ef_construction=80, ef_search=32,
                 recall_cap=1024):
        super().__init__()
        self.model = model
        self.n_items = model.n_items
        self.topk = int(topk)
        self.hnsw_m = int(hnsw_m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.recall_cap = int(recall_cap)
        self.reset_stats()

    def reset_stats(self):
        self.users = 0
        self.head_queries = 0
        self.build_seconds = 0.0
        self.search_seconds = 0.0
        self.distance_computations = 0.0
        self.distance_stats_available = True
        self.recall_sum = 0.0
        self.recall_n = 0

    def stats(self):
        return {
            "users": int(self.users),
            "head_queries": int(self.head_queries),
            "mean_exact_top16_recall": self.recall_sum / max(1, self.recall_n),
            "recall_queries": int(self.recall_n),
            "mean_distance_computations_per_head_query": (
                self.distance_computations / max(1, self.head_queries)
                if self.distance_stats_available else None
            ),
            "hnsw_build_seconds_total": float(self.build_seconds),
            "hnsw_search_seconds_total": float(self.search_seconds),
            "note": "Python/CPU diagnostic; build is amortizable online, runtime is not a serving benchmark",
        }

    @torch.inference_mode()
    def full_scores(self, seq, lengths):
        try:
            import faiss
        except ImportError as e:
            raise RuntimeError("pip install -q faiss-cpu") from e

        model = self.model
        block = model.temporal
        # Corrected Walker states, before temporal attention.
        h = super(SparseWalkerAttention, model).encode(seq).float()
        B, L, D = h.shape
        H = block.attn.num_heads
        HD = D // H

        pos_ids = torch.arange(L, device=seq.device)
        qkv_in = block.norm1(h + block.pos(pos_ids)[None, :, :])
        proj = F.linear(qkv_in, block.attn.in_proj_weight, block.attn.in_proj_bias)
        q, k, v = proj.chunk(3, dim=-1)
        q = q.view(B, L, H, HD).transpose(1, 2).contiguous()
        k = k.view(B, L, H, HD).transpose(1, 2).contiguous()
        v = v.view(B, L, H, HD).transpose(1, 2).contiguous()

        # HNSW is CPU in this diagnostic. Transfer once per batch, not per query.
        q_np = q.detach().cpu().numpy().astype("float32", copy=False)
        k_np = k.detach().cpu().numpy().astype("float32", copy=False)
        v_np = v.detach().cpu().numpy().astype("float32", copy=False)
        lens = lengths.detach().cpu().numpy().astype(np.int64, copy=False)

        ctx = np.zeros((B, H, HD), dtype=np.float32)
        for b in range(B):
            n = int(lens[b])
            if n <= 0:
                continue
            self.users += 1
            qi = n - 1
            for head in range(H):
                keys = np.ascontiguousarray(k_np[b, head, :n])
                values = v_np[b, head, :n]
                query = np.ascontiguousarray(q_np[b, head, qi:qi + 1])
                kk = min(self.topk, n)

                try:
                    index = faiss.IndexHNSWFlat(HD, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
                except TypeError:
                    index = faiss.IndexHNSWFlat(HD, self.hnsw_m)
                    index.metric_type = faiss.METRIC_INNER_PRODUCT
                index.hnsw.efConstruction = self.ef_construction
                index.hnsw.efSearch = max(self.ef_search, kk)

                t0 = time.perf_counter()
                index.add(keys)
                self.build_seconds += time.perf_counter() - t0

                # Search stats count distance evaluations where supported.
                try:
                    faiss.cvar.hnsw_stats.reset()
                except Exception:
                    self.distance_stats_available = False

                t0 = time.perf_counter()
                _, found = index.search(query, kk)
                self.search_seconds += time.perf_counter() - t0
                found = found[0]
                found = found[found >= 0].astype(np.int64, copy=False)
                self.head_queries += 1

                if self.distance_stats_available:
                    try:
                        self.distance_computations += float(faiss.cvar.hnsw_stats.ndis)
                    except Exception:
                        self.distance_stats_available = False

                # Exact Top-K recall is diagnostic only, capped so the full quality
                # path does not secretly depend on dense search.
                if self.recall_n < self.recall_cap * H:
                    dense_score = (keys @ query[0]) / math.sqrt(HD)
                    if kk == n:
                        exact = np.arange(n, dtype=np.int64)
                    else:
                        exact = np.argpartition(dense_score, -kk)[-kk:].astype(np.int64)
                    self.recall_sum += len(set(found.tolist()).intersection(exact.tolist())) / max(1, kk)
                    self.recall_n += 1

                # Re-score retrieved candidates exactly with the trained MHA rule.
                logits = (keys[found] @ query[0]) / math.sqrt(HD)
                weight = _softmax_np(logits)
                ctx[b, head] = weight @ values[found]

        ctx_t = torch.from_numpy(ctx.reshape(B, D)).to(seq.device)
        a = F.linear(ctx_t, block.attn.out_proj.weight, block.attn.out_proj.bias)
        rows = torch.arange(B, device=seq.device)
        last = (lengths - 1).clamp_min(0)
        h_last = h[rows, last]
        z = h_last + a
        z = z + block.ffn(block.norm2(z))
        return model.score_hidden(z)


def eval_hnsw(model, split, n_items, max_len, device, batch_size, ef_search, topk,
              hnsw_m, ef_construction, recall_cap):
    wrapper = TemporalHNSWScorer(
        model, topk=topk, hnsw_m=hnsw_m,
        ef_construction=ef_construction, ef_search=ef_search,
        recall_cap=recall_cap,
    ).to(device)
    wrapper.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    val = evaluate_full(
        wrapper, split["val_prefix"], split["val_target"], n_items,
        max_len, device, topks=(10,), batch_size=batch_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_s = time.perf_counter() - t0
    return val, wrapper.stats(), total_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument("--checkpoint", default="/content/drive/MyDrive/sparsewalker_attention_control/ml1m/seed42/best.pt")
    p.add_argument("--output", default="/content/drive/MyDrive/sparsewalker_temporal_hnsw/result.json")
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--hnsw-m", type=int, default=8)
    p.add_argument("--ef-construction", type=int, default=80)
    p.add_argument("--ef-search", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--recall-cap", type=int, default=1024)
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print("DEVICE", device, torch.cuda.get_device_name(0) if device.type == "cuda" else None,
          "bf16", torch.cuda.is_bf16_supported() if device.type == "cuda" else False, flush=True)

    data = load_dataset("ml1m", args.data_dir)
    split = split_data(data["sequences"])
    max_len = 200
    protocol = protocol_manifest(max_len, data["n_items"])
    print("PROTOCOL", protocol, flush=True)

    ck_path = Path(args.checkpoint)
    if not ck_path.exists():
        raise FileNotFoundError(ck_path)
    ck = torch.load(ck_path, map_location="cpu")
    model = SparseWalkerAttention(
        data["n_items"], max_len,
        d=64, layers=2, side=256, h=16, active=8, top_side=2,
        degree=4, fresh_weight=.25, attn_heads=2, ff_mult=4, dropout=.1,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print("CHECKPOINT", {"epoch": int(ck.get("epoch", -1)),
                         "saved_val_NDCG@10": ck.get("val_on", {}).get("NDCG@10")}, flush=True)

    manual_dense_equivalence(model, device)

    # Native dense and exact Top-16 are the two references.
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    dense = evaluate_full(
        model, split["val_prefix"], split["val_target"], data["n_items"],
        max_len, device, topks=(10,), batch_size=1024,
    )
    if device.type == "cuda": torch.cuda.synchronize()
    dense_s = time.perf_counter() - t0
    exact16, exact16_s = evaluate_topk(
        model, split, data["n_items"], max_len, device, 1024, args.topk
    )
    print("REFERENCES", {
        "dense_NDCG@10": float(dense["NDCG@10"]),
        "exact_top16_NDCG@10": float(exact16["NDCG@10"]),
        "dense_seconds": dense_s,
        "exact_top16_seconds": exact16_s,
    }, flush=True)

    rows = []
    for ef in args.ef_search:
        print("HNSW_EVAL_START", {"efSearch": int(ef), "M": args.hnsw_m,
                                  "topk": args.topk}, flush=True)
        val, search_stats, total_s = eval_hnsw(
            model, split, data["n_items"], max_len, device,
            args.eval_batch_size, int(ef), args.topk,
            args.hnsw_m, args.ef_construction, args.recall_cap,
        )
        row = {
            "efSearch": int(ef),
            "NDCG@10": float(val["NDCG@10"]),
            "HR@10": float(val["HR@10"]),
            "MRR@10": float(val["MRR@10"]),
            "delta_vs_dense": float(val["NDCG@10"] - dense["NDCG@10"]),
            "delta_vs_exact_top16": float(val["NDCG@10"] - exact16["NDCG@10"]),
            "retained_dense_quality_pct": float(val["NDCG@10"] / max(float(dense["NDCG@10"]), 1e-12) * 100),
            "total_eval_seconds": float(total_s),
            **search_stats,
        }
        rows.append(row)
        print("HNSW_EVAL", row, flush=True)

    viable = [r for r in rows if r["NDCG@10"] >= float(exact16["NDCG@10"]) - .003]
    best_cost = min(viable, key=lambda r: r["efSearch"], default=None)
    decision = {
        "dense_NDCG@10": float(dense["NDCG@10"]),
        "exact_top16_NDCG@10": float(exact16["NDCG@10"]),
        "smallest_viable_efSearch": None if best_cost is None else best_cost["efSearch"],
        "temporal_small_world_search_supported": best_cost is not None,
        "criterion": "HNSW NDCG within 0.003 absolute of exact Top-16",
    }
    print("DECISION", decision, flush=True)

    result = {
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "protocol_fingerprint": protocol["fingerprint"],
        "config": {"topk": args.topk, "M": args.hnsw_m,
                   "efConstruction": args.ef_construction},
        "dense": {**{k: float(v) for k, v in dense.items()}, "seconds": dense_s},
        "exact_top16": {**{k: float(v) for k, v in exact16.items()}, "seconds": exact16_s},
        "hnsw": rows,
        "decision": decision,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
