#!/usr/bin/env python
"""A100 benchmark for GPU temporal SWG attention.

Two stages:
1. quality/correctness on real trained ML-1M temporal keys from the one-layer
   Walker+Attention checkpoint. FAISS is used only to BUILD/export persistent
   HNSW level-0 adjacency; query-time search/read is Triton-only.
2. synthetic scaling microbenchmark comparing the Triton SWG read against a
   dense qK^T/softmax/V read at L={200,1000,10000}.

This is intentionally the temporal-memory primitive benchmark, not yet the full
end-to-end Walker-vs-SASRec serving benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sparsewalker.data import load_dataset, split_data
from sparsewalker.serving.temporal_swg_triton import (
    temporal_swg_search_read,
    reference_temporal_swg,
)
from run_ml1m_walker_attention_control import SparseWalkerAttention, seed_all


def _pad(prefixes, indices, max_len, device):
    rows = [list(prefixes[i])[-max_len:] for i in indices]
    lengths = torch.tensor([len(x) for x in rows], device=device, dtype=torch.long)
    L = int(lengths.max().item())
    x = torch.zeros(len(rows), L, dtype=torch.long, device=device)
    for r, s in enumerate(rows):
        x[r, :len(s)] = torch.tensor(s, device=device)
    return x, lengths


def _qkv_last(model, seq, lengths):
    block = model.temporal
    with torch.inference_mode():
        h = super(SparseWalkerAttention, model).encode(seq).float()
        B, L, D = h.shape
        H = block.attn.num_heads
        HD = D // H
        pos = torch.arange(L, device=seq.device)
        z = block.norm1(h + block.pos(pos)[None])
        p = F.linear(z, block.attn.in_proj_weight, block.attn.in_proj_bias)
        q, k, v = p.chunk(3, -1)
        q = q.view(B, L, H, HD).transpose(1, 2).contiguous()
        k = k.view(B, L, H, HD).transpose(1, 2).contiguous()
        v = v.view(B, L, H, HD).transpose(1, 2).contiguous()
        row = torch.arange(B, device=seq.device)
        last = (lengths - 1).clamp_min(0)
        qlast = q[row[:, None], torch.arange(H, device=seq.device)[None, :], last[:, None]]
    return qlast, k, v


def _hnsw_level0(keys: np.ndarray, M=8):
    import faiss
    d = keys.shape[1]
    try:
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    except TypeError:
        index = faiss.IndexHNSWFlat(d, M)
        index.metric_type = faiss.METRIC_INNER_PRODUCT
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 32
    index.add(np.ascontiguousarray(keys.astype('float32')))
    offsets = faiss.vector_to_array(index.hnsw.offsets).astype(np.int64)
    neighbors = faiss.vector_to_array(index.hnsw.neighbors).astype(np.int64)
    try:
        cap = int(index.hnsw.nb_neighbors(0))
    except Exception:
        cap = 2 * M
    adj = np.full((len(keys), cap), -1, dtype=np.int32)
    for i in range(len(keys)):
        a, b = int(offsets[i]), int(offsets[i + 1])
        row = neighbors[a:b]
        row = row[row >= 0][:cap]
        adj[i, :len(row)] = row.astype(np.int32)
    return index, adj, int(index.hnsw.entry_point)


def _entry_seeds(n, hnsw_entry, beam=16):
    vals = [int(hnsw_entry), n - 1]
    for gap in (1, 2, 4, 8, 16, 32, 64, 128):
        vals.append(max(0, n - 1 - gap))
    for frac in (0.0, .25, .5, .75):
        vals.append(min(n - 1, int(frac * max(0, n - 1))))
    out = []
    for x in vals:
        if 0 <= x < n and x not in out:
            out.append(x)
    while len(out) < beam:
        out.append(out[-1] if out else 0)
    return np.asarray(out[:beam], dtype=np.int32)


def _set_recall(got, exact):
    return len(set(map(int, got)).intersection(map(int, exact))) / max(1, len(exact))


def real_quality_probe(model, split, n_items, device, users=128, hops=4, beam=16):
    chosen = [i for i, p in enumerate(split['val_prefix']) if min(len(p), 200) > 100][:users]
    seq, lengths = _pad(split['val_prefix'], chosen, 200, device)
    qlast, k4, v4 = _qkv_last(model, seq, lengths)
    B, H, L, HD = k4.shape
    nq = B * H
    q = qlast.reshape(nq, HD)
    k = k4.reshape(nq, L, HD)
    v = v4.reshape(nq, L, HD)
    lens = lengths[:, None].expand(B, H).reshape(-1)

    kcpu = k.detach().float().cpu().numpy()
    qcpu = q.detach().float().cpu().numpy()
    adj = np.full((nq, L, 16), -1, dtype=np.int32)
    entry = np.zeros((nq, beam), dtype=np.int32)
    exact_ids = []
    faiss_ids = []
    for r in range(nq):
        n = int(lens[r].item())
        keys = np.ascontiguousarray(kcpu[r, :n])
        index, a, ep = _hnsw_level0(keys, M=8)
        take = min(adj.shape[-1], a.shape[1])
        adj[r, :n, :take] = a[:, :take]
        entry[r] = _entry_seeds(n, ep, beam)
        score = keys @ qcpu[r]
        kk = min(beam, n)
        exact = np.argpartition(score, -kk)[-kk:]
        exact = exact[np.argsort(score[exact])[::-1]]
        _, found = index.search(np.ascontiguousarray(qcpu[r:r+1]), kk)
        exact_ids.append(exact.astype(np.int32))
        faiss_ids.append(found[0].astype(np.int32))

    adj_t = torch.from_numpy(adj).to(device)
    entry_t = torch.from_numpy(entry).to(device)
    # BF16 mirrors intended serving storage; accumulation inside kernel is FP32.
    qb = q.to(torch.bfloat16)
    kb = k.to(torch.bfloat16)
    vb = v.to(torch.bfloat16)

    ref_ctx, ref_ids = reference_temporal_swg(qb, kb, vb, adj_t, entry_t, lens, hops=hops, beam=beam)
    tri_ctx, tri_ids = temporal_swg_search_read(qb, kb, vb, adj_t, entry_t, lens, hops=hops, beam=beam)
    torch.cuda.synchronize()
    max_ctx = float((ref_ctx - tri_ctx).abs().max().item())
    id_match = float((ref_ids == tri_ids).all(-1).float().mean().item())
    recalls = [_set_recall(tri_ids[r].cpu().tolist(), exact_ids[r]) for r in range(nq)]
    frecalls = [_set_recall(faiss_ids[r], exact_ids[r]) for r in range(nq)]
    out = {
        'users': B,
        'head_queries': nq,
        'history_max_L': L,
        'triton_vs_reference_max_abs_ctx': max_ctx,
        'triton_exact_id_row_match_rate': id_match,
        'triton_swg_exact_top16_recall': float(np.mean(recalls)),
        'faiss_hnsw_exact_top16_recall': float(np.mean(frecalls)),
        'hops': hops,
        'beam': beam,
        'degree': 16,
        'note': 'FAISS builds persistent graph only; Triton performs query-time walk/read',
    }
    print('REAL_QUALITY', out, flush=True)
    return out


def _bench(fn, warmup=25, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return float(ms)


def _dense_read(q, k, v):
    # Strong baseline: PyTorch SDPA/Flash-style kernel, one logical head per NQ.
    return F.scaled_dot_product_attention(
        q[:, None, None, :], k[:, None, :, :], v[:, None, :, :],
        dropout_p=0.0, is_causal=False,
    )[:, 0, 0, :]


def synthetic_case(L, users, device, hops=4, beam=16):
    # Two attention heads per user, HD=32, matching the trained block.
    nq = users * 2
    HD = 32
    rng = np.random.default_rng(1234 + L + users)
    base = rng.standard_normal((L, HD), dtype=np.float32)
    index, adj1, ep = _hnsw_level0(base, M=8)
    # Queries near actual memory keys so graph navigation is non-degenerate.
    targets = rng.integers(0, L, size=nq)
    qnp = base[targets] + .05 * rng.standard_normal((nq, HD), dtype=np.float32)
    vnp = rng.standard_normal((L, HD), dtype=np.float32)
    knp = np.broadcast_to(base[None], (nq, L, HD)).copy()
    vvnp = np.broadcast_to(vnp[None], (nq, L, HD)).copy()
    adj = np.broadcast_to(adj1[None], (nq, L, adj1.shape[1])).copy()
    entry1 = _entry_seeds(L, ep, beam)
    entry = np.broadcast_to(entry1[None], (nq, beam)).copy()

    q = torch.from_numpy(qnp).to(device=device, dtype=torch.bfloat16)
    k = torch.from_numpy(knp).to(device=device, dtype=torch.bfloat16)
    v = torch.from_numpy(vvnp).to(device=device, dtype=torch.bfloat16)
    a = torch.from_numpy(adj).to(device=device, dtype=torch.int32)
    e = torch.from_numpy(entry).to(device=device, dtype=torch.int32)
    lens = torch.full((nq,), L, device=device, dtype=torch.int32)

    tri = lambda: temporal_swg_search_read(q, k, v, a, e, lens, hops=hops, beam=beam)[0]
    dense = lambda: _dense_read(q, k, v)
    tri_ms = _bench(tri)
    dense_ms = _bench(dense)

    with torch.inference_mode():
        _, ids = temporal_swg_search_read(q, k, v, a, e, lens, hops=hops, beam=beam)
        score = torch.bmm(k.float(), q.float().unsqueeze(-1)).squeeze(-1)
        exact = score.topk(beam, -1).indices
        rec = []
        for r in range(nq):
            rec.append(_set_recall(ids[r].cpu().tolist(), exact[r].cpu().tolist()))
    return {
        'L': L,
        'users': users,
        'head_queries': nq,
        'triton_swg_ms': tri_ms,
        'dense_temporal_ms': dense_ms,
        'speedup_vs_dense': dense_ms / max(tri_ms, 1e-9),
        'triton_us_per_user': tri_ms * 1000 / users,
        'mean_exact_top16_recall_synthetic': float(np.mean(rec)),
        'search_distance_budget_upper_bound_per_head': beam * 16 * hops + beam,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-dir', default='/content/sparsewalker_data')
    p.add_argument('--checkpoint', default='/content/drive/MyDrive/sparsewalker_attention_control/ml1m/seed42/best.pt')
    p.add_argument('--quality-users', type=int, default=128)
    p.add_argument('--lengths', nargs='+', type=int, default=[200, 1000, 10000])
    p.add_argument('--users', nargs='+', type=int, default=[1, 32])
    p.add_argument('--hops', type=int, default=4)
    p.add_argument('--beam', type=int, default=16)
    args = p.parse_args()

    seed_all(args.seed)
    assert torch.cuda.is_available(), 'GPU required'
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    print('DEVICE', torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported(), flush=True)

    data = load_dataset('ml1m', args.data_dir)
    split = split_data(data['sequences'])
    ck = torch.load(args.checkpoint, map_location='cpu')
    model = SparseWalkerAttention(
        data['n_items'], 200, d=64, layers=2, side=256, h=16,
        active=8, top_side=2, degree=4, fresh_weight=.25,
        attn_heads=2, ff_mult=4, dropout=.1,
    ).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    print('CHECKPOINT', {'epoch': int(ck.get('epoch', -1)), 'val': ck.get('val_on', {}).get('NDCG@10')}, flush=True)

    quality = real_quality_probe(model, split, data['n_items'], device, args.quality_users, args.hops, args.beam)
    rows = []
    for L in args.lengths:
        for users in args.users:
            row = synthetic_case(int(L), int(users), device, args.hops, args.beam)
            rows.append(row)
            print('SPEED', row, flush=True)
            torch.cuda.empty_cache()

    result = {'quality': quality, 'speed': rows}
    out = Path('/content/drive/MyDrive/sparsewalker_triton_swg/result.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print('SAVED', out, flush=True)


if __name__ == '__main__':
    main()
