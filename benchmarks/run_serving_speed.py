#!/usr/bin/env python
"""A100 online-serving latency microbenchmark for SparseWalker vs SASRec.

Separates:
  1) SASRec full-prefix recomputation.
  2) SASRec exact incremental KV-cache step.
  3) SparseWalker persistent local update with repo Triton kernels.
  4) Sparse temporal retrieval compute proxy: score ~96 graph-visited candidates/head,
     keep Top-16, read values. This intentionally excludes graph pointer-chasing.
  5) Final retrieval: dense catalog matmul+topk vs Walker terminal sparse scoring.

This is a serving microbenchmark, not a production p99 claim.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from sparsewalker.models import SASRec
from sparsewalker.serving.walker_triton import (
    route as walker_route,
    walk as walker_walk,
    readout as walker_readout,
    term_block,
    term_merge,
)


def seed_all(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bench_cuda(fn, warmup=50, iters=300):
    """Average CUDA latency in microseconds, launch overhead included."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end) / iters
    return float(ms * 1000.0)


class IncrementalSASRec(nn.Module):
    """One-token SASRec step over pre-existing per-layer KV caches.

    Mirrors the repo SASBlock at dropout=0 but receives K/V caches directly,
    so this measures online serving cost rather than prefill cost.
    """

    def __init__(self, sasrec):
        super().__init__()
        self.blocks = sasrec.blocks
        self.norm = sasrec.norm
        self.d = sasrec.d_model

    def _block_step(self, x, block, k_cache, v_cache):
        B, H, L, HD = k_cache.shape
        z = block.n1(x)
        proj = F.linear(z, block.attn.in_proj_weight, block.attn.in_proj_bias)
        q, k_new, v_new = proj.chunk(3, dim=-1)
        q = q.view(B, H, HD)
        k_new = k_new.view(B, H, HD)
        v_new = v_new.view(B, H, HD)

        scores_old = torch.einsum("bhd,bhld->bhl", q, k_cache) / math.sqrt(HD)
        score_new = (q * k_new).sum(-1, keepdim=True) / math.sqrt(HD)
        scores = torch.cat([scores_old, score_new], dim=-1)
        prob = torch.softmax(scores.float(), dim=-1).to(x.dtype)

        ctx_old = torch.einsum("bhl,bhld->bhd", prob[..., :-1], v_cache)
        ctx = ctx_old + prob[..., -1:] * v_new
        ctx = ctx.reshape(B, self.d)
        a = F.linear(ctx, block.attn.out_proj.weight, block.attn.out_proj.bias)
        x = x + a
        x = x + block.ffn(block.n2(x))
        return x

    def forward(self, x, k1, v1, k2, v2):
        x = self._block_step(x, self.blocks[0], k1, v1)
        x = self._block_step(x, self.blocks[1], k2, v2)
        return self.norm(x)


@triton.jit
def temporal_candidate_read(
    Q, K, V, OUT,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HD: tl.constexpr,
    KEEP: tl.constexpr,
):
    """Score R visited temporal candidates and read exact Top-KEEP values.

    One program = one head. R=96 approximates the measured ef=64 HNSW budget
    (~95 distance computations/head/query). This is the GPU compute after/beside
    navigation, not the pointer-chasing implementation itself.
    """
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    d = tl.arange(0, HD)

    q = tl.load(Q + pid * HD + d).to(tl.float32)
    key = tl.load(
        K + (pid * R + r[:, None]) * HD + d[None, :],
        mask=(r[:, None] < R),
        other=0.0,
    ).to(tl.float32)
    score = tl.sum(key * q[None, :], axis=1) / 5.656854249492381
    score = tl.where(r < R, score, -1e20)

    jj = tl.arange(0, KEEP)
    sel_s = tl.full((KEEP,), -1e20, tl.float32)
    sel_i = tl.zeros((KEEP,), tl.int32)
    work = score
    for j in range(KEEP):
        ix = tl.argmax(work, axis=0)
        sv = tl.max(work, axis=0)
        sel_s = tl.where(jj == j, sv, sel_s)
        sel_i = tl.where(jj == j, ix, sel_i)
        work = tl.where(r == ix, -1e20, work)

    p = tl.exp(sel_s - tl.max(sel_s, axis=0))
    p = p / (tl.sum(p, axis=0) + 1e-8)

    ctx = tl.zeros((HD,), tl.float32)
    for j in range(KEEP):
        ix = tl.sum(tl.where(jj == j, sel_i, 0), axis=0)
        w = tl.sum(tl.where(jj == j, p, 0.0), axis=0)
        val = tl.load(V + (pid * R + ix) * HD + d).to(tl.float32)
        ctx += w * val
    tl.store(OUT + pid * HD + d, ctx)


def make_sasrec(device, max_len=2048):
    # Same geometry as the canonical ML-1M SASRec baseline.
    return SASRec(
        n_items=4096,
        max_len=max_len,
        d=64,
        layers=2,
        heads=2,
        inner=256,
        dropout=0.0,
        ligr=False,
    ).to(device=device, dtype=torch.bfloat16).eval()


def bench_sasrec_full(device, lengths):
    rows = []
    for L in lengths:
        model = make_sasrec(device, max_len=L)
        seq = torch.randint(1, 4097, (1, L), device=device)
        fn = lambda: model.encode(seq)[:, -1]
        us = bench_cuda(fn, warmup=10, iters=50 if L >= 1024 else 100)
        rows.append({"history": L, "latency_us": us})
        del model, seq
        torch.cuda.empty_cache()
    return rows


def bench_sasrec_incremental(device, lengths, compile_model=True):
    sas = make_sasrec(device, max_len=max(lengths) + 1)
    inc = IncrementalSASRec(sas).to(device=device, dtype=torch.bfloat16).eval()
    compiled = False
    if compile_model and hasattr(torch, "compile"):
        try:
            inc = torch.compile(inc, mode="reduce-overhead", fullgraph=False)
            compiled = True
        except Exception as e:
            print("TORCH_COMPILE_INCREMENTAL_SKIPPED", repr(e), flush=True)

    rows = []
    B, H, D, HD = 1, 2, 64, 32
    x = torch.randn(B, D, device=device, dtype=torch.bfloat16)
    for L in lengths:
        caches = [
            torch.randn(B, H, L, HD, device=device, dtype=torch.bfloat16)
            for _ in range(4)
        ]
        fn = lambda: inc(x, *caches)
        us = bench_cuda(fn, warmup=20, iters=200)
        rows.append({"history": L, "latency_us": us, "torch_compile": compiled})
        del caches
    return rows


def make_walker_buffers(device, n_items=4096):
    D, H, S, K, DEG = 64, 16, 256, 8, 4
    C = S * S
    item = torch.tensor([17], device=device, dtype=torch.int32)
    emb = torch.randn(n_items + 1, D, device=device, dtype=torch.bfloat16) * 0.02
    qw = torch.randn(3 * H, D, device=device, dtype=torch.bfloat16) * 0.02
    lk = torch.randn(S, H, device=device, dtype=torch.bfloat16)
    rk = torch.randn(S, H, device=device, dtype=torch.bfloat16)
    fi = torch.empty(4, device=device, dtype=torch.int32)
    fm = torch.empty(4, device=device, dtype=torch.float32)
    q = torch.empty(H, device=device, dtype=torch.float32)

    sid = torch.randint(0, C, (K,), device=device, dtype=torch.int32)
    sm = torch.rand(K, device=device, dtype=torch.float32)
    sm /= sm.sum()
    dest = torch.randint(0, C, (C * DEG,), device=device, dtype=torch.int32)
    edge = torch.randn(C * DEG, device=device, dtype=torch.bfloat16)
    dk = torch.randn(C, H, device=device, dtype=torch.bfloat16)
    ni = torch.empty(K, device=device, dtype=torch.int32)
    nm = torch.empty(K, device=device, dtype=torch.float32)

    cv = torch.randn(C, D, device=device, dtype=torch.bfloat16) * 0.02
    mw = torch.randn(D, D, device=device, dtype=torch.bfloat16) * 0.02
    nw = torch.ones(D, device=device, dtype=torch.bfloat16)
    nb = torch.zeros(D, device=device, dtype=torch.bfloat16)
    hid = torch.empty(D, device=device, dtype=torch.float32)

    return dict(
        D=D, H=H, S=S, K=K, DEG=DEG, C=C,
        item=item, emb=emb, qw=qw, lk=lk, rk=rk, fi=fi, fm=fm, q=q,
        sid=sid, sm=sm, dest=dest, edge=edge, dk=dk, ni=ni, nm=nm,
        cv=cv, mw=mw, nw=nw, nb=nb, hid=hid,
    )


def launch_walker_local(b):
    walker_route[(1,)](
        b["item"], b["emb"], b["qw"], b["lk"], b["rk"],
        b["fi"], b["fm"], b["q"], D=b["D"], H=b["H"], S=b["S"],
    )
    walker_walk[(1,)](
        b["sid"], b["sm"], b["fi"], b["fm"], b["q"],
        b["dest"], b["edge"], b["dk"], b["ni"], b["nm"],
        K=b["K"], DEG=b["DEG"], H=b["H"],
    )
    walker_readout[(1,)](
        b["item"], b["ni"], b["nm"], b["cv"], b["mw"], b["nw"], b["nb"],
        b["emb"], b["hid"], K=b["K"], D=b["D"],
    )


def bench_walker_local(device):
    b = make_walker_buffers(device)
    us = bench_cuda(lambda: launch_walker_local(b), warmup=50, iters=500)
    return us, b


def bench_temporal_sparse_proxy(device, R=96, layers=2):
    H, HD, KEEP = 2, 32, 16
    q = torch.randn(H, HD, device=device, dtype=torch.bfloat16)
    keys = [torch.randn(H, R, HD, device=device, dtype=torch.bfloat16) for _ in range(layers)]
    vals = [torch.randn(H, R, HD, device=device, dtype=torch.bfloat16) for _ in range(layers)]
    outs = [torch.empty(H, HD, device=device, dtype=torch.float32) for _ in range(layers)]

    def fn():
        for i in range(layers):
            temporal_candidate_read[(H,)](
                q, keys[i], vals[i], outs[i],
                R=R, BLOCK_R=128, HD=HD, KEEP=KEEP,
            )
    return bench_cuda(fn, warmup=50, iters=500)


def bench_dense_catalog(device, catalog_sizes):
    rows = []
    h = torch.randn(1, 64, device=device, dtype=torch.bfloat16)
    for n in catalog_sizes:
        emb = torch.randn(n, 64, device=device, dtype=torch.bfloat16)
        fn = lambda: torch.topk(h @ emb.T, k=10, dim=-1)
        us = bench_cuda(fn, warmup=20, iters=100 if n >= 100_000 else 300)
        rows.append({"catalog": n, "latency_us": us})
        del emb
        torch.cuda.empty_cache()
    return rows


def bench_walker_terminal(device, b, catalog=1_000_000, terminal_degree=64):
    K, D, C = b["K"], b["D"], b["C"]
    DEG = terminal_degree
    TOTAL = K * DEG
    KEEP_BLOCK = 16
    blocks = (TOTAL + 127) // 128
    support = torch.randint(1, catalog, (C * DEG,), device=device, dtype=torch.int32)
    emb = torch.randn(catalog, D, device=device, dtype=torch.bfloat16)
    bi = torch.empty(blocks * KEEP_BLOCK, device=device, dtype=torch.int32)
    bs = torch.empty(blocks * KEEP_BLOCK, device=device, dtype=torch.float32)
    out = torch.empty(10, device=device, dtype=torch.int32)

    def fn():
        term_block[(blocks,)](
            b["hid"], b["ni"], support, emb, bi, bs,
            DEG=DEG, D=D, TOTAL=TOTAL, KEEP=KEEP_BLOCK,
        )
        term_merge[(1,)](
            bi, bs, out, COUNT=blocks * KEEP_BLOCK, BLOCK=64,
        )
    return bench_cuda(fn, warmup=50, iters=300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/content/drive/MyDrive/sparsewalker_speed/serving_speed.json")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")
    seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print("DEVICE", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported(), flush=True)
    scope = {
        "batch": 1,
        "d_model": 64,
        "sasrec_layers": 2,
        "sasrec_heads": 2,
        "walker_K": 8,
        "walker_degree": 4,
        "walker_graph_hops": 2,
        "temporal_layers": 2,
        "temporal_topk": 16,
        "temporal_candidate_reads_per_head": 96,
    }
    print("BENCHMARK_SCOPE", scope, flush=True)

    full_lengths = [200, 512, 1024, 2048]
    long_lengths = [200, 512, 1024, 2048, 4096, 8192, 16384]

    sas_full = bench_sasrec_full(device, full_lengths)
    print("SASREC_FULL_PREFIX", sas_full, flush=True)

    sas_kv = bench_sasrec_incremental(device, long_lengths, compile_model=not args.no_compile)
    print("SASREC_INCREMENTAL_KV", sas_kv, flush=True)

    walker_local_us, wb = bench_walker_local(device)
    print("WALKER_TRITON_LOCAL_UPDATE_US", walker_local_us, flush=True)

    temporal_proxy_us = bench_temporal_sparse_proxy(device, R=96, layers=2)
    print("WALKER_TEMPORAL_SPARSE_2L_PROXY_US", temporal_proxy_us, flush=True)

    walker_encoder_proxy_us = walker_local_us + temporal_proxy_us
    print("WALKER_ENCODER_PROXY_US", walker_encoder_proxy_us, flush=True)

    catalogs = [3706, 100_000, 1_000_000]
    dense_catalog = bench_dense_catalog(device, catalogs)
    print("SASREC_DENSE_CATALOG", dense_catalog, flush=True)

    walker_terminal_us = bench_walker_terminal(device, wb, catalog=1_000_000, terminal_degree=64)
    print("WALKER_TERMINAL_512_CANDIDATES_US", walker_terminal_us, flush=True)

    result = {
        "device": torch.cuda.get_device_name(0),
        "scope": scope,
        "sasrec_full_prefix_us": sas_full,
        "sasrec_incremental_kv_us": sas_kv,
        "walker_triton_local_update_us": walker_local_us,
        "walker_temporal_sparse_2layer_gpu_proxy_us": temporal_proxy_us,
        "walker_encoder_gpu_proxy_us": walker_encoder_proxy_us,
        "sasrec_dense_catalog_us": dense_catalog,
        "walker_terminal_512_candidate_us": walker_terminal_us,
        "caveats": [
            "Walker temporal number is a fused GPU compute proxy over 96 visited candidates/head; it excludes actual graph-index pointer chasing.",
            "Existing FAISS HNSW prototype is CPU/per-user and is intentionally not treated as production latency.",
            "SASRec KV benchmark measures an exact one-token two-layer attention step over cached history.",
            "Final Walker retrieval scores 8*64=512 reachable items; SASRec dense retrieval scores the entire catalog.",
            "All numbers are batch=1 kernel/compute latency on the reported GPU, not service p99.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
