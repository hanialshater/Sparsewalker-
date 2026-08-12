#!/usr/bin/env python
"""Realistic apples-to-apples A100 serving benchmark.

Steady-state stateful online request contract
---------------------------------------------
Both systems are allowed persistent per-user state.

SASRec:
  new item -> 2-layer incremental causal update over a fixed-size KV ring
           -> either:
                (a) model-only hidden state
                (b) FREE-ANN-512 lower bound: score/top10 among 512 pre-retrieved items
                (c) exact dense catalog top10 reference
  Important: the FREE-ANN-512 path intentionally gives SASRec candidate discovery
  for free. It is therefore an optimistic lower bound on practical SASRec E2E latency.

SparseWalker:
  new item -> Triton route + 2-hop local concept walk + readout
           -> temporal block 1: QKV + real Triton SWG search/read + FFN
           -> temporal block 2: QKV + real Triton SWG search/read + FFN
           -> sparse terminal retrieval/scoring over K*degree reachable products
           -> top10
  There is NO full-catalog dense matmul in the Walker E2E path.

For both systems, historical serving state is preallocated and persistent. The timed
request includes K/V writes for the new event. Temporal SWG graph-link maintenance
(index insertion / neighbor selection) is not in the query critical path and is
reported as an explicit caveat.

Two execution modes are measured:
  - eager_same_harness: direct steady-state execution, same CUDA-event timing harness.
  - cuda_graph: the entire fixed-shape request path is captured and replayed for BOTH
    systems, removing Python/kernel-launch fragmentation symmetrically.

This is a systems microbenchmark, not a quality-equivalent recommender comparison.
The current 2-layer Walker quality was measured with dense full-catalog scoring;
the sparse terminal compiler must still be evaluated/distilled for quality.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

from sparsewalker.models import SASRec
from sparsewalker.serving.walker_triton import (
    route as walker_route,
    walk as walker_walk,
    readout as walker_readout,
    term_block,
    term_merge,
)
from sparsewalker.serving.temporal_swg_triton import _swg_search_read_kernel


def seed_all(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def latency_samples(fn, warmup=40, iters=250):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = np.empty(iters, dtype=np.float64)
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    for i in range(iters):
        a.record()
        fn()
        b.record()
        b.synchronize()
        vals[i] = a.elapsed_time(b) * 1000.0
    return {
        "mean_us": float(vals.mean()),
        "p50_us": float(np.percentile(vals, 50)),
        "p95_us": float(np.percentile(vals, 95)),
        "p99_us": float(np.percentile(vals, 99)),
    }


@torch.inference_mode()
def wall_throughput(fn, iters=2000):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "wall_us_per_request": float(dt * 1e6 / iters),
        "requests_per_second": float(iters / max(dt, 1e-12)),
    }


def capture_cuda_graph(fn):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph, graph.replay


class StatefulSASRec(nn.Module):
    """Steady-state one-token SASRec update with fixed-size persistent KV rings."""

    def __init__(self, sasrec: SASRec, history: int):
        super().__init__()
        if len(sasrec.blocks) != 2:
            raise ValueError("benchmark expects 2 SASRec layers")
        self.sas = sasrec
        self.history = int(history)
        self.d = sasrec.d_model
        self.heads = sasrec.blocks[0].attn.num_heads
        self.hd = self.d // self.heads
        self.register_buffer("k1", torch.randn(1, self.heads, history, self.hd, dtype=torch.bfloat16))
        self.register_buffer("v1", torch.randn(1, self.heads, history, self.hd, dtype=torch.bfloat16))
        self.register_buffer("k2", torch.randn(1, self.heads, history, self.hd, dtype=torch.bfloat16))
        self.register_buffer("v2", torch.randn(1, self.heads, history, self.hd, dtype=torch.bfloat16))
        self.slot = 0

    def _block_step(self, x, block, k_cache, v_cache):
        B = x.shape[0]
        z = block.n1(x)
        qkv = F.linear(z, block.attn.in_proj_weight, block.attn.in_proj_bias)
        q, k_new, v_new = qkv.chunk(3, dim=-1)
        q = q.view(B, self.heads, 1, self.hd)
        k_new = k_new.view(B, self.heads, self.hd)
        v_new = v_new.view(B, self.heads, self.hd)
        k_cache[:, :, self.slot, :].copy_(k_new)
        v_cache[:, :, self.slot, :].copy_(v_new)
        ctx = F.scaled_dot_product_attention(q, k_cache, v_cache, dropout_p=0.0, is_causal=False).reshape(B, self.d)
        a = F.linear(ctx, block.attn.out_proj.weight, block.attn.out_proj.bias)
        x = x + a
        x = x + block.ffn(block.n2(x))
        return x

    def forward(self, item_id, pos_id):
        x = self.sas.item(item_id) + self.sas.pos(pos_id)
        x = self._block_step(x, self.sas.blocks[0], self.k1, self.v1)
        x = self._block_step(x, self.sas.blocks[1], self.k2, self.v2)
        return self.sas.norm(x)


class TemporalMath(nn.Module):
    def __init__(self, d=64, heads=2, ff=256):
        super().__init__()
        self.d = d
        self.heads = heads
        self.hd = d // heads
        self.n1 = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)

    def prepare(self, h):
        qkv = self.in_proj(self.n1(h))
        q, k, v = qkv.chunk(3, -1)
        return q.view(self.heads, self.hd), k.view(self.heads, self.hd), v.view(self.heads, self.hd)

    def finish(self, h, ctx):
        z = h + self.out_proj(ctx.reshape(1, self.d).to(h.dtype))
        return z + self.ff2(F.gelu(self.ff1(self.n2(z))))


def make_sasrec(device, history):
    max_len = max(200, min(history, 10000)) + 1
    return SASRec(
        n_items=4096,
        max_len=max_len,
        d=64,
        layers=2,
        heads=1,
        inner=256,
        dropout=0.1,
        ligr=False,
    ).to(device=device, dtype=torch.bfloat16).eval()


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
    return dict(D=D,H=H,S=S,K=K,DEG=DEG,C=C,item=item,emb=emb,qw=qw,lk=lk,rk=rk,
                fi=fi,fm=fm,q=q,sid=sid,sm=sm,dest=dest,edge=edge,dk=dk,ni=ni,nm=nm,
                cv=cv,mw=mw,nw=nw,nb=nb,hid=hid)


def walker_local(b):
    walker_route[(1,)](b["item"],b["emb"],b["qw"],b["lk"],b["rk"],b["fi"],b["fm"],b["q"],
                       D=b["D"],H=b["H"],S=b["S"],num_warps=4)
    walker_walk[(1,)](b["sid"],b["sm"],b["fi"],b["fm"],b["q"],b["dest"],b["edge"],b["dk"],b["ni"],b["nm"],
                      K=b["K"],DEG=b["DEG"],H=b["H"],num_warps=4)
    walker_readout[(1,)](b["item"],b["ni"],b["nm"],b["cv"],b["mw"],b["nw"],b["nb"],b["emb"],b["hid"],
                         K=b["K"],D=b["D"],num_warps=4)


def make_temporal_state(device, history, layers=2, degree=16, beam=16):
    states = []
    for _ in range(layers):
        k = torch.randn(2, history, 32, device=device, dtype=torch.bfloat16)
        v = torch.randn(2, history, 32, device=device, dtype=torch.bfloat16)
        base = torch.arange(history, device=device, dtype=torch.int32)[:, None]
        jumps = torch.randint(1, max(2, history), (history, degree), device=device, dtype=torch.int32)
        adj = ((base + jumps) % history)[None].expand(2, -1, -1).contiguous()
        recent = torch.arange(max(0, history - beam), history, device=device, dtype=torch.int32)
        if recent.numel() < beam:
            recent = F.pad(recent, (beam - recent.numel(), 0), value=0)
        entry = recent[None].expand(2, -1).contiguous()
        lens = torch.full((2,), history, device=device, dtype=torch.int32)
        ctx = torch.empty(2, 32, device=device, dtype=torch.float32)
        out_ids = torch.empty(2, beam, device=device, dtype=torch.int32)
        states.append(dict(k=k,v=v,adj=adj,entry=entry,lens=lens,ctx=ctx,out_ids=out_ids))
    return states


def launch_swg(q, state, history, *, beam=16, hops=4, degree=16):
    cand = beam + beam * degree
    cand_pad = 1 << (cand - 1).bit_length()
    _swg_search_read_kernel[(2,)](
        q,state["k"],state["v"],state["adj"],state["entry"],state["lens"],state["ctx"],state["out_ids"],
        L=history,HD=32,DEG=degree,BEAM=beam,HOPS=hops,CAND_PAD=cand_pad,
        SCALE=1.0/math.sqrt(32),num_warps=4,
    )


def make_terminal(device, b, catalog, degree=64):
    K,D,C=b["K"],b["D"],b["C"]
    total=K*degree
    keep_block=16
    blocks=(total+127)//128
    support=torch.randint(1,catalog,(C*degree,),device=device,dtype=torch.int32)
    emb=torch.randn(catalog,D,device=device,dtype=torch.bfloat16)
    bi=torch.empty(blocks*keep_block,device=device,dtype=torch.int32)
    bs=torch.empty(blocks*keep_block,device=device,dtype=torch.float32)
    out=torch.empty(10,device=device,dtype=torch.int32)
    return dict(K=K,D=D,degree=degree,total=total,keep=keep_block,blocks=blocks,
                support=support,emb=emb,bi=bi,bs=bs,out=out)


def terminal_sparse(h, b, t):
    term_block[(t["blocks"],)](h.reshape(-1),b["ni"],t["support"],t["emb"],t["bi"],t["bs"],
                               DEG=t["degree"],D=t["D"],TOTAL=t["total"],KEEP=t["keep"],num_warps=4)
    block=triton.next_power_of_2(t["blocks"]*t["keep"])
    term_merge[(1,)](t["bi"],t["bs"],t["out"],COUNT=t["blocks"]*t["keep"],BLOCK=block,num_warps=4)


def state_memory(history, *, d=64, sas_layers=2, walker_layers=2, walker_heads=2, swg_degree=16, beam=16):
    sas_kv=sas_layers*2*history*d*2
    walker_kv=walker_layers*2*history*d*2
    walker_adj=walker_layers*walker_heads*history*swg_degree*4
    walker_entry=walker_layers*walker_heads*beam*4
    walker_local_state=8*(4+4)
    return {
        "sasrec_KV_MiB_per_user":sas_kv/2**20,
        "walker_temporal_KV_MiB_per_user":walker_kv/2**20,
        "walker_temporal_adj_MiB_per_user":walker_adj/2**20,
        "walker_local_plus_entry_KiB_per_user":(walker_local_state+walker_entry)/2**10,
        "note":"global model/item/concept weights excluded; current SWG adjacency is duplicated per head/layer",
    }


def bench_case(device, history, catalog, *, beam=16, hops=4, degree=16, terminal_degree=64):
    sas=make_sasrec(device,history)
    sas_state=StatefulSASRec(sas,history).to(device=device,dtype=torch.bfloat16).eval()
    item_id=torch.tensor([17],device=device,dtype=torch.long)
    pos_id=torch.tensor([min(history-1,sas.max_len-1)],device=device,dtype=torch.long)
    ann512=torch.randn(512,64,device=device,dtype=torch.bfloat16)
    dense_emb=torch.randn(catalog,64,device=device,dtype=torch.bfloat16)

    def sas_model():
        return sas_state(item_id,pos_id)

    def sas_ann512():
        h=sas_state(item_id,pos_id)
        return torch.topk(h@ann512.T,k=10,dim=-1)

    def sas_dense():
        h=sas_state(item_id,pos_id)
        return torch.topk(h@dense_emb.T,k=10,dim=-1)

    wb=make_walker_buffers(device)
    temporal=make_temporal_state(device,history,layers=2,degree=degree,beam=beam)
    tmath=[TemporalMath().to(device=device,dtype=torch.bfloat16).eval(),
           TemporalMath().to(device=device,dtype=torch.bfloat16).eval()]
    terminal=make_terminal(device,wb,catalog,degree=terminal_degree)

    def walker_model():
        walker_local(wb)
        h=wb["hid"].to(torch.bfloat16).reshape(1,64)
        for li in range(2):
            q,k_new,v_new=tmath[li].prepare(h)
            st=temporal[li]
            st["k"][:,0,:].copy_(k_new)
            st["v"][:,0,:].copy_(v_new)
            launch_swg(q,st,history,beam=beam,hops=hops,degree=degree)
            h=tmath[li].finish(h,st["ctx"])
        return h

    def walker_sparse_e2e():
        h=walker_model()
        terminal_sparse(h,wb,terminal)
        return terminal["out"]

    for fn in (sas_model,sas_ann512,sas_dense,walker_model,walker_sparse_e2e):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()

    eager={
        "sasrec_model":latency_samples(sas_model),
        "sasrec_freeANN512":latency_samples(sas_ann512),
        "sasrec_dense_exact":latency_samples(sas_dense,iters=100 if catalog>=1_000_000 else 200),
        "walker_model":latency_samples(walker_model),
        "walker_sparse_terminal":latency_samples(walker_sparse_e2e),
    }
    eager_wall={
        "sasrec_model":wall_throughput(sas_model,1000),
        "sasrec_freeANN512":wall_throughput(sas_ann512,1000),
        "walker_model":wall_throughput(walker_model,1000),
        "walker_sparse_terminal":wall_throughput(walker_sparse_e2e,1000),
    }

    graphs={}
    graph_fns={}
    for name,fn in {
        "sasrec_model":sas_model,
        "sasrec_freeANN512":sas_ann512,
        "sasrec_dense_exact":sas_dense,
        "walker_model":walker_model,
        "walker_sparse_terminal":walker_sparse_e2e,
    }.items():
        try:
            g,replay=capture_cuda_graph(fn)
            graphs[name]=g
            graph_fns[name]=replay
        except Exception as e:
            print("CUDA_GRAPH_SKIP",name,repr(e),flush=True)

    graph_stats={}
    graph_wall={}
    for name,replay in graph_fns.items():
        graph_stats[name]=latency_samples(replay,iters=100 if (name=="sasrec_dense_exact" and catalog>=1_000_000) else 250)
        if name!="sasrec_dense_exact":
            graph_wall[name]=wall_throughput(replay,2000)

    def ratio(a,b,mode="p50_us"):
        if a not in graph_stats or b not in graph_stats:
            return None
        return graph_stats[a][mode]/max(graph_stats[b][mode],1e-12)

    result={
        "history":history,
        "catalog":catalog,
        "canonical_sasrec_geometry":{"d":64,"layers":2,"heads":1,"ffn":256,"train_max_len":200},
        "walker_geometry":{"d":64,"local_K":8,"local_degree":4,"local_hops":2,
                           "temporal_layers":2,"temporal_heads":2,"temporal_beam":beam,
                           "temporal_degree":degree,"temporal_hops":hops,
                           "terminal_candidates":8*terminal_degree},
        "state_memory":state_memory(history,swg_degree=degree,beam=beam),
        "eager_same_harness":eager,
        "eager_wall":eager_wall,
        "cuda_graph":graph_stats,
        "cuda_graph_wall":graph_wall,
        "cuda_graph_p50_speedups":{
            "walker_model_vs_sasrec_model":ratio("sasrec_model","walker_model"),
            "walker_native_sparse_vs_sasrec_FREE_ANN512":ratio("sasrec_freeANN512","walker_sparse_terminal"),
            "walker_native_sparse_vs_sasrec_dense_exact":ratio("sasrec_dense_exact","walker_sparse_terminal"),
        },
        "contract":{
            "persistent_user_state_for_both":True,
            "new_event_KV_writes_in_timed_path":True,
            "sasrec_FREE_ANN512_candidate_discovery_cost":"EXCLUDED; optimistic SASRec lower bound",
            "walker_terminal_discovery_and_scoring":"INCLUDED",
            "walker_full_catalog_dense_matmul":False,
            "walker_temporal_graph_link_maintenance":"EXCLUDED from query path",
            "quality_equivalent":False,
            "quality_note":"Two-layer Walker NDCG was measured with dense full-catalog scoring. Sparse terminal support still needs distillation/evaluation on this checkpoint.",
        },
    }
    _=graphs
    return result


def compact(row):
    def p50(section,name):
        return row.get(section,{}).get(name,{}).get("p50_us")
    return {
        "history":row["history"],
        "catalog":row["catalog"],
        "eager_p50_us":{"sas_model":p50("eager_same_harness","sasrec_model"),
                        "sas_freeANN512":p50("eager_same_harness","sasrec_freeANN512"),
                        "walker_model":p50("eager_same_harness","walker_model"),
                        "walker_sparse":p50("eager_same_harness","walker_sparse_terminal")},
        "graph_p50_us":{"sas_model":p50("cuda_graph","sasrec_model"),
                        "sas_freeANN512":p50("cuda_graph","sasrec_freeANN512"),
                        "sas_dense":p50("cuda_graph","sasrec_dense_exact"),
                        "walker_model":p50("cuda_graph","walker_model"),
                        "walker_sparse":p50("cuda_graph","walker_sparse_terminal")},
        "speedups":row["cuda_graph_p50_speedups"],
        "state_memory":row["state_memory"],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="/content/drive/MyDrive/sparsewalker_speed/realistic_apples_serving.json")
    p.add_argument("--lengths",type=int,nargs="+",default=[200,1000,10000])
    p.add_argument("--catalogs",type=int,nargs="+",default=[3706,1000000])
    p.add_argument("--beam",type=int,default=16)
    p.add_argument("--hops",type=int,default=4)
    p.add_argument("--degree",type=int,default=16)
    p.add_argument("--terminal-degree",type=int,default=64)
    args=p.parse_args()

    assert torch.cuda.is_available(),"CUDA required"
    device=torch.device("cuda")
    seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision("high")
    print("DEVICE",torch.cuda.get_device_name(0),"torch",torch.__version__,"bf16",torch.cuda.is_bf16_supported(),flush=True)
    print("FAIR_CONTRACT",{"persistent_state":"both",
                           "sasrec":"canonical d64/l2/h1 incremental SDPA KV-ring",
                           "sasrec_retrieval":"FREE pre-retrieved 512 candidates (optimistic) + dense exact reference",
                           "walker":"Triton local + 2 Triton SWG temporal layers + sparse terminal",
                           "walker_dense_last_stage":False,
                           "execution":"same eager harness + same CUDA-graph replay optimization"},flush=True)

    rows=[]
    for L in args.lengths:
        for C in args.catalogs:
            print("BENCH_START",{"history":L,"catalog":C},flush=True)
            row=bench_case(device,L,C,beam=args.beam,hops=args.hops,degree=args.degree,terminal_degree=args.terminal_degree)
            rows.append(row)
            print("APPLES",json.dumps(compact(row),sort_keys=True),flush=True)

    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"rows":rows},indent=2,sort_keys=True))
    print("SAVED",out,flush=True)


if __name__=="__main__":
    main()
