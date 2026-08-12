#!/usr/bin/env python
"""End-to-end query-path serving microbenchmark: SparseWalker vs SASRec.

This benchmark measures one new-event request on A100, batch=1.

SASRec path:
  new item embedding + positional embedding
    -> exact 2-layer incremental KV-cache SASRec step
    -> dense catalog matmul + top-10

SparseWalker path:
  Triton route -> 2-hop concept walk -> readout
    -> temporal block #1: QKV projection -> real Triton SWG walk/read -> out-proj+FFN
    -> temporal block #2: QKV projection -> real Triton SWG walk/read -> out-proj+FFN
    -> Triton sparse terminal retrieval -> top-10

Persistent serving state (not rebuilt in the timed region):
- SASRec historical K/V caches
- Walker concept graph / terminal support
- temporal SWG adjacency and historical K/V tensors

The benchmark intentionally excludes temporal graph index construction/insertion from the
request critical path. It measures the query-path architecture, not production p99.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsewalker.models import SASRec
from sparsewalker.serving.walker_triton import (
    route as walker_route,
    walk as walker_walk,
    readout as walker_readout,
    term_block,
    term_merge,
)
from sparsewalker.serving.temporal_swg_triton import temporal_swg_search_read


def seed_all(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bench_cuda(fn, warmup=30, iters=200):
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
        return float(start.elapsed_time(end) * 1000.0 / iters)  # us


class IncrementalSASRec(nn.Module):
    """Exact one-token update for the repo's 2-layer SASRec with persistent KV caches."""
    def __init__(self, sasrec):
        super().__init__()
        self.blocks = sasrec.blocks
        self.norm = sasrec.norm
        self.d = sasrec.d_model

    def _step(self, x, block, k_cache, v_cache):
        B, H, L, HD = k_cache.shape
        z = block.n1(x)
        proj = F.linear(z, block.attn.in_proj_weight, block.attn.in_proj_bias)
        q, k_new, v_new = proj.chunk(3, dim=-1)
        q = q.view(B, H, HD)
        k_new = k_new.view(B, H, HD)
        v_new = v_new.view(B, H, HD)
        score_old = torch.einsum('bhd,bhld->bhl', q, k_cache) / math.sqrt(HD)
        score_new = (q * k_new).sum(-1, keepdim=True) / math.sqrt(HD)
        prob = torch.softmax(torch.cat([score_old, score_new], -1).float(), -1).to(x.dtype)
        ctx = torch.einsum('bhl,bhld->bhd', prob[..., :-1], v_cache)
        ctx = ctx + prob[..., -1:] * v_new
        a = F.linear(ctx.reshape(B, self.d), block.attn.out_proj.weight, block.attn.out_proj.bias)
        x = x + a
        x = x + block.ffn(block.n2(x))
        return x

    def forward(self, x, k1, v1, k2, v2):
        x = self._step(x, self.blocks[0], k1, v1)
        x = self._step(x, self.blocks[1], k2, v2)
        return self.norm(x)


class TemporalMath(nn.Module):
    """Dense math around the sparse SWG read, matching d=64/2-head temporal blocks."""
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
        return (
            q.view(self.heads, self.hd),
            k.view(self.heads, self.hd),
            v.view(self.heads, self.hd),
        )

    def finish(self, h, ctx):
        z = h + self.out_proj(ctx.reshape(1, self.d).to(h.dtype))
        return z + self.ff2(F.gelu(self.ff1(self.n2(z))))


def make_sasrec(device, max_len):
    return SASRec(
        n_items=4096,
        max_len=max_len + 1,
        d=64,
        layers=2,
        heads=1,
        inner=256,
        dropout=0.0,
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
    sm = torch.rand(K, device=device, dtype=torch.float32); sm /= sm.sum()
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
    walker_route[(1,)](b['item'],b['emb'],b['qw'],b['lk'],b['rk'],b['fi'],b['fm'],b['q'],
                       D=b['D'],H=b['H'],S=b['S'])
    walker_walk[(1,)](b['sid'],b['sm'],b['fi'],b['fm'],b['q'],b['dest'],b['edge'],b['dk'],b['ni'],b['nm'],
                      K=b['K'],DEG=b['DEG'],H=b['H'])
    walker_readout[(1,)](b['item'],b['ni'],b['nm'],b['cv'],b['mw'],b['nw'],b['nb'],b['emb'],b['hid'],
                         K=b['K'],D=b['D'])


def make_temporal_state(device, L, layers=2, degree=16, beam=16):
    states=[]
    for _ in range(layers):
        k=torch.randn(2,L,32,device=device,dtype=torch.bfloat16)
        v=torch.randn(2,L,32,device=device,dtype=torch.bfloat16)
        # Persistent small-world adjacency. Latency depends on shape, not graph semantics.
        base=torch.arange(L,device=device,dtype=torch.int32)[:,None]
        jumps=torch.randint(1,max(2,L),(L,degree),device=device,dtype=torch.int32)
        adj=((base+jumps)%L)[None].expand(2,-1,-1).contiguous()
        recent=torch.arange(max(0,L-beam),L,device=device,dtype=torch.int32)
        if recent.numel()<beam:
            recent=F.pad(recent,(beam-recent.numel(),0),value=0)
        entry=recent[None].expand(2,-1).contiguous()
        lens=torch.full((2,),L,device=device,dtype=torch.int32)
        states.append((k,v,adj,entry,lens))
    return states


def make_terminal(device, b, catalog, degree=64):
    K,D,C=b['K'],b['D'],b['C']
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
    walker_h=h.reshape(-1)
    term_block[(t['blocks'],)](walker_h,b['ni'],t['support'],t['emb'],t['bi'],t['bs'],
                               DEG=t['degree'],D=t['D'],TOTAL=t['total'],KEEP=t['keep'])
    term_merge[(1,)](t['bi'],t['bs'],t['out'],COUNT=t['blocks']*t['keep'],BLOCK=64)


def make_compiled_temporal_math(device):
    layers=[]
    for _ in range(2):
        m=TemporalMath().to(device=device,dtype=torch.bfloat16).eval()
        prep=m.prepare; finish=m.finish
        if hasattr(torch,'compile'):
            try:
                prep=torch.compile(prep,mode='reduce-overhead',fullgraph=False)
                finish=torch.compile(finish,mode='reduce-overhead',fullgraph=False)
            except Exception:
                pass
        layers.append((m,prep,finish))
    return layers


def bench_one(device,L,catalog,beam=16,hops=4):
    # ----- SASRec persistent state -----
    sas=make_sasrec(device,L)
    inc=IncrementalSASRec(sas).to(device=device,dtype=torch.bfloat16).eval()
    if hasattr(torch,'compile'):
        try:
            inc=torch.compile(inc,mode='reduce-overhead',fullgraph=False)
        except Exception:
            pass
    caches=[torch.randn(1,1,L,64,device=device,dtype=torch.bfloat16) for _ in range(4)]
    item_id=torch.tensor([17],device=device,dtype=torch.long)
    pos_id=torch.tensor([min(L,sas.max_len-1)],device=device,dtype=torch.long)
    catalog_emb=torch.randn(catalog,64,device=device,dtype=torch.bfloat16)

    def sas_fn():
        x=sas.item(item_id)+sas.pos(pos_id)
        h=inc(x,*caches)
        return torch.topk(h@catalog_emb.T,k=10,dim=-1)

    # ----- Walker persistent state -----
    wb=make_walker_buffers(device)
    temporal=make_temporal_state(device,L)
    tmath=make_compiled_temporal_math(device)
    terminal=make_terminal(device,wb,catalog)

    def walker_fn():
        walker_local(wb)
        h=wb['hid'].to(torch.bfloat16).reshape(1,64)
        for li in range(2):
            _,prep,finish=tmath[li]
            q,_,_=prep(h)
            k,v,adj,entry,lens=temporal[li]
            ctx,_=temporal_swg_search_read(q,k,v,adj,entry,lens,hops=hops,beam=beam)
            h=finish(h,ctx)
        terminal_sparse(h,wb,terminal)
        return terminal['out']

    # Compile/warm both paths before timed region.
    sas_fn(); walker_fn(); torch.cuda.synchronize()
    iters=80 if catalog>=1_000_000 else 160
    sas_us=bench_cuda(sas_fn,warmup=20,iters=iters)
    walker_us=bench_cuda(walker_fn,warmup=20,iters=iters)
    row={
        'history':L,
        'catalog':catalog,
        'sasrec_e2e_us':sas_us,
        'walker_e2e_us':walker_us,
        'walker_speedup_vs_sasrec':sas_us/max(walker_us,1e-9),
        'walker_temporal_layers':2,
        'walker_temporal_beam':beam,
        'walker_temporal_hops':hops,
        'walker_temporal_degree':16,
        'walker_terminal_candidates':8*64,
    }
    del sas,inc,caches,catalog_emb,wb,temporal,tmath,terminal
    torch.cuda.empty_cache()
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='/content/drive/MyDrive/sparsewalker_speed/e2e_serving_speed.json')
    p.add_argument('--lengths',type=int,nargs='+',default=[200,1000,10000])
    p.add_argument('--catalogs',type=int,nargs='+',default=[3706,100000,1000000])
    p.add_argument('--beam',type=int,default=16)
    p.add_argument('--hops',type=int,default=4)
    args=p.parse_args()

    assert torch.cuda.is_available(), 'CUDA required'
    device=torch.device('cuda')
    seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    print('DEVICE',torch.cuda.get_device_name(0),'bf16',torch.cuda.is_bf16_supported(),flush=True)
    print('SCOPE',{
        'batch':1,
        'sasrec':'2-layer exact incremental KV-cache + dense catalog top10',
        'walker':'Triton local update + 2 real Triton SWG reads + temporal FFNs + Triton sparse terminal top10',
        'persistent_state_excluded_from_timed_region':True,
        'graph_index_maintenance_excluded':True,
    },flush=True)

    rows=[]
    for L in args.lengths:
        for C in args.catalogs:
            print('BENCH_START',{'L':L,'catalog':C},flush=True)
            row=bench_one(device,L,C,beam=args.beam,hops=args.hops)
            rows.append(row)
            print('E2E',row,flush=True)

    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'rows':rows},indent=2,sort_keys=True))
    print('SAVED',out,flush=True)


if __name__=='__main__':
    main()
