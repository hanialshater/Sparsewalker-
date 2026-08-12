#!/usr/bin/env python
"""Final quality-equivalent serving audit for current ML-1M checkpoints.

This benchmark deliberately removes the invalid persistent-cache shortcut from BOTH
models after max_len=200 saturation.

Current trained semantics:
- SASRec: canonical evaluator uses the last <=200 items, learned absolute positions,
  and recomputes the 200-token Transformer window.
- SparseWalker+2Temporal: canonical evaluator also truncates to <=200, resets the
  local Walker at the window start, then applies two temporal blocks with learned
  absolute positions.

Therefore the latency headline here is the exact 200-window recompute path for both.
The fast persistent KV/SWG paths are NOT used for the headline because they are not
quality-equivalent after the window slides.

Serving output:
- SASRec gets 512 candidate embeddings for FREE and only pays score+top10.
- Walker uses native sparse terminal discovery/scoring over 8*64=512 reachable items.
  Terminal support quality is not yet distilled/evaluated for the 2-temporal checkpoint,
  so Walker NDCG below is the dense-score quality oracle for the core representation.

Both latency paths use BF16 inference, torch.compile when available, and CUDA Graph
capture/replay under the same CUDA-event timing harness. FP32/BF16 quality is reported
separately; training-precision provenance is printed as a fairness caveat.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sparsewalker.data import load_dataset, split_data
from sparsewalker.evaluation import evaluate_full
from sparsewalker.models import SASRec
from sparsewalker.serving.walker_triton import term_block, term_merge


@torch.inference_mode()
def latency_samples(fn, warm=40, n=250):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    vals=np.empty(n,dtype=np.float64)
    a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
    for i in range(n):
        a.record(); fn(); b.record(); b.synchronize(); vals[i]=a.elapsed_time(b)*1000.0
    return {"mean_us":float(vals.mean()),"p50_us":float(np.percentile(vals,50)),
            "p95_us":float(np.percentile(vals,95)),"p99_us":float(np.percentile(vals,99))}


def capture_cuda_graph(fn):
    for _ in range(30): fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize()
    return g,g.replay


class SASExactWindow(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model=model
    def forward(self, seq, lens):
        return self.model.last_hidden(seq,lens)


class WalkerExactWindow(nn.Module):
    """Return exact canonical 2-temporal hidden plus final local sparse concept IDs/mass."""
    def __init__(self, model):
        super().__init__(); self.model=model
    def forward(self, seq, lens):
        # _encode_impl is the inherited local Walker recurrence only; applying the
        # two temporal blocks below exactly reproduces the subclass encode path.
        H,I,M = self.model._encode_impl(seq,True)
        H = self.model.temporal1(H,seq.eq(0))
        H = self.model.temporal2(H,seq.eq(0))
        rows=torch.arange(seq.size(0),device=seq.device)
        last=(lens-1).clamp_min(0)
        return H[rows,last],I[rows,last],M[rows,last]


def make_terminal(model,device,catalog,degree=64):
    K=8; D=64; C=256*256; total=K*degree; keep=16; blocks=(total+127)//128
    support=torch.randint(1,catalog+1,(C*degree,),device=device,dtype=torch.int32)
    # Terminal kernels use 1-based item IDs, so keep an explicit row 0.
    emb=torch.zeros(catalog+1,D,device=device,dtype=torch.bfloat16)
    if catalog<=model.n_items:
        emb[1:catalog+1].copy_(model.item_weight[1:catalog+1].detach().to(torch.bfloat16))
    else:
        emb[1:].normal_(0.0,.02)
        emb[1:model.n_items+1].copy_(model.item_weight[1:model.n_items+1].detach().to(torch.bfloat16))
    bi=torch.empty(blocks*keep,device=device,dtype=torch.int32)
    bs=torch.empty(blocks*keep,device=device,dtype=torch.float32)
    out=torch.empty(10,device=device,dtype=torch.int32)
    return dict(K=K,D=D,degree=degree,total=total,keep=keep,blocks=blocks,
                support=support,emb=emb,bi=bi,bs=bs,out=out)


def terminal_sparse(h,ids,t):
    term_block[(t["blocks"],)](h.reshape(-1),ids.to(torch.int32),t["support"],t["emb"],t["bi"],t["bs"],
        DEG=t["degree"],D=t["D"],TOTAL=t["total"],KEEP=t["keep"],num_warps=4)
    block=1 << ((t["blocks"]*t["keep"]-1).bit_length())
    term_merge[(1,)](t["bi"],t["bs"],t["out"],COUNT=t["blocks"]*t["keep"],BLOCK=block,num_warps=4)
    return t["out"]


def build_sas(n_items,device,ckpt):
    m=SASRec(n_items,200,d=64,layers=2,heads=1,inner=256,dropout=.1).to(device).eval()
    m.load_state_dict(ckpt["model"]); return m


def build_walker(n_items,device,ckpt):
    exp="/content/Sparsewalker/experiments"
    if exp not in sys.path: sys.path.insert(0,exp)
    from run_ml1m_walker_two_temporal_layers import SparseWalkerTwoTemporal
    m=SparseWalkerTwoTemporal(n_items,200,d=64,layers=2,side=256,h=16,active=8,top_side=2,
        degree=4,fresh_weight=.25,attn_heads=2,ff_mult=4,dropout=.1).to(device).eval()
    m.load_state_dict(ckpt["model"]); m.temporal_depth=2; return m


def quality(model,split,n_items,device,bf16=False):
    kw={"autocast_dtype":torch.bfloat16} if bf16 else {}
    val=evaluate_full(model,split["val_prefix"],split["val_target"],n_items,200,device,
                      topks=(10,),batch_size=1024,**kw)
    test=evaluate_full(model,split["test_prefix"],split["test_target"],n_items,200,device,
                       topks=(10,),batch_size=1024,**kw)
    return {"val":{k:float(v) for k,v in val.items()},"test":{k:float(v) for k,v in test.items()}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="/content/sparsewalker_data")
    p.add_argument("--sasrec-checkpoint",default="/content/drive/MyDrive/sparsewalker_esasrec_2x2/ml1m/seed42/SASRec_FullCE/best.pt")
    p.add_argument("--walker-checkpoint",default="/content/drive/MyDrive/sparsewalker_two_temporal_layers/ml1m/seed42/best.pt")
    p.add_argument("--catalog",type=int,default=1_000_000)
    p.add_argument("--output",default="/content/drive/MyDrive/sparsewalker_speed/final_quality_equivalent_serving.json")
    args=p.parse_args()

    assert torch.cuda.is_available(); device=torch.device("cuda")
    torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32=True; torch.set_float32_matmul_precision("high")
    data=load_dataset("ml1m",args.data_dir); split=split_data(data["sequences"])
    sck=torch.load(args.sasrec_checkpoint,map_location="cpu")
    wck=torch.load(args.walker_checkpoint,map_location="cpu")
    sas32=build_sas(data["n_items"],device,sck); walker32=build_walker(data["n_items"],device,wck)

    fairness={
      "device":torch.cuda.get_device_name(0),"batch":1,"window":200,
      "inference_precision":"BF16 for BOTH timed paths",
      "sasrec_training_precision":"FP32",
      "walker_training_precision":"BF16 autocast",
      "training_precision_strictly_matched":False,
      "sasrec_semantics":"exact canonical last-200 recompute; persistent stale KV excluded",
      "walker_semantics":"exact canonical last-200 local reset/recompute + 2 temporal blocks; persistent SWG shortcut excluded from headline",
      "sasrec_retrieval":"FREE candidate discovery; score/top10 over 512 candidates included",
      "walker_retrieval":"native sparse terminal discovery+score over 512 reachable items included",
      "walker_terminal_quality_validated":False,
    }
    print("FAIRNESS",json.dumps(fairness,indent=2),flush=True)

    # Quality oracle: actual checkpoints; dense item scoring only here, not timed Walker path.
    sas_q32=quality(sas32,split,data["n_items"],device,False)
    sas_q16=quality(sas32,split,data["n_items"],device,True)
    wal_q32=quality(walker32,split,data["n_items"],device,False)
    wal_q16=quality(walker32,split,data["n_items"],device,True)
    q={"sasrec":{"epoch":int(sck.get("epoch",-1)),"FP32":sas_q32,"BF16":sas_q16},
       "walker":{"epoch":int(wck.get("epoch",-1)),"FP32":wal_q32,"BF16":wal_q16}}
    print("QUALITY",json.dumps(q,indent=2),flush=True)

    # Timed models are the same checkpoints cast to BF16.
    sas=build_sas(data["n_items"],device,sck).to(torch.bfloat16).eval()
    walker=build_walker(data["n_items"],device,wck).to(torch.bfloat16).eval()
    swrap=SASExactWindow(sas).eval(); wwrap=WalkerExactWindow(walker).eval()

    seq=torch.randint(1,data["n_items"]+1,(1,200),device=device,dtype=torch.long)
    lens=torch.tensor([200],device=device,dtype=torch.long)
    new_item=torch.tensor([17],device=device,dtype=torch.long)
    shift_scratch=torch.empty((1,199),device=device,dtype=torch.long)
    cand_ids=torch.randint(1,data["n_items"]+1,(512,),device=device,dtype=torch.long)
    cand_emb=sas.item_weight[cand_ids].detach().contiguous()
    terminal=make_terminal(walker,device,args.catalog,degree=64)

    # Compile exact recompute kernels first; fallback to eager if backend refuses a graph.
    srun=swrap; wrun=wwrap; compile_status={}
    try:
        srun=torch.compile(swrap,mode="reduce-overhead",fullgraph=False); srun(seq,lens); torch.cuda.synchronize(); compile_status["sasrec"]=True
    except Exception as e:
        compile_status["sasrec"]=False; compile_status["sasrec_error"]=repr(e)
    try:
        wrun=torch.compile(wwrap,mode="reduce-overhead",fullgraph=False); wrun(seq,lens); torch.cuda.synchronize(); compile_status["walker"]=True
    except Exception as e:
        compile_status["walker"]=False; compile_status["walker_error"]=repr(e)
    print("COMPILE",compile_status,flush=True)

    def shift_seq():
        # Fixed buffers only, so the request-state update is CUDA-Graph safe.
        shift_scratch.copy_(seq[:,1:]); seq[:,:-1].copy_(shift_scratch); seq[:,-1].copy_(new_item)

    def sas_model():
        shift_seq(); return srun(seq,lens)
    def sas_free_ann512():
        h=sas_model(); return torch.topk(h@cand_emb.T,k=10,dim=-1)
    def walker_model():
        # Canonical window reset/recompute already consumes current 200 IDs.
        return wrun(seq,lens)
    def walker_sparse():
        h,ids,mass=walker_model(); terminal_sparse(h,ids,terminal); return terminal["out"]

    # Validate compiled outputs against eager exact implementations before timing.
    with torch.inference_mode():
        hs0=swrap(seq,lens).float(); hs1=srun(seq,lens).float()
        hw0,ii0,mm0=wwrap(seq,lens); hw1,ii1,mm1=wrun(seq,lens)
        correctness={"sasrec_max_abs_hidden":float((hs0-hs1).abs().max().cpu()),
                     "walker_max_abs_hidden":float((hw0.float()-hw1.float()).abs().max().cpu()),
                     "walker_final_ids_match":bool(torch.equal(ii0,ii1))}
    print("COMPILED_CORRECTNESS",correctness,flush=True)

    eager={"sasrec_model":latency_samples(sas_model,n=120),
           "sasrec_freeANN512":latency_samples(sas_free_ann512,n=120),
           "walker_model":latency_samples(walker_model,n=80),
           "walker_sparse_terminal":latency_samples(walker_sparse,n=80)}
    print("EAGER",json.dumps(eager,indent=2),flush=True)

    graph={}; holders=[]
    for name,fn,iters in [("sasrec_model",sas_model,180),("sasrec_freeANN512",sas_free_ann512,180),
                          ("walker_model",walker_model,120),("walker_sparse_terminal",walker_sparse,120)]:
        try:
            g,replay=capture_cuda_graph(fn); holders.append(g); graph[name]=latency_samples(replay,n=iters)
        except Exception as e:
            graph[name]={"error":repr(e)}
    print("CUDA_GRAPH",json.dumps(graph,indent=2),flush=True)

    def p50(name):
        return graph.get(name,{}).get("p50_us")
    sas_p=p50("sasrec_freeANN512"); wal_p=p50("walker_sparse_terminal")
    headline={"sasrec_exact_window_plus_FREE_ANN512_p50_us":sas_p,
              "walker_exact_window_plus_native_sparse_terminal_p50_us":wal_p,
              "walker_speedup_vs_sasrec":None if not sas_p or not wal_p else sas_p/wal_p,
              "quality_contract":"exact current trained 200-window semantics on both sides",
              "important":"fast persistent KV/SWG numbers are excluded because learned absolute positions + canonical window reset make them non-equivalent after saturation"}
    print("HEADLINE",json.dumps(headline,indent=2),flush=True)

    result={"fairness":fairness,"quality":q,"compile":compile_status,"compiled_correctness":correctness,
            "eager":eager,"cuda_graph":graph,"headline":headline}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True))
    print("SAVED",out,flush=True)

if __name__=="__main__": main()
