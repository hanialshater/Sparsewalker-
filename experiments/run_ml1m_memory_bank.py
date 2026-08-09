#!/usr/bin/env python
"""ML-1M hypothesis test: Walker -> dynamic memory bank -> historical state.

This is intentionally a diagnostic, not a canonical baseline. It warm-starts
from the best plain K=8 Walker and tests whether sparse addressable access to
novel historical states fixes the long-history quality plateau.

Success target: val NDCG@10 >= 0.145. Diagnostics test *why*:
- dynamic bank occupancy / novelty writes
- retrieval age and similarity
- quality by effective history length
- counterfactual evaluation with memory reads disabled
"""
import argparse
import hashlib
import inspect
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sparsewalker.data import load_dataset, split_data, WindowDataset, collate_windows
from sparsewalker.evaluation import evaluate_full
from sparsewalker.models import SparseWalker, SparseWalkerMemoryBank
from sparsewalker.models.core import ar_training_loss


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def cpu_state_dict(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def protocol_manifest(max_len,n_items):
    source=inspect.getsource(split_data)+"\n"+inspect.getsource(evaluate_full)
    source_hash=hashlib.sha256(source.encode()).hexdigest()[:16]
    m={
        "protocol_version":"EVAL_CANONICAL_v1_candidate",
        "dataset":"ml1m",
        "split":"per-user leave-two-out: train=s[:-2], val=s[-2], test=s[-1]",
        "catalog":"all mapped item ids 1..n_items",
        "seen_item_masking":True,
        "validation_selection":"best validation full-catalog NDCG@10",
        "metrics":["HR@10","HR@20","HR@50","NDCG@10","NDCG@20","NDCG@50","MRR@10"],
        "max_len":int(max_len),"n_items":int(n_items),"implementation_hash":source_hash,
    }
    m["fingerprint"]=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()[:20]
    return m


def set_lr(opt,epoch,max_epochs,peak=5e-4,min_lr=1e-4,warmup=2):
    if epoch<=warmup: lr=peak*epoch/warmup
    else:
        p=(epoch-warmup)/max(1,max_epochs-warmup)
        lr=min_lr+.5*(peak-min_lr)*(1+math.cos(math.pi*p))
    for g in opt.param_groups: g["lr"]=lr
    return lr


def length_bucket_batches(dataset,batch_size,generator):
    idx=torch.randperm(len(dataset),generator=generator).tolist()
    idx.sort(key=lambda i:min(len(dataset.seqs[i]),dataset.max_len+1))
    batches=[idx[i:i+batch_size] for i in range(0,len(idx),batch_size)]
    order=torch.randperm(len(batches),generator=generator).tolist()
    return [batches[i] for i in order]


def make_loader(ds,batch_size,epoch):
    ds.set_epoch(epoch)
    g=torch.Generator(); g.manual_seed(ds.seed+epoch)
    return DataLoader(
        ds,batch_sampler=length_bucket_batches(ds,batch_size,g),
        collate_fn=collate_windows,pin_memory=True,
    )


def train_epoch_live(model,ds,opt,device,batch_size,epoch):
    loader=make_loader(ds,batch_size,epoch)
    model.train(); total=0.; positions=0; padded=0
    t0=time.perf_counter(); window=time.perf_counter()
    print(f"EPOCH {epoch} START batches={len(loader)}",flush=True)
    for bi,(tokens,lengths) in enumerate(loader,1):
        positions += int((lengths-1).clamp_min(0).sum())
        padded += int(tokens.size(0)*max(0,tokens.size(1)-1))
        tokens=tokens.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss=ar_training_loss(model,tokens,loss_mode="full")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        total += float(loss.detach())
        if bi==1 or bi%5==0 or bi==len(loader):
            torch.cuda.synchronize(); now=time.perf_counter()
            print({"epoch":epoch,"batch":bi,"of":len(loader),"batch_window_s":round(now-window,2),
                   "elapsed_s":round(now-t0,2),"avg_loss":round(total/bi,4),"batch_L":int(tokens.size(1))},flush=True)
            window=now
    torch.cuda.synchronize(); secs=time.perf_counter()-t0
    return {"loss":total/max(1,len(loader)),"seconds":secs,
            "positions_per_s":positions/max(secs,1e-9),"padding_efficiency":positions/max(1,padded)}


def profile_one_batch(cls,n_items,device,name,**kwargs):
    torch.manual_seed(123)
    model=cls(n_items,200,d=64,layers=2,side=256,h=16,active=8,top_side=2,degree=4,**kwargs).to(device).train()
    tokens=torch.randint(1,n_items+1,(128,201),device=device)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        _=model.encode(tokens[:4,:20])
    torch.cuda.synchronize(); model.zero_grad(set_to_none=True); t=time.perf_counter()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        loss=ar_training_loss(model,tokens,loss_mode="full")
    torch.cuda.synchronize(); forward=time.perf_counter()-t
    t=time.perf_counter(); loss.backward(); torch.cuda.synchronize(); backward=time.perf_counter()-t
    total=forward+backward
    print("SPEED_GATE",name,{"forward_s":round(forward,3),"backward_s":round(backward,3),"total_s":round(total,3)},flush=True)
    del model,tokens,loss; torch.cuda.empty_cache()
    return total


def bucket_eval(model,prefix,target,n_items,max_len,device,batch_size):
    buckets={"short_<=50":[],"medium_51_100":[],"long_101_200":[]}
    for i,p in enumerate(prefix):
        L=min(len(p),max_len)
        key="short_<=50" if L<=50 else ("medium_51_100" if L<=100 else "long_101_200")
        buckets[key].append(i)
    out={}
    for key,idx in buckets.items():
        if not idx: continue
        pp=[prefix[i] for i in idx]; tt=[target[i] for i in idx]
        r=evaluate_full(model,pp,tt,n_items,max_len,device,topks=(10,),batch_size=batch_size)
        out[key]={"n":len(idx),"NDCG@10":float(r["NDCG@10"]),"HR@10":float(r["HR@10"]),"MRR@10":float(r["MRR@10"])}
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--data-dir",default="/content/sparsewalker_data")
    p.add_argument("--base-root",default="/content/drive/MyDrive/sparsewalker_canonical_pair")
    p.add_argument("--output-dir",default="/content/drive/MyDrive/sparsewalker_memory_bank")
    p.add_argument("--max-epochs",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--eval-batch-size",type=int,default=1024)
    p.add_argument("--success-ndcg",type=float,default=.145)
    args=p.parse_args()

    seed_all(args.seed); device=torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32=True; torch.set_float32_matmul_precision("high")
    print("device",device,"GPU",torch.cuda.get_device_name(0),"bf16",torch.cuda.is_bf16_supported(),flush=True)

    # Fail fast on systems cost before loading the dataset.
    base_t=profile_one_batch(SparseWalker,3706,device,"BASE")
    bank_t=profile_one_batch(
        SparseWalkerMemoryBank,3706,device,"MEMORY_BANK",
        bank_size=16,memory_topk=2,novelty_threshold=.85,min_write_gap=8,initial_memory_share=.10,
    )
    ratio=bank_t/base_t
    print("SPEED_GATE_RATIO",round(ratio,3),flush=True)
    if ratio>2.5:
        raise RuntimeError(f"Memory bank is {ratio:.2f}x slower than base; aborting before full run")

    t=time.perf_counter(); data=load_dataset("ml1m",args.data_dir); split=split_data(data["sequences"])
    print("DATA READY",{"seconds":round(time.perf_counter()-t,2),"users":len(data["sequences"]),"n_items":data["n_items"]},flush=True)
    max_len=200; protocol=protocol_manifest(max_len,data["n_items"]); print("PROTOCOL",protocol,flush=True)

    base_path=Path(args.base_root)/"ml1m"/f"seed{args.seed}"/"SparseWalker_FullCE"/"best.pt"
    base_ckpt=torch.load(base_path,map_location="cpu")
    if base_ckpt.get("protocol",{}).get("fingerprint")!=protocol["fingerprint"]:
        raise RuntimeError("Base checkpoint protocol mismatch")
    base_val=float(base_ckpt["val"]["NDCG@10"])
    print("BASE CHECKPOINT",{"epoch":int(base_ckpt["epoch"]),"val_NDCG@10":base_val},flush=True)

    model=SparseWalkerMemoryBank(
        data["n_items"],max_len,d=64,layers=2,side=256,h=16,active=8,top_side=2,degree=4,fresh_weight=.25,
        bank_size=16,memory_topk=2,novelty_threshold=.85,min_write_gap=8,initial_memory_share=.10,
    ).to(device)
    missing,unexpected=model.load_state_dict(base_ckpt["model"],strict=False)
    expected={"memory_q.weight","memory_share_logit"}
    if set(missing)!=expected or unexpected:
        raise RuntimeError(f"warm-start mismatch missing={missing} unexpected={unexpected}")
    print("MEMORY_CONFIG",{"K":8,"bank_size":16,"topk_states":2,"novelty_threshold":.85,
          "min_write_gap":8,"initial_memory_share":.10,"new_params":sorted(missing)},flush=True)

    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    ds=WindowDataset(split["train"],max_len,args.seed)
    out=Path(args.output_dir)/"ml1m"/f"seed{args.seed}"; out.mkdir(parents=True,exist_ok=True)
    history=[]; best_ndcg=-1.; best_epoch=0; best_state=None

    # Epoch-0 tells us whether adding a random router damages the warm start.
    model.eval(); t=time.perf_counter()
    e0=evaluate_full(model,split["val_prefix"],split["val_target"],data["n_items"],max_len,device,topks=(10,),batch_size=args.eval_batch_size)
    print("EPOCH0 EVAL",{"NDCG@10":float(e0["NDCG@10"]),"base":base_val,"delta":float(e0["NDCG@10"])-base_val,
                        "seconds":round(time.perf_counter()-t,2)},flush=True)

    for epoch in range(1,args.max_epochs+1):
        lr=set_lr(opt,epoch,args.max_epochs)
        stats=train_epoch_live(model,ds,opt,device,args.batch_size,epoch)
        row={"epoch":epoch,"lr":lr,**stats,"memory_share":float(torch.sigmoid(model.memory_share_logit).detach().cpu())}
        print("TRAIN DONE",{k:round(v,6) if isinstance(v,float) else v for k,v in row.items()},flush=True)
        model.eval(); t=time.perf_counter()
        val=evaluate_full(model,split["val_prefix"],split["val_target"],data["n_items"],max_len,device,topks=(10,),batch_size=args.eval_batch_size)
        ndcg=float(val["NDCG@10"]); row.update(val); history.append(row)
        pd.DataFrame(history).to_csv(out/"history.csv",index=False)
        print("MEMORY EVAL",{"epoch":epoch,"NDCG@10":ndcg,"gain_vs_base_pct":100*(ndcg/base_val-1),
              "memory_share":row["memory_share"],"eval_seconds":round(time.perf_counter()-t,2)},flush=True)
        if ndcg>best_ndcg:
            best_ndcg=ndcg; best_epoch=epoch; best_state=cpu_state_dict(model)
            torch.save({"model":best_state,"epoch":epoch,"val":val,"protocol":protocol},out/"best.pt")
        # This is a falsification experiment, not an endless tune.
        if epoch>=6 and best_ndcg<base_val+.004:
            print("KILL_CRITERION",{"reason":"< +0.004 absolute after 6 epochs","best":best_ndcg,"base":base_val},flush=True)
            break
        if best_ndcg>=args.success_ndcg:
            print("SUCCESS_CRITERION",{"best":best_ndcg,"target":args.success_ndcg},flush=True)
            break

    if best_state is None: raise RuntimeError("No memory-bank checkpoint")
    model.load_state_dict(best_state); model.eval()

    # Counterfactual: same trained weights, but no memory jump.
    full_on=evaluate_full(model,split["val_prefix"],split["val_target"],data["n_items"],max_len,device,topks=(10,),batch_size=args.eval_batch_size)
    model.memory_enabled=False
    full_off=evaluate_full(model,split["val_prefix"],split["val_target"],data["n_items"],max_len,device,topks=(10,),batch_size=args.eval_batch_size)
    model.memory_enabled=True
    counterfactual={"memory_on":float(full_on["NDCG@10"]),"memory_off":float(full_off["NDCG@10"]),
                    "absolute_memory_contribution":float(full_on["NDCG@10"]-full_off["NDCG@10"])}
    print("COUNTERFACTUAL",counterfactual,flush=True)

    # If the hypothesis is right, gains should be largest on longer histories.
    base=SparseWalker(data["n_items"],max_len,d=64,layers=2,side=256,h=16,active=8,top_side=2,degree=4,fresh_weight=.25).to(device)
    base.load_state_dict(base_ckpt["model"]); base.eval()
    by_len_base=bucket_eval(base,split["val_prefix"],split["val_target"],data["n_items"],max_len,device,args.eval_batch_size)
    by_len_mem=bucket_eval(model,split["val_prefix"],split["val_target"],data["n_items"],max_len,device,args.eval_batch_size)
    by_len={}
    for key in by_len_base:
        by_len[key]={"n":by_len_base[key]["n"],"base_NDCG@10":by_len_base[key]["NDCG@10"],
                     "memory_NDCG@10":by_len_mem[key]["NDCG@10"],
                     "delta":by_len_mem[key]["NDCG@10"]-by_len_base[key]["NDCG@10"]}
    print("HISTORY_LENGTH_DIAGNOSTIC",json.dumps(by_len,indent=2),flush=True)
    del base; torch.cuda.empty_cache()

    # Routing diagnostics on long-history users only, capped to 256 users.
    long_idx=[i for i,pf in enumerate(split["val_prefix"]) if min(len(pf),max_len)>100][:256]
    diag_prefix=[split["val_prefix"][i] for i in long_idx]; diag_target=[split["val_target"][i] for i in long_idx]
    model.enable_diagnostics(True)
    _=evaluate_full(model,diag_prefix,diag_target,data["n_items"],max_len,device,topks=(10,),batch_size=256)
    routing=model.diagnostics_summary(); model.enable_diagnostics(False)
    print("ROUTING_DIAGNOSTIC",json.dumps(routing,indent=2),flush=True)

    test=evaluate_full(model,split["test_prefix"],split["test_target"],data["n_items"],max_len,device,topks=(10,20,50),batch_size=args.eval_batch_size)
    result={"cell":"SparseWalker+MemoryBank","selected_epoch":best_epoch,"base_val_NDCG@10":base_val,
            "best_val_NDCG@10":best_ndcg,"protocol_fingerprint":protocol["fingerprint"],"speed_ratio":ratio,
            "counterfactual":counterfactual,"history_length":by_len,"routing":routing,**test}
    (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True))
    print("FINAL RESULT",json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
