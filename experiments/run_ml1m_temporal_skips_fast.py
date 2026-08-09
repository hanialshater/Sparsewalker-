#!/usr/bin/env python
"""Fast ML-1M temporal-skip diagnostic.

Unlike the first diagnostic runner, this does NOT re-evaluate the plain Walker
before training. The baseline validation metric is read from its checkpoint.
This keeps startup cheap and isolates the temporal-memory experiment.
"""
import argparse, hashlib, inspect, json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full
from sparsewalker.models import SparseWalkerTemporalMemory
from sparsewalker.training import train_epoch


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


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--data-dir",default="/content/sparsewalker_data")
    p.add_argument("--base-root",default="/content/drive/MyDrive/sparsewalker_canonical_pair")
    p.add_argument("--output-dir",default="/content/drive/MyDrive/sparsewalker_temporal_skips_v2")
    p.add_argument("--max-epochs",type=int,default=12)
    p.add_argument("--eval-every",type=int,default=2)
    p.add_argument("--patience",type=int,default=6)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--eval-batch-size",type=int,default=1024)
    args=p.parse_args()

    seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device",device,"GPU",torch.cuda.get_device_name(0) if device.type=="cuda" else None,
          "bf16",torch.cuda.is_bf16_supported() if device.type=="cuda" else False,flush=True)

    max_len=200
    data=load_dataset("ml1m",args.data_dir)
    split=split_data(data["sequences"])
    protocol=protocol_manifest(max_len,data["n_items"])
    print("PROTOCOL",protocol,flush=True)

    base_path=Path(args.base_root)/"ml1m"/f"seed{args.seed}"/"SparseWalker_FullCE"/"best.pt"
    if not base_path.exists(): raise FileNotFoundError(base_path)
    base_ckpt=torch.load(base_path,map_location="cpu")
    bp=base_ckpt.get("protocol",{})
    if bp.get("fingerprint") and bp.get("fingerprint")!=protocol["fingerprint"]:
        raise RuntimeError(f"fingerprint mismatch {bp.get('fingerprint')} != {protocol['fingerprint']}")
    base_val=float(base_ckpt.get("val",{}).get("NDCG@10",float("nan")))
    print("BASE CHECKPOINT",{"epoch":int(base_ckpt.get("epoch",-1)),"val_NDCG@10":base_val},flush=True)

    model=SparseWalkerTemporalMemory(
        data["n_items"],max_len,d=64,layers=2,side=256,h=16,active=8,top_side=2,degree=4,fresh_weight=.25,
        memory_periods=(16,64,256),memory_offsets=(0,16,64),initial_memory_share=.25,
    ).to(device)
    missing,unexpected=model.load_state_dict(base_ckpt["model"],strict=False)
    expected={"memory_q.weight","memory_bias","memory_share_logit"}
    if set(missing)!=expected or unexpected:
        raise RuntimeError(f"warm-start mismatch missing={missing} unexpected={unexpected}")
    print("WARMSTART",{"base_epoch":int(base_ckpt.get("epoch",-1)),"new_params":sorted(missing),
                       "K":8,"memory_slots":3,"stored_slots":24,"detached_landmarks":True},flush=True)

    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    ds=WindowDataset(split["train"],max_len,args.seed)
    out=Path(args.output_dir)/"ml1m"/f"seed{args.seed}"; out.mkdir(parents=True,exist_ok=True)
    last_path=out/"last.pt"; best_path=out/"best.pt"
    start=1; best_ndcg=-1.; best_epoch=0; best_state=None; bad=0; history=[]
    if last_path.exists():
        last=torch.load(last_path,map_location="cpu")
        if last.get("protocol",{}).get("fingerprint")==protocol["fingerprint"]:
            model.load_state_dict(last["model"]); opt.load_state_dict(last["optimizer"])
            start=int(last["epoch"])+1; best_ndcg=float(last["best_ndcg"]); best_epoch=int(last["best_epoch"])
            best_state=last["best_state"]; bad=int(last.get("bad",0)); history=list(last.get("history",[]))
            print("RESUME",{"from_epoch":start-1,"best_epoch":best_epoch,"best_ndcg":best_ndcg},flush=True)

    for epoch in range(start,args.max_epochs+1):
        lr=set_lr(opt,epoch,args.max_epochs)
        print("EPOCH START",epoch,"lr",lr,flush=True)
        t0=time.perf_counter()
        stats=train_epoch("SparseWalkerTemporalMemory",model,ds,opt,device,batch_size=args.batch_size,
                          epoch=epoch,loss_mode="full",bucket_by_length=True,use_bf16=True,return_stats=True)
        if device.type=="cuda": torch.cuda.synchronize()
        secs=time.perf_counter()-t0
        row={"epoch":epoch,"loss":float(stats["loss"]),"seconds":secs,
             "positions_per_s":stats["positions"]/max(secs,1e-9),
             "padding_efficiency":stats["padding_efficiency"],
             "memory_share":float(torch.sigmoid(model.memory_share_logit).detach().cpu())}
        print("TEMPORAL TRAIN",row,flush=True)

        if epoch==1 or epoch%args.eval_every==0:
            print("EVAL START",epoch,flush=True)
            val=evaluate_full(model,split["val_prefix"],split["val_target"],data["n_items"],max_len,device,
                              topks=(10,),batch_size=args.eval_batch_size)
            ndcg=float(val["NDCG@10"])
            erow={**row,**val}; history.append(erow); pd.DataFrame(history).to_csv(out/"history.csv",index=False)
            print("TEMPORAL EVAL",{"epoch":epoch,"NDCG@10":ndcg,"HR@10":float(val["HR@10"]),
                                   "MRR@10":float(val["MRR@10"]),
                                   "gain_vs_base_val_pct":100*(ndcg/base_val-1) if base_val==base_val else None},flush=True)
            if ndcg>best_ndcg:
                best_ndcg=ndcg; best_epoch=epoch; best_state=cpu_state_dict(model); bad=0
                torch.save({"model":best_state,"epoch":epoch,"val":val,"protocol":protocol},best_path)
            else: bad+=args.eval_every
            torch.save({"model":cpu_state_dict(model),"optimizer":opt.state_dict(),"epoch":epoch,
                        "best_epoch":best_epoch,"best_ndcg":best_ndcg,"best_state":best_state,
                        "bad":bad,"history":history,"protocol":protocol},last_path)
            if bad>=args.patience:
                print("EARLY STOP best",best_epoch,best_ndcg,flush=True); break

    print("DONE TRAINING",{"best_epoch":best_epoch,"best_val_NDCG@10":best_ndcg,"base_val_NDCG@10":base_val},flush=True)
    if best_state is not None:
        model.load_state_dict(best_state); model.eval()
        print("TEST START",flush=True)
        test=evaluate_full(model,split["test_prefix"],split["test_target"],data["n_items"],max_len,device,
                           topks=(10,20,50),batch_size=args.eval_batch_size)
        result={"cell":"SparseWalker+TemporalSkips","selected_epoch":best_epoch,
                "canonical_val_NDCG@10":best_ndcg,"base_val_NDCG@10":base_val,
                "protocol_fingerprint":protocol["fingerprint"],"detached_landmarks":True,**test}
        (out/"done.json").write_text(json.dumps(result,indent=2,sort_keys=True))
        print("TEMPORAL RESULT",result,flush=True)

if __name__=="__main__": main()
