#!/usr/bin/env python
"""Canonical Beauty/ML-1M five-model comparison.

Re-evaluates the four trained SASRec/LiGR checkpoints with the current shared
full-catalog evaluator and trains Sparse Walker against the exact same
split/catalog/masking/metric contract.
"""
import argparse, hashlib, inspect, json, math, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full
from sparsewalker.models import SASRec, ESASRec, SparseWalker
from sparsewalker.training import train_epoch

BASELINE_CELLS={
    "SASRec+FullCE":(SASRec,"FullCE"),
    "SASRec+SS":(SASRec,"SS256"),
    "LiGR+FullCE":(ESASRec,"FullCE"),
    "eSASRec":(ESASRec,"SS256"),
}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def cpu_state_dict(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def save_json(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True))

def protocol_manifest(dataset,max_len,n_items):
    source=inspect.getsource(split_data)+"\n"+inspect.getsource(evaluate_full)
    source_hash=hashlib.sha256(source.encode()).hexdigest()[:16]
    manifest={
        "protocol_version":"EVAL_CANONICAL_v1_candidate",
        "dataset":dataset,
        "split":"per-user leave-two-out: train=s[:-2], val=s[-2], test=s[-1]",
        "catalog":"all mapped item ids 1..n_items",
        "seen_item_masking":True,
        "validation_selection":"best validation full-catalog NDCG@10",
        "metrics":["HR@10","HR@20","HR@50","NDCG@10","NDCG@20","NDCG@50","MRR@10"],
        "max_len":int(max_len),"n_items":int(n_items),"implementation_hash":source_hash,
    }
    manifest["fingerprint"]=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:20]
    return manifest

def baseline_model(cell,n_items,cfg):
    cls,_=BASELINE_CELLS[cell]
    return cls(n_items,int(cfg["max_len"]),d=int(cfg["d_model"]),layers=int(cfg["layers"]),heads=int(cfg["heads"]),inner=int(cfg["d_model"])*int(cfg["ff_mult"]),dropout=float(cfg["dropout"]))

def reevaluate_baseline(cell,data,split,cfg,ckpt_root,device,eval_batch,protocol):
    best=ckpt_root/cell.replace("+","_")/"best.pt"
    if not best.exists(): raise FileNotFoundError(f"Missing {best}. Run notebooks/reproductions/03_esasrec.ipynb first.")
    model=baseline_model(cell,data["n_items"],cfg).to(device).eval(); ckpt=torch.load(best,map_location=device); model.load_state_dict(ckpt["model"])
    val=evaluate_full(model,split["val_prefix"],split["val_target"],data["n_items"],int(cfg["max_len"]),device,topks=(10,),batch_size=eval_batch)
    test=evaluate_full(model,split["test_prefix"],split["test_target"],data["n_items"],int(cfg["max_len"]),device,topks=(10,20,50),batch_size=eval_batch)
    out={"cell":cell,"architecture":"LiGR" if cell.startswith("LiGR") or cell=="eSASRec" else "SASRec","loss":BASELINE_CELLS[cell][1],"params":sum(p.numel() for p in model.parameters()),"selected_epoch":int(ckpt.get("epoch",-1)),"canonical_val_NDCG@10":float(val["NDCG@10"]),"protocol_fingerprint":protocol["fingerprint"],**test}
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out

def walker_defaults(dataset):
    return dict(max_len=50,max_epochs=30,eval_every=2,patience=14,batch_size=512) if dataset=="beauty" else dict(max_len=200,max_epochs=50,eval_every=5,patience=20,batch_size=128)

def set_walker_lr(optimizer,epoch,max_epochs,peak=1e-3,min_lr=1e-4,warmup=3):
    if epoch<=warmup: lr=peak*epoch/warmup
    else:
        progress=(epoch-warmup)/max(1,max_epochs-warmup)
        lr=min_lr+.5*(peak-min_lr)*(1+math.cos(math.pi*progress))
    for group in optimizer.param_groups: group["lr"]=lr
    return lr

def train_walker(data,split,args,device,out_root,protocol):
    wd=walker_defaults(args.dataset); max_len=wd["max_len"]; cell_dir=out_root/"SparseWalker_FullCE"; cell_dir.mkdir(parents=True,exist_ok=True); done=cell_dir/"done.json"
    if done.exists() and not args.force:
        old=json.loads(done.read_text())
        if old.get("protocol_fingerprint")==protocol["fingerprint"]:
            print("SKIP completed SparseWalker+FullCE"); return old
    seed_all(args.seed)
    model=SparseWalker(data["n_items"],max_len,d=64,layers=2,side=256,h=16,active=8,top_side=2,degree=4,fresh_weight=.25).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); ds=WindowDataset(split["train"],max_len,args.seed)
    max_epochs=args.walker_max_epochs or wd["max_epochs"]; eval_every=args.walker_eval_every or wd["eval_every"]; patience=args.walker_patience or wd["patience"]; batch_size=args.walker_batch_size or wd["batch_size"]
    best_ndcg=-1.; best_epoch=0; best_state=None; bad_epochs=0; history=[]
    for epoch in range(1,max_epochs+1):
        lr=set_walker_lr(optimizer,epoch,max_epochs)
        loss=train_epoch("SparseWalker",model,ds,optimizer,device,batch_size=batch_size,epoch=epoch,loss_mode="full")
        pursued=model.graph.pursue(optimizer,refresh=2) if epoch>=4 and epoch%2==0 else 0
        if epoch==1 or epoch%eval_every==0:
            val=evaluate_full(model,split["val_prefix"],split["val_target"],data["n_items"],max_len,device,topks=(10,),batch_size=args.eval_batch_size)
            row={"cell":"SparseWalker+FullCE","epoch":epoch,"loss":loss,"lr":lr,"pursued_rows":pursued,**val}; print(row); history.append(row); pd.DataFrame(history).to_csv(cell_dir/"history.csv",index=False)
            ndcg=float(val["NDCG@10"])
            if ndcg>best_ndcg:
                best_ndcg=ndcg; best_epoch=epoch; best_state=cpu_state_dict(model); bad_epochs=0
                torch.save({"model":best_state,"epoch":epoch,"val":val,"protocol":protocol},cell_dir/"best.pt")
            else: bad_epochs+=eval_every
            torch.save({"model":cpu_state_dict(model),"optimizer":optimizer.state_dict(),"epoch":epoch,"best_epoch":best_epoch,"best_ndcg":best_ndcg,"best_state":best_state,"history":history,"protocol":protocol},cell_dir/"last.pt")
            if bad_epochs>=patience:
                print("EARLY STOP Walker best epoch",best_epoch); break
    if best_state is None: raise RuntimeError("Walker produced no validation checkpoint.")
    model.load_state_dict(best_state)
    val=evaluate_full(model,split["val_prefix"],split["val_target"],data["n_items"],max_len,device,topks=(10,),batch_size=args.eval_batch_size)
    test=evaluate_full(model,split["test_prefix"],split["test_target"],data["n_items"],max_len,device,topks=(10,20,50),batch_size=args.eval_batch_size)
    result={"cell":"SparseWalker+FullCE","architecture":"SparseWalker","loss":"FullCE","params":sum(p.numel() for p in model.parameters()),"selected_epoch":best_epoch,"canonical_val_NDCG@10":float(val["NDCG@10"]),"protocol_fingerprint":protocol["fingerprint"],**test}; save_json(done,result); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=["beauty","ml1m"],required=True); p.add_argument("--seed",type=int,default=42); p.add_argument("--data-dir",default="/content/sparsewalker_data"); p.add_argument("--baseline-root",default="/content/drive/MyDrive/sparsewalker_esasrec_2x2"); p.add_argument("--output-dir",default="/content/drive/MyDrive/sparsewalker_canonical_pair"); p.add_argument("--eval-batch-size",type=int,default=1024); p.add_argument("--walker-max-epochs",type=int,default=None); p.add_argument("--walker-eval-every",type=int,default=None); p.add_argument("--walker-patience",type=int,default=None); p.add_argument("--walker-batch-size",type=int,default=None); p.add_argument("--force",action="store_true"); args=p.parse_args()
    seed_all(args.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device",device)
    data=load_dataset(args.dataset,args.data_dir); split=split_data(data["sequences"])
    baseline_root=Path(args.baseline_root)/args.dataset/f"seed{args.seed}"; config_path=baseline_root/"config.json"
    if not config_path.exists(): raise FileNotFoundError(f"Missing {config_path}; complete the eSASRec 2x2 run first.")
    cfg=json.loads(config_path.read_text()); protocol=protocol_manifest(args.dataset,int(cfg["max_len"]),data["n_items"])
    out_root=Path(args.output_dir)/args.dataset/f"seed{args.seed}"; out_root.mkdir(parents=True,exist_ok=True); save_json(out_root/"protocol.json",protocol); print("PROTOCOL",json.dumps(protocol,indent=2))
    rows=[]
    for cell in BASELINE_CELLS:
        r=reevaluate_baseline(cell,data,split,cfg,baseline_root,device,args.eval_batch_size,protocol); rows.append(r); print("CANONICAL",r)
    rows.append(train_walker(data,split,args,device,out_root,protocol))
    df=pd.DataFrame(rows).sort_values("NDCG@10",ascending=False); df.to_csv(out_root/"summary.csv",index=False); print("\nCANONICAL SUMMARY"); print(df.to_string(index=False))
    best=float(df["NDCG@10"].max()); df2=df[["cell","NDCG@10","HR@10","MRR@10","params"]].copy(); df2["NDCG_gap_vs_best_pct"]=100*(df2["NDCG@10"]/best-1); print("\nQUALITY GAPS"); print(df2.to_string(index=False))

if __name__=="__main__": main()
