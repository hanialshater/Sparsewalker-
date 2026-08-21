#!/usr/bin/env python
"""Experiment 41: backward-free local-contrastive SparseWalker with EMA targets.

Same corrected SparseWalker v1.1 recurrence and same local learning rules as
Experiment 39. The only change is a slow target copy of the item representation.
Forward scoring uses the online item table; local teaching vectors, router target
contexts, and concept-value targets use the EMA table.

No warm start, optimizer, backward(), or autograd learning.
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import run_amazon_local_contrastive_walker as base
from sparsewalker.data import load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full

AMAZON=base.AMAZON

@torch.no_grad()
def ema_rows(target, online, rows, mu):
    if rows.numel()==0: return 1.0
    rows=torch.unique(rows.long())
    old=target[rows]
    src=online[rows].float()
    new=float(mu)*old+(1.0-float(mu))*src
    new=F.normalize(new,dim=-1)
    target.index_copy_(0,rows,new)
    target[0].zero_()
    return float(F.cosine_similarity(src,new,dim=-1).mean().item())

@torch.no_grad()
def contrastive_ema(model,target_table,h,pos_ids,neg_ids,lr,temp):
    h=F.normalize(h.float(),dim=-1)
    pos=F.normalize(target_table[pos_ids].float(),dim=-1)
    neg=F.normalize(target_table[neg_ids].float(),dim=-1)
    sp=(h*pos).sum(-1)/temp
    sn=(neg*h[:,None,:]).sum(-1)/temp
    pp=torch.sigmoid(sp); pn=torch.sigmoid(sn)
    signal=((1-pp)[:,None]*pos-(pn[:,:,None]*neg).mean(1))/temp

    # Online item table remains plastic exactly as in Experiment 39.
    pd=float(lr)*(1-pp)[:,None]*h/temp
    nd=-float(lr)*pn[:,:,None]*h[:,None,:]/(temp*neg_ids.size(1))
    ids=torch.cat([pos_ids,neg_ids.reshape(-1)])
    delta=torch.cat([pd,nd.reshape(-1,model.d_model)]).to(model.item.weight.dtype)
    model.item.weight.index_add_(0,ids,delta)
    base.normalize_rows(model.item.weight,ids)
    model.item.weight[0].zero_()
    return signal,{"margin":float((sp-sn.mean(-1)).mean()),"pp":float(pp.mean()),"pn":float(pn.mean())}

@torch.no_grad()
def train_epoch_ema(model,target_table,ds,device,args,epoch):
    model.eval()
    ngen=torch.Generator(device=device); ngen.manual_seed(args.seed*100003+epoch)
    total=steps=batches=0
    sm=sp=sn=rc=vc=cc=tc=0.0
    t0=time.perf_counter()

    for tokens,lengths in base.loader(ds,args.batch_size,epoch):
        tokens=tokens.to(device,non_blocking=True)
        x=tokens[:,:-1]; y=tokens[:,1:]; B,L=x.shape
        ids=torch.zeros(B,model.active,device=device,dtype=torch.long)
        mass=torch.zeros(B,model.active,device=device)

        for t in range(L):
            act=x[:,t].ne(0); valid=act&y[:,t].ne(0)
            if not act.any(): continue

            cur_ids=x[:,t]
            cur=model.item(cur_ids).float()
            context=cur*math.sqrt(model.d_model)
            fi,fm=model.router(context,model.space)
            af=act.float()[:,None]
            xids,xmass=model._merge(ids,mass*af,fi,fm*af)
            for _ in range(model.layers_n):
                xids,xmass=model.graph(xids,xmass,context,model.space,track_touched=False)
            ids=torch.where(act[:,None],xids,ids)
            mass=torch.where(act[:,None],xmass,mass)

            if valid.any():
                curv=cur[valid]
                target_ids=y[valid,t]
                target_pre=F.normalize(target_table[target_ids].float(),dim=-1)

                rc+=base.update_router_and_keys(model,curv,fi[valid],fm[valid].float(),args.prototype_lr,args.key_lr)
                target_context=target_pre*math.sqrt(model.d_model)
                tr,tm=model.router(target_context,model.space)
                cc+=base.update_context(model,curv,tr,tm.float(),args.context_lr)

                msg=(model.space.value(ids[valid])*mass[valid,:,None]).sum(1).float()
                h=model.norm(context[valid]+model.message_proj(msg)).float()
                neg=torch.randint(1,model.n_items+1,(int(valid.sum()),args.negatives),device=device,generator=ngen)
                signal,d=contrastive_ema(model,target_table,h,target_ids,neg,args.item_lr,args.temperature)

                base.update_current_items(model,cur_ids[valid],signal,args.input_lr)
                vc+=base.update_values(model,ids[valid],mass[valid].float(),target_pre,args.value_lr)
                base.update_message(model,msg,signal,args.message_lr)

                touched=torch.cat([cur_ids[valid],target_ids,neg.reshape(-1)])
                tc+=ema_rows(target_table,model.item.weight,touched,args.ema)

                sm+=d["margin"]; sp+=d["pp"]; sn+=d["pn"]
                steps+=1; total+=int(valid.sum())
        batches+=1

    if device.type=="cuda": torch.cuda.synchronize()
    sec=time.perf_counter()-t0
    grads=[n for n,p in model.named_parameters() if p.grad is not None]
    if grads: raise AssertionError(f"grad tensors created: {grads[:8]}")
    z=max(1,steps)
    return {
        "epoch":epoch,"positions":total,"seconds":sec,"positions_per_s":total/max(sec,1e-9),"batches":batches,
        "mean_contrastive_margin":sm/z,"mean_positive_prob":sp/z,"mean_negative_prob":sn/z,
        "mean_router_confidence":rc/z,"mean_value_update":vc/z,"mean_context_error":cc/z,
        "mean_online_target_cosine":tc/z,"ema":float(args.ema),
        "autograd_grad_tensors":0,"loss_backward_calls":0,"optimizer":None,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=AMAZON,default="beauty")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=70)
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--eval-batch-size",type=int,default=1024)
    p.add_argument("--eval-every",type=int,default=1)
    p.add_argument("--negatives",type=int,default=32)
    p.add_argument("--temperature",type=float,default=.15)
    p.add_argument("--item-lr",type=float,default=.02)
    p.add_argument("--input-lr",type=float,default=.004)
    p.add_argument("--prototype-lr",type=float,default=.025)
    p.add_argument("--key-lr",type=float,default=.015)
    p.add_argument("--value-lr",type=float,default=.035)
    p.add_argument("--context-lr",type=float,default=.006)
    p.add_argument("--message-lr",type=float,default=.0008)
    p.add_argument("--message-gain",type=float,default=8.0)
    p.add_argument("--ema",type=float,default=.995)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--data-dir",default="/content/sparsewalker_data")
    p.add_argument("--output-dir",default="/content/drive/MyDrive/sparsewalker_local_contrastive_ema")
    a=p.parse_args()

    assert torch.cuda.is_available(); device=torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32=True; torch.set_float32_matmul_precision("high")
    base.seed_all(a.seed)
    data=load_dataset(a.dataset,a.data_dir); split=split_data(data["sequences"])
    model=base.build(data["n_items"]).to(device); base.init_model(model,a.seed,a.message_gain)
    target_table=model.item.weight.detach().float().clone()
    out=Path(a.output_dir)/a.dataset/f"seed{a.seed}"/f"ema{a.ema:g}"; out.mkdir(parents=True,exist_ok=True)

    init_val,init_test=base.evaluate(model,split,data["n_items"],device,a.eval_batch_size)
    print("LCEMA_INIT",json.dumps({"ema":a.ema,"val":init_val,"test":init_test}),flush=True)

    ds=WindowDataset(split["train"],50,a.seed)
    cfg={"experiment":"LocalContrastiveWalker-EMA-v1","base":"Experiment39 exact local rules","ema":a.ema,
         "warm_start":False,"autograd":False,"optimizer":None,"loss_backward_calls":0,
         "reference_val_NDCG@10":{"LC_v1":0.040515991131704246,"SASRec":0.04296780165590764},"args":vars(a)}
    (out/"config.json").write_text(json.dumps(cfg,indent=2,sort_keys=True))

    best=float(init_val["NDCG@10"]); best_epoch=0; best_state=base.cpu_state(model); best_target=target_table.cpu().clone(); hist=[]; start=1
    last=out/"last.pt"
    if a.resume and last.exists():
        ck=torch.load(last,map_location=device)
        model.load_state_dict(ck["model"]); target_table=ck["target"].to(device)
        best=float(ck["best"]); best_epoch=int(ck["best_epoch"]); best_state=ck["best_state"]; best_target=ck["best_target"]
        hist=ck.get("history",[]); start=int(ck["epoch"])+1
        print("LCEMA_RESUME",json.dumps({"from_epoch":start-1,"best_epoch":best_epoch,"best":best}),flush=True)

    for e in range(start,a.epochs+1):
        s=train_epoch_ema(model,target_table,ds,device,a,e)
        val=evaluate_full(model,split["val_prefix"],split["val_target"],data["n_items"],50,device,topks=(10,),batch_size=a.eval_batch_size)
        row={**s,**{f"val_{k}":float(v) for k,v in val.items()}}
        hist.append(row); print("LCEMA_EPOCH",json.dumps(row),flush=True); (out/"history.json").write_text(json.dumps(hist,indent=2))
        nd=float(val["NDCG@10"])
        if nd>best:
            best=nd; best_epoch=e; best_state=base.cpu_state(model); best_target=target_table.cpu().clone()
            torch.save({"model":best_state,"target":best_target,"epoch":e,"val":val,"config":cfg},out/"best.pt")
        torch.save({"model":base.cpu_state(model),"target":target_table.cpu(),"epoch":e,"best":best,"best_epoch":best_epoch,
                    "best_state":best_state,"best_target":best_target,"history":hist,"config":cfg},last)

    model.load_state_dict(best_state); target_table=best_target.to(device)
    bv,bt=base.evaluate(model,split,data["n_items"],device,a.eval_batch_size)
    result={"config":cfg,"initial":{"val":init_val,"test":init_test},"best_epoch":best_epoch,
            "best_ema":{"val":bv,"test":bt},"vs_lc_v1_val_ratio":float(bv["NDCG@10"]/0.040515991131704246),
            "vs_sasrec_test_ndcg_ratio":float(bt["NDCG@10"]/0.031195719394901355)}
    (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True))
    print("LCEMA_RESULT",json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
