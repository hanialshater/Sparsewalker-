#!/usr/bin/env python
"""Experiment 39: from-scratch local-contrastive SparseWalker v1.1.

Same plain Amazon-winning recurrence (K=8, 65,536 concepts, degree=4, 2 hops,
fresh-once, duplicate coalescing), but no warm start, optimizer, or backward.
The graph topology stays random/fixed and static edge logits are zero, matching
our finding that learned edge logits contribute almost nothing on Beauty.

Local learning rules:
- item embeddings: analytic contrastive positive/negative update;
- router/key prototypes: winner-take-all competitive Hebbian update;
- concept values: active concepts predict the observed next-item embedding;
- graph context projection: current item predicts the next item's concept key;
- message projection: local delta rule from sparse message to contrastive signal.
"""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sparsewalker.data import load_dataset, split_data, WindowDataset, collate_windows
from sparsewalker.evaluation import evaluate_full
from sparsewalker.models import SparseWalker

AMAZON=("beauty","video_games","sports","toys")

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def cpu_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def build(n_items):
    return SparseWalker(n_items,50,d=64,layers=2,side=256,h=16,active=8,top_side=2,degree=4,fresh_weight=.25)

@torch.no_grad()
def normalize_rows(table, rows):
    if rows.numel()==0: return
    rows=torch.unique(rows.long()); v=table[rows]
    table.index_copy_(0,rows,v/v.norm(dim=-1,keepdim=True).clamp_min(1e-8))

@torch.no_grad()
def init_model(model,seed,message_gain):
    device=model.item.weight.device; g=torch.Generator(device=device); g.manual_seed(seed+4049)
    item=F.normalize(torch.randn(model.n_items+1,model.d_model,device=device,generator=g),dim=-1); item[0].zero_(); model.item.weight.copy_(item)
    model.router.left_q.weight.normal_(0,1/math.sqrt(model.d_model),generator=g)
    model.router.right_q.weight.normal_(0,1/math.sqrt(model.d_model),generator=g)
    model.graph.context_q.weight.copy_(.5*(model.router.left_q.weight+model.router.right_q.weight))
    model.router.scale.fill_(math.log(8.0)); model.graph.scale.fill_(math.log(1.0))
    for t in (model.space.left_router,model.space.right_router,model.space.left_key,model.space.right_key):
        t.copy_(F.normalize(torch.randn(t.shape,device=device,generator=g),dim=-1))
    model.space.left_value.weight.copy_(F.normalize(torch.randn(model.side,model.d_model,device=device,generator=g),dim=-1))
    model.space.right_value.weight.copy_(F.normalize(torch.randn(model.side,model.d_model,device=device,generator=g),dim=-1))
    model.space.value_proj.weight.zero_(); model.space.value_proj.bias.zero_(); eye=torch.eye(model.d_model,device=device)
    model.space.value_proj.weight[:,:model.d_model].copy_(.5*eye); model.space.value_proj.weight[:,model.d_model:].copy_(.5*eye)
    model.message_proj.weight.copy_(float(message_gain)*eye); model.norm.weight.fill_(1.0); model.norm.bias.zero_(); model.graph.edge_logits.weight.zero_()
    dest=torch.randint(0,model.n_concepts,(model.n_concepts,model.degree),device=device,dtype=torch.int32,generator=g)
    dest[:,0]=torch.arange(model.n_concepts,device=device,dtype=torch.int32); model.graph.destination.copy_(dest)
    for p in model.parameters(): p.requires_grad_(False); p.grad=None

def loader(ds,batch,epoch):
    ds.set_epoch(epoch); g=torch.Generator(); g.manual_seed(ds.seed+epoch)
    return DataLoader(ds,batch_size=batch,shuffle=True,generator=g,collate_fn=collate_windows,pin_memory=True)

def weighted_mean(index,value,weight,size):
    out=torch.zeros(size,value.size(-1),device=value.device,dtype=value.dtype); den=torch.zeros(size,device=value.device,dtype=value.dtype)
    out.index_add_(0,index.long(),value*weight[:,None]); den.index_add_(0,index.long(),weight); used=den>0; out[used]/=den[used,None]; return out,den

@torch.no_grad()
def contrastive(model,h,pos_ids,neg_ids,lr,temp):
    h=F.normalize(h.float(),dim=-1); pos=F.normalize(model.item.weight[pos_ids].float(),dim=-1); neg=F.normalize(model.item.weight[neg_ids].float(),dim=-1)
    sp=(h*pos).sum(-1)/temp; sn=(neg*h[:,None,:]).sum(-1)/temp; pp=torch.sigmoid(sp); pn=torch.sigmoid(sn)
    signal=((1-pp)[:,None]*pos-(pn[:,:,None]*neg).mean(1))/temp
    pd=float(lr)*(1-pp)[:,None]*h/temp; nd=-float(lr)*pn[:,:,None]*h[:,None,:]/(temp*neg_ids.size(1))
    ids=torch.cat([pos_ids,neg_ids.reshape(-1)]); delta=torch.cat([pd,nd.reshape(-1,model.d_model)]).to(model.item.weight.dtype)
    model.item.weight.index_add_(0,ids,delta); normalize_rows(model.item.weight,ids); model.item.weight[0].zero_()
    return signal,{"margin":float((sp-sn.mean(-1)).mean()),"pp":float(pp.mean()),"pn":float(pn.mean())}

@torch.no_grad()
def update_current_items(model,ids,signal,lr):
    model.item.weight.index_add_(0,ids,(float(lr)*signal).to(model.item.weight.dtype)); normalize_rows(model.item.weight,ids)

@torch.no_grad()
def update_router_and_keys(model,current,routed,mass,proto_lr,key_lr):
    ql=F.normalize(model.router.left_q(current),dim=-1); qr=F.normalize(model.router.right_q(current),dim=-1); l,r=model.space.split(routed); conf=float(mass.max(-1).values.mean())
    for idx,q,proto,key in ((l,ql,model.space.left_router,model.space.left_key),(r,qr,model.space.right_router,model.space.right_key)):
        flat=idx.reshape(-1); val=q[:,None,:].expand(-1,idx.size(1),-1).reshape(-1,model.h); w=mass.reshape(-1).float(); mean,den=weighted_mean(flat,val.float(),w,model.side); rows=(den>0).nonzero().squeeze(-1)
        if rows.numel():
            proto.index_copy_(0,rows,F.normalize((1-proto_lr)*proto[rows]+proto_lr*mean[rows],dim=-1))
            key.index_copy_(0,rows,F.normalize((1-key_lr)*key[rows]+key_lr*mean[rows],dim=-1))
    return conf

@torch.no_grad()
def update_values(model,ids,mass,target,lr):
    l,r=model.space.split(ids); val=target[:,None,:].expand(-1,ids.size(1),-1).reshape(-1,model.d_model); w=mass.reshape(-1).float(); change=0.0
    for idx,table in ((l.reshape(-1),model.space.left_value.weight),(r.reshape(-1),model.space.right_value.weight)):
        mean,den=weighted_mean(idx,val,w,model.side); rows=(den>0).nonzero().squeeze(-1)
        if rows.numel():
            old=table[rows]; new=F.normalize((1-lr)*old+lr*mean[rows],dim=-1); change+=float((new-old).abs().mean()); table.index_copy_(0,rows,new)
    return change

@torch.no_grad()
def update_context(model,current,target_route,target_mass,lr):
    key=model.space.key(target_route); target=F.normalize((key*target_mass[:,:,None]).sum(1),dim=-1); q=F.normalize(model.graph.context_q(current),dim=-1); err=target-q
    model.graph.context_q.weight.add_(float(lr)*(err.T@current)/max(1,current.size(0))); return float(err.abs().mean())

@torch.no_grad()
def update_message(model,msg,signal,lr):
    model.message_proj.weight.add_(float(lr)*(signal.T@msg.float())/max(1,msg.size(0))); model.message_proj.weight.clamp_(-12,12)

@torch.no_grad()
def train_epoch(model,ds,device,args,epoch):
    model.eval(); ngen=torch.Generator(device=device); ngen.manual_seed(args.seed*100003+epoch); total=steps=batches=0; sm=sp=sn=rc=vc=cc=0.0; t0=time.perf_counter()
    for tokens,lengths in loader(ds,args.batch_size,epoch):
        tokens=tokens.to(device,non_blocking=True); x=tokens[:,:-1]; y=tokens[:,1:]; B,L=x.shape
        ids=torch.zeros(B,model.active,device=device,dtype=torch.long); mass=torch.zeros(B,model.active,device=device)
        for t in range(L):
            act=x[:,t].ne(0); valid=act&y[:,t].ne(0)
            if not act.any(): continue
            cur_ids=x[:,t]; cur=model.item(cur_ids).float(); context=cur*math.sqrt(model.d_model); fi,fm=model.router(context,model.space); af=act.float()[:,None]
            xids,xmass=model._merge(ids,mass*af,fi,fm*af)
            for _ in range(model.layers_n): xids,xmass=model.graph(xids,xmass,context,model.space,track_touched=False)
            ids=torch.where(act[:,None],xids,ids); mass=torch.where(act[:,None],xmass,mass)
            if valid.any():
                curv=cur[valid]; target_ids=y[valid,t]; target_pre=F.normalize(model.item.weight[target_ids].float(),dim=-1)
                rc+=update_router_and_keys(model,curv,fi[valid],fm[valid].float(),args.prototype_lr,args.key_lr)
                target_context=target_pre*math.sqrt(model.d_model); tr,tm=model.router(target_context,model.space); cc+=update_context(model,curv,tr,tm.float(),args.context_lr)
                msg=(model.space.value(ids[valid])*mass[valid,:,None]).sum(1).float(); h=model.norm(context[valid]+model.message_proj(msg)).float()
                neg=torch.randint(1,model.n_items+1,(int(valid.sum()),args.negatives),device=device,generator=ngen); signal,d=contrastive(model,h,target_ids,neg,args.item_lr,args.temperature)
                update_current_items(model,cur_ids[valid],signal,args.input_lr); target_now=F.normalize(model.item.weight[target_ids].float(),dim=-1); vc+=update_values(model,ids[valid],mass[valid].float(),target_now,args.value_lr); update_message(model,msg,signal,args.message_lr)
                sm+=d["margin"]; sp+=d["pp"]; sn+=d["pn"]; steps+=1; total+=int(valid.sum())
        batches+=1
    if device.type=="cuda": torch.cuda.synchronize(); sec=time.perf_counter()-t0
    grads=[n for n,p in model.named_parameters() if p.grad is not None]
    if grads: raise AssertionError(f"grad tensors created: {grads[:8]}")
    z=max(1,steps)
    return {"epoch":epoch,"positions":total,"seconds":sec,"positions_per_s":total/max(sec,1e-9),"batches":batches,"mean_contrastive_margin":sm/z,"mean_positive_prob":sp/z,"mean_negative_prob":sn/z,"mean_router_confidence":rc/z,"mean_value_update":vc/z,"mean_context_error":cc/z,"autograd_grad_tensors":0,"loss_backward_calls":0,"optimizer":None}

def evaluate(model,split,n_items,device,batch):
    model.eval(); val=evaluate_full(model,split["val_prefix"],split["val_target"],n_items,50,device,topks=(10,),batch_size=batch); test=evaluate_full(model,split["test_prefix"],split["test_target"],n_items,50,device,topks=(10,20,50),batch_size=batch); return val,test

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=AMAZON,default="beauty"); p.add_argument("--seed",type=int,default=42); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch-size",type=int,default=512); p.add_argument("--eval-batch-size",type=int,default=1024); p.add_argument("--eval-every",type=int,default=1); p.add_argument("--negatives",type=int,default=32); p.add_argument("--temperature",type=float,default=.15); p.add_argument("--item-lr",type=float,default=.02); p.add_argument("--input-lr",type=float,default=.004); p.add_argument("--prototype-lr",type=float,default=.025); p.add_argument("--key-lr",type=float,default=.015); p.add_argument("--value-lr",type=float,default=.035); p.add_argument("--context-lr",type=float,default=.006); p.add_argument("--message-lr",type=float,default=.0008); p.add_argument("--message-gain",type=float,default=8.0); p.add_argument("--data-dir",default="/content/sparsewalker_data"); p.add_argument("--output-dir",default="/content/drive/MyDrive/sparsewalker_local_contrastive"); a=p.parse_args()
    assert torch.cuda.is_available(); device=torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32=True; torch.set_float32_matmul_precision("high"); seed_all(a.seed)
    data=load_dataset(a.dataset,a.data_dir); split=split_data(data["sequences"]); model=build(data["n_items"]).to(device); init_model(model,a.seed,a.message_gain); init_val,init_test=evaluate(model,split,data["n_items"],device,a.eval_batch_size); print("LC_INIT",json.dumps({"val":init_val,"test":init_test}),flush=True)
    ds=WindowDataset(split["train"],50,a.seed); out=Path(a.output_dir)/a.dataset/f"seed{a.seed}"; out.mkdir(parents=True,exist_ok=True); cfg={"experiment":"LocalContrastiveWalker-v1","base_architecture":"corrected SparseWalker v1.1 Amazon winner","warm_start":False,"pretrained_checkpoint":None,"autograd":False,"optimizer":None,"loss_backward_calls":0,"graph_topology":"random fixed","static_edge_logits":"zero","local_rules":"contrastive items + competitive router/keys + predictive values/context/message","references":{"PheromoneWalker_v1_val_NDCG@10":0.0023507147682577624,"SASRec_test_NDCG@10":0.031195719394901355,"SparseWalker_v11_test_NDCG@10":0.044882819399656555},"args":vars(a)}; (out/"config.json").write_text(json.dumps(cfg,indent=2,sort_keys=True))
    best=float(init_val["NDCG@10"]); best_epoch=0; best_state=cpu_state(model); hist=[]
    for e in range(1,a.epochs+1):
        s=train_epoch(model,ds,device,a,e)
        if e==1 or e%a.eval_every==0:
            val=evaluate_full(model,split["val_prefix"],split["val_target"],data["n_items"],50,device,topks=(10,),batch_size=a.eval_batch_size); row={**s,**{f"val_{k}":float(v) for k,v in val.items()}}; hist.append(row); print("LC_EPOCH",json.dumps(row),flush=True); (out/"history.json").write_text(json.dumps(hist,indent=2)); nd=float(val["NDCG@10"])
            if nd>best: best=nd; best_epoch=e; best_state=cpu_state(model); torch.save({"model":best_state,"epoch":e,"val":val,"config":cfg},out/"best.pt")
    model.load_state_dict(best_state); bv,bt=evaluate(model,split,data["n_items"],device,a.eval_batch_size); result={"config":cfg,"initial":{"val":init_val,"test":init_test},"best_epoch":best_epoch,"best_local_contrastive":{"val":bv,"test":bt},"vs_sasrec_test_ndcg_ratio":float(bt["NDCG@10"]/0.031195719394901355),"vs_backprop_walker_test_ndcg_ratio":float(bt["NDCG@10"]/0.044882819399656555)}; (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)); print("LC_RESULT",json.dumps(result,indent=2),flush=True)
if __name__=="__main__": main()
