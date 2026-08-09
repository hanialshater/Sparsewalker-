#!/usr/bin/env python
"""Train only the query head for HNSW/SWG navigation on frozen SparseWalker states.

Hypothesis: HNSW topology already makes next-item concepts reachable; the current
Walker query is simply not aligned with the next-item destination. Freeze the
Walker, concept geometry, router, and HNSW graph. Train a tiny 64->16 query head
so next-item routed concepts outrank sampled concept negatives, then test whether
4-hop state-started navigation reaches those concepts.
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sparsewalker.data import load_dataset, split_data
from sparsewalker.models import SparseWalker

# Reuse the already-tested HNSW graph/search helpers from the primitive experiment.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from run_hnsw_attention_primitive import (  # noqa: E402
    build_hnsw, extract_hnsw_adjacency, navigate_one, oracle_reachability,
)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def pad_batch(seqs, max_len, device):
    rows=[list(s)[-max_len:] for s in seqs]
    lens=[len(s) for s in rows]; L=max(lens)
    x=torch.zeros(len(rows),L,dtype=torch.long,device=device)
    for i,s in enumerate(rows): x[i,:len(s)]=torch.as_tensor(s,dtype=torch.long,device=device)
    return x,torch.as_tensor(lens,dtype=torch.long,device=device)


@torch.inference_mode()
def concept_keys(model, device):
    ids=torch.arange(model.n_concepts,device=device)
    k=model.space.key(ids).float().cpu().numpy().astype('float32')
    k/=np.linalg.norm(k,axis=1,keepdims=True).clip(min=1e-12)
    return k


@torch.inference_mode()
def target_concepts(model, item_ids):
    state=model.item(item_ids)*math.sqrt(model.d_model)
    ids,_=model.router(state,model.space)
    return ids


@torch.inference_mode()
def cache_train_features(model, train_seqs, max_len, device, batch_users=128, cap=300000):
    """Cache frozen hidden states for autoregressive positions and their next-item concepts."""
    hs=[]; pos=[]; total=0; t0=time.perf_counter()
    model.eval()
    for st in range(0,len(train_seqs),batch_users):
        seqs=[]
        for s in train_seqs[st:st+batch_users]:
            s=list(s)[-(max_len+1):]
            if len(s)>=2: seqs.append(s)
        if not seqs: continue
        tok,lens=pad_batch(seqs,max_len+1,device)
        x=tok[:,:-1]
        y=tok[:,1:]
        xl=(lens-1).clamp_min(0)
        use_amp=device.type=='cuda' and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=use_amp):
            H=model.encode(x)
        mask=torch.arange(x.size(1),device=device)[None,:] < xl[:,None]
        h=H[mask].float()
        yy=y[mask]
        pp=target_concepts(model,yy)
        hs.append(h.cpu().to(torch.float16)); pos.append(pp.cpu().to(torch.int32))
        total+=h.size(0)
        if total>=cap: break
    H=torch.cat(hs,0)[:cap]
    P=torch.cat(pos,0)[:cap]
    print('TRAIN_CACHE',{'examples':int(H.size(0)),'seconds':round(time.perf_counter()-t0,2),'MB':round((H.numel()*2+P.numel()*4)/1e6,1)},flush=True)
    return H,P


class QueryHead(nn.Module):
    def __init__(self,d,h,init_weight):
        super().__init__(); self.proj=nn.Linear(d,h,bias=False)
        with torch.no_grad(): self.proj.weight.copy_(init_weight)
        self.log_scale=nn.Parameter(torch.tensor(math.log(10.0)))
    def forward(self,x): return F.normalize(self.proj(x),dim=-1)


def train_head(head,H,P,keys,device,epochs=5,batch=2048,n_neg=512,lr=2e-3):
    ds=TensorDataset(H,P)
    loader=DataLoader(ds,batch_size=batch,shuffle=True,pin_memory=True)
    opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=1e-4)
    kt=torch.as_tensor(keys,device=device,dtype=torch.float32)
    history=[]
    for ep in range(1,epochs+1):
        head.train(); total=0.; n=0; t0=time.perf_counter()
        for h,p in loader:
            h=h.to(device,non_blocking=True).float(); p=p.to(device,non_blocking=True).long()
            q=head(h); scale=head.log_scale.exp().clamp(1.,50.)
            pk=kt[p]                                  # B x 4 x h
            ps=(pk*q[:,None,:]).sum(-1)*scale
            neg=torch.randint(0,kt.size(0),(h.size(0),n_neg),device=device)
            nk=kt[neg]
            ns=(nk*q[:,None,:]).sum(-1)*scale
            # Multi-positive InfoNCE: probability mass on any routed next-item concept.
            numer=torch.logsumexp(ps,dim=-1)
            denom=torch.logsumexp(torch.cat([ps,ns],dim=-1),dim=-1)
            loss=(denom-numer).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total+=float(loss.detach())*h.size(0); n+=h.size(0)
        row={'epoch':ep,'loss':total/max(1,n),'seconds':time.perf_counter()-t0,'scale':float(head.log_scale.exp().detach().cpu())}
        history.append(row); print('QUERY_TRAIN',row,flush=True)
    return history


@torch.inference_mode()
def collect_eval(model,head,prefixes,targets,indices,max_len,device,keys,batch=128):
    qs=[]; starts=[]; nextc=[]; dense=[]; lengths=[]
    kt=torch.as_tensor(keys,device=device,dtype=torch.float32)
    for st in range(0,len(indices),batch):
        idx=indices[st:st+batch]; seqs=[prefixes[i] for i in idx]
        x,l=pad_batch(seqs,max_len,device)
        use_amp=device.type=='cuda' and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=use_amp):
            H,I,_=model.encode_with_states(x)
        row=torch.arange(x.size(0),device=device); last=(l-1).clamp_min(0)
        h=H[row,last].float(); s=I[row,last]
        q=head(h).float()
        tgt=torch.as_tensor([targets[i] for i in idx],dtype=torch.long,device=device)
        nc=target_concepts(model,tgt)
        sc=q@kt.T; dt=sc.topk(10,dim=-1).indices
        qs.append(q.cpu()); starts.append(s.cpu()); nextc.append(nc.cpu()); dense.append(dt.cpu()); lengths.extend(l.cpu().tolist())
    return {'q':torch.cat(qs).numpy().astype('float32'),'starts':torch.cat(starts).numpy().astype('int64'),'next':torch.cat(nextc).numpy().astype('int64'),'dense':torch.cat(dense).numpy().astype('int64'),'lengths':np.asarray(lengths)}


def eval_navigation(ev,adj,keys,hops=4,beam=16):
    dense_rec=np.zeros(hops+1); next_seen=np.zeros(hops+1); next_out=np.zeros(hops+1); reads=np.zeros(hops+1)
    n=len(ev['q'])
    for i in range(n):
        snaps=navigate_one(ev['q'][i],ev['starts'][i],adj,keys,hops,beam)
        ds=set(map(int,ev['dense'][i])); ns=set(map(int,ev['next'][i]))
        for h,(top,visited,c) in enumerate(snaps):
            dense_rec[h]+=len(ds.intersection(map(int,top.tolist())))/10.0
            next_seen[h]+=float(bool(ns.intersection(visited)))
            next_out[h]+=float(bool(ns.intersection(map(int,top.tolist()))))
            reads[h]+=c
    return {str(h):{'dense_recall@10':float(dense_rec[h]/n),'next_item_concept_seen_rate':float(next_seen[h]/n),'next_item_concept_in_output_rate':float(next_out[h]/n),'mean_edge_reads':float(reads[h]/n)} for h in range(hops+1)}


def dense_target_alignment(ev):
    vals=[]
    for d,nc in zip(ev['dense'],ev['next']): vals.append(float(bool(set(map(int,d)).intersection(map(int,nc)))))
    return float(np.mean(vals))


def balanced_indices(prefixes,max_len,per_bucket,seed):
    rng=np.random.default_rng(seed); b=[[],[],[]]
    for i,p in enumerate(prefixes):
        n=min(len(p),max_len); j=0 if n<=50 else (1 if n<=100 else 2); b[j].append(i)
    out=[]
    for x in b:
        take=min(per_bucket,len(x)); out.extend(rng.choice(np.asarray(x),size=take,replace=False).tolist())
    rng.shuffle(out); return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--data-dir',default='/content/sparsewalker_data'); ap.add_argument('--checkpoint',default='/content/drive/MyDrive/sparsewalker_canonical_pair/ml1m/seed42/SparseWalker_FullCE/best.pt'); ap.add_argument('--output',default='/content/drive/MyDrive/sparsewalker_swg_query/result.json'); ap.add_argument('--epochs',type=int,default=5); ap.add_argument('--train-cap',type=int,default=300000); ap.add_argument('--per-bucket',type=int,default=128); ap.add_argument('--hnsw-m',type=int,default=8); ap.add_argument('--ef-construction',type=int,default=160); args=ap.parse_args()
    seed_all(args.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('DEVICE',device,torch.cuda.get_device_name(0) if device.type=='cuda' else None,flush=True)
    data=load_dataset('ml1m',args.data_dir); split=split_data(data['sequences']); ck=torch.load(args.checkpoint,map_location='cpu')
    model=SparseWalker(data['n_items'],200,d=64,layers=2,side=256,h=16,active=8,top_side=2,degree=4,fresh_weight=.25).to(device); model.load_state_dict(ck['model']); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    keys=concept_keys(model,device); index,build_s=build_hnsw(keys,args.hnsw_m,args.ef_construction); _,flat,stats=extract_hnsw_adjacency(index,model.n_concepts,args.hnsw_m)
    print('HNSW',{'build_s':round(build_s,2),**stats},flush=True)
    H,P=cache_train_features(model,split['train'],200,device,cap=args.train_cap)
    head=QueryHead(64,16,model.graph.context_q.weight.detach()).to(device)
    # Baseline validation before training the query head.
    idx=balanced_indices(split['val_prefix'],200,args.per_bucket,args.seed)
    pre=collect_eval(model,head,split['val_prefix'],split['val_target'],idx,200,device,keys)
    pre_nav=eval_navigation(pre,flat,keys); print('PRETRAIN_PANEL',{'dense_target_hit@10':dense_target_alignment(pre),'hop4':pre_nav['4']},flush=True)
    hist=train_head(head,H,P,keys,device,epochs=args.epochs)
    post=collect_eval(model,head,split['val_prefix'],split['val_target'],idx,200,device,keys)
    post_nav=eval_navigation(post,flat,keys)
    oracle_next=oracle_reachability(post['starts'],post['next'],flat,4)
    result={'pretrain':{'dense_target_hit@10':dense_target_alignment(pre),'navigation':pre_nav},'posttrain':{'dense_target_hit@10':dense_target_alignment(post),'navigation':post_nav},'oracle_next':oracle_next,'history':hist,'hnsw':{'build_s':build_s,**stats}}
    print('POSTTRAIN_PANEL',{'dense_target_hit@10':result['posttrain']['dense_target_hit@10'],'hop4':post_nav['4'],'oracle_next_hop4':oracle_next['4']},flush=True)
    # Hard interpretation signal.
    gain=post_nav['4']['next_item_concept_seen_rate']-pre_nav['4']['next_item_concept_seen_rate']
    print('DECISION',{'pre_next_seen_h4':pre_nav['4']['next_item_concept_seen_rate'],'post_next_seen_h4':post_nav['4']['next_item_concept_seen_rate'],'absolute_gain':gain,'dense_target_hit@10':result['posttrain']['dense_target_hit@10'],'oracle_next_h4':oracle_next['4']['any_target_hit_rate']},flush=True)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); torch.save({'query_head':head.state_dict()},out.with_suffix('.pt'))

if __name__=='__main__': main()
