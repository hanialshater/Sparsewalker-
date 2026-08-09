import math
import random
import numpy as np
import torch


def make_eval_batch(prefixes,indices,max_len):
    seqs=[prefixes[i][-max_len:] for i in indices]
    lens=torch.tensor([len(s) for s in seqs],dtype=torch.long)
    L=int(lens.max()); x=torch.zeros(len(seqs),L,dtype=torch.long)
    for r,s in enumerate(seqs): x[r,:len(s)]=torch.tensor(s)
    return x,lens

@torch.inference_mode()
def evaluate_full(model,prefixes,targets,n_items,max_len,device,topks=(10,20,50,200),batch_size=1024,autocast_dtype=None):
    model.eval(); topks=sorted(k for k in set(topks) if k<=n_items); maxk=max(topks)
    sums={f"HR@{k}":0. for k in topks}; sums.update({f"NDCG@{k}":0. for k in topks}); mrr10=0.; total=len(targets)
    for start in range(0,total,batch_size):
        end=min(total,start+batch_size); idxs=list(range(start,end)); seq,lens=make_eval_batch(prefixes,idxs,max_len)
        seq=seq.to(device); lens=lens.to(device); tgt=torch.tensor(targets[start:end],device=device)
        enabled=autocast_dtype is not None and device.type=="cuda"
        with torch.autocast("cuda",dtype=autocast_dtype,enabled=enabled): scores=model.full_scores(seq,lens)
        scores=scores.float()
        for r,i in enumerate(idxs):
            seen=set(prefixes[i]); truth=int(tgt[r]); seen.discard(truth)
            if seen: scores[r,torch.tensor(list(seen),device=device)]=-1e20
        top=scores.topk(maxk,dim=-1).indices.cpu().numpy(); tgt_np=tgt.cpu().numpy()
        for r,truth in enumerate(tgt_np):
            pos=np.where(top[r]==truth)[0]; rank=int(pos[0])+1 if len(pos) else None
            for k in topks:
                if rank is not None and rank<=k:
                    sums[f"HR@{k}"]+=1; sums[f"NDCG@{k}"]+=1/math.log2(rank+1)
            if rank is not None and rank<=10: mrr10+=1/rank
    out={k:v/total for k,v in sums.items()}; out["MRR@10"]=mrr10/total; return out

def deterministic_sampled_candidates(prefix,target,n_items,n_negs,user_idx,seed=12345):
    rng=random.Random(seed*1_000_003+user_idx); seen=set(prefix); seen.add(target); neg=[]
    while len(neg)<n_negs:
        x=rng.randint(1,n_items)
        if x not in seen: seen.add(x); neg.append(x)
    return [target]+neg

@torch.inference_mode()
def evaluate_sampled(model,prefixes,targets,n_items,max_len,device,n_negs=100,batch_size=1024):
    model.eval(); hits=ndcg=mrr=0.; total=len(targets)
    for start in range(0,total,batch_size):
        end=min(total,start+batch_size); idxs=list(range(start,end)); seq,lens=make_eval_batch(prefixes,idxs,max_len)
        cand=np.asarray([deterministic_sampled_candidates(prefixes[i],targets[i],n_items,n_negs,i) for i in idxs],dtype=np.int64)
        seq=seq.to(device); lens=lens.to(device); c=torch.tensor(cand,device=device); scores=model.full_scores(seq,lens).gather(1,c)
        pos_score=scores[:,0:1]; pos_id=c[:,0:1]; rank=((scores>pos_score)|((scores==pos_score)&(c<pos_id))).sum(1).cpu().numpy()
        hits+=float((rank<10).sum()); ndcg+=float(sum(1/math.log2(int(r)+2) for r in rank if r<10)); mrr+=float(sum(1/(int(r)+1) for r in rank if r<10))
    return {"sampled_HR@10":hits/total,"sampled_NDCG@10":ndcg/total,"sampled_MRR@10":mrr/total}
