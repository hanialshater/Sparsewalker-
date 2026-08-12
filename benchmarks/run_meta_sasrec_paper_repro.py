#!/usr/bin/env python
"""Paper-comparable Meta SASRec ML-1M reproduction + latency audit.

Pure-PyTorch mirror of the frozen Meta generative-recommenders recipe used for
its ML-1M SASRec row. The script first gates on paper-level NDCG, then measures
batch-1 A100 latency for the same trained model.
"""
import argparse, json, math, random, time, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PAPER_NDCG = 0.1603


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def meta_trunc_normal(x, mean, std):
    tmp=x.new_empty(x.shape+(4,)).normal_(); valid=(tmp<2)&(tmp>-2)
    ind=valid.max(-1,keepdim=True)[1]
    x.copy_(tmp.gather(-1,ind).squeeze(-1)); x.mul_(std).add_(mean); return x


def load_data(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    z=root/'ml-1m.zip'; d=root/'ml-1m'
    if not (d/'ratings.dat').exists():
        if not z.exists():
            urllib.request.urlretrieve('https://files.grouplens.org/datasets/movielens/ml-1m.zip',z)
        with zipfile.ZipFile(z) as f: f.extractall(root)
    ratings=pd.read_csv(d/'ratings.dat',sep='::',engine='python',
        names=['user_id','movie_id','rating','unix_timestamp'])
    movies=pd.read_csv(d/'movies.dat',sep='::',engine='python',
        names=['movie_id','title','genres'],encoding='latin-1')
    # Meta preprocessor: raw movie IDs; global timestamp sort then groupby user.
    rg=ratings.sort_values(by=['unix_timestamp']).groupby('user_id')
    seqs=[g.movie_id.astype(int).tolist() for _,g in rg]
    rated_unique=len(set(ratings.movie_id.tolist()))
    all_movie_ids=movies.movie_id.astype(int).tolist()  # Meta CandidateIndex source.
    return dict(seqs=seqs, all_movie_ids=all_movie_ids,
        rated_unique=rated_unique, max_item_id=max(all_movie_ids),
        users=len(seqs), ratings=len(ratings), candidate_movies=len(all_movie_ids),
        tied_timestamp_rows=int(ratings.unix_timestamp.duplicated(keep=False).sum()))


def make_tensors(data,max_len=200):
    n=len(data['seqs']); train=torch.zeros(n,max_len+1,dtype=torch.long)
    hist=torch.zeros(n,max_len,dtype=torch.long); lens=torch.zeros(n,dtype=torch.long)
    tgt=torch.zeros(n,dtype=torch.long)
    for i,s in enumerate(data['seqs']):
        tr=s[:-1][-(max_len+1):]  # train DatasetV2(ignore_last_n=1), target appended.
        train[i,:len(tr)]=torch.tensor(tr)
        h=s[:-1][-max_len:]; hist[i,:len(h)]=torch.tensor(h); lens[i]=len(h); tgt[i]=s[-1]
    return train,hist,lens,tgt


class FF(nn.Module):
    def __init__(self,d=50,p=.2):
        super().__init__(); self.c1=nn.Conv1d(d,d,1); self.c2=nn.Conv1d(d,d,1)
        self.a=nn.ReLU(); self.d1=nn.Dropout(p); self.d2=nn.Dropout(p)
    def forward(self,x):
        y=x.transpose(-1,-2); y=self.c1(y); y=self.a(y); y=self.d1(y); y=self.c2(y); y=self.d2(y)
        return y.transpose(-1,-2)+x


class MetaSASRec(nn.Module):
    def __init__(self,max_item_id=3952,npos=211,d=50,p=.2):
        super().__init__(); self.d=d
        self.item=nn.Embedding(max_item_id+1,d,padding_idx=0); self.pos=nn.Embedding(npos,d)
        self.edrop=nn.Dropout(p)
        self.attn=nn.ModuleList([nn.MultiheadAttention(d,1,dropout=p,batch_first=True) for _ in range(2)])
        self.ff=nn.ModuleList([FF(d,p) for _ in range(2)])
        self.register_buffer('mask',torch.triu(torch.ones(npos,npos,dtype=torch.bool),1),persistent=False)
        meta_trunc_normal(self.item.weight,0,.02); meta_trunc_normal(self.pos.weight,0,math.sqrt(1/d))
        for m in list(self.attn)+list(self.ff):
            for q in m.parameters():
                if q.dim()>=2: nn.init.xavier_normal_(q)
    def l2(self,x): return x/torch.clamp(torch.linalg.vector_norm(x,dim=-1,keepdim=True),min=1e-6)
    def item_norm(self,ids): return self.l2(self.item(ids))
    def encode_all(self,ids):
        B,N=ids.shape; x=self.item(ids)*math.sqrt(self.d)+self.pos(torch.arange(N,device=ids.device))[None]
        x=self.edrop(x); valid=(ids!=0).unsqueeze(-1).to(x.dtype); x=x*valid; am=self.mask[:N,:N]
        for a,f in zip(self.attn,self.ff):
            q=F.layer_norm(x,(self.d,),eps=1e-8)
            z,_=a(q,x,x,attn_mask=am,need_weights=False)
            x=f(F.layer_norm(q+z,(self.d,),eps=1e-8))*valid
        return self.l2(x)
    def encode_last(self,ids,lens):
        z=self.encode_all(ids); r=torch.arange(ids.size(0),device=ids.device)
        return z[r,(lens-1).clamp_min(0)]


def ss_loss(model,seq,candidates,nneg=128,temp=.05):
    out=model.encode_all(seq); sup=seq[:,1:]; q=out[:,:-1]; valid=sup!=0
    q=q[valid]; posid=sup[valid]
    pos=model.item_norm(posid); pl=(q*pos).sum(-1,keepdim=True)/temp
    off=torch.randint(0,candidates.numel(),(posid.numel(),nneg),device=seq.device)
    nid=candidates[off]; neg=model.item_norm(nid); nl=(q[:,None]*neg).sum(-1)/temp
    nl=torch.where(nid==posid[:,None],torch.full_like(nl,-5e4),nl)
    return -F.log_softmax(torch.cat([pl,nl],1),1)[:,0].mean()


@torch.inference_mode()
def evaluate(model,hist,lens,tgt,candidates,max_item_id,device,bs=512):
    model.eval(); c=candidates.to(device); ce=model.item_norm(c)
    id2col=torch.full((max_item_id+1,),-1,device=device,dtype=torch.long)
    id2col[c]=torch.arange(c.numel(),device=device)
    hit=ndcg=mrr=0.; n=tgt.numel()
    for st in range(0,n,bs):
        en=min(n,st+bs); x=hist[st:en].to(device); l=lens[st:en].to(device); y=tgt[st:en].to(device)
        s=model.encode_last(x,l)@ce.T
        cols=id2col[x.clamp(0,max_item_id)]; rows=torch.arange(en-st,device=device)[:,None].expand_as(cols)
        good=cols>=0; s[rows[good],cols[good]]=-1e20  # exact Meta invalid_ids behavior.
        top=s.topk(10,1).indices; tc=id2col[y]; eq=top.eq(tc[:,None]); ok=eq.any(1); rank=eq.float().argmax(1)+1
        hit+=float(ok.sum()); rr=rank[ok].float()
        if rr.numel(): ndcg+=float((1/torch.log2(rr+1)).sum()); mrr+=float((1/rr).sum())
    return {'HR@10':hit/n,'NDCG@10':ndcg/n,'MRR@10':mrr/n}


def stat(v):
    a=np.asarray(v); return {'mean_us':float(a.mean()),'p50_us':float(np.percentile(a,50)),
        'p95_us':float(np.percentile(a,95)),'p99_us':float(np.percentile(a,99))}


@torch.inference_mode()
def wall(fn,n=160,warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); v=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); torch.cuda.synchronize(); v.append((time.perf_counter()-t)*1e6)
    return stat(v)


@torch.inference_mode()
def gpu(fn,n=160,warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); v=[]; a=torch.cuda.Event(True); b=torch.cuda.Event(True)
    for _ in range(n):
        a.record(); fn(); b.record(); b.synchronize(); v.append(a.elapsed_time(b)*1000)
    return stat(v)


def latency(model,hist,lens,candidates,max_item_id,device):
    i=int(torch.where(lens==200)[0][0]); x=hist[i:i+1].to(device).contiguous(); l=lens[i:i+1].to(device)
    c=candidates.to(device); ce=model.item_norm(c).detach()
    id2col=torch.full((max_item_id+1,),-1,device=device,dtype=torch.long); id2col[c]=torch.arange(c.numel(),device=device)
    scols=id2col[x[0]]; scols=scols[scols>=0].unique(); smask=torch.zeros(1,c.numel(),device=device,dtype=torch.bool); smask[:,scols]=True
    result={}
    for tag,bf16 in [('FP32_TF32',False),('BF16_autocast',True)]:
        def model_fn():
            if bf16:
                with torch.autocast('cuda',dtype=torch.bfloat16): return model.encode_last(x,l)
            return model.encode_last(x,l)
        def mips_fn():
            if bf16:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    q=model.encode_last(x,l); s=(q@ce.T).masked_fill(smask,-1e20); return torch.topk(s,10,1)
            q=model.encode_last(x,l); s=(q@ce.T).masked_fill(smask,-1e20); return torch.topk(s,10,1)
        r={'eager':{'model_wall':wall(model_fn),'full_MIPS_wall':wall(mips_fn),
                    'model_gpu':gpu(model_fn),'full_MIPS_gpu':gpu(mips_fn)}}
        try:
            comp=torch.compile(mips_fn,mode='reduce-overhead',fullgraph=False); ref=mips_fn(); got=comp(); torch.cuda.synchronize()
            ok=bool(torch.equal(ref.indices,got.indices)); diff=float((ref.values.float()-got.values.float()).abs().max().cpu())
            r['compile_correctness']={'top10_ids_match':ok,'max_abs_score':diff,'accepted':ok and diff<.05}
            if r['compile_correctness']['accepted']:
                r['compiled']={'full_MIPS_wall':wall(comp),'full_MIPS_gpu':gpu(comp)}
        except Exception as e: r['compile_error']=repr(e)
        # Lower bound only; never headline.
        try:
            for _ in range(30): mips_fn()
            torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): mips_fn()
            r['cuda_graph_lower_bound']={'wall':wall(g.replay),'gpu':gpu(g.replay)}
        except Exception as e: r['cuda_graph_error']=repr(e)
        result[tag]=r
    return result


def save_ck(path,model,opt,epoch,best,history):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'history':history},path)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',default='/content/meta_ml1m')
    p.add_argument('--checkpoint',default='/content/drive/MyDrive/sparsewalker_meta_sasrec/ml1m/seed42/meta_sasrec.pt')
    p.add_argument('--output',default='/content/drive/MyDrive/sparsewalker_meta_sasrec/ml1m/seed42/result.json')
    p.add_argument('--epochs',type=int,default=101); p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=128); p.add_argument('--seed',type=int,default=42); p.add_argument('--force',action='store_true')
    a=p.parse_args(); assert torch.cuda.is_available(); dev=torch.device('cuda'); seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.set_float32_matmul_precision('high')
    data=load_data(a.data_dir); train,hist,lens,tgt=make_tensors(data); candidates=torch.tensor(data['all_movie_ids'],dtype=torch.long); cdev=candidates.to(dev)
    manifest={'paper_target_NDCG@10':PAPER_NDCG,'max_history':200,'d':50,'blocks':2,'heads':1,'ffn':50,
      'activation':'relu','dropout':.2,'loss':'local sampled softmax','negatives':128,'temperature':.05,
      'item_l2':True,'user_l2':True,'optimizer':'AdamW betas .9/.98 lr1e-3 wd0','epochs':101,
      'FP32_model':True,'TF32':True,'raw_ids_not_remapped':True,'users':data['users'],'ratings':data['ratings'],
      'rated_unique_items':data['rated_unique'],'candidate_movie_ids':data['candidate_movies'],'max_item_id':data['max_item_id'],
      'timestamp_tied_rows':data['tied_timestamp_rows'],'device':torch.cuda.get_device_name(0)}
    print('META_PROTOCOL',json.dumps(manifest,indent=2),flush=True)
    model=MetaSASRec(data['max_item_id']).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(.9,.98),weight_decay=0)
    start=1; best=-1.; history=[]; ck=Path(a.checkpoint)
    if ck.exists() and not a.force:
        z=torch.load(ck,map_location=dev); model.load_state_dict(z['model']); opt.load_state_dict(z['optimizer'])
        start=int(z['epoch'])+1; best=float(z.get('best',-1)); history=list(z.get('history',[])); print('RESUME',start,flush=True)
    n=train.size(0)
    for ep in range(start,a.epochs+1):
        model.train(); perm=torch.randperm(n); ls=0.; nb=0; t0=time.perf_counter()
        for st in range(0,n,a.batch_size):
            x=train[perm[st:st+a.batch_size]].to(dev); opt.zero_grad(set_to_none=True)
            loss=ss_loss(model,x,cdev); loss.backward(); opt.step(); ls+=float(loss.detach()); nb+=1
        if ep==1 or ep%a.eval_every==0 or ep==a.epochs:
            m=evaluate(model,hist,lens,tgt,candidates,data['max_item_id'],dev); row={'epoch':ep,'loss':ls/max(1,nb),'train_s':time.perf_counter()-t0,**m}
            history.append(row); best=max(best,m['NDCG@10']); print('EPOCH',row,flush=True); save_ck(ck,model,opt,ep,best,history)
    q=evaluate(model,hist,lens,tgt,candidates,data['max_item_id'],dev); gate=abs(q['NDCG@10']-PAPER_NDCG)<=.01
    print('META_SASREC_QUALITY',json.dumps({**q,'paper_NDCG@10':PAPER_NDCG,'delta':q['NDCG@10']-PAPER_NDCG,'quality_gate':gate},indent=2),flush=True)
    model.eval(); lat=latency(model,hist,lens,candidates,data['max_item_id'],dev); print('META_SASREC_LATENCY',json.dumps(lat,indent=2),flush=True)
    head={'NDCG@10':q['NDCG@10'],'paper_NDCG@10':PAPER_NDCG,'paper_comparable_quality_gate':gate,
      'latency_policy':'synchronized batch1 wall-clock; exact 200-window recompute + exhaustive raw movie-catalog MIPS + seen filtering',
      'eager_FP32_full_MIPS_p50_us':lat['FP32_TF32']['eager']['full_MIPS_wall']['p50_us'],
      'compiled_FP32_full_MIPS_p50_us':lat['FP32_TF32'].get('compiled',{}).get('full_MIPS_wall',{}).get('p50_us'),
      'BF16_eager_full_MIPS_p50_us':lat['BF16_autocast']['eager']['full_MIPS_wall']['p50_us'],
      'cuda_graph_is_lower_bound_only':True}
    print('META_SASREC_HEADLINE',json.dumps(head,indent=2),flush=True)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'manifest':manifest,'history':history,'quality':q,'latency':lat,'headline':head},indent=2,sort_keys=True)); print('SAVED',out,flush=True)

if __name__=='__main__': main()
