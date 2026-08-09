import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import ARRecommender, init_embedding, autoregressive_inputs


def _coalesced_topk(ids, mass, k):
    """Sum duplicate concept masses before pruning to the top-k states."""
    same = ids.unsqueeze(-1).eq(ids.unsqueeze(-2))
    summed = (same.to(mass.dtype) * mass.unsqueeze(-2)).sum(-1)
    repeated = torch.tril(same, diagonal=-1).any(-1)
    summed = summed.masked_fill(repeated, 0.0)
    v, slot = summed.topk(min(int(k), summed.size(-1)), dim=-1)
    out_ids = ids.gather(-1, slot)
    v = v / (v.sum(-1, keepdim=True) + 1e-8)
    return out_ids, v


class ConceptSpace(nn.Module):
    def __init__(self,d,side,h):
        super().__init__(); self.side=side; self.h=h
        self.left_router=nn.Parameter(torch.randn(side,h)/math.sqrt(h)); self.right_router=nn.Parameter(torch.randn(side,h)/math.sqrt(h))
        self.left_key=nn.Parameter(torch.randn(side,h)/math.sqrt(h)); self.right_key=nn.Parameter(torch.randn(side,h)/math.sqrt(h))
        self.left_value=nn.Embedding(side,d); self.right_value=nn.Embedding(side,d); init_embedding(self.left_value); init_embedding(self.right_value)
        self.value_proj=nn.Linear(2*d,d)
    def split(self,ids): return ids//self.side,ids%self.side
    def value(self,ids):
        l,r=self.split(ids); return self.value_proj(torch.cat([self.left_value(l),self.right_value(r)],-1))
    def key(self,ids):
        l,r=self.split(ids); return F.normalize(self.left_key[l]+self.right_key[r],dim=-1)


class Router(nn.Module):
    def __init__(self,d,h,top_side,side):
        super().__init__(); self.top_side=top_side; self.side=side
        self.left_q=nn.Linear(d,h,bias=False); self.right_q=nn.Linear(d,h,bias=False); self.scale=nn.Parameter(torch.tensor(math.log(10.0)))
    def forward(self,hidden,space):
        ql=F.normalize(self.left_q(hidden),dim=-1); qr=F.normalize(self.right_q(hidden),dim=-1)
        kl=F.normalize(space.left_router,dim=-1); kr=F.normalize(space.right_router,dim=-1); sc=torch.exp(self.scale)
        lv,li=(ql@kl.T*sc).topk(self.top_side,-1); rv,ri=(qr@kr.T*sc).topk(self.top_side,-1)
        ids=(li.unsqueeze(-1)*self.side+ri.unsqueeze(-2)).reshape(hidden.size(0),-1)
        logits=(lv.unsqueeze(-1)+rv.unsqueeze(-2)).reshape(hidden.size(0),-1)
        return ids,F.softmax(logits,-1)


class CompactGraph(nn.Module):
    def __init__(self,d,h,n_concepts,degree,active):
        super().__init__(); self.n_concepts=n_concepts; self.degree=degree; self.active=active
        self.edge_logits=nn.Embedding(n_concepts,degree); nn.init.normal_(self.edge_logits.weight,std=.02)
        dest=torch.randint(0,n_concepts,(n_concepts,degree),dtype=torch.int32); dest[:,0]=torch.arange(n_concepts,dtype=torch.int32)
        self.register_buffer("destination",dest); self.register_buffer("touched",torch.zeros(n_concepts,dtype=torch.bool),persistent=False)
        self.context_q=nn.Linear(d,h,bias=False); self.scale=nn.Parameter(torch.tensor(math.log(3.0)))
    def topk(self,ids,mass):
        return _coalesced_topk(ids,mass,self.active)
    @torch.no_grad()
    def mark_touched(self,ids):
        if ids.numel(): self.touched[ids.detach().reshape(-1)]=True
    def forward(self,ids,mass,context,space,track_touched=True):
        if self.training and track_touched: self.mark_touched(ids)
        dest=self.destination[ids].long(); static=self.edge_logits(ids); q=F.normalize(self.context_q(context),dim=-1); key=space.key(dest)
        score=static+torch.exp(self.scale)*(key*q[:,None,None,:]).sum(-1); prob=F.softmax(score,-1)
        B=ids.size(0); return self.topk(dest.reshape(B,-1),(mass.unsqueeze(-1)*prob).reshape(B,-1))
    @torch.no_grad()
    def pursue(self,opt,refresh):
        rows=self.touched.nonzero(as_tuple=False).squeeze(-1)
        if rows.numel()==0: return 0
        w=self.edge_logits.weight[rows]; repl=w.abs().argsort(-1)[:,:refresh]; rr=rows[:,None].expand_as(repl)
        self.destination[rr,repl]=torch.randint(0,self.n_concepts,repl.shape,device=self.destination.device,dtype=torch.int32)
        self.edge_logits.weight[rr,repl]=0
        st=opt.state.get(self.edge_logits.weight,{})
        for k in ("exp_avg","exp_avg_sq"):
            if st.get(k) is not None: st[k][rr,repl]=0
        n=int(rows.numel()); self.touched.zero_(); return n


class SparseWalker(ARRecommender):
    def __init__(self,n_items,max_len,d=64,layers=2,side=256,h=16,active=8,top_side=2,degree=4,fresh_weight=.25):
        super().__init__(n_items,max_len,d); self.layers_n=layers; self.side=side; self.h=h; self.active=active; self.degree=degree; self.fresh_weight=fresh_weight
        self.n_concepts=side*side; self.fresh_concepts=top_side*top_side
        self.item=nn.Embedding(n_items+1,d,padding_idx=0); init_embedding(self.item)
        self.space=ConceptSpace(d,side,h); self.router=Router(d,h,top_side,side); self.graph=CompactGraph(d,h,self.n_concepts,degree,active)
        self.message_proj=nn.Linear(d,d,bias=False); self.norm=nn.LayerNorm(d)
    @property
    def item_weight(self): return self.item.weight
    def _top(self,ids,mass):
        return _coalesced_topk(ids,mass,self.active)
    def _merge(self,oi,om,fi,fm):
        return self._top(torch.cat([oi,fi],-1),torch.cat([(1-self.fresh_weight)*om,self.fresh_weight*fm],-1))
    def _encode_impl(self,seq,return_states):
        B,L=seq.shape; valid=seq!=0; item_state=self.item(seq)*math.sqrt(self.d_model)
        fi,fm=self.router(item_state.reshape(B*L,self.d_model),self.space); fi=fi.view(B,L,-1); fm=fm.view(B,L,-1)
        ids=torch.zeros(B,self.active,dtype=torch.long,device=seq.device); mass=torch.zeros(B,self.active,dtype=item_state.dtype,device=seq.device)
        outs=[]; ids_hist=[] if return_states else None; mass_hist=[] if return_states else None
        touched_sources=[] if self.training else None
        for t in range(L):
            act=valid[:,t]; af=act.to(item_state.dtype)[:,None]; xids=ids; xmass=mass*af; fids=fi[:,t]; fmass=fm[:,t]*af

            # v1.1: inject the current item once per event, not once per graph hop.
            xids,xmass=self._merge(xids,xmass,fids,fmass)
            for _ in range(self.layers_n):
                if touched_sources is not None: touched_sources.append(xids.detach())
                xids,xmass=self.graph(xids,xmass,item_state[:,t],self.space,track_touched=False)

            ids=torch.where(act[:,None],xids,ids); mass=torch.where(act[:,None],xmass,mass)
            msg=(self.space.value(ids)*mass[:,:,None]).sum(1); h=self.norm(item_state[:,t]+self.message_proj(msg))*af
            outs.append(h)
            if return_states:
                ids_hist.append(ids); mass_hist.append(mass)
        if touched_sources:
            self.graph.mark_touched(torch.cat([x.reshape(-1) for x in touched_sources],0))
        H=torch.stack(outs,1)
        if return_states: return H,torch.stack(ids_hist,1),torch.stack(mass_hist,1)
        return H
    def encode_with_states(self,seq): return self._encode_impl(seq,True)
    def encode(self,seq): return self._encode_impl(seq,False)


class SparseWalkerE2E(SparseWalker):
    def __init__(self,n_items,max_len,d=64,layers=2,side=256,h=16,active=8,top_side=2,degree=4,fresh_weight=.25,terminal_degree=128):
        super().__init__(n_items,max_len,d,layers,side,h,active,top_side,degree,fresh_weight)
        self.terminal_degree=int(terminal_degree)
        init=torch.randint(1,n_items+1,(self.n_concepts,self.terminal_degree),dtype=torch.int32)
        self.register_buffer("terminal_items",init,persistent=True); self._evidence_pairs=[]; self._evidence_weights=[]
    @torch.no_grad()
    def initialize_terminal(self,popular_items):
        pop=torch.as_tensor(popular_items,dtype=torch.long,device=self.terminal_items.device); pop=pop[(pop>0)&(pop<=self.n_items)]
        if pop.numel()==0: pop=torch.arange(1,min(self.n_items,self.terminal_degree)+1,device=self.terminal_items.device)
        if pop.numel()<self.terminal_degree: pop=pop.repeat(math.ceil(self.terminal_degree/max(1,pop.numel())))
        row=pop[:self.terminal_degree].to(torch.int32); self.terminal_items.copy_(row[None,:].expand(self.n_concepts,-1))
    def begin_terminal_epoch(self): self._evidence_pairs=[]; self._evidence_weights=[]
    @torch.no_grad()
    def _record_terminal_evidence(self,ids,mass,target,max_rows):
        n=target.numel()
        if n==0: return
        if n>max_rows:
            pick=torch.randperm(n,device=target.device)[:max_rows]; ids=ids[pick]; mass=mass[pick]; target=target[pick]
        pair=ids.long()*(self.n_items+1)+target[:,None].long(); self._evidence_pairs.append(pair.reshape(-1).cpu()); self._evidence_weights.append(mass.float().reshape(-1).cpu())
    def sparse_training_loss(self,tokens,evidence_per_batch=2048,chunk=256):
        x,y=autoregressive_inputs(tokens,None); H,I,M=self.encode_with_states(x); valid=y!=0
        h=H[valid]; ids=I[valid]; mass=M[valid]; tgt=y[valid]
        if h.numel()==0: return H.sum()*0
        self._record_terminal_evidence(ids.detach(),mass.detach(),tgt.detach(),evidence_per_batch)
        total=h.new_zeros((),dtype=torch.float32); count=0; item_w=self.item.weight
        for st in range(0,h.size(0),chunk):
            en=min(h.size(0),st+chunk); hh=h[st:en]; ii=ids[st:en]; yy=tgt[st:en]
            cand=self.terminal_items[ii].long().reshape(en-st,-1); cs=(item_w[cand]*hh[:,None,:]).sum(-1).float()
            table=torch.full((en-st,self.n_items+1),-1e9,device=hh.device,dtype=torch.float32); table=table.scatter_reduce(1,cand,cs,reduce="amax",include_self=True)
            pos=(hh*item_w[yy]).sum(-1).float(); table=table.scatter_reduce(1,yy[:,None],pos[:,None],reduce="amax",include_self=True)
            total=total+F.cross_entropy(table[:,1:],yy-1,reduction="sum"); count+=en-st
        return total/max(1,count)
    @torch.no_grad()
    def refresh_terminal(self):
        if not self._evidence_pairs: return 0
        pairs=torch.cat(self._evidence_pairs); weights=torch.cat(self._evidence_weights).float(); order=torch.argsort(pairs); pairs=pairs[order]; weights=weights[order]
        up,inv=torch.unique_consecutive(pairs,return_inverse=True); agg=torch.zeros(up.numel(),dtype=torch.float32); agg.scatter_add_(0,inv,weights)
        concept=up//(self.n_items+1); item=up%(self.n_items+1); order=torch.argsort(agg,descending=True,stable=True); concept=concept[order]; item=item[order]
        order=torch.argsort(concept,stable=True); concept=concept[order]; item=item[order]
        counts=torch.bincount(concept,minlength=self.n_concepts); starts=torch.cumsum(counts,0)-counts; local=torch.arange(concept.numel())-torch.repeat_interleave(starts,counts); keep=(local<self.terminal_degree)&(item!=0)
        kc=concept[keep].to(self.terminal_items.device); ki=item[keep].to(self.terminal_items.device,dtype=torch.int32); kr=local[keep].to(self.terminal_items.device); self.terminal_items[kc,kr]=ki
        touched=int(torch.unique(concept).numel()); self.begin_terminal_epoch(); return touched
    @torch.inference_mode()
    def full_scores(self,seq,lengths):
        H,I,_=self.encode_with_states(seq); rows=torch.arange(seq.size(0),device=seq.device); ix=lengths-1; h=H[rows,ix]; ids=I[rows,ix]
        cand=self.terminal_items[ids].long().reshape(seq.size(0),-1); cs=(self.item.weight[cand]*h[:,None,:]).sum(-1).float()
        table=torch.full((seq.size(0),self.n_items+1),-1e9,device=seq.device,dtype=torch.float32); table=table.scatter_reduce(1,cand,cs,reduce="amax",include_self=True); table[:,0]=-1e9; return table
