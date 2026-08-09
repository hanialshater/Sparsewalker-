import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def truncated_normal_(x: torch.Tensor, mean: float, std: float):
    with torch.no_grad():
        size=x.shape
        tmp=x.new_empty(size+(4,)).normal_()
        valid=(tmp<2)&(tmp>-2)
        ind=valid.max(-1,keepdim=True)[1]
        x.copy_(tmp.gather(-1,ind).squeeze(-1))
        x.mul_(std).add_(mean)
    return x


class RelativeBucketedTimeAndPositionBias(nn.Module):
    def __init__(self,n:int,num_buckets:int=128):
        super().__init__(); self.n=n; self.num_buckets=num_buckets
        self.ts_w=nn.Parameter(torch.empty(num_buckets+1)); self.pos_w=nn.Parameter(torch.empty(2*n-1))
        nn.init.normal_(self.ts_w,mean=0.0,std=0.02); nn.init.normal_(self.pos_w,mean=0.0,std=0.02)
        i=torch.arange(n)[:,None]; j=torch.arange(n)[None,:]
        self.register_buffer("rel_pos_idx",(j-i+(n-1)).long(),persistent=False)
    def forward_from_buckets(self,time_buckets):
        return self.pos_w[self.rel_pos_idx].unsqueeze(0)+self.ts_w[time_buckets]


class HSTUBlock(nn.Module):
    def __init__(self,d_model,heads,dqk,dv,dropout,n):
        super().__init__(); self.d_model=d_model; self.heads=heads; self.dqk=dqk; self.dv=dv; self.dropout=dropout; self.n=n; self.eps=1e-6
        total=2*dv*heads+2*dqk*heads
        self.uvqk=nn.Parameter(torch.empty(d_model,total)); nn.init.normal_(self.uvqk,mean=0.0,std=0.02)
        self.rel_bias=RelativeBucketedTimeAndPositionBias(n=n,num_buckets=128)
        self.o=nn.Linear(dv*heads,d_model); nn.init.xavier_uniform_(self.o.weight)
        self.register_buffer("causal",torch.tril(torch.ones(n,n,dtype=torch.float32)),persistent=False)
    def forward(self,x,time_buckets,lengths):
        B,N,D=x.shape; assert N==self.n
        valid=(torch.arange(N,device=x.device)[None,:]<lengths[:,None]).unsqueeze(-1)
        x=x*valid.to(x.dtype)
        z=F.silu(F.layer_norm(x,[D],eps=self.eps)@self.uvqk)
        split=[self.dv*self.heads,self.dv*self.heads,self.dqk*self.heads,self.dqk*self.heads]
        u,v,q,k=torch.split(z,split,dim=-1)
        valid_f=valid.to(z.dtype); u=u*valid_f; v=v*valid_f; q=q*valid_f; k=k*valid_f
        q=q.view(B,N,self.heads,self.dqk).permute(0,2,1,3).contiguous()
        k=k.view(B,N,self.heads,self.dqk).permute(0,2,1,3).contiguous()
        v=v.view(B,N,self.heads,self.dv).permute(0,2,1,3).contiguous()
        qk=q@k.transpose(-2,-1)
        qk=qk+self.rel_bias.forward_from_buckets(time_buckets).unsqueeze(1)
        qk=F.silu(qk)/N
        qk=qk*self.causal.view(1,1,N,N)
        attn=(qk@v).permute(0,2,1,3).contiguous().reshape(B,N,self.heads*self.dv)
        attn=F.layer_norm(attn,[self.heads*self.dv],eps=self.eps)
        u=u.view(B,N,self.heads*self.dv)
        y=self.o(F.dropout(u*attn,p=self.dropout,training=self.training))+x
        return y*valid.to(y.dtype)


class HSTU(nn.Module):
    def __init__(self,max_item_id,d_model,n,layers,heads,dqk,dv,dropout,l2_eps):
        super().__init__(); self.max_item_id=max_item_id; self.d_model=d_model; self.n=n; self.l2_eps=l2_eps
        self.item=nn.Embedding(max_item_id+1,d_model,padding_idx=0); truncated_normal_(self.item.weight,mean=0.0,std=0.02)
        with torch.no_grad(): self.item.weight[0].zero_()
        self.pos=nn.Embedding(n,d_model); truncated_normal_(self.pos.weight,mean=0.0,std=math.sqrt(1.0/d_model))
        self.input_dropout=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([HSTUBlock(d_model,heads,dqk,dv,dropout,n) for _ in range(layers)])
    def item_embeddings(self,ids): return self.item(ids)
    def encode_all(self,ids,timestamps,lengths):
        B,N=ids.shape; pos=torch.arange(N,device=ids.device)[None,:].expand(B,N)
        x=self.input_dropout(self.item(ids)*math.sqrt(self.d_model)+self.pos(pos))
        x=x*(ids!=0).unsqueeze(-1).to(x.dtype)
        ext_ts=torch.cat([timestamps,timestamps[:,N-1:N]],dim=1)
        delta=ext_ts[:,1:].unsqueeze(2)-ext_ts[:,:-1].unsqueeze(1)
        time_buckets=(torch.log(torch.abs(delta).clamp(min=1).float())/0.301).long().clamp(min=0,max=128).detach()
        for block in self.blocks: x=block(x,time_buckets,lengths)
        return F.normalize(x.float(),p=2,dim=-1,eps=self.l2_eps).to(x.dtype)
    def current_embedding(self,ids,timestamps,lengths):
        x=self.encode_all(ids,timestamps,lengths); row=torch.arange(ids.size(0),device=ids.device); return x[row,lengths-1]
