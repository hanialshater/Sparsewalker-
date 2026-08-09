import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import ARRecommender, init_embedding

class FFN(nn.Module):
    def __init__(self,d,inner,dropout,swiglu=False):
        super().__init__(); self.swiglu=swiglu
        if swiglu:
            self.up=nn.Linear(d,inner,bias=False); self.gate=nn.Linear(d,inner,bias=False); self.down=nn.Linear(inner,d,bias=False)
        else:
            self.net=nn.Sequential(nn.Linear(d,inner),nn.GELU(),nn.Dropout(dropout),nn.Linear(inner,d))
        self.drop=nn.Dropout(dropout)
    def forward(self,x):
        if self.swiglu: return self.drop(self.down(F.silu(self.up(x))*self.gate(x)))
        return self.net(x)

class SASBlock(nn.Module):
    def __init__(self,d,heads,inner,dropout,ligr=False):
        super().__init__(); self.ligr=ligr
        self.n1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True)
        self.n2=nn.LayerNorm(d); self.ffn=FFN(d,inner,dropout,swiglu=ligr)
        self.d1=nn.Dropout(dropout); self.d2=nn.Dropout(dropout)
        if ligr:
            self.g1=nn.Linear(d,d,bias=False); self.g2=nn.Linear(d,d,bias=False)
    def forward(self,x,padding):
        L=x.size(1)
        causal=torch.triu(torch.ones(L,L,dtype=torch.bool,device=x.device),diagonal=1)
        z=self.n1(x)
        a,_=self.attn(z,z,z,attn_mask=causal,key_padding_mask=padding,need_weights=False)
        a=self.d1(a); x=x+(a*torch.sigmoid(self.g1(x)) if self.ligr else a)
        f=self.d2(self.ffn(self.n2(x)))
        return x+(f*torch.sigmoid(self.g2(x)) if self.ligr else f)

class SASRec(ARRecommender):
    def __init__(self,n_items,max_len,d=64,layers=2,heads=2,inner=256,dropout=.2,ligr=False):
        super().__init__(n_items,max_len,d)
        self.item=nn.Embedding(n_items+1,d,padding_idx=0); init_embedding(self.item)
        self.pos=nn.Embedding(max_len,d); init_embedding(self.pos)
        self.drop=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([SASBlock(d,heads,inner,dropout,ligr=ligr) for _ in range(layers)])
        self.norm=nn.LayerNorm(d)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    @property
    def item_weight(self): return self.item.weight
    def encode(self,seq):
        _,L=seq.shape; p=torch.arange(L,device=seq.device)[None]
        x=self.drop(self.item(seq)+self.pos(p)); padding=seq==0
        for b in self.blocks: x=b(x,padding)
        return self.norm(x)
