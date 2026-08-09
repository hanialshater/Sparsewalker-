import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import ARRecommender, init_embedding

class FMLPFilterBlock(nn.Module):
    def __init__(self,max_len,d,inner,dropout):
        super().__init__()
        self.real=nn.Parameter(torch.randn(1,max_len//2+1,d)*0.02)
        self.imag=nn.Parameter(torch.randn(1,max_len//2+1,d)*0.02)
        self.filter_drop=nn.Dropout(dropout); self.filter_ln=nn.LayerNorm(d,eps=1e-12)
        self.ff1=nn.Linear(d,inner); self.ff2=nn.Linear(inner,d)
        self.ff_drop=nn.Dropout(dropout); self.ff_ln=nn.LayerNorm(d,eps=1e-12)
    def forward(self,x,pad):
        _,L,_=x.shape
        xf=torch.fft.rfft(x.float(),dim=1,norm="ortho")
        w=torch.complex(self.real[:,:xf.size(1)],self.imag[:,:xf.size(1)])
        y=torch.fft.irfft(xf*w,n=L,dim=1,norm="ortho").to(x.dtype)
        h=self.filter_ln(x+self.filter_drop(y))
        f=self.ff2(self.ff_drop(F.gelu(self.ff1(h))))
        h=self.ff_ln(h+self.ff_drop(f))
        return h.masked_fill(pad[:,:,None],0.0)

class FMLPRec(ARRecommender):
    def __init__(self,n_items,max_len,d=64,layers=2,inner=256,dropout=.2):
        super().__init__(n_items,max_len,d)
        self.item=nn.Embedding(n_items+2,d,padding_idx=0); init_embedding(self.item)
        self.pos=nn.Embedding(max_len,d); init_embedding(self.pos)
        self.in_ln=nn.LayerNorm(d,eps=1e-12); self.drop=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([FMLPFilterBlock(max_len,d,inner,dropout) for _ in range(layers)])
    @property
    def item_weight(self): return self.item.weight
    def encode(self,seq):
        _,L=seq.shape; p=torch.arange(L,device=seq.device)[None]
        x=self.drop(self.in_ln(self.item(seq)+self.pos(p))); pad=seq==0
        for b in self.blocks: x=b(x,pad)
        return x
