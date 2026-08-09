import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import ARRecommender, init_embedding
try:
    from mamba_ssm import Mamba
    HAS_MAMBA=True
except Exception:
    Mamba=None
    HAS_MAMBA=False

class MambaRecLayer(nn.Module):
    def __init__(self,d,d_state,d_conv,expand,dropout,n_layers):
        super().__init__()
        if not HAS_MAMBA:
            raise RuntimeError('mamba_ssm is not installed; portable no-compile Mamba is a queued baseline task')
        self.n_layers=n_layers
        self.mamba=Mamba(d_model=d,d_state=d_state,d_conv=d_conv,expand=expand)
        self.drop=nn.Dropout(dropout); self.norm=nn.LayerNorm(d,eps=1e-12)
        self.ff1=nn.Linear(d,4*d); self.ff2=nn.Linear(4*d,d)
        self.ffdrop=nn.Dropout(dropout); self.ffnorm=nn.LayerNorm(d,eps=1e-12)
    def forward(self,x):
        h=self.mamba(x)
        h=self.norm(self.drop(h)+x) if self.n_layers>1 else self.norm(self.drop(h))
        f=self.ff2(self.ffdrop(F.gelu(self.ff1(h)))); f=self.ffdrop(f)
        return self.ffnorm(f+h)

class Mamba4Rec(ARRecommender):
    def __init__(self,n_items,max_len,d=64,layers=2,dropout=.2,d_state=32,d_conv=4,expand=2):
        super().__init__(n_items,max_len,d)
        self.item=nn.Embedding(n_items+2,d,padding_idx=0); init_embedding(self.item)
        self.in_norm=nn.LayerNorm(d,eps=1e-12); self.drop=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([MambaRecLayer(d,d_state,d_conv,expand,dropout,layers) for _ in range(layers)])
        self.apply(self._init)
        with torch.no_grad(): self.item.weight[0].zero_()
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,0,0.02)
        if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m,nn.LayerNorm): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    @property
    def item_weight(self): return self.item.weight
    def encode(self,seq):
        x=self.in_norm(self.drop(self.item(seq)))
        for b in self.blocks: x=b(x)
        return x.masked_fill((seq==0)[:,:,None],0.0)
