import torch.nn as nn
from .core import ARRecommender, init_embedding

class GRU4Rec(ARRecommender):
    def __init__(self,n_items,max_len,d=64,layers=2,dropout=.2):
        super().__init__(n_items,max_len,d)
        self.item=nn.Embedding(n_items+2,d,padding_idx=0); init_embedding(self.item)
        self.drop=nn.Dropout(dropout)
        self.gru=nn.GRU(d,d,num_layers=layers,batch_first=True,dropout=dropout if layers>1 else 0)
        self.norm=nn.LayerNorm(d)
    @property
    def item_weight(self): return self.item.weight
    def encode(self,seq):
        x=self.drop(self.item(seq)); h,_=self.gru(x); return self.norm(h)
