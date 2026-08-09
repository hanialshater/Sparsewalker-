import torch
import torch.nn as nn

class MostPop(nn.Module):
    def __init__(self,n_items,freq):
        super().__init__(); self.n_items=n_items
        scores=torch.zeros(n_items+1)
        for i,c in freq.items():
            if 0<int(i)<=n_items: scores[int(i)]=float(c)
        scores[0]=-1e9
        self.register_buffer("scores",scores)
    def full_scores(self,seq,lengths):
        return self.scores[None].expand(seq.size(0),-1)
