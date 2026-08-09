import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import init_embedding, sampled_softmax_from_hidden

class BERT4Rec(nn.Module):
    def __init__(self,n_items,max_len,d=64,layers=2,heads=2,inner=256,dropout=.2):
        super().__init__(); self.n_items=n_items; self.max_len=max_len; self.d_model=d; self.mask_id=n_items+1
        self.item=nn.Embedding(n_items+2,d,padding_idx=0); init_embedding(self.item)
        self.pos=nn.Embedding(max_len,d); init_embedding(self.pos)
        layer=nn.TransformerEncoderLayer(d,heads,inner,dropout,activation="gelu",batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(layer,layers); self.norm=nn.LayerNorm(d); self.drop=nn.Dropout(dropout)
    @property
    def item_weight(self): return self.item.weight
    def encode(self,seq):
        _,L=seq.shape; p=torch.arange(L,device=seq.device)[None]
        x=self.drop(self.item(seq)+self.pos(p)); x=self.enc(x,src_key_padding_mask=(seq==0)); return self.norm(x)
    def masked_loss(self,tokens,lengths,mask_prob=.2,loss_mode="full",n_negs=256):
        x=tokens[:,-self.max_len:].clone(); real=x!=0; rand=torch.rand_like(x.float()); mask=(rand<mask_prob)&real
        for r in range(x.size(0)):
            idx=real[r].nonzero(as_tuple=False).flatten()
            if idx.numel() and not mask[r].any(): mask[r,idx[torch.randint(idx.numel(),(1,),device=x.device)]]=True
        target=x.clone(); inp=x.clone(); u=torch.rand_like(inp.float())
        inp[mask&(u<0.8)]=self.mask_id
        random_mask=mask&(u>=0.8)&(u<0.9)
        inp[random_mask]=torch.randint(1,self.n_items+1,(int(random_mask.sum()),),device=x.device)
        H=self.encode(inp); h=H[mask]; y=target[mask]
        if loss_mode=="ss": return sampled_softmax_from_hidden(H,target.masked_fill(~mask,0),self.item_weight,self.n_items,n_negs)
        logits=h@self.item.weight[:self.n_items+1].T; logits[:,0]=-1e9
        return F.cross_entropy(logits,y)
    def full_scores(self,seq,lengths):
        B=seq.size(0); out=[]
        for r in range(B):
            n=int(lengths[r].item()); hist=seq[r,:n][-(self.max_len-1):]
            z=torch.zeros(self.max_len,dtype=torch.long,device=seq.device); z[:hist.numel()]=hist; z[hist.numel()]=self.mask_id; out.append(z)
        z=torch.stack(out); H=self.encode(z)
        idx=torch.tensor([min(int(l.item()),self.max_len-1) for l in lengths],device=seq.device); rows=torch.arange(B,device=seq.device); h=H[rows,idx]
        scores=h@self.item.weight[:self.n_items+1].T; scores[:,0]=-1e9; return scores
