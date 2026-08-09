import torch
from torch.utils.data import DataLoader
from sparsewalker.data import collate_windows
from sparsewalker.models.core import ar_training_loss


def train_epoch(name,model,dataset,optimizer,device,batch_size=512,epoch=1,loss_mode="full",n_negs=256,mask_prob=.2,grad_clip=5.0):
    dataset.set_epoch(epoch)
    g=torch.Generator(); g.manual_seed(dataset.seed+epoch)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,generator=g,collate_fn=collate_windows,pin_memory=device.type=="cuda")
    model.train(); total=0.; n=0
    if name=="SparseWalker-E2E": model.begin_terminal_epoch()
    for tokens,lengths in loader:
        tokens=tokens.to(device); lengths=lengths.to(device); optimizer.zero_grad(set_to_none=True)
        if name=="BERT4Rec": loss=model.masked_loss(tokens,lengths,mask_prob,loss_mode,n_negs)
        elif name=="SparseWalker-E2E": loss=model.sparse_training_loss(tokens)
        else: loss=ar_training_loss(model,tokens,lengths,loss_mode,n_negs)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip); optimizer.step(); total+=float(loss.detach()); n+=1
    return total/max(1,n)
