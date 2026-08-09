#!/usr/bin/env python
import argparse
import random
import numpy as np
import torch

from sparsewalker.data import DATASETS, load_dataset, split_data, WindowDataset
from sparsewalker.evaluation import evaluate_full
from sparsewalker.training import train_epoch
from model_registry import LEARNED, build_model, loss_mode


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=DATASETS,default="beauty")
    p.add_argument("--model",choices=LEARNED,default="SASRec")
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--data-dir",default="/content/sparsewalker_data")
    args=p.parse_args(); seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=load_dataset(args.dataset,args.data_dir); split=split_data(data["sequences"]); max_len=data["spec"].default_max_len
    model=build_model(args.model,data["n_items"],max_len).to(device)
    ds=WindowDataset(split["train"],max_len,args.seed); opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=-1.; best_state=None
    for epoch in range(1,args.epochs+1):
        loss=train_epoch(args.model,model,ds,opt,device,args.batch_size,epoch,loss_mode(args.model))
        val=evaluate_full(model,split["val_prefix"],split["val_target"],data["n_items"],max_len,device,topks=(10,),batch_size=1024)
        print({"epoch":epoch,"loss":loss,**val})
        if val["NDCG@10"]>best: best=val["NDCG@10"]; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    test=evaluate_full(model,split["test_prefix"],split["test_target"],data["n_items"],max_len,device,topks=(10,20,50),batch_size=1024)
    print("TEST",test)

if __name__=="__main__": main()
