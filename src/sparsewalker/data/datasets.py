import gzip
import json
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    url: str
    filename: str
    archive_member: Optional[str]=None
    default_max_len: int=50

DATASETS={
    "beauty":DatasetSpec("beauty","amazon","https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz","reviews_Beauty_5.json.gz",default_max_len=50),
    "video_games":DatasetSpec("video_games","amazon","https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz","reviews_Video_Games_5.json.gz",default_max_len=50),
    "sports":DatasetSpec("sports","amazon","https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz","reviews_Sports_and_Outdoors_5.json.gz",default_max_len=50),
    "toys":DatasetSpec("toys","amazon","https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz","reviews_Toys_and_Games_5.json.gz",default_max_len=50),
    "ml1m":DatasetSpec("ml1m","ml1m","https://files.grouplens.org/datasets/movielens/ml-1m.zip","ml-1m.zip","ml-1m/ratings.dat",200),
    "ml20m":DatasetSpec("ml20m","ml20m","https://files.grouplens.org/datasets/movielens/ml-20m.zip","ml-20m.zip","ml-20m/ratings.csv",200),
}

def _download(url,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists(): urllib.request.urlretrieve(url,path)
    return path

def _finalize(events,min_user_len=5):
    by_user=defaultdict(list)
    for user,item,ts in events: by_user[str(user)].append((float(ts),str(item)))
    users=[]
    for u,vals in by_user.items():
        vals.sort(key=lambda x:x[0]); seq=[]; prev=None
        for _,it in vals:
            if it!=prev: seq.append(it); prev=it
        if len(seq)>=min_user_len: users.append((u,seq))
    item_map={it:i+1 for i,it in enumerate(sorted({it for _,s in users for it in s}))}
    seqs=[[item_map[it] for it in s] for _,s in users]
    freq=Counter(i for s in seqs for i in s)
    return {"sequences":seqs,"n_items":len(item_map),"frequency":freq}

def load_dataset(name,data_dir="/content/sparsewalker_data",min_user_len=5):
    spec=DATASETS[name]; path=_download(spec.url,Path(data_dir)/spec.filename)
    events=[]
    if spec.kind=="amazon":
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                r=json.loads(line); events.append((r["reviewerID"],r["asin"],r.get("unixReviewTime",0)))
    else:
        with zipfile.ZipFile(path) as z:
            with z.open(spec.archive_member) as fh:
                if spec.kind=="ml1m":
                    import io
                    df=pd.read_csv(io.TextIOWrapper(fh,encoding="latin-1"),sep="::",engine="python",names=["user","item","rating","ts"])
                else:
                    df=pd.read_csv(fh); df=df.rename(columns={"userId":"user","movieId":"item","timestamp":"ts"})
        events=list(df[["user","item","ts"]].itertuples(index=False,name=None))
    out=_finalize(events,min_user_len); out["spec"]=spec; return out

def split_data(sequences):
    train=[]; val_prefix=[]; val_target=[]; test_prefix=[]; test_target=[]
    for s in sequences:
        train.append(s[:-2]); val_prefix.append(s[:-2]); val_target.append(s[-2]); test_prefix.append(s[:-1]); test_target.append(s[-1])
    return {"train":train,"val_prefix":val_prefix,"val_target":val_target,"test_prefix":test_prefix,"test_target":test_target}

class WindowDataset(Dataset):
    def __init__(self,seqs,max_len,seed=42):
        self.seqs=[s for s in seqs if len(s)>=2]; self.max_len=max_len; self.seed=seed; self.epoch=0
    def set_epoch(self,epoch): self.epoch=epoch
    def __len__(self): return len(self.seqs)
    def __getitem__(self,idx):
        s=self.seqs[idx]; rng=random.Random(self.seed*1_000_003+self.epoch*97_409+idx)
        end=len(s) if len(s)<=self.max_len+1 else rng.randint(2,len(s)); start=max(0,end-(self.max_len+1)); seg=s[start:end]
        if len(seg)<2: seg=s[:2]
        return torch.tensor(seg,dtype=torch.long)

def collate_windows(batch):
    L=max(x.numel() for x in batch); out=torch.zeros(len(batch),L,dtype=torch.long); lengths=torch.empty(len(batch),dtype=torch.long)
    for r,x in enumerate(batch): out[r,:x.numel()]=x; lengths[r]=x.numel()
    return out,lengths
