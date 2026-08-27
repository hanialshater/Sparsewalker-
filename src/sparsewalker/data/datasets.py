import gzip
import json
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
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


def _load_ml1m_fast(path, data_dir, min_user_len=5):
    """Protocol-preserving fast parser for MovieLens-1M.

    The legacy loader used pandas' Python CSV engine for the '::' delimiter,
    which is very slow on Colab. This implementation parses all four integer
    columns with NumPy, preserves first-seen user order, timestamp ordering
    (stable for ties), consecutive-item de-duplication, and the existing
    lexicographic string item-ID remapping used by `_finalize`.

    The processed result is cached beside the archive. Subsequent runs skip the
    million-row parse entirely.
    """
    data_dir = Path(data_dir)
    cache = data_dir / f"ml1m_preprocessed_protocol_v1_min{int(min_user_len)}.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=False)

    with zipfile.ZipFile(path) as z:
        raw = z.read("ml-1m/ratings.dat")

    # ~24 MB input: replacing the delimiter and parsing in NumPy is much faster
    # than pandas engine='python'. Columns are user, item, rating, timestamp.
    text = raw.replace(b"::", b" ").decode("ascii")
    arr = np.fromstring(text, sep=" ", dtype=np.int64)
    if arr.size % 4:
        raise ValueError(f"Unexpected ML-1M field count: {arr.size}")
    arr = arr.reshape(-1, 4)
    user = arr[:, 0]
    item = arr[:, 1]
    ts = arr[:, 3]
    original_row = np.arange(arr.shape[0], dtype=np.int64)

    # Preserve the dict insertion order of the old `_finalize`: user order is
    # first occurrence in the source file, while events within a user are sorted
    # by timestamp and retain source order for timestamp ties.
    users_unique, first = np.unique(user, return_index=True)
    users_in_source_order = users_unique[np.argsort(first)]
    max_user = int(user.max())
    rank = np.empty(max_user + 1, dtype=np.int64)
    rank.fill(-1)
    rank[users_in_source_order] = np.arange(users_in_source_order.size)
    user_rank = rank[user]
    order = np.lexsort((original_row, ts, user_rank))
    user_s = user[order]
    item_s = item[order]

    starts = np.flatnonzero(np.r_[True, user_s[1:] != user_s[:-1]])
    ends = np.r_[starts[1:], user_s.size]
    raw_sequences = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        s = item_s[a:b]
        if s.size:
            keep = np.r_[True, s[1:] != s[:-1]]
            s = s[keep]
        if s.size >= min_user_len:
            raw_sequences.append(s)

    if not raw_sequences:
        raise ValueError("ML-1M preprocessing produced no users")

    # Match old string-based sorted item map exactly: '1','10','100',...,'2'.
    used_raw = np.unique(np.concatenate(raw_sequences))
    lex_items = sorted((int(x) for x in used_raw.tolist()), key=lambda x: str(x))
    lookup = np.zeros(int(used_raw.max()) + 1, dtype=np.int64)
    for mapped, raw_id in enumerate(lex_items, start=1):
        lookup[raw_id] = mapped

    seqs = [lookup[s].tolist() for s in raw_sequences]
    flat = np.concatenate([lookup[s] for s in raw_sequences])
    counts = np.bincount(flat, minlength=len(lex_items) + 1)
    freq = Counter({i: int(counts[i]) for i in range(1, len(counts)) if counts[i]})
    out = {"sequences": seqs, "n_items": len(lex_items), "frequency": freq}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out


def load_dataset(name,data_dir="/content/sparsewalker_data",min_user_len=5):
    spec=DATASETS[name]; path=_download(spec.url,Path(data_dir)/spec.filename)
    if spec.kind=="ml1m":
        out=_load_ml1m_fast(path,data_dir,min_user_len); out["spec"]=spec; return out

    events=[]
    if spec.kind=="amazon":
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                r=json.loads(line); events.append((r["reviewerID"],r["asin"],r.get("unixReviewTime",0)))
    else:
        with zipfile.ZipFile(path) as z:
            with z.open(spec.archive_member) as fh:
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