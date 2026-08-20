#!/usr/bin/env python
"""Experiment 33 Amazon training-parallelism sweep.

Matched A100/BF16 throughput benchmark for streaming SparseWalker+2 temporal
layers versus SASRec on Beauty, Video Games, Sports, and Toys. Both models see
the same length-bucketed users, histories are capped to 50 prediction positions,
and both use FullCE. This is a systems comparison, separate from the optional
corrected v1.1 quality rerun in run_amazon_quality_pair.py.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "experiments"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))

from sparsewalker.data import load_dataset, split_data
from sparsewalker.models import SASRec
from run_ml1m_walker_streaming_native import make_batches, seed_all
from run_ml1m_walker_streaming_speed import build_model, cpu_state_dict, pad_xy_fast, train_epoch_fast

AMAZON = ("beauty", "video_games", "sports", "toys")


def cap_train(seqs, max_len=50):
    return [list(s)[-(max_len + 1):] for s in seqs if len(s) >= 2]


def train_sasrec_cells(n_items, seqs, device, batch_sizes, max_batches, seed, max_len=50):
    rows = []
    for bs in batch_sizes:
        seed_all(seed)
        model = SASRec(n_items, max_len, d=64, layers=2, heads=2, inner=256, dropout=.2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        batches = make_batches(seqs, bs, seed, 1)[:max_batches]
        # One warm batch, then reset model/optimizer and measure from identical init.
        base = cpu_state_dict(model)
        warm = batches[:1]
        for ids in warm:
            x, y, lengths, valid_total = pad_xy_fast(seqs, ids, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                h = model.encode(x); valid = y.ne(0)
                loss = F.cross_entropy(model.score_hidden(h[valid]).float(), y[valid])
            loss.backward(); opt.step()
        if device.type == "cuda": torch.cuda.synchronize()
        model.load_state_dict(base); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
        pos = 0; loss_sum = torch.zeros((), device=device); t0 = time.perf_counter()
        for ids in batches:
            x, y, lengths, valid_total = pad_xy_fast(seqs, ids, device); pos += valid_total
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                h = model.encode(x); valid = y.ne(0)
                logits = model.score_hidden(h[valid])
                loss = F.cross_entropy(logits.float(), y[valid])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            loss_sum += loss.detach() * float(valid_total)
        if device.type == "cuda": torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
        row = {"model":"SASRec","batch_size":bs,"positions":pos,"seconds":sec,
               "positions_per_s":pos/max(sec,1e-9),"peak_allocated_GB":peak,
               "loss":float(loss_sum.cpu())/max(1,pos),"status":"OK"}
        print("AMAZON_SPEED_CELL", json.dumps(row), flush=True); rows.append(row)
        del model, opt
        if device.type == "cuda": torch.cuda.empty_cache()
    return rows


def train_walker_cells(n_items, seqs, device, batch_sizes, max_batches, seed, chunk_size=50, memory_size=128):
    rows = []
    seed_all(seed)
    base_model = build_model(n_items, memory_size).to(device); base = cpu_state_dict(base_model); del base_model
    for bs in batch_sizes:
        try:
            seed_all(seed)
            model = build_model(n_items, memory_size).to(device); model.load_state_dict(base); model.set_local_compile(False)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            # Warm one batch then restore exact initial weights.
            train_epoch_fast(model, seqs, opt, device, batch_size=bs, chunk_size=chunk_size,
                             seed=seed, epoch=1, backward_group_chunks=1, max_batches=1,
                             fixed_training_chunk=False)
            model.load_state_dict(base); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
            st = train_epoch_fast(model, seqs, opt, device, batch_size=bs, chunk_size=chunk_size,
                                  seed=seed, epoch=1, backward_group_chunks=1,
                                  max_batches=max_batches, fixed_training_chunk=False)
            peak = torch.cuda.max_memory_allocated()/(1024**3) if device.type == "cuda" else 0.0
            row = {**st, "model":"StreamingSparseWalker+2T", "peak_allocated_GB":peak, "status":"OK"}
        except torch.cuda.OutOfMemoryError as e:
            if device.type == "cuda": torch.cuda.empty_cache()
            row = {"model":"StreamingSparseWalker+2T","batch_size":bs,"status":"OOM","error":repr(e)}
        except Exception as e:
            row = {"model":"StreamingSparseWalker+2T","batch_size":bs,"status":"ERROR","error":repr(e)}
        print("AMAZON_SPEED_CELL", json.dumps(row), flush=True); rows.append(row)
        if device.type == "cuda": torch.cuda.empty_cache()
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--datasets",default=",".join(AMAZON))
    p.add_argument("--batch-sizes",default="128,256,512,1024")
    p.add_argument("--benchmark-batches",type=int,default=24)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--data-dir",default="/content/sparsewalker_data")
    p.add_argument("--output",default="/content/drive/MyDrive/sparsewalker_amazon_parallel_scaling/result.json")
    a=p.parse_args(); assert torch.cuda.is_available(), "GPU required"
    device=torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32=True; torch.set_float32_matmul_precision("high")
    datasets=[x.strip() for x in a.datasets.split(",") if x.strip()]
    batches=[int(x) for x in a.batch_sizes.split(",") if x.strip()]
    all_rows=[]
    print("DEVICE",torch.cuda.get_device_name(0),"torch",torch.__version__,"bf16",torch.cuda.is_bf16_supported(),flush=True)
    for d in datasets:
        if d not in AMAZON: raise ValueError(d)
        data=load_dataset(d,a.data_dir); split=split_data(data["sequences"]); seqs=cap_train(split["train"],50)
        lens=np.asarray([len(s)-1 for s in seqs])
        meta={"dataset":d,"users":len(seqs),"items":data["n_items"],"mean_train_positions":float(lens.mean()),"p95_train_positions":float(np.percentile(lens,95)),"max_positions":int(lens.max())}
        print("AMAZON_DATA",json.dumps(meta),flush=True)
        wr=train_walker_cells(data["n_items"],seqs,device,batches,a.benchmark_batches,a.seed)
        sr=train_sasrec_cells(data["n_items"],seqs,device,batches,a.benchmark_batches,a.seed)
        for r in wr+sr: all_rows.append({"dataset":d,**meta,**r})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(all_rows,indent=2))
    print("AMAZON_PARALLEL_SCALING",json.dumps(all_rows,indent=2),flush=True); print("SAVED",out,flush=True)

if __name__=="__main__": main()
