#!/usr/bin/env python
"""ML-1M diagnostic: frozen SparseWalker + one residual HNSW/SWG layer.

The original K=8 Walker is left untouched. A query derived from its hidden
working-memory state navigates a fixed HNSW level-0 concept graph starting from
the Walker's active concepts. Retrieved concept values are added through a
learned gated residual before the existing item scorer.

Trainable only:
  - SWG query head (initialized from the previous navigation-query experiment)
  - concept read projection
  - residual gate

Diagnostics:
  - canonical-style val NDCG@10 / HR@10 / MRR@10
  - exact same trained weights with SWG ON vs OFF
  - quality by short / medium / long history
  - next-item routed-concept hit inside the sparse retrieved beam
  - residual gate and sparse edge reads

Success target: val NDCG@10 >= 0.145 with a clear long-history gain.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sparsewalker.data import load_dataset, split_data
from sparsewalker.models import SparseWalker

from run_hnsw_attention_primitive import build_hnsw, extract_hnsw_adjacency
from run_swg_query_training import QueryHead, concept_keys, target_concepts


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_batch(seqs, max_len, device):
    rows = [list(s)[-max_len:] for s in seqs]
    lens = [len(s) for s in rows]
    L = max(lens)
    x = torch.zeros(len(rows), L, dtype=torch.long, device=device)
    for i, s in enumerate(rows):
        if s:
            x[i, :len(s)] = torch.as_tensor(s, dtype=torch.long, device=device)
    return x, torch.as_tensor(lens, dtype=torch.long, device=device)


@torch.inference_mode()
def cache_train_features(model, train_seqs, max_len, device, cap=120000, batch_users=128):
    """Frozen Walker states, active concepts, and next-item ids."""
    hs, starts, ys = [], [], []
    total = 0
    t0 = time.perf_counter()
    model.eval()
    for st in range(0, len(train_seqs), batch_users):
        seqs = []
        for s in train_seqs[st:st + batch_users]:
            s = list(s)[-(max_len + 1):]
            if len(s) >= 2:
                seqs.append(s)
        if not seqs:
            continue
        tok, lens = pad_batch(seqs, max_len + 1, device)
        x, y = tok[:, :-1], tok[:, 1:]
        xl = (lens - 1).clamp_min(0)
        use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            H, I, _ = model.encode_with_states(x)
        mask = torch.arange(x.size(1), device=device)[None, :] < xl[:, None]
        h = H[mask].float()
        ii = I[mask]
        yy = y[mask]
        hs.append(h.cpu().to(torch.float16))
        starts.append(ii.cpu().to(torch.int32))
        ys.append(yy.cpu().to(torch.int32))
        total += int(h.size(0))
        if total >= cap:
            break
    H = torch.cat(hs, 0)[:cap]
    I = torch.cat(starts, 0)[:cap]
    Y = torch.cat(ys, 0)[:cap]
    mb = (H.numel() * 2 + I.numel() * 4 + Y.numel() * 4) / 1e6
    print("TRAIN_CACHE", {"examples": int(H.size(0)), "seconds": round(time.perf_counter() - t0, 2), "MB": round(mb, 1)}, flush=True)
    return H, I, Y


class ResidualSWGLayer(nn.Module):
    def __init__(self, model, keys_np, adjacency_np, hops=3, beam=8, gate_init=0.10):
        super().__init__()
        self.model = model
        self.hops = int(hops)
        self.beam = int(beam)
        self.query = QueryHead(model.d_model, model.h, model.graph.context_q.weight.detach())
        self.read_proj = nn.Linear(model.d_model, model.d_model, bias=False)
        nn.init.normal_(self.read_proj.weight, std=0.02)
        g = float(gate_init)
        self.gate_logit = nn.Parameter(torch.tensor(math.log(g / (1.0 - g))))
        self.register_buffer("keys", torch.as_tensor(keys_np, dtype=torch.float32), persistent=False)
        self.register_buffer("adjacency", torch.as_tensor(adjacency_np, dtype=torch.long), persistent=False)

    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def walk(self, h, starts):
        """GPU beam navigation from the Walker's active concepts."""
        q = self.query(h.float())
        B = h.size(0)
        beam = starts.long()
        total_edge_reads = 0
        for _ in range(self.hops):
            nbr = self.adjacency[beam]              # B x beam x degree
            total_edge_reads += int(beam.size(1) * self.adjacency.size(1))
            cand = torch.cat([beam, nbr.reshape(B, -1)], dim=-1)
            valid = cand >= 0
            safe = cand.clamp_min(0)
            ck = self.keys[safe]
            score = (ck * q[:, None, :]).sum(-1)
            score = score.masked_fill(~valid, -1e9)
            k = min(self.beam, score.size(1))
            topi = score.topk(k, dim=-1).indices
            beam = safe.gather(1, topi)

        bk = self.keys[beam]
        scale = self.query.log_scale.exp().clamp(1.0, 50.0)
        score = (bk * q[:, None, :]).sum(-1) * scale
        weight = F.softmax(score, dim=-1)
        values = self.model.space.value(beam)
        read = (values * weight[..., None]).sum(1)
        refined = h.float() + self.gate() * self.read_proj(read.float())
        return refined, beam, total_edge_reads


def concept_aux_loss(layer, h, target_items, n_neg=256):
    q = layer.query(h.float())
    scale = layer.query.log_scale.exp().clamp(1.0, 50.0)
    with torch.no_grad():
        pos_ids = target_concepts(layer.model, target_items.long())
    pk = layer.keys[pos_ids]
    ps = (pk * q[:, None, :]).sum(-1) * scale
    neg = torch.randint(0, layer.keys.size(0), (h.size(0), n_neg), device=h.device)
    nk = layer.keys[neg]
    ns = (nk * q[:, None, :]).sum(-1) * scale
    numer = torch.logsumexp(ps, dim=-1)
    denom = torch.logsumexp(torch.cat([ps, ns], dim=-1), dim=-1)
    return (denom - numer).mean()


def train_layer(layer, H, I, Y, device, epochs=4, batch_size=1024, lr=1e-3, aux_weight=0.25):
    ds = TensorDataset(H, I, Y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    params = [p for p in layer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    item_w = layer.model.item.weight
    hist = []
    for ep in range(1, epochs + 1):
        layer.train()
        total = ce_total = aux_total = 0.0
        n = 0
        reads = 0
        t0 = time.perf_counter()
        for h, starts, y in loader:
            h = h.to(device, non_blocking=True).float()
            starts = starts.to(device, non_blocking=True).long()
            y = y.to(device, non_blocking=True).long()
            refined, _, r = layer.walk(h, starts)
            logits = refined @ item_w[1:].T
            ce = F.cross_entropy(logits, y - 1)
            aux = concept_aux_loss(layer, h, y)
            loss = ce + aux_weight * aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            bs = h.size(0)
            total += float(loss.detach()) * bs
            ce_total += float(ce.detach()) * bs
            aux_total += float(aux.detach()) * bs
            n += bs
            reads += r * bs
        row = {
            "epoch": ep,
            "loss": total / max(1, n),
            "item_ce": ce_total / max(1, n),
            "concept_aux": aux_total / max(1, n),
            "gate": float(layer.gate().detach().cpu()),
            "mean_edge_reads": reads / max(1, n),
            "seconds": time.perf_counter() - t0,
        }
        hist.append(row)
        print("HYBRID_TRAIN", row, flush=True)
    return hist


def metrics_from_ranks(top, targets):
    hit = 0.0
    ndcg = 0.0
    mrr = 0.0
    for row, tgt in zip(top, targets):
        where = np.where(row == int(tgt))[0]
        if where.size:
            r = int(where[0]) + 1
            if r <= 10:
                hit += 1.0
                ndcg += 1.0 / math.log2(r + 1.0)
                mrr += 1.0 / r
    n = max(1, len(targets))
    return {"HR@10": hit / n, "NDCG@10": ndcg / n, "MRR@10": mrr / n}


@torch.inference_mode()
def evaluate_hybrid(model, layer, prefixes, targets, n_items, max_len, device, batch_size=512, swg_on=True, indices=None):
    if indices is None:
        indices = list(range(len(prefixes)))
    tops = []
    tgts = []
    next_seen = 0
    total = 0
    edge_reads = 0
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for st in range(0, len(indices), batch_size):
        idx = indices[st:st + batch_size]
        seqs = [prefixes[i] for i in idx]
        x, lengths = pad_batch(seqs, max_len, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            H, I, _ = model.encode_with_states(x)
        row = torch.arange(x.size(0), device=device)
        last = (lengths - 1).clamp_min(0)
        h = H[row, last].float()
        starts = I[row, last]
        if swg_on:
            refined, beam, r = layer.walk(h, starts)
            y_t = torch.as_tensor([targets[i] for i in idx], dtype=torch.long, device=device)
            nc = target_concepts(model, y_t)
            seen = (beam[:, :, None] == nc[:, None, :]).any(dim=(1, 2))
            next_seen += int(seen.sum().item())
            edge_reads += r * len(idx)
        else:
            refined = h

        logits = refined @ model.item.weight[1:].T
        # canonical seen-item masking
        for b, original_i in enumerate(idx):
            seen_items = list(prefixes[original_i])[-max_len:]
            if seen_items:
                ids = torch.as_tensor(seen_items, dtype=torch.long, device=device)
                ids = ids[(ids > 0) & (ids <= n_items)]
                logits[b, ids - 1] = -1e9
        k = min(50, n_items)
        top = logits.topk(k, dim=-1).indices.add(1).cpu().numpy()
        tops.append(top)
        tgts.extend([targets[i] for i in idx])
        total += len(idx)
    top = np.concatenate(tops, axis=0)
    out = metrics_from_ranks(top, np.asarray(tgts))
    out["next_item_concept_seen_rate"] = next_seen / max(1, total) if swg_on else None
    out["mean_edge_reads"] = edge_reads / max(1, total) if swg_on else 0.0
    return out


def history_buckets(prefixes, max_len):
    out = {"short_<=50": [], "medium_51_100": [], "long_101_200": []}
    for i, p in enumerate(prefixes):
        n = min(len(p), max_len)
        if n <= 50:
            out["short_<=50"].append(i)
        elif n <= 100:
            out["medium_51_100"].append(i)
        else:
            out["long_101_200"].append(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="/content/sparsewalker_data")
    ap.add_argument("--checkpoint", default="/content/drive/MyDrive/sparsewalker_canonical_pair/ml1m/seed42/SparseWalker_FullCE/best.pt")
    ap.add_argument("--query-checkpoint", default="/content/drive/MyDrive/sparsewalker_swg_query/result.pt")
    ap.add_argument("--output", default="/content/drive/MyDrive/sparsewalker_hybrid_swg/result.json")
    ap.add_argument("--train-cap", type=int, default=120000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--hnsw-m", type=int, default=8)
    ap.add_argument("--ef-construction", type=int, default=160)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--beam", type=int, default=8)
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print("DEVICE", device, torch.cuda.get_device_name(0) if device.type == "cuda" else None, flush=True)

    data = load_dataset("ml1m", args.data_dir)
    split = split_data(data["sequences"])
    ck = torch.load(args.checkpoint, map_location="cpu")
    model = SparseWalker(data["n_items"], 200, d=64, layers=2, side=256, h=16, active=8, top_side=2, degree=4, fresh_weight=.25).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("BASE_CHECKPOINT", {"epoch": int(ck.get("epoch", -1)), "val_NDCG@10": ck.get("val", {}).get("NDCG@10")}, flush=True)

    keys = concept_keys(model, device)
    index, build_s = build_hnsw(keys, args.hnsw_m, args.ef_construction)
    level0, _, stats = extract_hnsw_adjacency(index, model.n_concepts, args.hnsw_m)
    print("HNSW", {"build_s": round(build_s, 2), "using": "level0", **stats}, flush=True)

    layer = ResidualSWGLayer(model, keys, level0, hops=args.hops, beam=args.beam, gate_init=0.10).to(device)
    qpath = Path(args.query_checkpoint)
    if qpath.exists():
        qck = torch.load(qpath, map_location="cpu")
        if "query_head" in qck:
            layer.query.load_state_dict(qck["query_head"])
            print("QUERY_INIT", "loaded previous SWG query head", flush=True)
    else:
        print("QUERY_INIT", "previous query head not found; using Walker context_q initialization", flush=True)

    base = evaluate_hybrid(model, layer, split["val_prefix"], split["val_target"], data["n_items"], 200, device, args.eval_batch_size, swg_on=False)
    pre = evaluate_hybrid(model, layer, split["val_prefix"], split["val_target"], data["n_items"], 200, device, args.eval_batch_size, swg_on=True)
    print("PRETRAIN_EVAL", {"base": base, "hybrid": pre, "gate": float(layer.gate().detach().cpu())}, flush=True)

    H, I, Y = cache_train_features(model, split["train"], 200, device, cap=args.train_cap)
    hist = train_layer(layer, H, I, Y, device, epochs=args.epochs, batch_size=args.batch_size)

    post = evaluate_hybrid(model, layer, split["val_prefix"], split["val_target"], data["n_items"], 200, device, args.eval_batch_size, swg_on=True)
    off = evaluate_hybrid(model, layer, split["val_prefix"], split["val_target"], data["n_items"], 200, device, args.eval_batch_size, swg_on=False)
    buckets = history_buckets(split["val_prefix"], 200)
    by_len = {}
    for name, idx in buckets.items():
        b0 = evaluate_hybrid(model, layer, split["val_prefix"], split["val_target"], data["n_items"], 200, device, args.eval_batch_size, swg_on=False, indices=idx)
        bh = evaluate_hybrid(model, layer, split["val_prefix"], split["val_target"], data["n_items"], 200, device, args.eval_batch_size, swg_on=True, indices=idx)
        by_len[name] = {"n": len(idx), "base_NDCG@10": b0["NDCG@10"], "hybrid_NDCG@10": bh["NDCG@10"], "delta": bh["NDCG@10"] - b0["NDCG@10"]}

    decision = {
        "base_NDCG@10": off["NDCG@10"],
        "hybrid_NDCG@10": post["NDCG@10"],
        "absolute_gain": post["NDCG@10"] - off["NDCG@10"],
        "target_0.145_reached": post["NDCG@10"] >= 0.145,
        "next_item_concept_seen_rate": post["next_item_concept_seen_rate"],
        "gate": float(layer.gate().detach().cpu()),
        "mean_edge_reads": post["mean_edge_reads"],
        "long_history_delta": by_len["long_101_200"]["delta"],
    }
    print("POSTTRAIN_EVAL", {"swg_on": post, "swg_off": off}, flush=True)
    print("HISTORY_LENGTH", json.dumps(by_len, indent=2), flush=True)
    print("DECISION", decision, flush=True)

    result = {"pretrain": pre, "base": base, "posttrain": post, "counterfactual_off": off, "history_length": by_len, "training": hist, "decision": decision, "hnsw": {"build_s": build_s, **stats}, "config": vars(args)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    torch.save({"layer": layer.state_dict()}, out.with_suffix(".pt"))


if __name__ == "__main__":
    main()
