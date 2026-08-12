#!/usr/bin/env python
"""Streaming-native SparseWalker on ML-1M.

Purpose
-------
Train the model with the same forward semantics we ultimately want to serve:
  event -> persistent local Walker state -> cacheable temporal memory #1
        -> cacheable temporal memory #2 -> tied item scorer.

Unlike the previous two-temporal experiment:
- local Walker state is carried across the user's full chronological sequence;
- temporal K/V state is carried across chunks;
- there are NO learned absolute positional embeddings;
- temporal ordering uses a learned relative-lag score bias only;
- K/V for old memories therefore never need re-encoding when the window slides;
- training uses chunked TBPTT: forward state is exact, gradients are detached at
  chunk boundaries, and parameters are updated only after all chunks of a user
  batch have been processed.

This first experiment uses exact dense retrieval over a bounded temporal cache.
If quality is good, serving can replace that scan with the already validated SWG
Top-K search without changing the trained state semantics.
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

from sparsewalker.data import load_dataset, split_data
from sparsewalker.models import SparseWalker


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_lr(opt, epoch, max_epochs, peak=1e-3, min_lr=1e-4, warmup=3):
    if epoch <= warmup:
        lr = peak * epoch / warmup
    else:
        p = (epoch - warmup) / max(1, max_epochs - warmup)
        lr = min_lr + .5 * (peak - min_lr) * (1 + math.cos(math.pi * p))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


class StreamingTemporalLayer(nn.Module):
    """Causal, cache-exact temporal layer with relative-lag bias.

    Stored K/V depend only on the historical representation, never on the
    historical token's *current* absolute position. Relative age enters only as
    a query-time score bias, so old K/V remain valid forever.
    """

    def __init__(self, d=64, heads=2, ff_mult=4, dropout=.1, memory_size=512):
        super().__init__()
        assert d % heads == 0
        self.d = int(d)
        self.heads = int(heads)
        self.hd = d // heads
        self.memory_size = int(memory_size)
        self.dropout = float(dropout)

        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * ff_mult, d),
            nn.Dropout(dropout),
        )
        # bias[lag, head]. lag=0 is current position; lag >= memory_size is masked.
        self.rel_bias = nn.Embedding(memory_size, heads)
        nn.init.zeros_(self.rel_bias.weight)

    def _empty_state(self, x):
        B = x.size(0)
        return {
            "k": x.new_empty((B, self.heads, 0, self.hd)),
            "v": x.new_empty((B, self.heads, 0, self.hd)),
            "pos": torch.empty((B, 0), device=x.device, dtype=torch.long),
            "valid": torch.empty((B, 0), device=x.device, dtype=torch.bool),
        }

    @staticmethod
    def detach_state(state):
        if state is None:
            return None
        return {
            "k": state["k"].detach(),
            "v": state["v"].detach(),
            "pos": state["pos"],
            "valid": state["valid"],
        }

    def forward_chunk(self, x, valid, state=None, start_pos=0):
        B, C, D = x.shape
        if state is None:
            state = self._empty_state(x)

        n = self.norm1(x)
        qkv = self.qkv(n)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, C, self.heads, self.hd).transpose(1, 2)
        k = k.view(B, C, self.heads, self.hd).transpose(1, 2)
        v = v.view(B, C, self.heads, self.hd).transpose(1, 2)

        cur_pos = torch.arange(start_pos, start_pos + C, device=x.device, dtype=torch.long)
        cur_pos = cur_pos.unsqueeze(0).expand(B, -1)

        all_k = torch.cat([state["k"], k], dim=2)
        all_v = torch.cat([state["v"], v], dim=2)
        all_pos = torch.cat([state["pos"], cur_pos], dim=1)
        all_valid = torch.cat([state["valid"], valid], dim=1)

        # [B,C,S]. Negative lags are future positions inside the current chunk.
        raw_lag = cur_pos[:, :, None] - all_pos[:, None, :]
        causal = raw_lag >= 0
        within_memory = raw_lag < self.memory_size
        allowed = valid[:, :, None] & all_valid[:, None, :] & causal & within_memory
        lag = raw_lag.clamp(min=0, max=self.memory_size - 1)

        # Tensor-core QK where autocast is active; accumulate masking/softmax in fp32.
        score = torch.matmul(q, all_k.transpose(-2, -1)) / math.sqrt(self.hd)
        score = score.float()
        bias = self.rel_bias(lag).permute(0, 3, 1, 2).float()
        score = score + bias
        score = score.masked_fill(~allowed[:, None, :, :], -1e4)
        prob = torch.softmax(score, dim=-1)
        prob = prob * valid[:, None, :, None].to(prob.dtype)
        if self.training and self.dropout > 0:
            prob = F.dropout(prob, p=self.dropout)
        ctx = torch.matmul(prob.to(all_v.dtype), all_v)
        ctx = ctx.transpose(1, 2).reshape(B, C, D)

        z = x + self.out(ctx)
        z = z + self.ffn(self.norm2(z))
        z = z * valid[:, :, None].to(z.dtype)

        # Keep a fixed cache. Exact semantics are enforced by the lag mask above,
        # so chunk boundaries cannot change which keys a query is allowed to see.
        keep = min(self.memory_size, all_k.size(2))
        new_state = {
            "k": all_k[:, :, -keep:, :],
            "v": all_v[:, :, -keep:, :],
            "pos": all_pos[:, -keep:],
            "valid": all_valid[:, -keep:],
        }
        return z, new_state


class StreamingSparseWalker(SparseWalker):
    def __init__(
        self,
        n_items,
        d=64,
        graph_hops=2,
        side=256,
        h=16,
        active=8,
        top_side=2,
        degree=4,
        fresh_weight=.25,
        temporal_heads=2,
        ff_mult=4,
        dropout=.1,
        memory_size=512,
    ):
        # max_len is metadata only for SparseWalker; streaming code does not truncate.
        super().__init__(
            n_items, memory_size, d=d, layers=graph_hops, side=side, h=h,
            active=active, top_side=top_side, degree=degree,
            fresh_weight=fresh_weight,
        )
        self.temporal1 = StreamingTemporalLayer(
            d=d, heads=temporal_heads, ff_mult=ff_mult,
            dropout=dropout, memory_size=memory_size,
        )
        self.temporal2 = StreamingTemporalLayer(
            d=d, heads=temporal_heads, ff_mult=ff_mult,
            dropout=dropout, memory_size=memory_size,
        )
        self.memory_size = int(memory_size)

    def local_chunk(self, seq, state=None, return_ids=False):
        B, C = seq.shape
        valid = seq.ne(0)
        item_state = self.item(seq) * math.sqrt(self.d_model)
        fi, fm = self.router(item_state.reshape(B * C, self.d_model), self.space)
        fi = fi.view(B, C, -1)
        fm = fm.view(B, C, -1)

        if state is None:
            ids = torch.zeros(B, self.active, dtype=torch.long, device=seq.device)
            mass = torch.zeros(B, self.active, dtype=item_state.dtype, device=seq.device)
        else:
            ids, mass = state

        outs = []
        ids_hist = [] if return_ids else None
        for t in range(C):
            act = valid[:, t]
            af = act.to(item_state.dtype)[:, None]
            xids = ids
            xmass = mass * af
            xids, xmass = self._merge(xids, xmass, fi[:, t], fm[:, t] * af)
            for _ in range(self.layers_n):
                xids, xmass = self.graph(
                    xids, xmass, item_state[:, t], self.space,
                    track_touched=False,
                )
            ids = torch.where(act[:, None], xids, ids)
            mass = torch.where(act[:, None], xmass, mass)
            msg = (self.space.value(ids) * mass[:, :, None]).sum(1)
            ht = self.norm(item_state[:, t] + self.message_proj(msg)) * af
            outs.append(ht)
            if return_ids:
                ids_hist.append(ids)
        H = torch.stack(outs, dim=1)
        I = torch.stack(ids_hist, dim=1) if return_ids else None
        return H, (ids, mass), I

    @staticmethod
    def detach_local(state):
        if state is None:
            return None
        return state[0], state[1].detach()

    def forward_chunk(self, seq, local_state=None, temporal_states=None, start_pos=0, return_ids=False):
        valid = seq.ne(0)
        h0, local_state, ids_hist = self.local_chunk(seq, local_state, return_ids=return_ids)
        s1 = None if temporal_states is None else temporal_states[0]
        s2 = None if temporal_states is None else temporal_states[1]
        h1, s1 = self.temporal1.forward_chunk(h0, valid, s1, start_pos=start_pos)
        h2, s2 = self.temporal2.forward_chunk(h1, valid, s2, start_pos=start_pos)
        return h2, local_state, (s1, s2), ids_hist

    def detach_states(self, local_state, temporal_states):
        local_state = self.detach_local(local_state)
        if temporal_states is None:
            return local_state, None
        return local_state, (
            self.temporal1.detach_state(temporal_states[0]),
            self.temporal2.detach_state(temporal_states[1]),
        )

    def stream_last_hidden(self, seq, lengths, chunk_size=64, return_final_ids=False):
        B, T = seq.shape
        local_state = None
        temporal_states = None
        last_h = torch.zeros(B, self.d_model, device=seq.device, dtype=self.item.weight.dtype)
        final_ids = torch.zeros(B, self.active, device=seq.device, dtype=torch.long)
        rows = torch.arange(B, device=seq.device)
        for st in range(0, T, chunk_size):
            en = min(T, st + chunk_size)
            h, local_state, temporal_states, ids_hist = self.forward_chunk(
                seq[:, st:en], local_state, temporal_states, start_pos=st,
                return_ids=return_final_ids,
            )
            rel = lengths - 1 - st
            hit = (rel >= 0) & (rel < en - st)
            if hit.any():
                pick = rel.clamp(min=0, max=en - st - 1)
                cand = h[rows, pick]
                last_h = torch.where(hit[:, None], cand, last_h)
                if return_final_ids:
                    cand_i = ids_hist[rows, pick]
                    final_ids = torch.where(hit[:, None], cand_i, final_ids)
        return (last_h, final_ids) if return_final_ids else last_h


@torch.inference_mode()
def chunk_invariance_test(model, device):
    model.eval()
    # Keep length below memory_size so one-chunk and multi-chunk should be identical.
    L = min(192, model.memory_size - 1)
    x = torch.randint(1, model.n_items + 1, (3, L), device=device)
    lens = torch.tensor([L, L - 17, L - 41], device=device)
    for r, n in enumerate(lens.tolist()):
        if n < L:
            x[r, n:] = 0
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        a = model.stream_last_hidden(x, lens, chunk_size=L)
        b = model.stream_last_hidden(x, lens, chunk_size=64)
        c = model.stream_last_hidden(x, lens, chunk_size=31)
    ab = float((a.float() - b.float()).abs().max().cpu())
    ac = float((a.float() - c.float()).abs().max().cpu())
    out = {"max_abs_one_vs_64": ab, "max_abs_one_vs_31": ac}
    print("CHUNK_INVARIANCE", out, flush=True)
    if max(ab, ac) > 2e-2:
        raise AssertionError(f"streaming chunk invariance failed: {out}")


@torch.inference_mode()
def causal_test(model, device):
    model.eval()
    x = torch.randint(1, model.n_items + 1, (2, 64), device=device)
    valid = x.ne(0)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        a, _, _, _ = model.forward_chunk(x, None, None, start_pos=0)
        y = x.clone()
        y[:, 48:] = torch.randint(1, model.n_items + 1, y[:, 48:].shape, device=device)
        b, _, _, _ = model.forward_chunk(y, None, None, start_pos=0)
    diff = float((a[:, :48].float() - b[:, :48].float()).abs().max().cpu())
    print("CAUSAL_TEST", {"max_abs_prefix_diff": diff}, flush=True)
    if diff > 2e-2:
        raise AssertionError(f"causal test failed: {diff}")


def make_batches(seqs, batch_size, seed, epoch):
    ids = [i for i, s in enumerate(seqs) if len(s) >= 2]
    # Strong length bucketing keeps full-history padding under control.
    ids.sort(key=lambda i: len(seqs[i]))
    batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    rng = random.Random(seed * 1_000_003 + epoch * 97_409)
    rng.shuffle(batches)
    for b in batches:
        rng.shuffle(b)
    return batches


def pad_xy(seqs, ids, device):
    xs = [seqs[i][:-1] for i in ids]
    ys = [seqs[i][1:] for i in ids]
    lens = torch.tensor([len(x) for x in xs], device=device, dtype=torch.long)
    T = int(lens.max().item())
    x = torch.zeros(len(ids), T, device=device, dtype=torch.long)
    y = torch.zeros(len(ids), T, device=device, dtype=torch.long)
    for r, (a, b) in enumerate(zip(xs, ys)):
        x[r, :len(a)] = torch.tensor(a, device=device)
        y[r, :len(b)] = torch.tensor(b, device=device)
    return x, y, lens


def train_epoch_streaming(model, seqs, opt, device, batch_size, chunk_size, seed, epoch, use_bf16=True):
    model.train()
    batches = make_batches(seqs, batch_size, seed, epoch)
    total_loss = 0.0
    total_positions = 0
    t0 = time.perf_counter()

    for ids in batches:
        x, y, lens = pad_xy(seqs, ids, device)
        valid_total = int(y.ne(0).sum().item())
        if valid_total == 0:
            continue
        opt.zero_grad(set_to_none=True)
        local_state = None
        temporal_states = None
        batch_loss_sum = 0.0

        for st in range(0, x.size(1), chunk_size):
            en = min(x.size(1), st + chunk_size)
            xc = x[:, st:en]
            yc = y[:, st:en]
            valid = yc.ne(0)
            if not valid.any():
                break
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
                h, local_state, temporal_states, _ = model.forward_chunk(
                    xc, local_state, temporal_states, start_pos=st,
                )
                hv = h[valid]
                yv = yc[valid]
                logits = model.score_hidden(hv)
                loss_sum = F.cross_entropy(logits.float(), yv, reduction="sum")
                loss = loss_sum / valid_total
            loss.backward()
            batch_loss_sum += float(loss_sum.detach().cpu())
            # Forward state is carried exactly; only gradient history is truncated.
            local_state, temporal_states = model.detach_states(local_state, temporal_states)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total_loss += batch_loss_sum
        total_positions += valid_total

    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    return {
        "loss": total_loss / max(1, total_positions),
        "positions": total_positions,
        "seconds": sec,
        "positions_per_s": total_positions / max(sec, 1e-9),
    }


def eval_batches(prefixes, batch_size, history_limit=None):
    ids = list(range(len(prefixes)))
    ids.sort(key=lambda i: min(len(prefixes[i]), history_limit) if history_limit else len(prefixes[i]))
    return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]


@torch.inference_mode()
def evaluate_streaming(model, prefixes, targets, n_items, device, batch_size=64, chunk_size=64, history_limit=None):
    model.eval()
    total = len(targets)
    hit = ndcg = mrr = 0.0
    for ids in eval_batches(prefixes, batch_size, history_limit):
        seqs = []
        for i in ids:
            p = prefixes[i]
            if history_limit is not None:
                p = p[-history_limit:]
            seqs.append(p)
        lens = torch.tensor([len(s) for s in seqs], device=device, dtype=torch.long)
        T = int(lens.max().item())
        x = torch.zeros(len(ids), T, device=device, dtype=torch.long)
        for r, s in enumerate(seqs):
            x[r, :len(s)] = torch.tensor(s, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h = model.stream_last_hidden(x, lens, chunk_size=chunk_size)
            scores = model.score_hidden(h)
        scores = scores.float()
        tgt = torch.tensor([targets[i] for i in ids], device=device, dtype=torch.long)
        for r, i in enumerate(ids):
            seen = set(prefixes[i])
            truth = int(targets[i])
            seen.discard(truth)
            if seen:
                scores[r, torch.tensor(list(seen), device=device)] = -1e20
        top = scores.topk(10, dim=-1).indices.cpu().numpy()
        truth_np = tgt.cpu().numpy()
        for r, truth in enumerate(truth_np):
            pos = np.where(top[r] == truth)[0]
            if len(pos):
                rank = int(pos[0]) + 1
                hit += 1.0
                ndcg += 1.0 / math.log2(rank + 1)
                mrr += 1.0 / rank
    return {"HR@10": hit / total, "NDCG@10": ndcg / total, "MRR@10": mrr / total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_streaming_native")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--memory-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print("DEVICE", device, torch.cuda.get_device_name(0) if device.type == "cuda" else None,
          "bf16", torch.cuda.is_bf16_supported() if device.type == "cuda" else False, flush=True)

    data = load_dataset("ml1m", args.data_dir)
    split = split_data(data["sequences"])
    model = StreamingSparseWalker(
        data["n_items"], d=64, graph_hops=2, side=256, h=16, active=8,
        top_side=2, degree=4, fresh_weight=.25, temporal_heads=2,
        ff_mult=4, dropout=.1, memory_size=args.memory_size,
    ).to(device)

    params = {
        "total": sum(p.numel() for p in model.parameters()),
        "temporal1": sum(p.numel() for p in model.temporal1.parameters()),
        "temporal2": sum(p.numel() for p in model.temporal2.parameters()),
    }
    params["local_walker"] = params["total"] - params["temporal1"] - params["temporal2"]
    config = {
        "architecture": "StreamingSparseWalker-v1",
        "local_state": "persistent K=8 concept IDs+masses",
        "pursuit": False,
        "absolute_position_embeddings": False,
        "relative_temporal_feature": "learned per-head lag bias at query time",
        "temporal_layers": 2,
        "temporal_memory": args.memory_size,
        "temporal_training": "exact dense cache read; SWG deferred until quality established",
        "chunk_size": args.chunk_size,
        "tbptt": "detach state between chunks; optimizer step after full user batch",
        "full_user_history_training": True,
        "dense_catalog": "quality/training only; terminal sparse retrieval deferred",
        "params": params,
    }
    print("STREAMING_CONFIG", json.dumps(config, indent=2), flush=True)

    chunk_invariance_test(model, device)
    causal_test(model, device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out = Path(args.output_dir) / "ml1m" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    best = -1.0
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        lr = set_lr(opt, epoch, args.epochs, peak=args.lr)
        tr = train_epoch_streaming(
            model, split["train"], opt, device,
            batch_size=args.batch_size, chunk_size=args.chunk_size,
            seed=args.seed, epoch=epoch, use_bf16=True,
        )
        tr["epoch"] = epoch
        tr["lr"] = lr
        print("TRAIN", tr, flush=True)

        if epoch == 1 or epoch % args.eval_every == 0:
            t0 = time.perf_counter()
            val_full = evaluate_streaming(
                model, split["val_prefix"], split["val_target"], data["n_items"],
                device, batch_size=args.eval_batch_size, chunk_size=args.chunk_size,
                history_limit=None,
            )
            val_reset200 = evaluate_streaming(
                model, split["val_prefix"], split["val_target"], data["n_items"],
                device, batch_size=args.eval_batch_size, chunk_size=args.chunk_size,
                history_limit=200,
            )
            ev = {
                **tr,
                "val_stream_full_NDCG@10": float(val_full["NDCG@10"]),
                "val_stream_full_HR@10": float(val_full["HR@10"]),
                "val_stream_full_MRR@10": float(val_full["MRR@10"]),
                "val_reset200_NDCG@10": float(val_reset200["NDCG@10"]),
                "persistent_history_gain": float(val_full["NDCG@10"] - val_reset200["NDCG@10"]),
                "eval_seconds": time.perf_counter() - t0,
            }
            history.append(ev)
            print("EVAL", ev, flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=2))
            if val_full["NDCG@10"] > best:
                best = float(val_full["NDCG@10"])
                best_epoch = epoch
                best_state = cpu_state_dict(model)
                torch.save({
                    "model": best_state,
                    "epoch": epoch,
                    "val_stream_full": val_full,
                    "val_reset200": val_reset200,
                    "config": config,
                }, out / "best.pt")
            torch.save({
                "model": cpu_state_dict(model),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_ndcg": best,
                "config": config,
                "history": history,
            }, out / "last.pt")
            model.train()

    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    val_full = evaluate_streaming(
        model, split["val_prefix"], split["val_target"], data["n_items"], device,
        batch_size=args.eval_batch_size, chunk_size=args.chunk_size,
    )
    test_full = evaluate_streaming(
        model, split["test_prefix"], split["test_target"], data["n_items"], device,
        batch_size=args.eval_batch_size, chunk_size=args.chunk_size,
    )
    test_reset200 = evaluate_streaming(
        model, split["test_prefix"], split["test_target"], data["n_items"], device,
        batch_size=args.eval_batch_size, chunk_size=args.chunk_size, history_limit=200,
    )
    result = {
        "selected_epoch": best_epoch,
        "config": config,
        "val_stream_full": val_full,
        "test_stream_full": test_full,
        "test_reset200": test_reset200,
        "references": {
            "previous_two_temporal_test_NDCG@10": 0.16235588745115664,
            "previous_two_temporal_val_NDCG@10_epoch30": 0.16813880880288698,
            "meta_sasrec_paper_protocol_NDCG@10": 0.15965351016316193,
            "note": "Meta SASRec number is a different protocol; included only as external reference.",
        },
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("FINAL", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
