#!/usr/bin/env python
"""Train SparseWalker + one residual SWG layer end-to-end from scratch on ML-1M.

Design:
- Random initialization. No pretrained Walker or query head.
- Original SparseWalker trains on every autoregressive position with FullCE.
- On the final valid prediction position of each sampled training window, replace
  the plain Walker CE with a residual SWG-refined CE. This makes the hybrid
  train end-to-end while paying for one sparse global walk per user/window,
  not one walk per token.
- HNSW topology is rebuilt from the current concept keys every epoch because
  the concept geometry moves during scratch training.
- SWG uses flattened HNSW edges, 4 hops, beam 16: the configuration that
  previously recovered ~9.6% next-item concepts from a trained query and ~70%
  of dense top-10 concepts.
- Canonical evaluation uses the shared evaluate_full function. The scratch
  runner verifies that SWG-off wrapper scores exactly reproduce the base model
  before training.

This is an architectural experiment, not yet the final serving implementation.
"""
import argparse
import hashlib
import inspect
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sparsewalker.data import load_dataset, split_data, WindowDataset, collate_windows
from sparsewalker.evaluation import evaluate_full
from sparsewalker.models import SparseWalker

from run_hnsw_attention_primitive import build_hnsw, extract_hnsw_adjacency


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state_dict(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def protocol_manifest(max_len, n_items):
    source = inspect.getsource(split_data) + "\n" + inspect.getsource(evaluate_full)
    source_hash = hashlib.sha256(source.encode()).hexdigest()[:16]
    m = {
        "protocol_version": "EVAL_CANONICAL_v1_candidate",
        "dataset": "ml1m",
        "split": "per-user leave-two-out: train=s[:-2], val=s[-2], test=s[-1]",
        "catalog": "all mapped item ids 1..n_items",
        "seen_item_masking": True,
        "validation_selection": "best validation full-catalog NDCG@10",
        "metrics": ["HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50", "MRR@10"],
        "max_len": int(max_len),
        "n_items": int(n_items),
        "implementation_hash": source_hash,
    }
    m["fingerprint"] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()[:20]
    return m


def set_lr(opt, epoch, max_epochs, peak=1e-3, min_lr=1e-4, warmup=3):
    if epoch <= warmup:
        lr = peak * epoch / warmup
    else:
        progress = (epoch - warmup) / max(1, max_epochs - warmup)
        lr = min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * progress))
    for group in opt.param_groups:
        group["lr"] = lr
    return lr


def length_bucket_batches(dataset, batch_size, epoch):
    g = torch.Generator()
    g.manual_seed(dataset.seed + epoch)
    ids = torch.randperm(len(dataset), generator=g).tolist()
    ids.sort(key=lambda i: min(len(dataset.seqs[i]), dataset.max_len + 1))
    batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    if len(batches) > 1:
        order = torch.randperm(len(batches), generator=g).tolist()
        batches = [batches[i] for i in order]
    return batches


def concept_keys_np(model, device):
    with torch.no_grad():
        ids = torch.arange(model.n_concepts, device=device)
        keys = model.space.key(ids).float().cpu().numpy().astype("float32")
    keys /= np.linalg.norm(keys, axis=1, keepdims=True).clip(min=1e-12)
    return keys


class ScratchResidualSWG(nn.Module):
    """One residual sparse-global-read layer over a rebuilt HNSW adjacency."""
    def __init__(self, model, hops=4, beam=16, gate_init=0.05):
        super().__init__()
        self.model = model
        self.hops = int(hops)
        self.beam = int(beam)
        self.query = nn.Linear(model.d_model, model.h, bias=False)
        nn.init.normal_(self.query.weight, std=1.0 / math.sqrt(model.d_model))
        self.log_scale = nn.Parameter(torch.tensor(math.log(7.0)))
        self.read_proj = nn.Linear(model.d_model, model.d_model, bias=False)
        nn.init.normal_(self.read_proj.weight, std=0.02)
        g = float(gate_init)
        self.gate_logit = nn.Parameter(torch.tensor(math.log(g / (1.0 - g))))
        self.register_buffer("adjacency", torch.empty(0, 0, dtype=torch.long), persistent=False)
        self.mean_degree = 0.0

    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def set_adjacency(self, adj_np):
        dev = next(self.parameters()).device
        self.adjacency = torch.as_tensor(adj_np, dtype=torch.long, device=dev)
        self.mean_degree = float((adj_np >= 0).sum(1).mean())

    def query_vec(self, h):
        return F.normalize(self.query(h.float()), dim=-1)

    def walk(self, h, starts):
        if self.adjacency.numel() == 0:
            raise RuntimeError("SWG adjacency has not been built")
        q = self.query_vec(h)
        B = h.size(0)
        beam = starts.long()
        reads_per_example = 0
        for _ in range(self.hops):
            nbr = self.adjacency[beam]  # B x beam_or_K x degree
            reads_per_example += int(beam.size(1) * self.adjacency.size(1))
            cand = torch.cat([beam, nbr.reshape(B, -1)], dim=-1)
            valid = cand >= 0
            safe = cand.clamp_min(0)
            # Use live concept keys so selected candidates can still train their geometry.
            ck = self.model.space.key(safe)
            score = (ck.float() * q[:, None, :]).sum(-1)
            score = score.masked_fill(~valid, -1e9)
            k = min(self.beam, score.size(1))
            topi = score.topk(k, dim=-1).indices
            beam = safe.gather(1, topi)

        bk = self.model.space.key(beam).float()
        scale = self.log_scale.exp().clamp(1.0, 50.0)
        score = (bk * q[:, None, :]).sum(-1) * scale
        weight = F.softmax(score, dim=-1)
        values = self.model.space.value(beam)
        read = (values.float() * weight[..., None]).sum(1)
        refined = h.float() + self.gate() * self.read_proj(read)
        return refined, beam, reads_per_example


class HybridEvalWrapper(nn.Module):
    """evaluate_full-compatible wrapper that applies SWG only at final state."""
    def __init__(self, model, swg, enabled=True):
        super().__init__()
        self.model = model
        self.swg = swg
        self.enabled = bool(enabled)

    @property
    def n_items(self):
        return self.model.n_items

    def eval(self):
        self.model.eval()
        self.swg.eval()
        return super().eval()

    @torch.inference_mode()
    def full_scores(self, seq, lengths):
        H, I, _ = self.model.encode_with_states(seq)
        row = torch.arange(seq.size(0), device=seq.device)
        ix = lengths - 1
        h = H[row, ix].float()
        if self.enabled:
            starts = I[row, ix]
            h, _, _ = self.swg.walk(h, starts)
        scores = h @ self.model.item.weight.T
        scores[:, 0] = -1e20
        return scores


def rebuild_hnsw(model, swg, device, M, ef_construction):
    keys = concept_keys_np(model, device)
    index, build_s = build_hnsw(keys, M, ef_construction)
    _, flat, stats = extract_hnsw_adjacency(index, model.n_concepts, M)
    swg.set_adjacency(flat)
    return {"seconds": build_s, **stats}


def hybrid_training_loss(model, swg, tokens, lengths):
    """FullCE on all positions; final position is replaced by SWG-refined FullCE."""
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    H, I, _ = model.encode_with_states(x)
    valid = y != 0
    if not valid.any():
        return H.sum() * 0.0, {"base_ce": 0.0, "hybrid_final_ce": 0.0, "final_count": 0, "edge_reads": 0}

    h_valid = H[valid].float()
    y_valid = y[valid]
    logits = h_valid @ model.item.weight[1:].T
    ce_each = F.cross_entropy(logits, y_valid - 1, reduction="none")
    total_loss = ce_each.sum()
    total_count = int(ce_each.numel())

    row = torch.arange(tokens.size(0), device=tokens.device)
    final_ix = (lengths - 2).clamp_min(0)
    has_final = lengths >= 2
    row = row[has_final]
    final_ix = final_ix[has_final]
    if row.numel() == 0:
        return total_loss / max(1, total_count), {"base_ce": float(ce_each.mean().detach()), "hybrid_final_ce": 0.0, "final_count": 0, "edge_reads": 0}

    h_final = H[row, final_ix].float()
    starts = I[row, final_ix]
    y_final = y[row, final_ix]
    refined, _, reads_per_example = swg.walk(h_final, starts)
    logits_h = refined @ model.item.weight[1:].T
    ce_h = F.cross_entropy(logits_h, y_final - 1, reduction="none")

    # Remove base CE for those final positions, replace with hybrid CE.
    # Compute their flattened valid-position indexes using cumulative counts.
    valid_counts = valid.long().sum(1)
    base_offsets = torch.cumsum(valid_counts, dim=0) - valid_counts
    flat_final = base_offsets[row] + final_ix
    old_final = ce_each[flat_final]
    total_loss = total_loss - old_final.sum() + ce_h.sum()

    stats = {
        "base_ce": float(ce_each.mean().detach()),
        "hybrid_final_ce": float(ce_h.mean().detach()),
        "final_count": int(row.numel()),
        "edge_reads": int(reads_per_example),
    }
    return total_loss / max(1, total_count), stats


@torch.no_grad()
def next_concept_seen_rate(model, swg, prefixes, targets, max_len, device, batch_size=512):
    model.eval(); swg.eval()
    total = hit = reads = 0
    for st in range(0, len(prefixes), batch_size):
        ps = prefixes[st:st + batch_size]
        lens = torch.tensor([min(len(p), max_len) for p in ps], dtype=torch.long)
        L = int(lens.max())
        seq = torch.zeros(len(ps), L, dtype=torch.long)
        for r, p in enumerate(ps):
            s = p[-max_len:]
            seq[r, :len(s)] = torch.tensor(s, dtype=torch.long)
        seq = seq.to(device); lens = lens.to(device)
        H, I, _ = model.encode_with_states(seq)
        row = torch.arange(seq.size(0), device=device); ix = lens - 1
        h = H[row, ix].float(); starts = I[row, ix]
        _, beam, rp = swg.walk(h, starts)
        tgt = torch.tensor(targets[st:st + len(ps)], dtype=torch.long, device=device)
        tgt_state = model.item(tgt) * math.sqrt(model.d_model)
        nc, _ = model.router(tgt_state, model.space)
        seen = (beam[:, :, None] == nc[:, None, :]).any(dim=(1, 2))
        hit += int(seen.sum().item()); total += len(ps); reads += rp * len(ps)
    return {"next_item_concept_seen_rate": hit / max(1, total), "mean_edge_reads": reads / max(1, total)}


def bucket_indices(prefixes, max_len):
    b = {"short_<=50": [], "medium_51_100": [], "long_101_200": []}
    for i, p in enumerate(prefixes):
        n = min(len(p), max_len)
        if n <= 50: b["short_<=50"].append(i)
        elif n <= 100: b["medium_51_100"].append(i)
        else: b["long_101_200"].append(i)
    return b


def subset(xs, idx):
    return [xs[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="/content/sparsewalker_data")
    ap.add_argument("--output-dir", default="/content/drive/MyDrive/sparsewalker_hybrid_swg_scratch")
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--hnsw-m", type=int, default=8)
    ap.add_argument("--ef-construction", type=int, default=160)
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--beam", type=int, default=16)
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print("DEVICE", device, torch.cuda.get_device_name(0) if device.type == "cuda" else None,
          "bf16", torch.cuda.is_bf16_supported() if device.type == "cuda" else False, flush=True)

    max_len = 200
    data = load_dataset("ml1m", args.data_dir)
    split = split_data(data["sequences"])
    protocol = protocol_manifest(max_len, data["n_items"])
    print("PROTOCOL", protocol, flush=True)

    model = SparseWalker(
        data["n_items"], max_len, d=64, layers=2, side=256, h=16,
        active=8, top_side=2, degree=4, fresh_weight=.25,
    ).to(device)
    swg = ScratchResidualSWG(model, hops=args.hops, beam=args.beam, gate_init=0.05).to(device)
    print("SCRATCH_CONFIG", {
        "walker_params": sum(p.numel() for p in model.parameters()),
        "swg_trainable_params": sum(p.numel() for p in swg.parameters() if p.requires_grad) - sum(p.numel() for p in model.parameters()),
        "K": 8, "hops": args.hops, "beam": args.beam,
        "hnsw_M": args.hnsw_m, "graph_rebuild": "every_epoch",
        "objective": "FullCE all positions; final position replaced by SWG-refined CE",
        "warm_start": False,
    }, flush=True)

    hstats = rebuild_hnsw(model, swg, device, args.hnsw_m, args.ef_construction)
    print("HNSW_INIT", {k: round(v, 4) if isinstance(v, float) else v for k, v in hstats.items()}, flush=True)

    # Exact evaluator contract check: wrapper OFF must equal canonical base evaluator.
    base0 = evaluate_full(model, split["val_prefix"], split["val_target"], data["n_items"], max_len,
                          device, topks=(10,), batch_size=args.eval_batch_size,
                          autocast_dtype=torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None)
    off_wrapper = HybridEvalWrapper(model, swg, enabled=False).to(device)
    off0 = evaluate_full(off_wrapper, split["val_prefix"], split["val_target"], data["n_items"], max_len,
                         device, topks=(10,), batch_size=args.eval_batch_size,
                         autocast_dtype=torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None)
    diff = abs(float(base0["NDCG@10"]) - float(off0["NDCG@10"]))
    print("EVALUATOR_ASSERT", {"base": base0["NDCG@10"], "wrapper_off": off0["NDCG@10"], "abs_diff": diff}, flush=True)
    if diff > 1e-10:
        raise RuntimeError("Hybrid evaluator does not reproduce canonical base evaluator")

    ds = WindowDataset(split["train"], max_len, args.seed)
    params = list(model.parameters()) + [p for n, p in swg.named_parameters() if not n.startswith("model.")]
    # Deduplicate shared model params because swg holds a model reference.
    seen = set(); uniq = []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p)); uniq.append(p)
    opt = torch.optim.AdamW(uniq, lr=1e-3, weight_decay=1e-4)

    out = Path(args.output_dir) / "ml1m" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True))

    best_ndcg = -1.0; best_epoch = 0; best_model = None; best_swg = None; bad = 0; history = []
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    for epoch in range(1, args.max_epochs + 1):
        # Rebuild from current geometry at the beginning of every epoch.
        hs = rebuild_hnsw(model, swg, device, args.hnsw_m, args.ef_construction)
        lr = set_lr(opt, epoch, args.max_epochs)
        ds.set_epoch(epoch)
        loader = DataLoader(ds, batch_sampler=length_bucket_batches(ds, args.batch_size, epoch),
                            collate_fn=collate_windows, pin_memory=device.type == "cuda")
        model.train(); swg.train()
        t0 = time.perf_counter(); loss_sum = base_sum = hy_sum = 0.0; nb = 0; examples = positions = padded = 0; reads = 0
        print("EPOCH_START", {"epoch": epoch, "lr": lr, "hnsw_build_s": round(hs["seconds"], 3),
                              "gate": float(swg.gate().detach().cpu())}, flush=True)
        for bi, (tokens, lengths) in enumerate(loader, 1):
            examples += int(tokens.size(0)); positions += int((lengths - 1).clamp_min(0).sum().item())
            padded += int(tokens.size(0) * max(0, tokens.size(1) - 1))
            tokens = tokens.to(device, non_blocking=True); lengths = lengths.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss, st = hybrid_training_loss(model, swg, tokens, lengths)
            loss.backward(); torch.nn.utils.clip_grad_norm_(uniq, 1.0); opt.step()
            loss_sum += float(loss.detach()); base_sum += st["base_ce"]; hy_sum += st["hybrid_final_ce"]; reads += st["edge_reads"] * st["final_count"]; nb += 1
            if bi == 1 or bi % 12 == 0 or bi == len(loader):
                print("BATCH", {"epoch": epoch, "batch": bi, "of": len(loader), "loss": round(loss_sum / nb, 5),
                                "batch_L": int(tokens.size(1)), "gate": round(float(swg.gate().detach().cpu()), 4)}, flush=True)
        if device.type == "cuda": torch.cuda.synchronize()
        secs = time.perf_counter() - t0

        pursued = model.graph.pursue(opt, refresh=2) if epoch >= 4 and epoch % 2 == 0 else 0
        row = {
            "epoch": epoch, "loss": loss_sum / max(1, nb), "base_position_ce": base_sum / max(1, nb),
            "hybrid_final_ce": hy_sum / max(1, nb), "lr": lr, "seconds": secs,
            "positions_per_s": positions / max(secs, 1e-9), "padding_efficiency": positions / max(1, padded),
            "gate": float(swg.gate().detach().cpu()), "mean_swg_edge_reads": reads / max(1, examples),
            "pursued_rows": pursued, "hnsw_build_s": hs["seconds"],
        }
        print("TRAIN", row, flush=True)

        if epoch == 1 or epoch % args.eval_every == 0:
            # Rebuild after the epoch so eval topology matches the current concept geometry.
            ev_h = rebuild_hnsw(model, swg, device, args.hnsw_m, args.ef_construction)
            on = HybridEvalWrapper(model, swg, enabled=True).to(device)
            off = HybridEvalWrapper(model, swg, enabled=False).to(device)
            val_on = evaluate_full(on, split["val_prefix"], split["val_target"], data["n_items"], max_len,
                                   device, topks=(10,), batch_size=args.eval_batch_size,
                                   autocast_dtype=torch.bfloat16 if use_bf16 else None)
            val_off = evaluate_full(off, split["val_prefix"], split["val_target"], data["n_items"], max_len,
                                    device, topks=(10,), batch_size=args.eval_batch_size,
                                    autocast_dtype=torch.bfloat16 if use_bf16 else None)
            route = next_concept_seen_rate(model, swg, split["val_prefix"], split["val_target"], max_len, device,
                                           batch_size=args.eval_batch_size)
            erow = {**row, "val_on_NDCG@10": float(val_on["NDCG@10"]), "val_off_NDCG@10": float(val_off["NDCG@10"]),
                    "residual_contribution": float(val_on["NDCG@10"] - val_off["NDCG@10"]),
                    "next_item_concept_seen_rate": route["next_item_concept_seen_rate"],
                    "eval_edge_reads": route["mean_edge_reads"], "eval_hnsw_build_s": ev_h["seconds"]}
            history.append(erow); pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            print("EVAL", erow, flush=True)

            ndcg = float(val_on["NDCG@10"])
            if ndcg > best_ndcg:
                best_ndcg = ndcg; best_epoch = epoch; bad = 0
                best_model = cpu_state_dict(model); best_swg = cpu_state_dict(swg)
                torch.save({"model": best_model, "swg": best_swg, "epoch": epoch, "val_on": val_on,
                            "val_off": val_off, "protocol": protocol}, out / "best.pt")
            else:
                bad += args.eval_every if epoch != 1 else 1
            if bad >= args.patience:
                print("EARLY_STOP", {"best_epoch": best_epoch, "best_ndcg": best_ndcg}, flush=True)
                break

    if best_model is None:
        raise RuntimeError("No validation checkpoint")
    model.load_state_dict(best_model); swg.load_state_dict(best_swg, strict=False)
    rebuild_hnsw(model, swg, device, args.hnsw_m, args.ef_construction)
    on = HybridEvalWrapper(model, swg, enabled=True).to(device)
    off = HybridEvalWrapper(model, swg, enabled=False).to(device)
    final_val = evaluate_full(on, split["val_prefix"], split["val_target"], data["n_items"], max_len, device,
                              topks=(10,), batch_size=args.eval_batch_size,
                              autocast_dtype=torch.bfloat16 if use_bf16 else None)
    final_off = evaluate_full(off, split["val_prefix"], split["val_target"], data["n_items"], max_len, device,
                              topks=(10,), batch_size=args.eval_batch_size,
                              autocast_dtype=torch.bfloat16 if use_bf16 else None)
    test = evaluate_full(on, split["test_prefix"], split["test_target"], data["n_items"], max_len, device,
                         topks=(10,20,50), batch_size=args.eval_batch_size,
                         autocast_dtype=torch.bfloat16 if use_bf16 else None)
    route = next_concept_seen_rate(model, swg, split["val_prefix"], split["val_target"], max_len, device,
                                   batch_size=args.eval_batch_size)

    hb = bucket_indices(split["val_prefix"], max_len); length_diag = {}
    for name, idx in hb.items():
        pon = subset(split["val_prefix"], idx); ton = subset(split["val_target"], idx)
        mon = evaluate_full(on, pon, ton, data["n_items"], max_len, device, topks=(10,), batch_size=args.eval_batch_size,
                            autocast_dtype=torch.bfloat16 if use_bf16 else None)
        mof = evaluate_full(off, pon, ton, data["n_items"], max_len, device, topks=(10,), batch_size=args.eval_batch_size,
                            autocast_dtype=torch.bfloat16 if use_bf16 else None)
        length_diag[name] = {"n": len(idx), "swg_on_NDCG@10": float(mon["NDCG@10"]),
                             "swg_off_NDCG@10": float(mof["NDCG@10"]),
                             "residual_delta": float(mon["NDCG@10"] - mof["NDCG@10"])}
    print("HISTORY_LENGTH", json.dumps(length_diag, indent=2), flush=True)

    result = {
        "cell": "SparseWalkerScratch+ResidualSWG",
        "selected_epoch": best_epoch,
        "val_on_NDCG@10": float(final_val["NDCG@10"]),
        "val_off_NDCG@10": float(final_off["NDCG@10"]),
        "residual_contribution": float(final_val["NDCG@10"] - final_off["NDCG@10"]),
        "target_0.145_reached": float(final_val["NDCG@10"]) >= 0.145,
        "next_item_concept_seen_rate": route["next_item_concept_seen_rate"],
        "gate": float(swg.gate().detach().cpu()),
        "history_length": length_diag,
        "protocol_fingerprint": protocol["fingerprint"],
        **test,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("FINAL_RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
