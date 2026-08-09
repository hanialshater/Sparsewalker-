#!/usr/bin/env python
"""Test HNSW-grown small-world concept graphs as a sparse attention primitive.

This does NOT train a recommender. It loads the best trained ML-1M SparseWalker,
uses its learned concept geometry, and asks a narrower question:

  Can a query derived from the current working-memory state recover the concepts
  that dense query-key attention would select, by navigating only a few sparse
  graph edges?

Controls / diagnostics:
- exact dense top-10 concepts under the working-memory query
- standard HNSW ANN search recall (global-entry control)
- current learned degree-4 Walker graph
- HNSW level-0 graph
- HNSW flattened graph including upper-layer shortcut edges
- oracle BFS reachability vs query-guided beam navigation
- actual next-item routed-concept reachability

Interpretation:
- high oracle, low navigation => routing/search policy is the problem
- low oracle => topology / starting state is the problem
- high HNSW global recall, low state-started recall => entry/working-memory locality is the problem
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sparsewalker.data import load_dataset, split_data
from sparsewalker.models import SparseWalker


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_balanced(prefixes, max_len, per_bucket, seed):
    rng = np.random.default_rng(seed)
    buckets = {"short_<=50": [], "medium_51_100": [], "long_101_200": []}
    for i, p in enumerate(prefixes):
        n = min(len(p), max_len)
        if n <= 50:
            buckets["short_<=50"].append(i)
        elif n <= 100:
            buckets["medium_51_100"].append(i)
        else:
            buckets["long_101_200"].append(i)
    out = []
    meta = {}
    for name, ids in buckets.items():
        take = min(per_bucket, len(ids))
        pick = rng.choice(np.asarray(ids), size=take, replace=False).tolist() if take else []
        out.extend(pick)
        meta[name] = take
    rng.shuffle(out)
    return out, meta


def pad_prefix_batch(prefixes, indices, max_len, device):
    rows = []
    lengths = []
    for i in indices:
        s = list(prefixes[i])[-max_len:]
        lengths.append(len(s))
        rows.append(s)
    L = max(lengths)
    x = torch.zeros(len(rows), L, dtype=torch.long, device=device)
    for r, s in enumerate(rows):
        x[r, : len(s)] = torch.as_tensor(s, dtype=torch.long, device=device)
    return x, torch.as_tensor(lengths, dtype=torch.long, device=device)


@torch.inference_mode()
def collect_queries(model, prefixes, targets, selected, max_len, device, batch_size, concept_keys):
    all_q, all_starts, all_dense, all_next, all_lengths = [], [], [], [], []
    keys_t = torch.as_tensor(concept_keys, device=device, dtype=torch.float32)
    for st in range(0, len(selected), batch_size):
        ids = selected[st : st + batch_size]
        seq, lengths = pad_prefix_batch(prefixes, ids, max_len, device)
        use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            H, I, M = model.encode_with_states(seq)
            row = torch.arange(seq.size(0), device=device)
            last = (lengths - 1).clamp_min(0)
            h = H[row, last]
            starts = I[row, last]
            q = F.normalize(model.graph.context_q(h), dim=-1)

            tgt = torch.as_tensor([targets[i] for i in ids], dtype=torch.long, device=device)
            tgt_state = model.item(tgt) * math.sqrt(model.d_model)
            next_ids, _ = model.router(tgt_state, model.space)

        # exact dense concept retrieval under the same query used for navigation
        dense = []
        q32 = q.float()
        for qst in range(0, q32.size(0), 128):
            score = q32[qst : qst + 128] @ keys_t.T
            dense.append(score.topk(10, dim=-1).indices.cpu())
        dense = torch.cat(dense, 0)

        all_q.append(q32.cpu())
        all_starts.append(starts.cpu())
        all_dense.append(dense)
        all_next.append(next_ids.cpu())
        all_lengths.extend(lengths.cpu().tolist())

    return {
        "q": torch.cat(all_q, 0).numpy().astype("float32"),
        "starts": torch.cat(all_starts, 0).numpy().astype("int64"),
        "dense_top10": torch.cat(all_dense, 0).numpy().astype("int64"),
        "next_concepts": torch.cat(all_next, 0).numpy().astype("int64"),
        "lengths": np.asarray(all_lengths, dtype=np.int64),
    }


def build_hnsw(keys, M, ef_construction):
    try:
        import faiss
    except ImportError as e:
        raise RuntimeError("Install faiss-cpu first: pip install -q faiss-cpu") from e

    d = keys.shape[1]
    try:
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    except TypeError:
        index = faiss.IndexHNSWFlat(d, M)
        index.metric_type = faiss.METRIC_INNER_PRODUCT
    index.hnsw.efConstruction = int(ef_construction)
    t0 = time.perf_counter()
    index.add(np.ascontiguousarray(keys.astype("float32")))
    build_s = time.perf_counter() - t0
    return index, build_s


def extract_hnsw_adjacency(index, n_nodes, M):
    import faiss
    offsets = faiss.vector_to_array(index.hnsw.offsets).astype(np.int64)
    neighbors = faiss.vector_to_array(index.hnsw.neighbors).astype(np.int64)
    try:
        level0_cap = int(index.hnsw.nb_neighbors(0))
    except Exception:
        level0_cap = 2 * int(M)

    deg_all = offsets[1:] - offsets[:-1]
    max_all = int(deg_all.max())
    level0 = np.full((n_nodes, level0_cap), -1, dtype=np.int64)
    flat = np.full((n_nodes, max_all), -1, dtype=np.int64)
    for i in range(n_nodes):
        a, b = int(offsets[i]), int(offsets[i + 1])
        row = neighbors[a:b]
        row = row[row >= 0]
        if row.size:
            flat[i, : row.size] = row
            l0 = row[:level0_cap]
            level0[i, : l0.size] = l0
    stats = {
        "level0_cap": level0_cap,
        "flat_max_degree": max_all,
        "level0_mean_degree": float((level0 >= 0).sum(1).mean()),
        "flat_mean_degree": float((flat >= 0).sum(1).mean()),
    }
    return level0, flat, stats


def exact_set_recall(found, target):
    return len(set(map(int, found)).intersection(map(int, target))) / max(1, len(target))


def standard_hnsw_control(index, queries, dense_top10, efs=(8, 16, 32, 64)):
    out = {}
    for ef in efs:
        index.hnsw.efSearch = int(ef)
        t0 = time.perf_counter()
        _, got = index.search(np.ascontiguousarray(queries), 10)
        dt = time.perf_counter() - t0
        recalls = [exact_set_recall(got[i], dense_top10[i]) for i in range(len(got))]
        out[str(ef)] = {
            "dense_recall@10": float(np.mean(recalls)),
            "queries_per_s": float(len(got) / max(dt, 1e-9)),
        }
    return out


def oracle_reachability(starts, target_sets, adjacency, max_hops):
    # Fraction of each target set physically reachable within <=h hops, ignoring routing scores.
    sums = np.zeros(max_hops + 1, dtype=np.float64)
    hits = np.zeros(max_hops + 1, dtype=np.float64)
    mean_visited = np.zeros(max_hops + 1, dtype=np.float64)
    for qi in range(len(starts)):
        target = set(map(int, target_sets[qi]))
        frontier = np.unique(starts[qi]).astype(np.int64)
        visited = set(map(int, frontier.tolist()))
        for h in range(max_hops + 1):
            inter = len(target.intersection(visited))
            sums[h] += inter / max(1, len(target))
            hits[h] += float(inter > 0)
            mean_visited[h] += len(visited)
            if h == max_hops or frontier.size == 0:
                continue
            nxt = adjacency[frontier].reshape(-1)
            nxt = nxt[nxt >= 0]
            if nxt.size == 0:
                frontier = np.empty(0, dtype=np.int64)
                continue
            nxt = np.unique(nxt)
            frontier = np.asarray([x for x in nxt.tolist() if int(x) not in visited], dtype=np.int64)
            visited.update(map(int, frontier.tolist()))
    n = max(1, len(starts))
    return {
        str(h): {
            "target_recall": float(sums[h] / n),
            "any_target_hit_rate": float(hits[h] / n),
            "mean_nodes_reachable": float(mean_visited[h] / n),
        }
        for h in range(max_hops + 1)
    }


def navigate_one(q, starts, adjacency, keys, max_hops, beam):
    current = np.unique(starts).astype(np.int64)
    visited = set(map(int, current.tolist()))
    comparisons = 0
    snapshots = []
    for h in range(max_hops + 1):
        ids = np.fromiter(visited, dtype=np.int64)
        scores = keys[ids] @ q
        k = min(10, ids.size)
        top = ids[np.argpartition(scores, -k)[-k:]] if k else np.empty(0, dtype=np.int64)
        if k:
            top = top[np.argsort(keys[top] @ q)[::-1]]
        snapshots.append((top.copy(), set(visited), comparisons))
        if h == max_hops or current.size == 0:
            continue

        nbr = adjacency[current].reshape(-1)
        nbr = nbr[nbr >= 0]
        comparisons += int(nbr.size)
        if nbr.size == 0:
            current = np.empty(0, dtype=np.int64)
            continue
        cand = np.unique(np.concatenate([current, nbr]))
        cscore = keys[cand] @ q
        b = min(int(beam), cand.size)
        current = cand[np.argpartition(cscore, -b)[-b:]] if b else np.empty(0, dtype=np.int64)
        if b:
            current = current[np.argsort(keys[current] @ q)[::-1]]
        visited.update(map(int, current.tolist()))
    return snapshots


def navigation_metrics(queries, starts, dense_top10, next_concepts, adjacency, keys, max_hops, beam):
    dense_rec = np.zeros(max_hops + 1)
    dense_hit = np.zeros(max_hops + 1)
    next_seen = np.zeros(max_hops + 1)
    next_out = np.zeros(max_hops + 1)
    comps = np.zeros(max_hops + 1)
    visited_n = np.zeros(max_hops + 1)
    n = len(queries)
    for i in range(n):
        snaps = navigate_one(queries[i], starts[i], adjacency, keys, max_hops, beam)
        dset = set(map(int, dense_top10[i]))
        nset = set(map(int, next_concepts[i]))
        for h, (top, visited, c) in enumerate(snaps):
            dint = len(dset.intersection(map(int, top.tolist())))
            dense_rec[h] += dint / 10.0
            dense_hit[h] += float(dint > 0)
            next_seen[h] += float(bool(nset.intersection(visited)))
            next_out[h] += float(bool(nset.intersection(map(int, top.tolist()))))
            comps[h] += c
            visited_n[h] += len(visited)
    n = max(1, n)
    return {
        str(h): {
            "dense_recall@10": float(dense_rec[h] / n),
            "dense_any_hit_rate": float(dense_hit[h] / n),
            "next_item_concept_seen_rate": float(next_seen[h] / n),
            "next_item_concept_in_output_rate": float(next_out[h] / n),
            "mean_edge_reads": float(comps[h] / n),
            "mean_visited_nodes": float(visited_n[h] / n),
        }
        for h in range(max_hops + 1)
    }


def by_length_metrics(data, fn):
    out = {}
    masks = {
        "short_<=50": data["lengths"] <= 50,
        "medium_51_100": (data["lengths"] > 50) & (data["lengths"] <= 100),
        "long_101_200": data["lengths"] > 100,
    }
    for name, mask in masks.items():
        idx = np.where(mask)[0]
        if idx.size:
            out[name] = fn(idx)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument("--checkpoint", default="/content/drive/MyDrive/sparsewalker_canonical_pair/ml1m/seed42/SparseWalker_FullCE/best.pt")
    p.add_argument("--output", default="/content/drive/MyDrive/sparsewalker_swg_attention/result.json")
    p.add_argument("--per-bucket", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hnsw-m", type=int, default=8)
    p.add_argument("--ef-construction", type=int, default=160)
    p.add_argument("--max-hops", type=int, default=4)
    p.add_argument("--beams", type=int, nargs="+", default=[8, 16])
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print("DEVICE", device, torch.cuda.get_device_name(0) if device.type == "cuda" else None, flush=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    data = load_dataset("ml1m", args.data_dir)
    split = split_data(data["sequences"])
    max_len = 200
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = SparseWalker(
        data["n_items"], max_len, d=64, layers=2, side=256, h=16,
        active=8, top_side=2, degree=4, fresh_weight=.25,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("CHECKPOINT", {"epoch": int(ckpt.get("epoch", -1)), "val_NDCG@10": ckpt.get("val", {}).get("NDCG@10")}, flush=True)

    with torch.inference_mode():
        ids = torch.arange(model.n_concepts, device=device)
        keys = model.space.key(ids).float().cpu().numpy().astype("float32")
    # defensive renormalization for cosine/IP equivalence
    keys /= np.linalg.norm(keys, axis=1, keepdims=True).clip(min=1e-12)
    current_adj = model.graph.destination.detach().cpu().numpy().astype(np.int64)
    print("CONCEPT_SPACE", {"n": int(keys.shape[0]), "dim": int(keys.shape[1]), "current_degree": int(current_adj.shape[1])}, flush=True)

    selected, sample_meta = sample_balanced(split["val_prefix"], max_len, args.per_bucket, args.seed)
    t0 = time.perf_counter()
    qdata = collect_queries(
        model, split["val_prefix"], split["val_target"], selected,
        max_len, device, args.batch_size, keys,
    )
    print("QUERY_SAMPLE", {"n": len(selected), "buckets": sample_meta, "seconds": round(time.perf_counter() - t0, 3)}, flush=True)

    index, build_s = build_hnsw(keys, args.hnsw_m, args.ef_construction)
    h0, hflat, hstats = extract_hnsw_adjacency(index, len(keys), args.hnsw_m)
    print("HNSW_BUILD", {"M": args.hnsw_m, "efConstruction": args.ef_construction, "seconds": round(build_s, 3), **hstats}, flush=True)

    global_control = standard_hnsw_control(index, qdata["q"], qdata["dense_top10"])
    print("GLOBAL_HNSW_CONTROL", json.dumps(global_control, indent=2), flush=True)

    graphs = {
        "current_degree4": current_adj,
        "hnsw_level0": h0,
        "hnsw_flattened": hflat,
    }
    result = {
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_val_NDCG@10": ckpt.get("val", {}).get("NDCG@10"),
        "sample": sample_meta,
        "n_queries": len(selected),
        "hnsw": {"M": args.hnsw_m, "efConstruction": args.ef_construction, "build_seconds": build_s, **hstats},
        "global_hnsw_control": global_control,
        "graphs": {},
    }

    for name, adj in graphs.items():
        print("\nGRAPH", name, {"max_degree": int(adj.shape[1]), "mean_degree": float((adj >= 0).sum(1).mean())}, flush=True)
        oracle_dense = oracle_reachability(qdata["starts"], qdata["dense_top10"], adj, args.max_hops)
        oracle_next = oracle_reachability(qdata["starts"], qdata["next_concepts"], adj, args.max_hops)
        print("ORACLE_DENSE", json.dumps(oracle_dense, indent=2), flush=True)
        print("ORACLE_NEXT", json.dumps(oracle_next, indent=2), flush=True)

        nav = {}
        nav_by_len = {}
        for beam in args.beams:
            m = navigation_metrics(
                qdata["q"], qdata["starts"], qdata["dense_top10"], qdata["next_concepts"],
                adj, keys, args.max_hops, beam,
            )
            nav[str(beam)] = m
            print("NAVIGATION", {"graph": name, "beam": beam, "metrics": m}, flush=True)

            def subset_fn(idx):
                return navigation_metrics(
                    qdata["q"][idx], qdata["starts"][idx], qdata["dense_top10"][idx], qdata["next_concepts"][idx],
                    adj, keys, args.max_hops, beam,
                )
            nav_by_len[str(beam)] = by_length_metrics(qdata, subset_fn)

        result["graphs"][name] = {
            "max_degree": int(adj.shape[1]),
            "mean_degree": float((adj >= 0).sum(1).mean()),
            "oracle_dense": oracle_dense,
            "oracle_next": oracle_next,
            "navigation": nav,
            "navigation_by_history_length": nav_by_len,
        }

    # Compact decision panel at 4 hops / largest beam.
    beam = str(max(args.beams))
    hop = str(args.max_hops)
    panel = {}
    for name in graphs:
        g = result["graphs"][name]
        panel[name] = {
            "oracle_dense_recall": g["oracle_dense"][hop]["target_recall"],
            "nav_dense_recall@10": g["navigation"][beam][hop]["dense_recall@10"],
            "oracle_next_hit": g["oracle_next"][hop]["any_target_hit_rate"],
            "nav_next_seen": g["navigation"][beam][hop]["next_item_concept_seen_rate"],
            "edge_reads": g["navigation"][beam][hop]["mean_edge_reads"],
        }
    result["decision_panel"] = panel
    print("\nDECISION_PANEL", json.dumps(panel, indent=2), flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
