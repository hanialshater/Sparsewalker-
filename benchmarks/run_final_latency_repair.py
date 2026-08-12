#!/usr/bin/env python
"""Latency-only repair for the final SparseWalker vs SASRec serving audit.

Fixes three problems in notebook 28:
1) keep checkpoint weights in FP32 and use matched BF16 autocast for BOTH models,
   exactly matching the quality audit policy;
2) never nest explicit CUDA Graph capture around torch.compile(reduce-overhead);
3) use static inputs for both models, so no asymmetric request-window mutation occurs.

Headline mode is an explicit CUDA Graph around the eager exact-semantics BF16-autocast
path for both models. If either capture fails or changes outputs, the benchmark falls
back to the matched eager BF16-autocast path. torch.compile is reported separately and
is never used for the headline unless it is numerically/discretely equivalent.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sparsewalker.data import load_dataset
from run_final_quality_equivalent_serving import (
    SASExactWindow,
    WalkerExactWindow,
    build_sas,
    build_walker,
    make_terminal,
    terminal_sparse,
)


@torch.inference_mode()
def latency_samples(fn, warm=20, n=120):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    vals = np.empty(n, dtype=np.float64)
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    for i in range(n):
        a.record(); fn(); b.record(); b.synchronize()
        vals[i] = a.elapsed_time(b) * 1000.0
    return {
        "mean_us": float(vals.mean()),
        "p50_us": float(np.percentile(vals, 50)),
        "p95_us": float(np.percentile(vals, 95)),
        "p99_us": float(np.percentile(vals, 99)),
    }


def capture_graph(fn):
    # Warm on a side stream, then capture the uncompiled eager path. This avoids
    # nesting CUDAGraph capture around torch.compile(mode='reduce-overhead').
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(8):
            fn()
    side.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        static_out = fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    return g, g.replay, static_out, side


def tensor_max_abs(a, b):
    return float((a.float() - b.float()).abs().max().cpu())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument("--sasrec-checkpoint", default="/content/drive/MyDrive/sparsewalker_esasrec_2x2/ml1m/seed42/SASRec_FullCE/best.pt")
    p.add_argument("--walker-checkpoint", default="/content/drive/MyDrive/sparsewalker_two_temporal_layers/ml1m/seed42/best.pt")
    p.add_argument("--catalog", type=int, default=1_000_000)
    p.add_argument("--output", default="/content/drive/MyDrive/sparsewalker_speed/final_latency_repair.json")
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    data = load_dataset("ml1m", args.data_dir)
    sck = torch.load(args.sasrec_checkpoint, map_location="cpu")
    wck = torch.load(args.walker_checkpoint, map_location="cpu")

    # IMPORTANT: keep checkpoint weights FP32. BF16 inference is autocast, exactly
    # as in evaluate_full(..., autocast_dtype=torch.bfloat16).
    sas = build_sas(data["n_items"], device, sck).eval()
    walker = build_walker(data["n_items"], device, wck).eval()
    swrap = SASExactWindow(sas).eval()
    wwrap = WalkerExactWindow(walker).eval()

    seq_s = torch.randint(1, data["n_items"] + 1, (1, 200), device=device, dtype=torch.long)
    seq_w = seq_s.clone()
    lens_s = torch.tensor([200], device=device, dtype=torch.long)
    lens_w = lens_s.clone()
    cand_ids = torch.randint(1, data["n_items"] + 1, (512,), device=device, dtype=torch.long)
    cand_emb = sas.item_weight[cand_ids].detach().contiguous()
    terminal = make_terminal(walker, device, args.catalog, degree=64)

    def sas_model_eager():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return swrap(seq_s, lens_s)

    def sas_e2e_eager():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = swrap(seq_s, lens_s)
            return torch.topk(h @ cand_emb.T, k=10, dim=-1)

    def walker_model_eager():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return wwrap(seq_w, lens_w)

    def walker_e2e_eager():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, ids, mass = wwrap(seq_w, lens_w)
        terminal_sparse(h, ids, terminal)
        return terminal["out"]

    # Stable eager reference used for both correctness and matched fallback timing.
    s_ref = sas_model_eager().detach().clone()
    w_ref_h, w_ref_i, w_ref_m = walker_model_eager()
    w_ref_h = w_ref_h.detach().clone(); w_ref_i = w_ref_i.detach().clone(); w_ref_m = w_ref_m.detach().clone()
    torch.cuda.synchronize()

    print("POLICY", json.dumps({
        "weights": "FP32 checkpoints for BOTH",
        "timed_inference": "BF16 autocast for BOTH",
        "semantics": "exact canonical last-200 recompute for BOTH",
        "sasrec_retrieval": "FREE discovery + score/top10 over 512 candidates",
        "walker_retrieval": "native sparse terminal discovery+score over 512 reachable items",
        "headline_execution": "explicit CUDA Graph around UNCOMPILED eager path for BOTH",
        "compile_execution": "reported separately; never nested inside explicit CUDA Graph",
        "quality_recheck": "not repeated; notebook 28 full quality audit already passed",
    }, indent=2), flush=True)

    # Matched eager exact-semantics baseline.
    eager = {
        "sasrec_model": latency_samples(sas_model_eager, n=80),
        "sasrec_freeANN512": latency_samples(sas_e2e_eager, n=80),
        "walker_model": latency_samples(walker_model_eager, n=40),
        "walker_sparse_terminal": latency_samples(walker_e2e_eager, n=40),
    }
    print("MATCHED_EAGER", json.dumps(eager, indent=2), flush=True)

    # torch.compile is diagnostic only. It must pass output checks before latency
    # is considered. Walker requires exact final concept IDs because hard Top-K
    # routing makes an ID mismatch a semantic change, not harmless floating noise.
    compile_result = {}
    try:
        scomp = torch.compile(sas_model_eager, mode="reduce-overhead", fullgraph=False)
        for _ in range(3): scomp()
        torch.cuda.synchronize()
        sc = scomp()
        s_err = tensor_max_abs(s_ref, sc)
        s_ok = s_err <= 0.05
        compile_result["sasrec_correctness"] = {"max_abs_hidden": s_err, "accepted": s_ok}
        if s_ok:
            compile_result["sasrec_model_latency"] = latency_samples(scomp, n=100)
    except Exception as e:
        compile_result["sasrec_error"] = repr(e)

    try:
        wcomp = torch.compile(walker_model_eager, mode="reduce-overhead", fullgraph=False)
        for _ in range(2): wcomp()
        torch.cuda.synchronize()
        wh, wi, wm = wcomp()
        w_err = tensor_max_abs(w_ref_h, wh)
        w_ids = bool(torch.equal(w_ref_i, wi))
        w_ok = w_ids and w_err <= 0.05
        compile_result["walker_correctness"] = {
            "max_abs_hidden": w_err,
            "final_ids_match": w_ids,
            "accepted": w_ok,
        }
        if w_ok:
            compile_result["walker_model_latency"] = latency_samples(wcomp, n=50)
    except Exception as e:
        compile_result["walker_error"] = repr(e)
    print("COMPILE_SEPARATE", json.dumps(compile_result, indent=2), flush=True)

    # Symmetric explicit CUDA Graph over the exact eager/autocast functions.
    graph = {}
    holders = []
    graph_ok = True

    try:
        g, replay, out, side = capture_graph(sas_model_eager)
        holders += [g, side]
        replay(); torch.cuda.synchronize()
        err = tensor_max_abs(s_ref, out)
        ok = err <= 0.05
        graph["sasrec_model_correctness"] = {"max_abs_hidden": err, "accepted": ok}
        if ok:
            graph["sasrec_model"] = latency_samples(replay, n=150)
        else:
            graph_ok = False
    except Exception as e:
        graph["sasrec_model_error"] = repr(e); graph_ok = False

    try:
        # E2E capture output is top-k tuple; correctness here only verifies capture
        # succeeds. Model numerical correctness was verified immediately above.
        g, replay, out, side = capture_graph(sas_e2e_eager)
        holders += [g, side]
        replay(); torch.cuda.synchronize()
        graph["sasrec_freeANN512"] = latency_samples(replay, n=150)
    except Exception as e:
        graph["sasrec_freeANN512_error"] = repr(e); graph_ok = False

    try:
        g, replay, out, side = capture_graph(walker_model_eager)
        holders += [g, side]
        replay(); torch.cuda.synchronize()
        wh, wi, wm = out
        err = tensor_max_abs(w_ref_h, wh)
        ids_ok = bool(torch.equal(w_ref_i, wi))
        ok = ids_ok and err <= 0.05
        graph["walker_model_correctness"] = {
            "max_abs_hidden": err, "final_ids_match": ids_ok, "accepted": ok
        }
        if ok:
            graph["walker_model"] = latency_samples(replay, n=80)
        else:
            graph_ok = False
    except Exception as e:
        graph["walker_model_error"] = repr(e); graph_ok = False

    try:
        ref_out = walker_e2e_eager().detach().clone()
        g, replay, out, side = capture_graph(walker_e2e_eager)
        holders += [g, side]
        replay(); torch.cuda.synchronize()
        ids_ok = bool(torch.equal(ref_out, out))
        graph["walker_sparse_correctness"] = {"top10_ids_match": ids_ok, "accepted": ids_ok}
        if ids_ok:
            graph["walker_sparse_terminal"] = latency_samples(replay, n=80)
        else:
            graph_ok = False
    except Exception as e:
        graph["walker_sparse_terminal_error"] = repr(e); graph_ok = False

    print("EXPLICIT_CUDA_GRAPH", json.dumps(graph, indent=2), flush=True)

    def gp50(name):
        return graph.get(name, {}).get("p50_us")
    if graph_ok and gp50("sasrec_freeANN512") and gp50("walker_sparse_terminal"):
        source = "explicit_cuda_graph_eager_exact_bf16_autocast"
        sas_p = gp50("sasrec_freeANN512")
        walker_p = gp50("walker_sparse_terminal")
    else:
        source = "matched_eager_exact_bf16_autocast_fallback"
        sas_p = eager["sasrec_freeANN512"]["p50_us"]
        walker_p = eager["walker_sparse_terminal"]["p50_us"]

    headline = {
        "source": source,
        "sasrec_exact_window_plus_FREE_ANN512_p50_us": sas_p,
        "walker_exact_window_plus_native_sparse_terminal_p50_us": walker_p,
        "walker_speedup_vs_sasrec": sas_p / walker_p,
        "quality_contract": "exact current trained last-200 semantics; full quality already verified in notebook 28",
        "precision_contract": "FP32 checkpoint weights + BF16 autocast on BOTH",
        "terminal_quality_caveat": "Walker sparse terminal latency is native, but its 2-temporal terminal support quality is not yet validated/distilled",
        "persistent_path_caveat": "fast persistent KV/SWG paths remain excluded because they are not quality-equivalent after window saturation for current learned-position checkpoints",
    }
    print("REPAIRED_HEADLINE", json.dumps(headline, indent=2), flush=True)

    result = {"policy": "matched BF16 autocast", "eager": eager, "compile": compile_result,
              "explicit_cuda_graph": graph, "headline": headline}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("SAVED", out, flush=True)
    _ = holders


if __name__ == "__main__":
    main()
