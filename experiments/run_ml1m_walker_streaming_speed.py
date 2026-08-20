#!/usr/bin/env python
"""Speed audit and optimized trainer for streaming-native SparseWalker on ML-1M.

This file intentionally keeps the model semantics from
`run_ml1m_walker_streaming_native.py` unchanged:
- corrected SparseWalker v1.1 local recurrence
- persistent K=8 local state across full chronological histories
- two dense causal temporal layers with learned relative-lag bias
- TBPTT detach at chunk boundaries
- optimizer step only after the full user batch has streamed through

The changes here are systems-only:
1. build each padded batch on CPU and transfer it to CUDA once;
2. derive valid counts/chunk limits from Python sequence lengths, avoiding
   GPU->CPU synchronization in the chunk loop;
3. aggregate detached loss on GPU and materialize it only once per epoch;
4. group several independent TBPTT chunk losses into one backward() call;
5. optionally torch.compile the fixed-shape local Walker chunk;
6. benchmark batch sizes and compile modes before committing to a training run.

The benchmark always checks eager/compiled local-state parity before timing.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sparsewalker.data import load_dataset, split_data

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_ml1m_walker_streaming_native import (  # noqa: E402
    StreamingSparseWalker,
    causal_test,
    chunk_invariance_test,
    cpu_state_dict,
    evaluate_streaming,
    make_batches,
    seed_all,
    set_lr,
)


class FastStreamingSparseWalker(StreamingSparseWalker):
    """StreamingSparseWalker with an optionally compiled local chunk.

    The local recurrence is copied exactly from PR17's `local_chunk`, except
    that state is passed as tensors (never None) to give torch.compile a stable
    signature. The temporal layers are unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compiled_local = None
        self._use_compiled_local = False

    def _local_chunk_from_state(self, seq, ids, mass):
        B, C = seq.shape
        valid = seq.ne(0)
        item_state = self.item(seq) * math.sqrt(self.d_model)
        fi, fm = self.router(item_state.reshape(B * C, self.d_model), self.space)
        fi = fi.view(B, C, -1)
        fm = fm.view(B, C, -1)

        outs = []
        for t in range(C):
            act = valid[:, t]
            af = act.to(item_state.dtype)[:, None]
            xids = ids
            xmass = mass * af
            xids, xmass = self._merge(xids, xmass, fi[:, t], fm[:, t] * af)
            for _ in range(self.layers_n):
                xids, xmass = self.graph(
                    xids,
                    xmass,
                    item_state[:, t],
                    self.space,
                    track_touched=False,
                )
            ids = torch.where(act[:, None], xids, ids)
            mass = torch.where(act[:, None], xmass, mass)
            msg = (self.space.value(ids) * mass[:, :, None]).sum(1)
            ht = self.norm(item_state[:, t] + self.message_proj(msg)) * af
            outs.append(ht)
        return torch.stack(outs, dim=1), ids, mass

    def enable_local_compile(self, mode="reduce-overhead"):
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")
        # dynamic=False deliberately specializes for the common [B,64] training
        # chunk. The trainer pads only the final chunk to keep C fixed.
        self._compiled_local = torch.compile(
            self._local_chunk_from_state,
            mode=mode,
            fullgraph=False,
            dynamic=False,
        )
        self._use_compiled_local = True

    def set_local_compile(self, enabled):
        self._use_compiled_local = bool(enabled) and self._compiled_local is not None

    def local_chunk(self, seq, state=None, return_ids=False):
        # Preserve the PR17 path for the uncommon ID-history diagnostic.
        if return_ids:
            return super().local_chunk(seq, state, return_ids=True)

        B = seq.size(0)
        if state is None:
            ids = torch.zeros(B, self.active, dtype=torch.long, device=seq.device)
            mass = torch.zeros(
                B, self.active, dtype=self.item.weight.dtype, device=seq.device
            )
        else:
            ids, mass = state

        fn = (
            self._compiled_local
            if self._use_compiled_local and self._compiled_local is not None
            else self._local_chunk_from_state
        )
        h, ids, mass = fn(seq, ids, mass)
        return h, (ids, mass), None


def build_model(n_items, memory_size=512):
    return FastStreamingSparseWalker(
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
        memory_size=memory_size,
    )


def pad_xy_fast(seqs, ids, device):
    """Build one pinned CPU batch, then perform one H2D transfer per tensor."""
    rows = [seqs[i] for i in ids]
    lengths = [len(s) - 1 for s in rows]
    T = max(lengths)
    pin = device.type == "cuda"
    x = torch.zeros(len(rows), T, dtype=torch.long, pin_memory=pin)
    y = torch.zeros(len(rows), T, dtype=torch.long, pin_memory=pin)
    for r, s in enumerate(rows):
        v = torch.as_tensor(s, dtype=torch.long)
        n = v.numel() - 1
        x[r, :n] = v[:-1]
        y[r, :n] = v[1:]
    non_blocking = device.type == "cuda"
    x = x.to(device, non_blocking=non_blocking)
    y = y.to(device, non_blocking=non_blocking)
    return x, y, lengths, sum(lengths)


def _pad_chunk(xc, yc, chunk_size):
    """Pad only the final training chunk so local compile sees fixed C."""
    c = xc.size(1)
    if c == chunk_size:
        return xc, yc
    pad = chunk_size - c
    xc = F.pad(xc, (0, pad), value=0)
    yc = F.pad(yc, (0, pad), value=0)
    return xc, yc


def train_epoch_fast(
    model,
    seqs,
    opt,
    device,
    *,
    batch_size,
    chunk_size,
    seed,
    epoch,
    use_bf16=True,
    backward_group_chunks=4,
    max_batches=None,
    fixed_training_chunk=True,
):
    """Streaming TBPTT with no inner-loop CPU synchronization.

    Because local + temporal states are detached after every chunk, chunk
    computational graphs are independent. Summing N chunk losses before one
    backward() produces the same accumulated gradient as N backward() calls;
    it only reduces Python/autograd launch overhead.
    """
    model.train()
    batches = make_batches(seqs, batch_size, seed, epoch)
    if max_batches is not None:
        batches = batches[: int(max_batches)]

    total_positions = 0
    epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    backward_calls = 0
    chunk_count = 0
    t0 = time.perf_counter()

    for ids in batches:
        x, y, lengths, valid_total = pad_xy_fast(seqs, ids, device)
        if valid_total <= 0:
            continue

        # No CUDA reduction is needed to decide how many chunks exist.
        T = max(lengths)
        opt.zero_grad(set_to_none=True)
        local_state = None
        temporal_states = None
        pending = None
        pending_chunks = 0

        for st in range(0, T, chunk_size):
            en = min(T, st + chunk_size)
            xc = x[:, st:en]
            yc = y[:, st:en]
            if fixed_training_chunk:
                xc, yc = _pad_chunk(xc, yc, chunk_size)

            valid = yc.ne(0)
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=bool(use_bf16 and device.type == "cuda"),
            ):
                h, local_state, temporal_states, _ = model.forward_chunk(
                    xc,
                    local_state,
                    temporal_states,
                    start_pos=st,
                )
                hv = h[valid]
                yv = yc[valid]
                logits = model.score_hidden(hv)
                loss_sum = F.cross_entropy(logits.float(), yv, reduction="sum")
                loss = loss_sum / float(valid_total)

            # Keep telemetry on GPU until the end of the epoch.
            epoch_loss_sum = epoch_loss_sum + loss_sum.detach()
            pending = loss if pending is None else pending + loss
            pending_chunks += 1
            chunk_count += 1

            # Preserve PR17 TBPTT semantics exactly.
            local_state, temporal_states = model.detach_states(
                local_state, temporal_states
            )

            last_chunk = en >= T
            if pending_chunks >= max(1, int(backward_group_chunks)) or last_chunk:
                pending.backward()
                pending = None
                pending_chunks = 0
                backward_calls += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total_positions += valid_total

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    loss = float(epoch_loss_sum.detach().cpu()) / max(1, total_positions)
    return {
        "loss": loss,
        "positions": int(total_positions),
        "seconds": float(seconds),
        "positions_per_s": float(total_positions / max(seconds, 1e-9)),
        "batches": int(len(batches)),
        "chunks": int(chunk_count),
        "backward_calls": int(backward_calls),
        "backward_group_chunks": int(backward_group_chunks),
        "batch_size": int(batch_size),
        "compiled_local": bool(model._use_compiled_local),
        "fixed_training_chunk": bool(fixed_training_chunk),
    }


@torch.inference_mode()
def local_parity_test(model, device, batch=8, chunk=64):
    """Fast eager and compiled local recurrence must match the PR17 reference."""
    model.eval()
    seq = torch.randint(1, model.n_items + 1, (batch, chunk), device=device)
    seq[0, -7:] = 0
    ids = torch.zeros(batch, model.active, device=device, dtype=torch.long)
    mass = torch.zeros(
        batch, model.active, device=device, dtype=model.item.weight.dtype
    )

    # Call the parent method explicitly so this cannot compare the optimized
    # implementation against itself.
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        href, (iref, mref), _ = StreamingSparseWalker.local_chunk(
            model,
            seq,
            (ids.clone(), mass.clone()),
            return_ids=False,
        )
        h0, i0, m0 = model._local_chunk_from_state(
            seq, ids.clone(), mass.clone()
        )

    out = {
        "fast_eager_hidden_max_abs": float(
            (href.float() - h0.float()).abs().max().cpu()
        ),
        "fast_eager_mass_max_abs": float(
            (mref.float() - m0.float()).abs().max().cpu()
        ),
        "fast_eager_ids_match": bool(torch.equal(iref, i0)),
        "compiled": model._compiled_local is not None,
    }

    eager_ok = (
        out["fast_eager_ids_match"]
        and out["fast_eager_hidden_max_abs"] <= 1e-5
        and out["fast_eager_mass_max_abs"] <= 1e-6
    )

    if model._compiled_local is None:
        out.update(
            {
                "compiled_hidden_max_abs": 0.0,
                "compiled_mass_max_abs": 0.0,
                "compiled_ids_match": True,
                "accepted": eager_ok,
            }
        )
        return out

    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        h1, i1, m1 = model._compiled_local(
            seq, ids.clone(), mass.clone()
        )
    out.update(
        {
            "compiled_hidden_max_abs": float(
                (href.float() - h1.float()).abs().max().cpu()
            ),
            "compiled_mass_max_abs": float(
                (mref.float() - m1.float()).abs().max().cpu()
            ),
            "compiled_ids_match": bool(torch.equal(iref, i1)),
        }
    )
    out["accepted"] = (
        eager_ok
        and out["compiled_ids_match"]
        and out["compiled_hidden_max_abs"] <= 5e-2
        and out["compiled_mass_max_abs"] <= 5e-3
    )
    return out


def grouped_backward_parity_test(n_items, device, memory_size=512):
    """One optimizer step: backward-per-chunk vs grouped backward."""
    seed_all(777)
    base = build_model(n_items, memory_size).to(device)
    state = cpu_state_dict(base)
    del base

    # Synthetic histories force three 64-event chunks.
    rng = np.random.default_rng(123)
    seqs = [
        rng.integers(1, n_items + 1, size=150).tolist()
        for _ in range(4)
    ]

    def one(group):
        seed_all(999)
        m = build_model(n_items, memory_size).to(device)
        m.load_state_dict(state)
        m.set_local_compile(False)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        seed_all(2026)
        stats = train_epoch_fast(
            m,
            seqs,
            opt,
            device,
            batch_size=4,
            chunk_size=64,
            seed=42,
            epoch=1,
            use_bf16=True,
            backward_group_chunks=group,
            max_batches=1,
        )
        return cpu_state_dict(m), stats

    s1, a = one(1)
    sg, b = one(4)
    max_abs = 0.0
    worst = None
    for k in s1:
        if not torch.is_floating_point(s1[k]):
            continue
        d = float((s1[k].float() - sg[k].float()).abs().max())
        if d > max_abs:
            max_abs = d
            worst = k
    return {
        "max_abs_parameter_delta": max_abs,
        "worst_parameter": worst,
        "group1_backward_calls": a["backward_calls"],
        "group4_backward_calls": b["backward_calls"],
        "accepted": max_abs <= 2e-5,
    }


def _warm_for_benchmark(
    model,
    base_state,
    seqs,
    device,
    *,
    batch_size,
    chunk_size,
    seed,
    compile_local,
    backward_group_chunks,
):
    """Warm kernels/compile once, then restore weights for fair timing."""
    model.load_state_dict(base_state)
    if compile_local:
        if model._compiled_local is None:
            model.enable_local_compile()
        model.set_local_compile(True)
    else:
        model.set_local_compile(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_epoch_fast(
        model,
        seqs,
        opt,
        device,
        batch_size=batch_size,
        chunk_size=chunk_size,
        seed=seed,
        epoch=1,
        backward_group_chunks=backward_group_chunks,
        max_batches=1,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_speed(
    data,
    split,
    device,
    *,
    seed,
    chunk_size,
    memory_size,
    batch_sizes,
    benchmark_batches,
    backward_group_chunks,
    try_compile,
):
    """Short controlled sweep from identical initial weights."""
    seed_all(seed)
    model = build_model(data["n_items"], memory_size).to(device)
    base_state = cpu_state_dict(model)
    rows = []

    modes = [False, True] if try_compile else [False]
    for compile_local in modes:
        for batch_size in batch_sizes:
            try:
                _warm_for_benchmark(
                    model,
                    base_state,
                    split["train"],
                    device,
                    batch_size=batch_size,
                    chunk_size=chunk_size,
                    seed=seed,
                    compile_local=compile_local,
                    backward_group_chunks=backward_group_chunks,
                )
                model.load_state_dict(base_state)
                model.set_local_compile(compile_local)
                parity = local_parity_test(
                    model, device, batch=min(8, batch_size), chunk=chunk_size
                )
                if not parity["accepted"]:
                    rows.append(
                        {
                            "batch_size": batch_size,
                            "compiled_local": compile_local,
                            "status": "PARITY_FAIL",
                            "parity": parity,
                        }
                    )
                    model.set_local_compile(False)
                    continue

                opt = torch.optim.AdamW(
                    model.parameters(), lr=1e-3, weight_decay=1e-4
                )
                stats = train_epoch_fast(
                    model,
                    split["train"],
                    opt,
                    device,
                    batch_size=batch_size,
                    chunk_size=chunk_size,
                    seed=seed,
                    epoch=1,
                    backward_group_chunks=backward_group_chunks,
                    max_batches=benchmark_batches,
                )
                row = {**stats, "status": "OK", "parity": parity}
                rows.append(row)
                print("SPEED_CELL", json.dumps(row), flush=True)
            except Exception as exc:
                model.set_local_compile(False)
                row = {
                    "batch_size": batch_size,
                    "compiled_local": compile_local,
                    "status": "ERROR",
                    "error": repr(exc),
                }
                rows.append(row)
                print("SPEED_CELL_ERROR", json.dumps(row), flush=True)

    return rows


def train_full(args, data, split, device):
    seed_all(args.seed)
    model = build_model(data["n_items"], args.memory_size).to(device)
    if args.compile_local:
        try:
            model.enable_local_compile()
            parity = local_parity_test(
                model,
                device,
                batch=min(8, args.batch_size),
                chunk=args.chunk_size,
            )
            print("LOCAL_COMPILE_PARITY", json.dumps(parity), flush=True)
            if not parity["accepted"]:
                print("LOCAL_COMPILE_DISABLED parity gate failed", flush=True)
                model.set_local_compile(False)
        except Exception as exc:
            print("LOCAL_COMPILE_DISABLED", repr(exc), flush=True)
            model.set_local_compile(False)

    # Existing PR17 correctness gates run on the eager reference semantics.
    compile_state = model._use_compiled_local
    model.set_local_compile(False)
    chunk_invariance_test(model, device)
    causal_test(model, device)
    model.set_local_compile(compile_state)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out = Path(args.output_dir) / "ml1m" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "StreamingSparseWalker-v1-speed",
        "semantics": "identical to PR17 streaming-native model",
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "memory_size": args.memory_size,
        "backward_group_chunks": args.backward_group_chunks,
        "compile_local_requested": bool(args.compile_local),
        "compile_local_active": bool(model._use_compiled_local),
        "trainer_fixes": [
            "single pinned H2D batch transfer",
            "no inner-loop CUDA->CPU sync",
            "GPU epoch loss accumulation",
            "grouped TBPTT backward calls",
            "fixed-size padded training chunks",
        ],
    }
    (out / "speed_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True)
    )

    best = -1.0
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        lr = set_lr(opt, epoch, args.epochs, peak=args.lr)
        tr = train_epoch_fast(
            model,
            split["train"],
            opt,
            device,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            seed=args.seed,
            epoch=epoch,
            use_bf16=True,
            backward_group_chunks=args.backward_group_chunks,
        )
        tr["epoch"] = epoch
        tr["lr"] = lr
        print("TRAIN_FAST", json.dumps(tr), flush=True)

        if epoch == 1 or epoch % args.eval_every == 0:
            # Evaluation uses eager local chunks to avoid variable-length
            # recompiles; weights and model semantics are unchanged.
            comp = model._use_compiled_local
            model.set_local_compile(False)
            val_full = evaluate_streaming(
                model,
                split["val_prefix"],
                split["val_target"],
                data["n_items"],
                device,
                batch_size=args.eval_batch_size,
                chunk_size=args.chunk_size,
                history_limit=None,
            )
            val_reset200 = evaluate_streaming(
                model,
                split["val_prefix"],
                split["val_target"],
                data["n_items"],
                device,
                batch_size=args.eval_batch_size,
                chunk_size=args.chunk_size,
                history_limit=200,
            )
            model.set_local_compile(comp)

            row = {
                **tr,
                "val_stream_full_NDCG@10": float(val_full["NDCG@10"]),
                "val_stream_full_HR@10": float(val_full["HR@10"]),
                "val_stream_full_MRR@10": float(val_full["MRR@10"]),
                "val_reset200_NDCG@10": float(val_reset200["NDCG@10"]),
                "persistent_history_gain": float(
                    val_full["NDCG@10"] - val_reset200["NDCG@10"]
                ),
            }
            history.append(row)
            print("EVAL_FAST", json.dumps(row), flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=2))

            if val_full["NDCG@10"] > best:
                best = float(val_full["NDCG@10"])
                best_epoch = epoch
                best_state = cpu_state_dict(model)
                torch.save(
                    {
                        "model": best_state,
                        "epoch": epoch,
                        "val_stream_full": val_full,
                        "val_reset200": val_reset200,
                        "config": config,
                    },
                    out / "best.pt",
                )

            torch.save(
                {
                    "model": cpu_state_dict(model),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_ndcg": best,
                    "config": config,
                    "history": history,
                },
                out / "last.pt",
            )

    if best_state is None:
        raise RuntimeError("no checkpoint selected")

    model.load_state_dict(best_state)
    model.set_local_compile(False)
    model.eval()
    test_full = evaluate_streaming(
        model,
        split["test_prefix"],
        split["test_target"],
        data["n_items"],
        device,
        batch_size=args.eval_batch_size,
        chunk_size=args.chunk_size,
    )
    test_reset200 = evaluate_streaming(
        model,
        split["test_prefix"],
        split["test_target"],
        data["n_items"],
        device,
        batch_size=args.eval_batch_size,
        chunk_size=args.chunk_size,
        history_limit=200,
    )
    result = {
        "selected_epoch": best_epoch,
        "best_val_NDCG@10": best,
        "test_stream_full": test_full,
        "test_reset200": test_reset200,
        "config": config,
    }
    (out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    print("FINAL_FAST", json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="/content/sparsewalker_data")
    p.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/sparsewalker_streaming_speed",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--memory-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backward-group-chunks", type=int, default=4)
    p.add_argument(
        "--compile-local", action=argparse.BooleanOptionalAction, default=False
    )

    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--benchmark-batches", type=int, default=24)
    p.add_argument(
        "--benchmark-batch-sizes",
        default="32,64,128",
        help="comma-separated batch sizes",
    )
    p.add_argument(
        "--benchmark-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = p.parse_args()

    assert torch.cuda.is_available(), "GPU runtime required for this speed audit"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed_all(args.seed)

    print(
        "DEVICE",
        torch.cuda.get_device_name(0),
        "torch",
        torch.__version__,
        "bf16",
        torch.cuda.is_bf16_supported(),
        flush=True,
    )

    data = load_dataset("ml1m", args.data_dir)
    split = split_data(data["sequences"])

    grad_gate = grouped_backward_parity_test(
        data["n_items"], device, memory_size=args.memory_size
    )
    print("GROUPED_BACKWARD_PARITY", json.dumps(grad_gate), flush=True)
    if not grad_gate["accepted"]:
        raise AssertionError(f"grouped backward parity failed: {grad_gate}")

    if args.benchmark_only:
        batch_sizes = [
            int(x.strip())
            for x in args.benchmark_batch_sizes.split(",")
            if x.strip()
        ]
        rows = benchmark_speed(
            data,
            split,
            device,
            seed=args.seed,
            chunk_size=args.chunk_size,
            memory_size=args.memory_size,
            batch_sizes=batch_sizes,
            benchmark_batches=args.benchmark_batches,
            backward_group_chunks=args.backward_group_chunks,
            try_compile=args.benchmark_compile,
        )
        out = (
            Path(args.output_dir)
            / "ml1m"
            / f"seed{args.seed}"
            / "speed_sweep.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
        print("SPEED_SWEEP", json.dumps(rows, indent=2), flush=True)
        print("SAVED", out, flush=True)
        return

    train_full(args, data, split, device)


if __name__ == "__main__":
    main()
