"""GPU-native temporal small-world search/read primitive.

The temporal graph is persistent serving state. This module optimizes only the
latency-critical query path: navigate a fixed-degree graph in Triton, retain a
small beam of high-scoring historical keys, then perform the exact softmax/value
read over that beam. Graph construction is intentionally outside the kernel.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _swg_search_read_kernel(
        Q, K, V, ADJ, ENTRY, LENGTHS, OUT, OUT_IDS,
        L: tl.constexpr,
        HD: tl.constexpr,
        DEG: tl.constexpr,
        BEAM: tl.constexpr,
        HOPS: tl.constexpr,
        CAND_PAD: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        d = tl.arange(0, HD)
        b = tl.arange(0, BEAM)
        c = tl.arange(0, CAND_PAD)
        neg = -1.0e20
        n = tl.load(LENGTHS + pid).to(tl.int32)
        q = tl.load(Q + pid * HD + d).to(tl.float32)

        # Start from persistent graph entry points. ENTRY is BEAM-wide so serving
        # can mix an HNSW entry point, recent nodes, and coarse temporal anchors.
        ids0 = tl.load(ENTRY + pid * BEAM + b).to(tl.int32)
        valid0 = (ids0 >= 0) & (ids0 < n)
        safe0 = tl.where(valid0, ids0, 0)
        k0 = tl.load(
            K + (pid * L + safe0[:, None]) * HD + d[None, :],
            mask=valid0[:, None], other=0.0,
        ).to(tl.float32)
        score0 = tl.sum(k0 * q[None, :], axis=1)
        score0 = tl.where(valid0, score0, neg)

        # Triton loop-carried values must retain a fixed tensor shape. Keep the
        # BEAM-wide entry-selection scratch distinct from the CAND_PAD-wide
        # per-hop candidate scratch below.
        beam_ids = tl.full((BEAM,), -1, tl.int32)
        work_entry = score0
        for j in range(BEAM):
            ix = tl.argmax(work_entry, axis=0)
            cid = tl.sum(tl.where(b == ix, safe0, 0), axis=0).to(tl.int32)
            beam_ids = tl.where(b == j, cid, beam_ids)
            work_entry = tl.where(safe0 == cid, neg, work_entry)

        # Fixed-budget beam walk. Each hop rescans the current beam plus all of
        # its fixed-degree neighbors, then keeps the unique top BEAM by q·k.
        for _ in range(HOPS):
            self_slot = c < BEAM
            edge_slot = (c >= BEAM) & (c < BEAM + BEAM * DEG)
            src = tl.where(self_slot, c, (c - BEAM) // DEG)
            src = tl.minimum(tl.maximum(src, 0), BEAM - 1)
            src_id = tl.gather(beam_ids, src, axis=0)
            edge = tl.maximum(c - BEAM, 0) % DEG
            nbr = tl.load(
                ADJ + (pid * L + tl.maximum(src_id, 0)) * DEG + edge,
                mask=edge_slot & (src_id >= 0), other=-1,
            ).to(tl.int32)
            cand = tl.where(self_slot, src_id, tl.where(edge_slot, nbr, -1))
            valid = (cand >= 0) & (cand < n)
            safe = tl.where(valid, cand, 0)
            kk = tl.load(
                K + (pid * L + safe[:, None]) * HD + d[None, :],
                mask=valid[:, None], other=0.0,
            ).to(tl.float32)
            score_cand = tl.sum(kk * q[None, :], axis=1)
            score_cand = tl.where(valid, score_cand, neg)

            next_ids = tl.full((BEAM,), -1, tl.int32)
            work_cand = score_cand
            for j in range(BEAM):
                ix = tl.argmax(work_cand, axis=0)
                cid = tl.sum(tl.where(c == ix, safe, 0), axis=0).to(tl.int32)
                next_ids = tl.where(b == j, cid, next_ids)
                # Remove duplicate occurrences of the selected graph node.
                work_cand = tl.where(safe == cid, neg, work_cand)
            beam_ids = next_ids

        validf = (beam_ids >= 0) & (beam_ids < n)
        safef = tl.where(validf, beam_ids, 0)
        kf = tl.load(
            K + (pid * L + safef[:, None]) * HD + d[None, :],
            mask=validf[:, None], other=0.0,
        ).to(tl.float32)
        logits = tl.sum(kf * q[None, :], axis=1) * SCALE
        logits = tl.where(validf, logits, neg)
        mx = tl.max(logits, axis=0)
        w = tl.exp(logits - mx) * validf.to(tl.float32)
        w = w / (tl.sum(w, axis=0) + 1.0e-8)
        vf = tl.load(
            V + (pid * L + safef[:, None]) * HD + d[None, :],
            mask=validf[:, None], other=0.0,
        ).to(tl.float32)
        ctx = tl.sum(vf * w[:, None], axis=0)
        tl.store(OUT + pid * HD + d, ctx)
        tl.store(OUT_IDS + pid * BEAM + b, beam_ids)


def temporal_swg_search_read(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    adjacency: torch.Tensor,
    entry: torch.Tensor,
    lengths: torch.Tensor,
    *,
    hops: int = 4,
    beam: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the fused Triton temporal graph walk + value read.

    Shapes:
      q          [NQ, HD]
      k, v       [NQ, L, HD]
      adjacency  [NQ, L, DEG] int32
      entry      [NQ, BEAM] int32
      lengths    [NQ] int32/int64
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is unavailable")
    if not (q.is_cuda and k.is_cuda and v.is_cuda and adjacency.is_cuda and entry.is_cuda):
        raise ValueError("all search tensors must be CUDA tensors")
    if q.ndim != 2 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("expected q=[NQ,HD], k=v=[NQ,L,HD]")
    nq, hd = q.shape
    if k.shape[0] != nq or k.shape[2] != hd:
        raise ValueError("q/k shape mismatch")
    L = int(k.shape[1])
    if adjacency.ndim != 3 or adjacency.shape[:2] != (nq, L):
        raise ValueError("adjacency must be [NQ,L,DEG]")
    deg = int(adjacency.shape[2])
    if entry.shape != (nq, beam):
        raise ValueError(f"entry must be [{nq},{beam}]")
    if hd not in (16, 32, 64):
        raise ValueError("current kernel supports HD in {16,32,64}")
    if beam not in (8, 16, 32):
        raise ValueError("current kernel supports beam in {8,16,32}")
    if deg not in (4, 8, 16, 32):
        raise ValueError("current kernel supports degree in {4,8,16,32}")
    cand = beam + beam * deg
    cand_pad = 1 << (cand - 1).bit_length()
    if cand_pad > 1024:
        raise ValueError("candidate frontier too large for this first kernel")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    adjacency = adjacency.to(dtype=torch.int32).contiguous()
    entry = entry.to(dtype=torch.int32).contiguous()
    lengths = lengths.to(device=q.device, dtype=torch.int32).contiguous()
    out = torch.empty((nq, hd), device=q.device, dtype=torch.float32)
    out_ids = torch.empty((nq, beam), device=q.device, dtype=torch.int32)
    _swg_search_read_kernel[(nq,)](
        q, k, v, adjacency, entry, lengths, out, out_ids,
        L=L, HD=hd, DEG=deg, BEAM=beam, HOPS=int(hops), CAND_PAD=cand_pad,
        SCALE=1.0 / math.sqrt(hd), num_warps=4,
    )
    return out, out_ids


def reference_temporal_swg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    adjacency: torch.Tensor,
    entry: torch.Tensor,
    lengths: torch.Tensor,
    *,
    hops: int = 4,
    beam: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slow deterministic reference for correctness tests."""
    device = q.device
    nq, hd = q.shape
    out = []
    ids_out = []
    scale = 1.0 / math.sqrt(hd)
    for r in range(nq):
        n = int(lengths[r].item())
        cur = []
        for x in entry[r].tolist():
            x = int(x)
            if 0 <= x < n and x not in cur:
                cur.append(x)
        if not cur:
            cur = [max(0, n - 1)]
        scores = (k[r, cur].float() @ q[r].float())
        order = torch.argsort(scores, descending=True).tolist()
        cur = [cur[i] for i in order[:beam]]
        while len(cur) < beam:
            cur.append(cur[-1])
        for _ in range(int(hops)):
            cand = []
            for x in cur:
                if x not in cand:
                    cand.append(x)
                for y in adjacency[r, x].tolist():
                    y = int(y)
                    if 0 <= y < n and y not in cand:
                        cand.append(y)
            score = k[r, cand].float() @ q[r].float()
            order = torch.argsort(score, descending=True).tolist()
            cur = [cand[i] for i in order[:beam]]
            while len(cur) < beam:
                cur.append(cur[-1])
        ids = torch.tensor(cur[:beam], device=device, dtype=torch.long)
        logits = (k[r, ids].float() @ q[r].float()) * scale
        w = torch.softmax(logits, dim=0)
        out.append((w[:, None] * v[r, ids].float()).sum(0))
        ids_out.append(ids)
    return torch.stack(out), torch.stack(ids_out).to(torch.int32)
