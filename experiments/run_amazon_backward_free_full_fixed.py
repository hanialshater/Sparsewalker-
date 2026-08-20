#!/usr/bin/env python
"""Corrected entry point for full backward-free SparseWalker.

This keeps experiment 35's implementation intact while fixing one prototype
rewiring detail: concept ID 0 is a valid concept ID and must not be filtered.
"""
import torch

import run_amazon_backward_free_full as base


@torch.no_grad()
def _hebbian_rewire_fixed(model, source_ids, source_mass, teacher_ids, teacher_weight, epoch, rate):
    if source_ids.numel() == 0 or rate <= 0:
        return 0
    B, K = source_ids.shape
    T = teacher_ids.size(1)
    src = source_ids[:, :, None].expand(B, K, T).reshape(-1)
    tgt = teacher_ids[:, None, :].expand(B, K, T).reshape(-1)
    score = (source_mass[:, :, None] * teacher_weight[:, None, :]).reshape(-1).float()

    # Concept 0 is valid; only zero-mass source/target pairs are invalid here.
    valid = score > 0
    if not valid.any():
        return 0
    src = src[valid]
    tgt = tgt[valid]
    score = score[valid]

    pair = src * model.n_concepts + tgt
    up, inv = torch.unique(pair, return_inverse=True)
    agg = torch.zeros(up.numel(), device=score.device, dtype=torch.float32)
    agg.scatter_add_(0, inv, score)
    usrc = up // model.n_concepts
    utgt = up % model.n_concepts

    best_score = torch.full(
        (model.n_concepts,), -float("inf"), device=score.device, dtype=torch.float32
    )
    best_score.scatter_reduce_(0, usrc, agg, reduce="amax", include_self=True)
    is_best = agg >= best_score[usrc]
    sentinel = int(model.n_concepts)
    best_target = torch.full(
        (model.n_concepts,), sentinel, device=score.device, dtype=torch.long
    )
    best_target.scatter_reduce_(
        0, usrc[is_best], utgt[is_best], reduce="amin", include_self=True
    )
    usrc = (best_target < sentinel).nonzero(as_tuple=False).squeeze(-1)
    if usrc.numel() == 0:
        return 0
    utgt = best_target[usrc]

    if rate < 1.0:
        gate = ((usrc * 1103515245 + int(epoch) * 12345) % 10000).float() < float(rate) * 10000
        usrc = usrc[gate]
        utgt = utgt[gate]
    if usrc.numel() == 0:
        return 0

    if model.degree <= 1:
        slot = torch.zeros_like(usrc)
    else:
        slot = 1 + ((usrc + int(epoch)) % (model.degree - 1))
    model.graph.destination[usrc, slot] = utgt.to(torch.int32)
    return int(usrc.numel())


base._hebbian_rewire = _hebbian_rewire_fixed

if __name__ == "__main__":
    base.main()
