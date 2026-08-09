# Sparse Walker Research Roadmap

Status legend: `✅ done` · `🟡 active` · `⬜ queued` · `🧪 exploratory`

## Repository / reproducibility

| Status | Action |
|---|---|
| ✅ | Refactor implementations into importable modules with notebooks as orchestration |
| ✅ | Recover the all-algorithms model code into the shared package/model registry |
| ✅ | Record current validated/provisional quality + latency evidence in README |
| 🟡 | Finish canonical integration tests and freeze first `BASELINES_CANONICAL_v1` |

## Baselines

| Status | Workstream | Current evidence / next action |
|---|---|---|
| ✅ | SASRec FullCE | Beauty NDCG@10 0.04675; cached SDPA serving benchmarked |
| ✅ | HSTU core quality | ML-1M NDCG@10 0.16975 vs published ~0.1720 |
| ✅ | HSTU large quality | ML-1M NDCG@10 0.18827 vs published ~0.1893 |
| ✅ | HSTU cached inference | full↔cached parity validated |
| ✅ | HSTU Triton inference | cached Torch↔Triton parity validated; 1k–10k latency curve recorded |
| 🟡 | HSTU canonical stamp | regenerate durable mature checkpoints, rerun finalizer, freeze |
| 🟡 | SASRec sampled softmax | per-position catalog-uniform SS256 fixed; Beauty 2×2 reproduction running |
| 🟡 | eSASRec / LiGR | LiGR reference gates/SwiGLU aligned; run SASRec/LiGR × FullCE/SS, then ML-1M literature sanity |
| ⬜ | Mamba quality | portable no-custom-compile reference implementation |
| ⬜ | Mamba recurrent inference | constant-state one-event path + parity |

## Sparse Walker

| Status | Action | Evidence / goal |
|---|---|---|
| ✅ | Dense Walker Beauty baseline | NDCG@10 0.04990, ~+6.7% over strong SASRec seed42 |
| ✅ | Fused state-update latency sanity | ~0.097 ms A100 B1, flat from 1k→10k history labels |
| ✅ | Fused d128 sparse-terminal latency sanity | ~0.146 ms A100 B1; 6.7×→18× SASRec speedup from 1k→10k |
| 🟡 | Compiled sparse terminal quality | rerun clean quality-retention sweep with corrected terminal implementation |
| ⬜ | E2E sparse terminal | sparse support participates in both training and serving |
| ⬜ | Dynamic-support ablation | isolate support selection, graph, persistence, pursuit |
| ⬜ | History/popularity analysis | identify where Walker gains/loses |
| ⬜ | Concept specialization | inspect semantics/behavior of concepts and paths |

## Shared benchmark

1. Port only independently validated implementations into the shared harness.
2. Beauty first; then Video Games, Sports, Toys, ML-1M, ML-20M.
3. Three seeds for main quality table; full-catalog evaluation primary.
4. Separate request-from-history, cached streaming, retrieval, memory, and training cost.
5. Compare fused serving against fused serving.

## Research after baseline freeze

- 🧪 S6/block-parallel Sparse Walker with train-K > serve-K.
- 🧪 Structural reparameterization: larger K/degree/branches during training → tiny serving graph.
- 🧪 Increasingly local/no-BPTT learning and reward-modulated pursuit.
