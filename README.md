# Sparse Walker

**Sparse Walker** is a dynamically sparse recurrent sequence model for recommendation. A user occupies only a small number of active latent concepts (`K=8` in the current experiments) inside a much larger concept space, and state evolves through a learned sparse graph.

The repository is organized around one rule:

> **implementations live in Python modules; notebooks reproduce or orchestrate experiments.**

## Current evidence

These numbers are **validated/provisional**, not yet the final multi-seed paper table. `RESULTS.md` is reserved for stamped canonical results.

### Quality

| Dataset / protocol | Model | NDCG@10 | Notes |
|---|---|---:|---|
| Amazon Beauty, full-catalog temporal LOO | SASRec | **0.04675** | strong reconciled seed 42 |
| Amazon Beauty, full-catalog temporal LOO | Dense Sparse Walker | **0.04990** | ~+6.7% vs SASRec, seed 42 |
| ML-1M, public HSTU protocol | HSTU-core | **0.16975** | published target ~0.1720 |
| ML-1M, public HSTU protocol | HSTU-large | **0.18827** | published target ~0.1893 |

### A100 cached streaming latency, batch=1

The sparse-terminal row includes state update + bounded concept→item retrieval (`degree=128`, 1024 reachable slots before unique top-k).

| History | SASRec | HSTU-core | Sparse Walker state | Walker + terminal | Walker terminal vs SASRec |
|---:|---:|---:|---:|---:|---:|
| 1k | 0.985 ms | 1.165 ms | **0.097 ms** | **0.146 ms** | **6.7×** |
| 2k | 0.967 ms | 1.144 ms | **0.097 ms** | **0.146 ms** | **6.6×** |
| 5k | 1.521 ms | 1.555 ms | **0.097 ms** | **0.146 ms** | **10.4×** |
| 10k | 2.640 ms | 2.471 ms | **0.097 ms** | **0.146 ms** | **18.0×** |

Logical per-user Walker state is `8 × (int32 concept id + fp32 mass) = 64 bytes`; global concept graph and terminal tables are shared model state and are reported separately.

## Repository layout

```text
src/sparsewalker/
  models/        model implementations
  data/          dataset loaders and temporal splits
  evaluation/    full-catalog and sampled metrics
  training/      shared objectives/utilities
  serving/       cached/fused serving implementations
benchmarks/      command-line benchmark entry points
notebooks/
  reproductions/ one notebook per important algorithm
  benchmarks/    cross-model quality/systems notebooks
experiments/     active non-canonical research
```

## Model status

| Model | Quality reproduction | Serving | Status |
|---|---|---|---|
| SASRec | strong Beauty baseline | cached SDPA | usable |
| HSTU-core / large | ML-1M reproduced | cached + Triton parity validated | canonical stamp pending durable checkpoints |
| eSASRec / LiGR | implementation recovered | — | reproduction queued |
| Mamba4Rec | legacy wrapper recovered | — | portable no-compile baseline queued |
| Sparse Walker | strong Beauty result | fused Triton state + sparse terminal | active |

See `ROADMAP.md` for priorities and `RESULTS.md` for the strict stamped ledger.
