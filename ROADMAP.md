# Sparse Walker Research Roadmap

Status legend: `✅ done` · `🟡 active` · `⬜ queued` · `🧪 exploratory`

## Critical path: trustworthy baselines first

| Status | Workstream | Action | Success criterion |
|---|---|---|---|
| ✅ | HSTU core quality | Reproduce HSTU-core on ML-1M | Pure-PyTorch best NDCG@10 0.16975 vs published ~0.172; wider metrics near parity |
| 🟡 | HSTU large quality | Finish HSTU-large ML-1M reproduction | Reach ~0.188–0.190 NDCG@10, or stable best close to published ~0.1893 |
| ⬜ | HSTU canonical stamp | Freeze `HSTU_CANONICAL_v1` | Core + large quality reproduction documented and code frozen |
| ⬜ | HSTU cached inference | Add exact incremental/cached inference | Cached output numerically matches full-history output |
| ⬜ | HSTU fused inference | Add fused/Triton incremental kernel | Fused output matches cached reference; latency curve recorded |
| ⬜ | SASRec sampled softmax | Fix/reproduce SASRec+SS | Reproduce expected SS behavior on a published protocol |
| ⬜ | eSASRec / LiGR | Implement and reproduce canonical eSASRec | Literature-aligned lift over matched SASRec+SS |
| ⬜ | Mamba quality | Add no-custom-compile reference Mamba | Portable implementation with credible quality |
| ⬜ | Mamba inference | Add fast recurrent/fused serving path | Constant-state incremental output matches reference |
| ⬜ | Baseline stamp | Freeze `BASELINES_CANONICAL_v1` | SASRec FullCE/SS, eSASRec, HSTU, Mamba independently validated |

## Shared benchmark

| Status | Action | Success criterion |
|---|---|---|
| ⬜ | Port stamped baselines unchanged into shared harness | Integration parity tests pass |
| ⬜ | Rerun Amazon Beauty | Trustworthy ordering for SASRec, SS, eSASRec, HSTU core/large, Mamba, Sparse Walker |
| ⬜ | Expand datasets | Amazon categories + ML-1M + ML-20M; 3 seeds; full-catalog primary evaluation |
| ⬜ | Separate systems metrics | Training cost, request-from-history latency, streaming latency, memory |
| ⬜ | Fused-vs-fused serving Pareto | HSTU/Mamba/SASRec serving baselines compared fairly with fused Sparse Walker |

## Sparse Walker core

| Status | Action | Goal |
|---|---|---|
| 🟡 | Dense Sparse Walker baseline | Preserve/extend current strong quality results |
| 🟡 | Compiled sparse terminal | Improve quality retention with bounded concept→item support |
| ⬜ | E2E sparse terminal | Same sparse terminal support participates in training and serving |
| ⬜ | Dynamic-support ablation | Isolate contribution of changing active latent support |
| ⬜ | History/popularity analysis | Understand where Walker gains and loses |
| ⬜ | Concept specialization | Measure semantic/behavioral structure of learned concepts and paths |

## Dynamic-support SSM research

Working formalization:

`S_t = (I_t, m_t)`, where `I_t ⊂ {1,…,C}` and `|I_t| = K ≪ C`.

Sparse Walker differs from ordinary fixed-coordinate SSMs by learning **which latent state variables are alive** at each timestep.

Key ablations:

1. dense state;
2. fixed sparse support;
3. sparse transition + fixed support;
4. dynamic support without graph;
5. dynamic support + graph;
6. dynamic support + graph + pursuit.

## S6 / selective-scan direction

🧪 Rewrite the unpruned update as approximately:

`s_t = A_t s_{t-1} + b_t`

Then explore:

- block sizes 2/4/8/16/32;
- larger temporary support inside blocks;
- pruning only at block boundaries;
- bounded sparse operator composition;
- train-K > serve-K;
- distillation/compilation back to K=8.

Goal: **parallel training + fixed-state recurrent serving**.

## Structural reparameterization

🧪 Train a richer Walker than the serving graph, then collapse/compile it:

- larger K during training → K=8 serving;
- larger degree → degree=4;
- multiple graph branches → one graph;
- two hops → compiled one-hop transition when quality permits;
- dense/rich readout → sparse terminal support.

Distinguish:

- **exact structural reparameterization** before nonlinear pruning;
- **compiled structural reparameterization** after approximate sparse compression.

## Backward-pass-free research

Progression:

1. full BPTT baseline;
2. detach recurrent state each event;
3. local/current-step gradients only;
4. local credit assignment;
5. reward-modulated pursuit / ACO-like structural updates;
6. gradient-free topology;
7. fully backward-pass-free Sparse Walker.

Long-term question: can sparse local credit assignment replace global backward-through-history without destroying recommendation quality?

## Publication ladder

1. **RecSys/systems version** — recommender quality + bounded-state serving Pareto.
2. **Broader ML version** — dynamic-support SSM formalization + parallel sparse scan.
3. **Follow-up** — local or backward-pass-free learning.

## Discipline

- Keep baseline reproduction notebooks isolated until stamped.
- Do not silently modify `CANONICAL_v1` implementations.
- Record dataset, protocol, seed, commit SHA, checkpoint, hardware, and metric definition for every stamped result.
- Compare fused serving implementations against fused serving implementations.
- Do not mix sampled-evaluation numbers with full-catalog numbers in the same quality table without explicit labeling.
