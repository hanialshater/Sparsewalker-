# HSTU Reproduction Status

HSTU remains isolated here until it is stamped canonical.

## Published ML-1M targets

| Model | HR@10 | NDCG@10 |
|---|---:|---:|
| SASRec | 0.2853 | 0.1603 |
| HSTU | 0.3097 | 0.1720 |
| HSTU-large | 0.3294 | 0.1893 |

## Pure-PyTorch reproduction

### HSTU-core

Configuration:

- ML-1M original movie IDs
- max history 200
- HSTU tensor width 211
- d=50
- 2 blocks
- 1 head
- dqk=dv=50
- relative bucketed time + position bias
- sampled softmax, 128 local negatives
- temperature 0.05
- L2-normalized user/item embeddings
- AdamW β=(0.9,0.98), LR=1e-3, WD=0
- 101 epochs
- full-corpus evaluation with seen-item filtering

Best observed:

- best epoch: 95
- NDCG@10: **0.1697457** vs published 0.1720
- HR@10 at epoch 95: **0.3044702** vs published 0.3097
- epoch-101 NDCG@50: **0.2307305** vs published 0.2307
- epoch-101 HR@50: **0.5779801** vs published 0.5754

Status: **reproduction accepted provisionally; waiting to stamp together with large + inference parity.**

### HSTU-large

Configuration:

- same ML-1M protocol
- d=50
- 8 blocks
- 2 heads
- dqk=dv=25

Current trajectory:

- epoch 20 NDCG@10: 0.17307
- epoch 25 NDCG@10: 0.17862
- epoch 30 NDCG@10: 0.18024
- published target: 0.1893

Status: **active**.

## Pure-PyTorch optimization

Reference implementation training time was ~110 s/epoch for HSTU-core because sampled-softmax chunks repeatedly backpropagated through the same encoder graph.

Optimized implementation:

- one backward per user batch;
- vectorized SS128 scoring;
- time-bucket IDs computed once per forward and shared structurally across layers;
- batched matmul for QK and AV;
- FP32 + TF32 retained for parity.

HSTU-core training time dropped to ~28.2 s/epoch with essentially unchanged quality trajectory (~3.9× implementation speedup).

HSTU-large currently costs ~111 s/epoch, consistent with 4× the HSTU depth.

## Remaining stamp criteria

- [ ] Finish HSTU-large reproduction close to 0.1893 NDCG@10.
- [ ] Freeze quality code as `HSTU_CANONICAL_v1`.
- [ ] Implement incremental cached inference.
- [ ] Verify cached output against full-history output.
- [ ] Implement fused/Triton inference path.
- [ ] Verify fused output against cached reference.
- [ ] Record latency and cache-memory scaling across history lengths.

Only after all above should HSTU be copied into the shared Sparse Walker benchmark.
