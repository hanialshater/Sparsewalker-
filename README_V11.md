# SparseWalker v1.1 recurrence control

This branch tests three recurrence corrections against the canonical ML-1M setup:

1. inject fresh item mass once per event, independent of graph hop count;
2. coalesce duplicate concept IDs by summing their masses before every TopK prune;
3. disable pursuit rewiring for the control run.

Run `notebooks/benchmarks/18_ml1m_walker_v11_fixes.ipynb` on Colab. The experiment writes results under `MyDrive/sparsewalker_v11/ml1m/seed42`.
