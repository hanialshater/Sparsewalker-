## SparseWalker v1.1 diagnostic fixes

- fresh item routed mass is merged once per timestep, then propagated through the configured graph hops
- duplicate concept IDs are coalesced before TopK in both merge pruning and graph-transition pruning
- the canonical v1.1 ML-1M control disables pursuit rewiring
- added fail-fast tests and an in-process Colab launcher
