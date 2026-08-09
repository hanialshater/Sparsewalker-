# Sparse Walker

Research repository for **Sparse Walker**, a dynamically sparse recurrent sequence model with persistent latent state, sparse graph transitions, and bounded streaming state.

## Current research questions

- Can dynamic-support latent state match or beat strong sequential recommenders?
- Can Sparse Walker preserve quality while replacing full-catalog scoring with sparse terminal retrieval?
- Can the recurrent update be parallelized with S6/Mamba-style selective-scan ideas?
- Can richer training structure be structurally reparameterized into a tiny serving graph?
- How much of learning can be made local, no-BPTT, or eventually backward-pass-free?

## Repository discipline

Canonical baselines are reproduced and stamped **before** being integrated into the shared benchmark. Once an implementation is stamped `CANONICAL_v1`, it is not silently modified; changes create a new version.

The first baseline being stamped is HSTU, kept isolated under `experiments/hstu_reproduction/` until both quality and inference parity are validated.

See [ROADMAP.md](ROADMAP.md) for the active research agenda and [RESULTS.md](RESULTS.md) for stamped results only.
