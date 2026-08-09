# Tests

Tests should protect canonical equivalence, not just code execution.

Priority parity tests:

- full-history vs cached HSTU inference;
- cached vs fused HSTU inference;
- full-sequence vs recurrent Mamba inference;
- reference vs fused Sparse Walker step;
- shared-harness baseline outputs vs canonical implementations;
- terminal candidate/scoring parity;
- deterministic evaluation/tie handling.

Canonical implementations should not be promoted without parity coverage for the path used in reported experiments.
