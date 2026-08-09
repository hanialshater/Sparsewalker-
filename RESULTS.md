# Stamped Results

This file contains **stamped results only**. Provisional reproduction numbers belong in the corresponding experiment status file until the implementation is frozen.

## Current status

No baseline suite has been globally stamped yet.

The first candidate for stamping is `HSTU_CANONICAL_v1`. See `experiments/hstu_reproduction/STATUS.md`.

## Result-record template

Every stamped result should include:

- model/version;
- dataset and source;
- preprocessing and split;
- evaluation protocol;
- seed(s);
- parameter count;
- training objective;
- checkpoint identifier;
- repository commit SHA;
- hardware/runtime;
- HR/NDCG/MRR/coverage as applicable;
- training time;
- request-from-history latency;
- streaming latency;
- state/cache memory;
- notes on comparability to literature.
