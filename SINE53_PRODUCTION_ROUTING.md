# SINE53 frozen batch routing

Current production routing is frozen as:

- `n < 1600`: current SINE53 evaluator
- `n >= 1600`: frozen custom permanent 2-core scheduler

The final threshold is backed by exact Intel Xeon 6973P-C three-way benchmarks.

Focused 1600-1900 evidence:

- run `33568489785`
- exact-Xeon shard `61`
- at every tested size from `n=1600` through `n=1900`, custom2 beat current SINE53 in all six requested sign/range cells
- custom2 was bit-identical to current SINE53 on the tested grid
- at `n=1600`, the all-six average advantage over current was about 2.31%, increasing through the tested range

Earlier boundary evidence:

- run `33567930913`
- exact-Xeon shard `50`
- at `n=1500`, current SINE53 remained faster on the six-case average
- at every tested size from `n=2000` through `n=4500`, custom2 beat current SINE53 in all six requested sign/range cells
- custom2 was bit-identical to current SINE53 on the boundary grid

Broad large-batch evidence:

- run `33567193984`
- exact-Xeon shards `37` and `25`
- at every tested size from `n=5000` through `n=4000000`, custom2 beat current SINE53 in all six requested sign/range cells on both shards
- custom2 was bit-identical to current SINE53 over the tested cells

Frozen production files:

- `sine53_batch_production.hpp`
- `sine53_custom_2core_1600_frozen.hpp`

Historical `sine53_custom_2core_2000_frozen.hpp` and `sine53_custom_2core_5000_frozen.hpp` are retained as previous freeze records. The experimental `sine53_custom_2core.hpp` remains available for research but is not the frozen production scheduler.

Do not change the 1,600 threshold or frozen scheduler without a new benchmark and explicit production promotion.
