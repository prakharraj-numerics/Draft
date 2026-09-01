# SINE53 frozen batch routing

Current production routing is frozen as:

- `n < 5000`: current SINE53 evaluator
- `n >= 5000`: frozen custom permanent 2-core scheduler

The routing is backed by the exact Intel Xeon 6973P-C three-way benchmark:

- run `33567193984`
- exact-Xeon shards `37` and `25`

Across both exact-Xeon artifacts, custom2 was bit-identical to the current SINE53 evaluator over all 72 requested cells. At every tested point from 5,000 through 4,000,000, custom2 beat current SINE53 in all six requested sign/range cases on both shards. At 1,200, current SINE53 remained faster than custom2, so no lower crossover is inferred from this benchmark.

Frozen production files:

- `sine53_batch_production.hpp`
- `sine53_custom_2core_5000_frozen.hpp`

The experimental `sine53_custom_2core.hpp` remains available for research but is not the frozen production scheduler.

Do not change the 5,000 threshold or frozen scheduler without a new benchmark and explicit production promotion.
