# SINE53 frozen batch routing

Current production routing is frozen as:

- `n < 4500`: current SINE53 evaluator; the custom2 helper is not constructed
- `n >= 4500`: lazily construct/use the frozen custom permanent 2-core scheduler

Once constructed, the helper remains alive for later large batches.

## Current production evidence

The 4,500 resource-elastic boundary is backed by the full native + Intel SDE benchmark:

- run `33693912241`
- exact Intel Xeon 6973P-C shards `17`, `44`, `67`, and `73`
- tested sizes `100, 700, 3500, 4500, 5000, 8000, 15000, 20000, 25000, 30000, 50000, 1000000, 2000000`
- comparison stacks: temporary lazy-4500 candidate, frozen 1600 production control, and Intel oneMKL `vmdSin(..., VML_HA)` sequential
- metrics: wall ns/el, process CPU ns/el, effective cores, instruction count/el, logical memory bytes/el, helper state, and ULP difference versus Intel

The experiment established that deferring helper construction below 4,500 substantially improves small-load CPU-resource proportionality. In particular, the temporary lazy path kept process CPU-time efficiency versus Intel above 1x throughout the measured no-helper region. This is intentionally preferred over the old speed-only 1,600 crossover policy even though part of the 1,600-4,499 interval gives up wall speed versus custom2.

At `n=4500` and above, the exact same frozen custom2 scheduler is used and the strong wall-speed advantage returns.

The maximum observed comparator difference in the promotion benchmark was `2 ULP` versus Intel oneMKL `VML_HA`. This is a difference versus Intel output, not a claim of a <=2 ULP absolute error bound relative to correctly rounded mathematical sine.

## Historical speed-only evidence

The previous 1,600 threshold remains useful historical evidence for the wall-time crossover:

- run `33568489785`, exact-Xeon shard `61`: custom2 beat current SINE53 in all six requested sign/range cells from `n=1600` through `n=1900`
- run `33567930913`, exact-Xeon shard `50`: current remained faster at `n=1500`; custom2 beat current from `n=2000` through `n=4500`
- run `33567193984`, exact-Xeon shards `37` and `25`: custom2 beat current throughout the tested `n=5000` through `n=4000000` range

That evidence has not been invalidated. The production objective changed from minimum wall time at every small batch to a balance of wall performance and CPU-resource proportionality.

## Frozen files

- `sine53_batch_production.hpp`
- `sine53_custom_2core_1600_frozen.hpp`

The scheduler filename retains `1600` because it is the already-validated frozen custom2 implementation; its internal scheduling code was not changed during the 4,500 lifecycle promotion. Only when that scheduler is constructed/activated changed.

Historical `sine53_custom_2core_2000_frozen.hpp` and `sine53_custom_2core_5000_frozen.hpp` remain previous freeze records. The experimental `sine53_custom_2core.hpp` remains research-only.

Do not change the `<4500` lazy region, the 4,500 activation boundary, or the frozen scheduler without a new benchmark and explicit production promotion.
