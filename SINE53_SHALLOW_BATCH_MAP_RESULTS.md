# SINE53 shallow batch map vs Intel oneMKL VML_HA

## Run

- GitHub Actions run: `33566118184`
- Exact Xeon shard: `57`
- Artifact: `sine53-shallow-batch-map-57`
- Commit: `5bb4fd149f8fb9e4260b9e118cb805d5aebe8aa8`
- CPU: Intel(R) Xeon(R) 6973P-C
- Compiler: Intel oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)
- Intel comparator: sequential oneMKL `vmdSin(..., VML_HA)`
- CPU affinity: logical CPU 0
- MKL/OMP threads: 1
- Production split used: X50 for `0<|x|<1`; X67 for `1<|x|<500` and `1000<|x|<10000`

## Shallow input design

At each batch size, exactly six deterministic random batch realizations were timed:

1. `unit_pos`: `0<x<1`
2. `unit_neg`: `-1<x<0`
3. `mid_pos`: `1<x<500`
4. `mid_neg`: `-500<x<-1`
5. `far_pos`: `1000<x<10000`
6. `far_neg`: `-10000<x<-1000`

This is intentionally a shallow performance map. Timing repetitions reuse the same six arrays.

`speedup = Intel ns/value / SINE53 ns/value`, so values above 1 mean SINE53 is faster.

## Results

| Batch n | `|x|<1` SINE53 / Intel ns | Speedup | `1<|x|<500` SINE53 / Intel ns | Speedup | `1000<|x|<10000` SINE53 / Intel ns | Speedup | All-six speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1.060305 / 1.152084 | 1.0866x | 1.061292 / 1.085917 | 1.0232x | 1.061061 / 1.080952 | 1.0187x | 1.0428x |
| 250 | 0.876305 / 0.782981 | 0.8935x | 0.611692 / 0.782487 | 1.2792x | 0.611978 / 0.782887 | 1.2793x | 1.1183x |
| 1,200 | 0.528250 / 0.725811 | 1.3740x | 0.472268 / 0.725527 | 1.5363x | 0.468343 / 0.720899 | 1.5393x | 1.4789x |
| 5,000 | 0.526005 / 0.717527 | 1.3641x | 0.458461 / 0.715214 | 1.5600x | 0.465972 / 0.715369 | 1.5352x | 1.4810x |
| 10,000 | 0.523512 / 0.717507 | 1.3706x | 0.464321 / 0.717707 | 1.5457x | 0.460185 / 0.715159 | 1.5541x | 1.4850x |
| 30,000 | 0.523317 / 0.711744 | 1.3601x | 0.460209 / 0.715436 | 1.5546x | 0.459106 / 0.711799 | 1.5504x | 1.4827x |
| 50,000 | 0.524688 / 0.713633 | 1.3601x | 0.458330 / 0.715624 | 1.5614x | 0.457662 / 0.714714 | 1.5617x | 1.4882x |
| 100,000 | 0.523816 / 0.719881 | 1.3743x | 0.536515 / 0.810989 | 1.5116x | 0.668948 / 0.893247 | 1.3353x | 1.4018x |
| 500,000 | 0.545083 / 0.728228 | 1.3360x | 0.541583 / 0.727897 | 1.3440x | 0.543335 / 0.728183 | 1.3402x | 1.3401x |
| 1,000,000 | 0.551011 / 0.718941 | 1.3048x | 0.540843 / 0.724269 | 1.3391x | 0.541156 / 0.728616 | 1.3464x | 1.3300x |
| 2,000,000 | 0.542254 / 0.726375 | 1.3395x | 0.540753 / 0.723662 | 1.3382x | 0.539471 / 0.725519 | 1.3449x | 1.3409x |
| 4,000,000 | 0.554554 / 0.742930 | 1.3397x | 0.547995 / 0.726390 | 1.3255x | 0.543696 / 0.725713 | 1.3348x | 1.3334x |

## Cell result

- SINE53 wins: **70 / 72** individual sign/range cells.
- Intel wins: **2 / 72**.
- The only Intel wins are `unit_pos` and `unit_neg` at `n=250`.
- SINE53 wins all six cells at the other 11 tested batch sizes.

## Interpretation

The current SINE53 baseline is ahead of Intel oneMKL VML_HA across almost the entire shallow map. There is one narrow unit-domain weakness at `n=250`; both wide-domain pairs still win there, so the all-six average at 250 remains a 1.118x SINE53 win.

From `n=1200` through `n=50000`, the all-six advantage is roughly 1.48x, with the wide bands around 1.54x-1.56x. At the largest tested sizes, 0.5M through 4M, the all-six advantage remains about 1.33x-1.34x.

Because the test uses only six deterministic random batch realizations per size, it should be treated as a shallow performance map rather than an exhaustive statement about every possible input distribution.
