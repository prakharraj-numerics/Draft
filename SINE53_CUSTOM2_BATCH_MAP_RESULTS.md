# SINE53 custom2 batch map vs current and Intel oneMKL VML_HA

## Run

- GitHub Actions run: `33567193984`
- Commit: `78c238c64ebdf142256542027dd9d876ff29f43a`
- Exact Intel Xeon 6973P-C artifacts: shards `37` and `25`
- Current SINE53: X50 for `0<|x|<1`, X67 for `1<|x|<500` and `1000<|x|<10000`
- custom2: permanent CPU0+CPU2 scheduler, 32-double aligned split
- Intel: sequential oneMKL `vmdSin(..., VML_HA)` on CPU0
- current/custom2/Intel timed in separate processes
- three rotated outer repetitions; per-cell median used
- exactly six deterministic random batches per size

custom2 output was bit-identical to current SINE53 over all 72 requested cells on both exact-Xeon artifacts.

## All-six averages — shard 37

| n | Current ns/value | custom2 ns/value | Intel ns/value | Current/custom2 | Intel/custom2 | Winner |
|---:|---:|---:|---:|---:|---:|:---|
| 50 | 1.061117 | 1.055617 | 1.087870 | 1.0052x | 1.0306x | custom2 |
| 250 | 0.699559 | 2.093122 | 0.785631 | 0.3342x | 0.3753x | Current |
| 1,200 | 0.492568 | 0.574578 | 0.723745 | 0.8573x | 1.2596x | Current |
| 5,000 | 0.481257 | 0.309309 | 0.715878 | 1.5559x | 2.3144x | custom2 |
| 10,000 | 0.482018 | 0.275847 | 0.715867 | 1.7474x | 2.5952x | custom2 |
| 30,000 | 0.480530 | 0.256802 | 0.713660 | 1.8712x | 2.7790x | custom2 |
| 50,000 | 0.479377 | 0.249033 | 0.713441 | 1.9250x | 2.8648x | custom2 |
| 100,000 | 0.489168 | 0.246909 | 0.715893 | 1.9812x | 2.8994x | custom2 |
| 500,000 | 0.543017 | 0.285558 | 0.721409 | 1.9016x | 2.5263x | custom2 |
| 1,000,000 | 0.551159 | 0.293461 | 0.730486 | 1.8781x | 2.4892x | custom2 |
| 2,000,000 | 0.545688 | 0.276819 | 0.720862 | 1.9713x | 2.6041x | custom2 |
| 4,000,000 | 0.543621 | 0.273283 | 0.721295 | 1.9892x | 2.6394x | custom2 |

Cell counts, shard 37:

- custom2 vs current: `57/72`
- custom2 vs Intel: `66/72`
- current vs Intel: `70/72`
- from `n=5000` through `n=4000000`: custom2 beats current `54/54` and Intel `54/54`

## All-six averages — shard 25 replication

| n | Current ns/value | custom2 ns/value | Intel ns/value | Current/custom2 | Intel/custom2 | Winner |
|---:|---:|---:|---:|---:|---:|:---|
| 50 | 1.106455 | 1.063596 | 1.190762 | 1.0403x | 1.1196x | custom2 |
| 250 | 0.715786 | 2.004023 | 0.863326 | 0.3572x | 0.4308x | Current |
| 1,200 | 0.492622 | 0.564599 | 0.796267 | 0.8725x | 1.4103x | Current |
| 5,000 | 0.491998 | 0.310088 | 0.774554 | 1.5866x | 2.4979x | custom2 |
| 10,000 | 0.489787 | 0.278629 | 0.778994 | 1.7579x | 2.7958x | custom2 |
| 30,000 | 0.480924 | 0.260890 | 0.780960 | 1.8434x | 2.9934x | custom2 |
| 50,000 | 0.480197 | 0.261143 | 0.775959 | 1.8388x | 2.9714x | custom2 |
| 100,000 | 0.487807 | 0.259797 | 0.776748 | 1.8776x | 2.9898x | custom2 |
| 500,000 | 0.544380 | 0.281389 | 0.768376 | 1.9346x | 2.7307x | custom2 |
| 1,000,000 | 0.557596 | 0.281850 | 0.781671 | 1.9783x | 2.7734x | custom2 |
| 2,000,000 | 0.558479 | 0.281709 | 0.772091 | 1.9825x | 2.7407x | custom2 |
| 4,000,000 | 0.642422 | 0.288974 | 0.820906 | 2.2231x | 2.8408x | custom2 |

Cell counts, shard 25:

- custom2 vs current: `58/72`
- custom2 vs Intel: `66/72`
- current vs Intel: `68/72`
- from `n=5000` through `n=4000000`: custom2 again beats current `54/54` and Intel `54/54`

## Replicated interpretation

Both exact-Xeon artifacts agree on the important routing pattern:

- `n=50`: custom2 is essentially tied/slightly ahead on the all-six average, with mixed individual cells.
- `n=250`: custom2 is decisively worse than both current and Intel.
- `n=1200`: current SINE53 is faster than custom2; custom2 still beats Intel in all six cells.
- `n>=5000`: custom2 wins all six cells against both current SINE53 and Intel at every tested size through 4M.

This benchmark establishes a strong large-batch custom2 region but does not yet locate the exact crossover between 1200 and 5000. No production routing threshold is frozen by this result alone.
