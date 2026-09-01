# SINE53 custom2 boundary sweep: 1500-4500

## Run

- GitHub Actions run: `33567930913`
- Commit: `3e8c5fa6098d6b78c10e8423bddd881759a805dc`
- Exact Intel Xeon 6973P-C shard: `50`
- Artifact: `sine53-custom2-boundary-1500-4500-50`
- Current SINE53: X50 for `0<|x|<1`, X67 for `1<|x|<500` and `1000<|x|<10000`
- custom2: permanent CPU0+CPU2 scheduler, 32-double aligned split
- Intel: sequential oneMKL `vmdSin(..., VML_HA)` on CPU0
- current/custom2/Intel timed in separate processes
- three rotated outer repetitions; per-cell median used
- exactly six deterministic random batches per size

custom2 output was bit-identical to current SINE53 over all 42 requested cells.

## All-six averages

| n | Current ns/value | custom2 ns/value | Intel ns/value | Current/custom2 | Intel/custom2 | Winner |
|---:|---:|---:|---:|---:|---:|:---|
| 1,500 | 0.505884 | 0.512269 | 0.727048 | 0.9875x | 1.4193x | Current |
| 2,000 | 0.488207 | 0.432921 | 0.719661 | 1.1277x | 1.6623x | custom2 |
| 2,500 | 0.484761 | 0.388459 | 0.721026 | 1.2479x | 1.8561x | custom2 |
| 3,000 | 0.486741 | 0.365525 | 0.718752 | 1.3316x | 1.9664x | custom2 |
| 3,500 | 0.487911 | 0.344602 | 0.717121 | 1.4159x | 2.0810x | custom2 |
| 4,000 | 0.483862 | 0.328920 | 0.716404 | 1.4711x | 2.1780x | custom2 |
| 4,500 | 0.485205 | 0.320739 | 0.715441 | 1.5128x | 2.2306x | custom2 |

## Cell-level result

- `n=1500`: custom2 beats current `1/6`; current wins the all-six average.
- `n=2000`: custom2 beats current `6/6` and Intel `6/6`.
- `n=2500`: custom2 beats current `6/6` and Intel `6/6`.
- `n=3000`: custom2 beats current `6/6` and Intel `6/6`.
- `n=3500`: custom2 beats current `6/6` and Intel `6/6`.
- `n=4000`: custom2 beats current `6/6` and Intel `6/6`.
- `n=4500`: custom2 beats current `6/6` and Intel `6/6`.

Overall:

- custom2 vs current: `37/42`
- custom2 vs Intel: `42/42`
- current vs Intel: `42/42`

## Interpretation

The tested crossover lies between 1500 and 2000. At 1500, current SINE53 is about 1.26% faster on the all-six average. At 2000, custom2 is already about 12.8% faster than current and wins all six individual cells. From there the custom2 advantage grows monotonically on this tested grid, reaching about 51.3% over current and 2.23x over Intel at 4500.

The existing frozen production threshold remains `n>=5000` unless explicitly promoted.
