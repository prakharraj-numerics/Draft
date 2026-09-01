# SINE53 custom2 boundary 1600-1900 results

## Run

- GitHub Actions run: `33568489785`
- Head: `62d96e11269b13848e78efe178c34f3661f280a3`
- Exact Intel Xeon 6973P-C shard: `61`
- Artifact: `9824079671` (`sine53-custom2-boundary-1600-1900-61`)
- Current SINE53 vs permanent CPU0+CPU2 custom2 vs sequential oneMKL `vmdSin(..., VML_HA)`
- Same six deterministic batches per size: unit +/-; 1..500 +/-; 1000..10000 +/-
- Three isolated/rotated outer repetitions; per-cell median timing
- custom2 was bit-identical to current SINE53 on all tested outputs (`total_bitdiff=0` for both X50 and X67 adapters)

## All-six averages

| n | Current ns/value | custom2 ns/value | Intel ns/value | Current/custom2 | Intel/custom2 | Intel/current | Winner |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 1600 | 0.488344833 | 0.477302750 | 0.723568417 | 1.023134x | 1.515953x | 1.481675x | custom2 |
| 1700 | 0.487414299 | 0.466260754 | 0.722807206 | 1.045368x | 1.550221x | 1.482942x | custom2 |
| 1800 | 0.485016418 | 0.455372787 | 0.720127013 | 1.065098x | 1.581401x | 1.484748x | custom2 |
| 1900 | 0.489878760 | 0.452160296 | 0.720937396 | 1.083418x | 1.594429x | 1.471665x | custom2 |

## Cell counts

Across 24 sign/range/size cells:

- custom2 beats current: `24/24`
- custom2 beats Intel: `24/24`
- current beats Intel: `24/24`

At `n=1600`, custom2 wins all six individual cells, although the all-six average advantage over current is only about 2.31%. The advantage grows through the tested grid: 4.54% at 1700, 6.51% at 1800, and 8.34% at 1900 by current/custom2 ratio.

This is benchmark evidence for the 1600-1900 grid only. The production threshold remains frozen at 2000 unless explicitly promoted.
