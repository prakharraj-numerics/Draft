# SINE53 Apple EARTH — FROZEN

Frozen Apple sine baseline as of 2026-09-07.

## Frozen implementation
- Kernel: `benchmark_support/apple_sine53_earth_frozen.cpp`
- Kernel blob at freeze: `947dd9db80ccfe6601eb931d1e16b007abb42530`
- Source state commit before freeze marker: `25276a4e50aac05e1f6b08d9115c8db496bb3c7d`
- Validation/benchmark reference run: `34134092002`
- Hardware: Namespace `nscloud-macos-tahoe-arm64-6x14`, Apple M4 Pro (Virtual), SME=1, SME2=1, SVL=64 bytes

## Contract
- Apple SME streaming kernel
- 8 FP64 lanes at 64-byte SVL
- terms=1 / degree=3 cubic
- tuned `EARTH_MH` / `EARTH_M6`
- original EARTH Horner/FMA ordering
- cosine anchor streams remapped minimally for sine inside the kernel
- speed is the selection metric; CPU-efficiency is not part of the sine EARTH decision criterion
- comparator: Apple Accelerate `vvsin`, `VECLIB_MAXIMUM_THREADS=1`

## Frozen speed reference
Run `34134092002` wall-time results (ns/el):

| n | SINE EARTH | Apple vvsin | speedup |
|---:|---:|---:|---:|
| 100 | 1.799576389 | 0.852277778 | 0.473599x |
| 400 | 0.688576389 | 0.819305556 | 1.189854x |
| 1000 | 0.706336806 | 0.864385417 | 1.223758x |
| 4000 | 0.690315972 | 0.822048611 | 1.190829x |
| 10000 | 0.702177083 | 0.888958333 | 1.266003x |
| 15000 | 0.704864583 | 0.907527778 | 1.287521x |
| 29999 | 0.705617271 | 0.833420142 | 1.181122x |
| 30000 | 0.706208333 | 0.895996528 | 1.268743x |
| 50000 | 0.698552083 | 0.879152778 | 1.258536x |
| 175000 | 0.701032913 | 0.879863445 | 1.255096x |
| 100000 | 0.709569444 | 0.961013889 | 1.354362x |
| 500000 | 0.702916667 | 0.880961806 | 1.253295x |
| 1000000 | 0.733895833 | 0.903854167 | 1.231584x |

## Freeze rule
Do not modify the frozen kernel for future experiments. Any new sine/SME optimization must use a separate experimental file/path and be compared against this frozen state.
