#!/usr/bin/env bash
set -euo pipefail

PRE=670bb969604e6acd994cba8033eb5171467e0e38
POST=e5803aaa88c1ff37f3f52e45f2abf2f29b0de87a
FLINT_PREFIX=${FLINT_PREFIX:-/home/runner/flint36}
MKLROOT=/opt/intel/oneapi/mkl/latest
CC=/opt/intel/oneapi/compiler/latest/bin/icx
CXX=/opt/intel/oneapi/compiler/latest/bin/icpx

git fetch origin backup-pre-cleanup-20260830:refs/remotes/origin/backup-pre-cleanup-20260830
git show origin/backup-pre-cleanup-20260830:bench_sine_53_wide_fast2.c > bench_sine_53_wide_fast2.c
git show origin/backup-pre-cleanup-20260830:bench_sine_53_wide_intel.c > bench_sine_53_wide_intel.c
git show origin/backup-pre-cleanup-20260830:sine_53_coeff_source.c > sine_53_coeff_source.c
sed -i 's/^int main(void){int cpu=pin();mkl_set_num_threads_local(1);printf("S53F2_DOMAIN/int s53f2_disabled_main(void){int cpu=pin();mkl_set_num_threads_local(1);printf("S53F2_DOMAIN/' bench_sine_53_wide_fast2.c
sed -i 's/#define SF_K 12/#define SF_K 8/' sine_53_coeff_source.c
sed -i 's/#define SF_LUT_N ((1UL << SF_K) + 1UL)/#define SF_LUT_N 403UL/' sine_53_coeff_source.c
sed -i 's/#define KGRID 4096.0/#define KGRID 256.0/' bench_sine_53_wide_intel.c
sed -i 's|#define INVK (1.0/4096.0)|#define INVK (1.0/256.0)|' bench_sine_53_wide_intel.c

git show "$PRE":sine53_x67_wide_production.c > x67_pre.c
git show "$POST":sine53_x67_wide_production.c > x67_post.c
printf 'PRE_SOURCE_SHA256 '; sha256sum x67_pre.c
printf 'POST_SOURCE_SHA256 '; sha256sum x67_post.c

for p in x67_pre x67_post; do
  cp "$p.c" "$p.gen.c"
  python3 sine53_production_inject_wide3200.py "$p.gen.c"
done
python3 - <<'PY'
from pathlib import Path
for name in ('x67_pre.gen.c','x67_post.gen.c'):
    p=Path(name); s=p.read_text(); needle='int main(void){'
    if s.count(needle)!=1: raise SystemExit(f'{name}: main count={s.count(needle)}')
    p.write_text(s.replace(needle,'int sine53_embedded_main(void){',1))
PY

mkdir -p /tmp/x67check
INC="-DNDEBUG -I$FLINT_PREFIX/include -I$MKLROOT/include -I."
COMMON='-O3 -xHost -qopt-zmm-usage=high -fp-model=precise -fno-math-errno -fPIC'
LIBS="-L$FLINT_PREFIX/lib -Wl,-rpath,$FLINT_PREFIX/lib -lflint -lmpfr -lgmp -L$MKLROOT/lib -Wl,-rpath,$MKLROOT/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
for v in x67_pre x67_post; do
  "$CC" $COMMON $INC -DSINE53_GENERATED_SOURCE='"'"$v"'.gen.c"' -shared sine53_engine_adapter.c -o "/tmp/x67check/$v.so" $LIBS
done
"$CXX" -std=c++17 -O3 experiments/x67_prepost_single_check.cpp -o /tmp/x67check/check -lmpfr -lgmp -ldl
export LD_LIBRARY_PATH="$FLINT_PREFIX/lib:/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/mkl/latest/lib:${LD_LIBRARY_PATH:-}"
taskset -c 0 /tmp/x67check/check | tee /tmp/x67check/result.txt
