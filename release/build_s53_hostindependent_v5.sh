#!/usr/bin/env bash
set -Eeuo pipefail

LOG=/tmp/s53-build.log
mkdir -p /tmp/s53release
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; printf "FAIL rc=%d line=%d cmd=%q\n" "$rc" "$LINENO" "$BASH_COMMAND"; exit "$rc"' ERR

CC=/opt/intel/oneapi/compiler/latest/bin/icx
MKLROOT=/opt/intel/oneapi/mkl/latest
IPPROOT=/opt/intel/oneapi/ipp/latest
export SOURCE_DATE_EPOCH=1

X50_SHA=ba79230dc0154f198129b21eb87fd0f96de0457e0a7f08f6e78ec2d29dd1806c
X67_SHA=737ef77ff9ca07e3e198675fab80a10b04c568f425af925a8f2e96e888bae602

test "$(sha256sum sine53_x50_unit_production.c | awk '{print $1}')" = "$X50_SHA"
test "$(sha256sum sine53_x67_wide_production.c | awk '{print $1}')" = "$X67_SHA"

git fetch -q origin backup-pre-cleanup-20260830:refs/remotes/origin/backup-pre-cleanup-20260830
git show origin/backup-pre-cleanup-20260830:bench_sine_53_wide_fast2.c > bench_sine_53_wide_fast2.c
git show origin/backup-pre-cleanup-20260830:bench_sine_53_wide_intel.c > bench_sine_53_wide_intel.c
git show origin/backup-pre-cleanup-20260830:sine_53_coeff_source.c > sine_53_coeff_source.c
sed -i 's/^int main(void){int cpu=pin();mkl_set_num_threads_local(1);printf("S53F2_DOMAIN/int s53f2_disabled_main(void){int cpu=pin();mkl_set_num_threads_local(1);printf("S53F2_DOMAIN/' bench_sine_53_wide_fast2.c
sed -i 's/#define SF_K 12/#define SF_K 8/' sine_53_coeff_source.c
sed -i 's/#define SF_LUT_N ((1UL << SF_K) + 1UL)/#define SF_LUT_N 403UL/' sine_53_coeff_source.c
sed -i 's/#define KGRID 4096.0/#define KGRID 256.0/' bench_sine_53_wide_intel.c
sed -i 's|#define INVK (1.0/4096.0)|#define INVK (1.0/256.0)|' bench_sine_53_wide_intel.c

cp sine53_x50_unit_production.c generated_x50.c
python3 sine53_production_inject_unit3200.py generated_x50.c
cp sine53_x67_wide_production.c generated_x67.c
python3 sine53_production_inject_wide3200.py generated_x67.c
python3 - <<'PY'
from pathlib import Path
for name in ('generated_x50.c','generated_x67.c'):
    p=Path(name)
    s=p.read_text()
    needle='int main(void){'
    if s.count(needle) != 1:
        raise SystemExit(f'{name}: main count={s.count(needle)}')
    p.write_text(s.replace(needle,'int embedded_entry(void){',1))
PY
sha256sum generated_x50.c generated_x67.c | tee /tmp/generated_sha256.txt

cat > /tmp/dump_s53_tables.c <<'C'
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpfr.h>
#include "sine_53_coeff_source.c"
#define REDN2 4096
static void *a64(size_t n){void*p=NULL;if(posix_memalign(&p,64,n?n:64))return NULL;return p;}
static double cv(const mp_limb_t q[2],int neg){long double v=ldexpl((long double)q[1],64-103)+ldexpl((long double)q[0],-103);double d=(double)v;return neg?-d:d;}
static void out(const char*n,const double*a,size_t z){printf("static const double %s[%zu] __attribute__((aligned(64)))={",n,z);for(size_t i=0;i<z;i++){if((i&3)==0)putchar('\n');printf("%a%s",a[i],i+1==z?"":",");}printf("\n};\n");}
int main(void){
    double *rh=a64((size_t)REDN2*sizeof(double)),*rl=a64((size_t)REDN2*sizeof(double));
    if(!rh||!rl)return 2;
    mpfr_t pi,t;mpfr_init2(pi,256);mpfr_init2(t,256);mpfr_const_pi(pi,MPFR_RNDN);
    for(unsigned q=0;q<REDN2;q++){mpfr_mul_ui(t,pi,q,MPFR_RNDN);double h=mpfr_get_d(t,MPFR_RNDN);rh[q]=h;mpfr_sub_d(t,t,h,MPFR_RNDN);rl[q]=mpfr_get_d(t,MPFR_RNDN);}
    mpfr_clear(t);mpfr_clear(pi);
    sine_fixed_ctx *ctx=s53_coeff_create_terms(2);if(!ctx)return 3;
    int deg=ctx->poly_deg;
    double *tab=a64((size_t)(deg+1)*SF_LUT_N*sizeof(double));if(!tab)return 4;
    for(unsigned a=0;a<SF_LUT_N;a++){size_t off=(size_t)a*(size_t)(deg+1);for(int j=0;j<=deg;j++)tab[(size_t)j*SF_LUT_N+(size_t)a]=cv(ctx->coef+2*(off+(size_t)j),ctx->coef_sign[off+(size_t)j]!=0);}
    printf("#define S53_FROZEN_DEG %d\n",deg);
    out("s53_r0",rh,REDN2);out("s53_r1",rl,REDN2);out("s53_r2",tab,(size_t)(deg+1)*SF_LUT_N);
    s53_coeff_destroy(ctx);free(tab);free(rl);free(rh);return 0;
}
C
/usr/bin/gcc -O2 -march=x86-64-v2 -I"$FLINT_PREFIX/include" -I. /tmp/dump_s53_tables.c \
  -L"$FLINT_PREFIX/lib" -Wl,-rpath,"$FLINT_PREFIX/lib" -lflint -lmpfr -lgmp -lm -o /tmp/dump_s53_tables
LD_LIBRARY_PATH="$FLINT_PREFIX/lib:${LD_LIBRARY_PATH:-}" /tmp/dump_s53_tables > /tmp/s53_frozen_tables.h
grep -q '^#define S53_FROZEN_DEG 5$' /tmp/s53_frozen_tables.h
grep -q 's53_r0\[4096\]' /tmp/s53_frozen_tables.h
grep -q 's53_r1\[4096\]' /tmp/s53_frozen_tables.h
grep -q 's53_r2\[2418\]' /tmp/s53_frozen_tables.h
sha256sum /tmp/s53_frozen_tables.h | tee /tmp/s53_tables_sha256.txt

INC="-I$FLINT_PREFIX/include -I$MKLROOT/include -I$IPPROOT/include -I$IPPROOT/include/ipp -I."
COMMON='-O3 -xGRANITERAPIDS -qopt-zmm-usage=high -fp-model=precise -fno-math-errno -DNDEBUG'
HARD="$COMMON -fPIC -fvisibility=hidden -fno-semantic-interposition -fno-ident -ffunction-sections -fdata-sections -fno-asynchronous-unwind-tables -fno-unwind-tables -ffile-prefix-map=$PWD=. -fdebug-prefix-map=$PWD=."

"$CC" $HARD $INC -DENGINE_PREFIX=u -DSINE53_GENERATED_SOURCE='"generated_x50.c"' -DS53_FROZEN_TABLES_HEADER='"/tmp/s53_frozen_tables.h"' -c release/s53_frozen_engine.c -o /tmp/s53release/u.o
"$CC" $HARD $INC -DENGINE_PREFIX=w -DSINE53_GENERATED_SOURCE='"generated_x67.c"' -DS53_FROZEN_TABLES_HEADER='"/tmp/s53_frozen_tables.h"' -c release/s53_frozen_engine.c -o /tmp/s53release/w.o
"$CC" $HARD -c release/s53_frozen_api.c -o /tmp/s53release/api.o

/usr/bin/gcc -shared -static-libgcc -o /tmp/s53release/libs53.so \
  /tmp/s53release/u.o /tmp/s53release/w.o /tmp/s53release/api.o \
  -Wl,--gc-sections -Wl,--version-script=release/s53_exports.map -Wl,--build-id=none \
  -Wl,-z,relro,-z,now,-z,noexecstack -Wl,-Bsymbolic-functions -Wl,--as-needed -Wl,-z,defs -lm

strip --strip-all /tmp/s53release/libs53.so
objcopy --remove-section=.comment --remove-section=.note.gnu.build-id --remove-section=.gnu_debuglink --remove-section=.gnu_debugaltlink /tmp/s53release/libs53.so
chmod 755 /tmp/s53release/libs53.so

LIB=/tmp/s53release/libs53.so
{
  file "$LIB"
  sha256sum "$LIB"
  echo '--- exports ---'
  nm -D --defined-only "$LIB"
  echo '--- undefined ---'
  nm -D --undefined-only "$LIB"
  echo '--- needed ---'
  readelf -d "$LIB" | grep NEEDED || true
  echo '--- program headers ---'
  readelf -l "$LIB"
  echo '--- sections ---'
  readelf -S "$LIB"
} | tee /tmp/s53release/audit.txt

nm -D --defined-only --format=posix "$LIB" | awk '{print $1}' | sort > /tmp/actual_exports
printf '%s\n' s53_close s53_eval s53_init | sort > /tmp/expected_exports
diff -u /tmp/expected_exports /tmp/actual_exports

if readelf -S "$LIB" | grep -Eq '\.(debug|symtab|strtab|comment)'; then echo FORBIDDEN_SECTION >&2; exit 31; fi
if readelf -d "$LIB" | grep -Eiq '(flint|mpfr|gmp|mkl|ipp|svml|imf|irng|intlc|iomp|tbb)'; then echo FORBIDDEN_RUNTIME_DEPENDENCY >&2; exit 32; fi
if strings -a "$LIB" | grep -Eiq '(chatgpt|openai|gpt[-_ ]?[0-9]|anthropic|claude|gemini|copilot|github|workflow|draft|benchmark|generated|mode.?5|secant|x50|x67|s53f|s53w|sine53_|flint|mpfr|gmp|mkl|ipp|svml|oneapi|intel|octant_eval|kernel_create|redtab)'; then echo FORBIDDEN_RELEASE_STRING_FOUND >&2; exit 33; fi

readelf -W -l "$LIB" | grep -q 'GNU_RELRO'
readelf -W -d "$LIB" | grep -q 'BIND_NOW'
if readelf -W -l "$LIB" | awk '/GNU_STACK/{print}' | grep -q 'RWE'; then echo EXECUTABLE_STACK >&2; exit 34; fi

cat > /tmp/s53release/s53.h <<'H'
#ifndef S53_H
#define S53_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
int s53_init(void);
void s53_eval(double *out, const double *in, size_t n, unsigned profile);
void s53_close(void);
#ifdef __cplusplus
}
#endif
#endif
H
cat > /tmp/s53release/USAGE.txt <<'U'
S53 binary interface

Call s53_init() once before evaluation.
Call s53_eval(out, in, n, 0) for the unit-range profile.
Call s53_eval(out, in, n, 1) for the wide-range profile.
Call s53_close() when finished.
U
sha256sum "$LIB" > /tmp/s53release/SHA256.txt
{
  echo 'frozen_source_commit=9f4b58c18eecb99c9220099eb0cd312702400e8f'
  echo "x50_sha256=$X50_SHA"
  echo "x67_sha256=$X67_SHA"
  cat /tmp/generated_sha256.txt
  cat /tmp/s53_tables_sha256.txt
} > /tmp/s53release/source-build-hashes.txt

echo S53_HARDENED_BUILD_OK
