#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/s53release
CC=/opt/intel/oneapi/compiler/latest/bin/icx
CXX=/opt/intel/oneapi/compiler/latest/bin/icpx
MKLROOT=/opt/intel/oneapi/mkl/latest
IPPROOT=/opt/intel/oneapi/ipp/latest
export SOURCE_DATE_EPOCH=1

test "$(sha256sum sine53_x50_unit_production.c | awk '{print $1}')" = 'ba79230dc0154f198129b21eb87fd0f96de0457e0a7f08f6e78ec2d29dd1806c'
test "$(sha256sum sine53_x67_wide_production.c | awk '{print $1}')" = '737ef77ff9ca07e3e198675fab80a10b04c568f425af925a8f2e96e888bae602'

git fetch origin backup-pre-cleanup-20260830:refs/remotes/origin/backup-pre-cleanup-20260830
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
    p=Path(name); s=p.read_text(); needle='int main(void){'
    if s.count(needle)!=1: raise SystemExit(f'{name}: main count={s.count(needle)}')
    p.write_text(s.replace(needle,'int embedded_entry(void){',1))
PY
sha256sum generated_x50.c generated_x67.c | tee /tmp/generated_sha256.txt

cat > /tmp/dump_tables.c <<'C'
#define _GNU_SOURCE
#include <stdio.h>
#include "generated_x67.c"
static void out(const char *n,const double *a,size_t z){
    printf("static const double %s[%zu] __attribute__((aligned(64)))={",n,z);
    for(size_t i=0;i<z;i++){ if((i&3)==0) putchar('\n'); printf("%a%s",a[i],i+1==z?"":","); }
    printf("\n};\n");
}
int main(void){
    if(!redtab2_init()) return 2;
    s53w_kernel *k=kernel_create(2); if(!k) return 3;
    printf("#define S53_FROZEN_DEG %d\n",k->deg);
    out("s53_r0",pih2,REDN2); out("s53_r1",pil2,REDN2);
    out("s53_r2",k->tab,(size_t)(k->deg+1)*LUTN);
    return 0;
}
C
INC="-I$FLINT_PREFIX/include -I$MKLROOT/include -I$IPPROOT/include -I$IPPROOT/include/ipp -I."
TABLELIBS="-L$FLINT_PREFIX/lib -Wl,-rpath,$FLINT_PREFIX/lib -lflint -lmpfr -lgmp -L$MKLROOT/lib -Wl,-rpath,$MKLROOT/lib -lmkl_rt -lm -lpthread -ldl"
"$CC" -O2 -ffunction-sections -fdata-sections $INC /tmp/dump_tables.c -Wl,--gc-sections -o /tmp/dump_tables $TABLELIBS
LD_LIBRARY_PATH="$FLINT_PREFIX/lib:$MKLROOT/lib:/opt/intel/oneapi/compiler/latest/lib:${LD_LIBRARY_PATH:-}" /tmp/dump_tables > /tmp/s53_frozen_tables.h
grep -q '^#define S53_FROZEN_DEG 5$' /tmp/s53_frozen_tables.h
grep -q 's53_r0\[4096\]' /tmp/s53_frozen_tables.h
grep -q 's53_r1\[4096\]' /tmp/s53_frozen_tables.h
grep -q 's53_r2\[2418\]' /tmp/s53_frozen_tables.h

COMMON='-O3 -xHost -qopt-zmm-usage=high -fp-model=precise -fno-math-errno -DNDEBUG'
REFLIBS="$TABLELIBS"
"$CC" $COMMON -fPIC -shared -ffunction-sections -fdata-sections $INC -DSINE53_GENERATED_SOURCE='"generated_x50.c"' sine53_engine_adapter.c -Wl,--gc-sections -o /tmp/s53release/ref_u.so $REFLIBS
"$CC" $COMMON -fPIC -shared -ffunction-sections -fdata-sections $INC -DSINE53_GENERATED_SOURCE='"generated_x67.c"' sine53_engine_adapter.c -Wl,--gc-sections -o /tmp/s53release/ref_w.so $REFLIBS
HARD="$COMMON -fPIC -fvisibility=hidden -fno-semantic-interposition -fno-ident -ffunction-sections -fdata-sections -fno-asynchronous-unwind-tables -fno-unwind-tables -ffile-prefix-map=$PWD=. -fdebug-prefix-map=$PWD=."
"$CC" $HARD $INC -DENGINE_PREFIX=u -DSINE53_GENERATED_SOURCE='"generated_x50.c"' -DS53_FROZEN_TABLES_HEADER='"/tmp/s53_frozen_tables.h"' -c release/s53_frozen_engine.c -o /tmp/s53release/u.o
"$CC" $HARD $INC -DENGINE_PREFIX=w -DSINE53_GENERATED_SOURCE='"generated_x67.c"' -DS53_FROZEN_TABLES_HEADER='"/tmp/s53_frozen_tables.h"' -c release/s53_frozen_engine.c -o /tmp/s53release/w.o
"$CC" $HARD -c release/s53_frozen_api.c -o /tmp/s53release/api.o
/usr/bin/gcc -shared -static-libgcc -o /tmp/s53release/libs53.so /tmp/s53release/u.o /tmp/s53release/w.o /tmp/s53release/api.o \
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
  echo '--- sections ---'
  readelf -S "$LIB"
} | tee /tmp/s53release/audit.txt
nm -D --defined-only --format=posix "$LIB" | awk '{print $1}' | sort > /tmp/actual_exports
printf '%s\n' s53_close s53_eval s53_init | sort > /tmp/expected_exports
diff -u /tmp/expected_exports /tmp/actual_exports
if readelf -S "$LIB" | grep -Eq '\.(debug|symtab|strtab|comment)'; then echo FORBIDDEN_SECTION >&2; exit 1; fi
if readelf -d "$LIB" | grep -Eiq '(flint|mpfr|gmp|mkl|ipp|svml|imf|irng|intlc|iomp|tbb)'; then echo FORBIDDEN_RUNTIME_DEPENDENCY >&2; exit 1; fi
if strings -a "$LIB" | grep -Eiq '(chatgpt|openai|gpt[-_ ]?[0-9]|anthropic|claude|gemini|copilot|github|workflow|draft|benchmark|generated|mode.?5|secant|x50|x67|s53f|s53w|sine53_|flint|mpfr|gmp|mkl|ipp|svml|oneapi|intel|octant_eval|kernel_create|redtab)'; then
  echo FORBIDDEN_RELEASE_STRING_FOUND >&2; exit 1
fi

cat > /tmp/validate.cpp <<'CPP'
#include <mpfr.h>
#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using init_t=int(*)(void);using eval_t=void(*)(double*,const double*,size_t);using pub_init_t=int(*)(void);using pub_eval_t=void(*)(double*,const double*,size_t,unsigned);using pub_close_t=void(*)(void);
struct R{void*h;init_t init;eval_t eval;};
static R ref(const char*p){R r{};r.h=dlopen(p,RTLD_NOW|RTLD_LOCAL);if(!r.h){std::fprintf(stderr,"dlopen %s\n",dlerror());std::exit(2);}r.init=(init_t)dlsym(r.h,"sine53_engine_init");r.eval=(eval_t)dlsym(r.h,"sine53_engine_eval");if(!r.init||!r.eval||!r.init())std::exit(3);return r;}
static uint64_t ord(double x){uint64_t u;std::memcpy(&u,&x,8);return(u>>63)?~u:(u|(1ULL<<63));}
static uint64_t ulp(double a,double b){uint64_t A=ord(a),B=ord(b);return A>B?A-B:B-A;}
static double rs(double x){mpfr_t a,r;mpfr_init2(a,256);mpfr_init2(r,256);mpfr_set_d(a,x,MPFR_RNDN);mpfr_sin(r,a,MPFR_RNDN);double y=mpfr_get_d(r,MPFR_RNDN);mpfr_clear(a);mpfr_clear(r);return y;}
int main(){
  void*h=dlopen("/tmp/s53release/libs53.so",RTLD_NOW|RTLD_LOCAL);if(!h){std::fprintf(stderr,"%s\n",dlerror());return 4;}
  auto pi=(pub_init_t)dlsym(h,"s53_init");auto pe=(pub_eval_t)dlsym(h,"s53_eval");auto pc=(pub_close_t)dlsym(h,"s53_close");if(!pi||!pe||!pc||!pi())return 5;
  R u=ref("/tmp/s53release/ref_u.so"),w=ref("/tmp/s53release/ref_w.so");
  struct B{double lo,hi;unsigned profile;const char*name;uint64_t seed;};
  B bs[]={{0,1,0,"0_1",0x5317C0DE1234ULL},{1,500,1,"1_500",0x5317C0DE2234ULL},{1000,10000,1,"1000_10000",0x5317C0DE3234ULL}};
  for(auto b:bs){int half=20000,n=40000;std::mt19937_64 g(b.seed);std::uniform_real_distribution<double>d(b.lo,b.hi);std::vector<double>x(n),a(n),z(n);for(int i=0;i<half;i++){double m=d(g);if(m==0)m=std::nextafter(0.0,1.0);x[i]=m;x[i+half]=-m;}pe(a.data(),x.data(),n,b.profile);(b.profile?w:u).eval(z.data(),x.data(),n);uint64_t neq=0,mx=0,gt1=0;for(int i=0;i<n;i++){if(std::memcmp(&a[i],&z[i],8))neq++;double r=rs(x[i]);uint64_t q=ulp(a[i],r);if(q>mx)mx=q;if(q>1)gt1++;}std::printf("BAND %s N=%d BIT_MISMATCH_REF=%llu MAX_ULP=%llu GT1=%llu\n",b.name,n,(unsigned long long)neq,(unsigned long long)mx,(unsigned long long)gt1);if(neq||gt1)return 10;}
  pc();return 0;
}
CPP
"$CXX" -O2 /tmp/validate.cpp -L$FLINT_PREFIX/lib -Wl,-rpath,$FLINT_PREFIX/lib -lmpfr -lgmp -ldl -lm -o /tmp/validate
LD_LIBRARY_PATH="$FLINT_PREFIX/lib:/opt/intel/oneapi/compiler/latest/lib:$MKLROOT/lib:${LD_LIBRARY_PATH:-}" taskset -c 0 /tmp/validate | tee /tmp/s53release/accuracy.txt

cat > /tmp/perf.cpp <<'CPP'
#include <ipp.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <vector>
using init_t=int(*)(void);using eval_t=void(*)(double*,const double*,size_t,unsigned);using close_t=void(*)(void);
static volatile double sink=0;static uint64_t mix(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static double u01(uint64_t x){return((double)(mix(x)>>11)+0.5)*0x1p-53;}static double ns(){timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec*1e9+t.tv_nsec;}static double med(double*v){std::sort(v,v+5);return v[2];}
static void fill(double*x,size_t n,int c){double e=0x1p-20,lo=c<2?e:(c<4?1+e:1000+e),hi=c<2?1-e:(c<4?500-e:10000-e);uint64_t s=0x53154a32d192ed03ULL^(n*0x94d049bb133111ebULL)^((uint64_t)c<<58);for(size_t i=0;i<n;i++){double v=lo+(hi-lo)*u01(s+i*0x9e3779b97f4a7c15ULL);x[i]=(c&1)?-v:v;}}
int main(){cpu_set_t s;CPU_ZERO(&s);CPU_SET(0,&s);if(pthread_setaffinity_np(pthread_self(),sizeof(s),&s))return 2;if(ippInit()!=ippStsNoErr)return 3;void*h=dlopen("/tmp/s53release/libs53.so",RTLD_NOW|RTLD_LOCAL);if(!h)return 4;auto ini=(init_t)dlsym(h,"s53_init");auto ev=(eval_t)dlsym(h,"s53_eval");auto cl=(close_t)dlsym(h,"s53_close");if(!ini||!ev||!cl||!ini())return 5;size_t sizes[]={100,400,1000,5000,20000,40000,70000,200000,500000,1000000};std::puts("PROFILE,N,S53_NS_EL,IPP_NS_EL,SPEEDUP");for(unsigned p=0;p<2;p++)for(size_t n:sizes){int c0=p?2:0,c1=p?6:2,cases=c1-c0;std::vector<std::vector<double>>x(cases,std::vector<double>(n)),y(cases,std::vector<double>(n));for(int j=0;j<cases;j++)fill(x[j].data(),n,c0+j);size_t reps=4000000/(n*cases);if(reps<1)reps=1;if(reps>20000)reps=20000;for(int q=0;q<3;q++)for(int j=0;j<cases;j++)ev(y[j].data(),x[j].data(),n,p);double a[5],b[5];for(int t=0;t<5;t++){double t0=ns();for(size_t r=0;r<reps;r++)for(int j=0;j<cases;j++)ev(y[j].data(),x[j].data(),n,p);double t1=ns();a[t]=(t1-t0)/(reps*n*cases);t0=ns();for(size_t r=0;r<reps;r++)for(int j=0;j<cases;j++)ippsSin_64f_A53(x[j].data(),y[j].data(),(int)n);t1=ns();b[t]=(t1-t0)/(reps*n*cases);sink+=y[0][n/2];}double A=med(a),B=med(b);std::printf("%u,%zu,%.9f,%.9f,%.6f\n",p,n,A,B,B/A);}cl();return sink==12345;}
CPP
"$CXX" -O3 -xHost -qopt-zmm-usage=high -fp-model=precise -I$IPPROOT/include -I$IPPROOT/include/ipp /tmp/perf.cpp -L$IPPROOT/lib -Wl,-rpath,$IPPROOT/lib -lippvm -lipps -lippcore -ldl -lpthread -lm -o /tmp/perf
LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/latest/lib:$IPPROOT/lib:${LD_LIBRARY_PATH:-}" taskset -c 0 /tmp/perf | tee /tmp/s53release/performance.csv

cat > /tmp/s53release/s53.h <<'H'
#pragma once
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
H
cat > /tmp/s53release/USAGE.txt <<'T'
Linux x86-64 AVX-512 binary64 sine library.
Call s53_init() before evaluation and s53_close() at shutdown.
s53_eval(out,in,n,0): batch known to satisfy |x| < 1.
s53_eval(out,in,n,1): general validated path for |x| up to 10000; use for mixed batches.
Input/output arrays contain binary64 values. In-place use is not certified.
T
sha256sum /tmp/s53release/libs53.so > /tmp/s53release/SHA256.txt
cp /tmp/generated_sha256.txt /tmp/s53release/source-build-hashes.txt
rm -f /tmp/s53release/ref_u.so /tmp/s53release/ref_w.so /tmp/s53release/*.o
