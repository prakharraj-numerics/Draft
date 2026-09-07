#include <mach/mach_time.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mpfr.h>

#include "apple_sine53_constants.h"
#include "apple_sine53_earth_prepare.hpp"

extern "C" void sine53_earth_raw(const double*, double*, size_t,
                                  const double*, const double*, const double*,
                                  const uint64_t*);

static volatile double g_sink = 0.0;

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31; return x;
}
static double unit52(uint64_t h) { return ((double)(h >> 12) + 0.5) * 0x1p-52; }
static uint64_t ticks() { return mach_absolute_time(); }
static double ticks_ns(uint64_t t) {
    static mach_timebase_info_data_t tb = [] { mach_timebase_info_data_t v{}; mach_timebase_info(&v); return v; }();
    return (double)t * (double)tb.numer / (double)tb.denom;
}
static uint64_t ordered_bits(double x) {
    uint64_t u; std::memcpy(&u, &x, 8);
    return (u >> 63) ? ~u : (u | UINT64_C(0x8000000000000000));
}
static uint64_t ulpd(double a, double b) {
    if (a == b) return 0;
    uint64_t x = ordered_bits(a), y = ordered_bits(b);
    return x > y ? x-y : y-x;
}

struct B {
    size_t n;
    std::vector<double> x, y, d, c0, c1;
    std::vector<uint64_t> sign;
    explicit B(size_t nn): n(nn), x(nn), y(nn), d(nn), c0(nn), c1(nn), sign(nn) {}
};

static apple_sine53_earth::PrepareTables tables() {
    return {apple_sine53_c0, apple_sine53_c1, APPLE_SINE53_LUTN,
            apple_sine53_pih, apple_sine53_pil, APPLE_SINE53_REDN};
}

static void fill(B& b, uint64_t seed) {
    for (size_t i = 0; i < b.n; ++i) {
        const int band = (int)(i % 3);
        const double lo[3] = {0.0, 1.0, 1000.0};
        const double hi[3] = {1.0, 500.0, 10000.0};
        double u = unit52(mix64(seed + i * UINT64_C(0x9e3779b97f4a7c15)));
        double v = std::fma(hi[band]-lo[band], u, lo[band]);
        b.x[i] = (i & 1) ? -v : v;
    }
}

static size_t prep(B& b) {
    return apple_sine53_earth::prepare(b.x.data(), b.n, tables(),
                                       b.d.data(), b.c0.data(), b.c1.data(), b.sign.data());
}
static void earth(B& b) {
    sine53_earth_raw(b.x.data(), b.y.data(), b.n,
                     b.d.data(), b.c0.data(), b.c1.data(), b.sign.data());
}

static int validate() {
    B b(9600); fill(b, UINT64_C(202609070053));
    size_t bad = prep(b);
    earth(b);
    if (bad) { std::printf("SINE53_EARTH_VALIDATE prep_bad=%zu\n", bad); return 4; }

    mpfr_t z, r; mpfr_init2(z,256); mpfr_init2(r,256);
    uint64_t maxulp=0; size_t exact=0, le1=0, le2=0;
    for (size_t i=0;i<b.n;++i) {
        mpfr_set_d(z,b.x[i],MPFR_RNDN); mpfr_sin(r,z,MPFR_RNDN);
        double ref=mpfr_get_d(r,MPFR_RNDN);
        uint64_t u=ulpd(b.y[i],ref);
        maxulp=std::max(maxulp,u); exact += (u==0); le1 += (u<=1); le2 += (u<=2);
    }
    mpfr_clear(r); mpfr_clear(z);
    std::printf("SINE53_EARTH_VALIDATE cases=%zu exact=%zu le1=%zu le2=%zu maxulp=%llu reference=MPFR256\n",
                b.n,exact,le1,le2,(unsigned long long)maxulp);
    return maxulp<=2 ? 0 : 5;
}

static double median7(double x[7]) { std::sort(x,x+7); return x[3]; }
static int bench(size_t n) {
    B b(n); fill(b, UINT64_C(202609070054)+n); if (prep(b)) return 6;
    for(int w=0;w<20;++w) earth(b);
    size_t reps=12000000/(n?n:1); if(reps<3) reps=3; if(reps>20000) reps=20000;
    double e[7], full[7];
    for(int t=0;t<7;++t) {
        uint64_t a=ticks(); for(size_t r=0;r<reps;++r) earth(b); uint64_t c=ticks();
        e[t]=ticks_ns(c-a)/(double)(reps*n);
        a=ticks(); for(size_t r=0;r<reps;++r){prep(b);earth(b);} c=ticks();
        full[t]=ticks_ns(c-a)/(double)(reps*n);
        g_sink += b.y[(t*97)%n];
    }
    std::printf("SINE53_EARTH_BENCH n=%zu earth_ns_el=%.9f prep_plus_earth_ns_el=%.9f prep_est_ns_el=%.9f reps=%zu sink=%.17g\n",
                n,median7(e),median7(full),median7(full)-median7(e),reps,(double)g_sink);
    return 0;
}

int main(int argc,char**argv) {
    if(argc==2 && std::string(argv[1])=="validate") return validate();
    if(argc==3 && std::string(argv[1])=="bench") return bench((size_t)std::strtoull(argv[2],nullptr,10));
    return 2;
}
