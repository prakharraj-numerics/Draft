#define _GNU_SOURCE
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <time.h>
#include <vector>

extern "C" int sine53_engine_init(void);
extern "C" void sine53_engine_eval(double *, const double *, size_t);
extern "C" void sine53_engine_cleanup(void);

static volatile double g_sink = 0.0;

static uint64_t mix64(uint64_t x){
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
static double u01(uint64_t x){ return ((double)(mix64(x) >> 11) + 0.5) * 0x1p-53; }
static double now_ns(){ timespec t{}; clock_gettime(CLOCK_MONOTONIC_RAW,&t); return (double)t.tv_sec*1e9+t.tv_nsec; }
static void *al64(size_t bytes){ void *p=nullptr; return posix_memalign(&p,64,bytes)==0?p:nullptr; }

static void bounds(const std::string &band, double &lo, double &hi){
    const double e = 0x1p-20;
    if(band=="0_1")      { lo=e;       hi=1.0-e; }
    else if(band=="1_500")    { lo=1.0+e;   hi=500.0-e; }
    else if(band=="500_10000") { lo=500.0+e; hi=10000.0-e; }
    else { std::fprintf(stderr,"bad band %s\n",band.c_str()); std::exit(2); }
}

static void fill(double *x,size_t n,const std::string &band){
    double lo,hi; bounds(band,lo,hi);
    uint64_t seed = UINT64_C(0x7f4a7c15d1b54a32) ^ ((uint64_t)n*UINT64_C(0x94d049bb133111eb));
    for(char c:band) seed=mix64(seed^(unsigned char)c);
    for(size_t i=0;i<n;i++){
        uint64_t r=mix64(seed+i*UINT64_C(0x9e3779b97f4a7c15));
        double mag=lo+(hi-lo)*u01(r);
        x[i]=(r&1)?-mag:mag;
    }
}

static size_t reps_for(size_t n){
    const size_t target = 8000000;
    size_t r = target/n;
    if(r<1) r=1;
    if(r>100000) r=100000;
    return r;
}

int main(int argc,char **argv){
    if(argc!=4){ std::fprintf(stderr,"usage: bench variant band n\n"); return 2; }
    const std::string variant=argv[1], band=argv[2];
    const size_t n=strtoull(argv[3],nullptr,10);
    if(!n) return 2;
    double *x=(double*)al64(n*sizeof(double));
    double *y=(double*)al64(n*sizeof(double));
    if(!x||!y) return 3;
    fill(x,n,band);
    if(!sine53_engine_init()) return 4;

    for(int w=0;w<7;w++) sine53_engine_eval(y,x,n);
    const size_t reps=reps_for(n);
    std::vector<double> samples;
    samples.reserve(9);
    for(int t=0;t<9;t++){
        double t0=now_ns();
        for(size_t r=0;r<reps;r++) sine53_engine_eval(y,x,n);
        double t1=now_ns();
        samples.push_back((t1-t0)/((double)reps*(double)n));
        g_sink += y[(size_t)((t*1315423911u)%n)];
    }
    std::sort(samples.begin(),samples.end());
    const double med=samples[samples.size()/2];
    const double p25=samples[2], p75=samples[6];
    std::printf("X67FIX_SPEED variant=%s band=%s n=%zu reps=%zu median_ns_el=%.9f p25=%.9f p75=%.9f spread_pct=%.4f sink=%.17g\n",
                variant.c_str(),band.c_str(),n,reps,med,p25,p75,med?100.0*(p75-p25)/med:0.0,(double)g_sink);
    sine53_engine_cleanup();
    free(x); free(y);
    return 0;
}
