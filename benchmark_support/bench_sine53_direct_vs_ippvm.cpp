#define _GNU_SOURCE
#include <ipp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <time.h>
#include <sys/resource.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <vector>
#include "sine53_batch_x50_x67_direct.hpp"

extern "C" int sine53_engine_init(void);
extern "C" void sine53_engine_eval(double *, const double *, size_t);
extern "C" void sine53_engine_cleanup(void);

#ifndef SINE53_ENGINE_WIDE
#define SINE53_ENGINE_WIDE 0
#endif

static volatile double g_sink=0.0;
extern "C" __attribute__((noinline)) void sine53_profile_start(void){asm volatile("":::"memory");}
extern "C" __attribute__((noinline)) void sine53_profile_stop(void){asm volatile("":::"memory");}

static uint64_t mix64(uint64_t x){x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);} 
static double u01(uint64_t x){return ((double)(mix64(x)>>11)+0.5)*0x1p-53;}
static void pin0(){cpu_set_t s;CPU_ZERO(&s);CPU_SET(0,&s);if(pthread_setaffinity_np(pthread_self(),sizeof(s),&s)!=0){perror("pthread_setaffinity_np");std::exit(8);}}
static void* al64(size_t b){void*p=nullptr;if(posix_memalign(&p,64,b)!=0)return nullptr;return p;}
static double clk(clockid_t id){timespec t;clock_gettime(id,&t);return(double)t.tv_sec*1e9+(double)t.tv_nsec;}
static double med5(std::vector<double>&v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static uint64_t ord(double x){uint64_t u;std::memcpy(&u,&x,8);return(u>>63)?~u:(u|(UINT64_C(1)<<63));}
static uint64_t ulp(double a,double b){uint64_t A=ord(a),B=ord(b);return A>B?A-B:B-A;}

static void fill(double*x,size_t n,int c){
    const double e=0x1p-20; double lo,hi;
    if(c<2){lo=e;hi=1.0-e;} else if(c<4){lo=1.0+e;hi=500.0-e;} else {lo=1000.0+e;hi=10000.0-e;}
    uint64_t seed=UINT64_C(0xd1b54a32d192ed03)^((uint64_t)n*UINT64_C(0x94d049bb133111eb))^((uint64_t)c<<58);
    for(size_t i=0;i<n;++i){double q=u01(seed+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));double v=lo+(hi-lo)*q;x[i]=(c&1)?-v:v;}
}

struct B{std::vector<double*>x,y;size_t n=0;int c0=0,c1=0;~B(){for(auto*p:x)free(p);for(auto*p:y)free(p);}};
static bool alloc(B&b,size_t n){b.n=n;b.c0=SINE53_ENGINE_WIDE?2:0;b.c1=SINE53_ENGINE_WIDE?6:2;int nc=b.c1-b.c0;b.x.resize(nc);b.y.resize(nc);for(int j=0;j<nc;++j){b.x[j]=(double*)al64(n*sizeof(double));b.y[j]=(double*)al64(n*sizeof(double));if(!b.x[j]||!b.y[j])return false;fill(b.x[j],n,b.c0+j);}return true;}
static size_t reps_for(size_t n,size_t cases){size_t r=4000000/(n*cases);if(r<1)r=1;if(r>20000)r=20000;return r;}

static void ours_call(Sine53BatchX50X67Direct&d,B&b){for(size_t j=0;j<b.x.size();++j)d.run(b.y[j],b.x[j],b.n);}
static void ipp_call(B&b){for(size_t j=0;j<b.x.size();++j){IppStatus st=ippsSin_64f_A53((const Ipp64f*)b.x[j],(Ipp64f*)b.y[j],(int)b.n);if(st!=ippStsNoErr){std::fprintf(stderr,"ippsSin_64f_A53 failed status=%d\n",(int)st);std::exit(9);}}}

static int native(const std::string&stack,size_t n){
    pin0(); B b;if(!alloc(b,n))return 10; size_t cases=b.x.size(),reps=reps_for(n,cases); Sine53BatchX50X67Direct direct(sine53_engine_eval);
    for(int w=0;w<3;++w){if(stack=="ours")ours_call(direct,b);else ipp_call(b);} 
    std::vector<double>wv,cv;wv.reserve(5);cv.reserve(5);
    for(int t=0;t<5;++t){double w0=clk(CLOCK_MONOTONIC_RAW),c0=clk(CLOCK_PROCESS_CPUTIME_ID);for(size_t r=0;r<reps;++r){if(stack=="ours")ours_call(direct,b);else ipp_call(b);}double c1=clk(CLOCK_PROCESS_CPUTIME_ID),w1=clk(CLOCK_MONOTONIC_RAW);double den=(double)reps*n*cases;wv.push_back((w1-w0)/den);cv.push_back((c1-c0)/den);g_sink+=b.y[t%cases][(n*7u/11u)%n];}
    rusage ru{};getrusage(RUSAGE_SELF,&ru);double mw=med5(wv),mc=med5(cv);
    std::printf("SINEIPP_NATIVE engine=%s stack=%s n=%zu cases=%zu reps=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f maxrss_kib=%ld\n",SINE53_ENGINE_WIDE?"wide":"unit",stack.c_str(),n,cases,reps,mw,mc,mw?mc/mw:0.0,ru.ru_maxrss);return 0;
}

static int paircheck(size_t n){
    pin0();B b;if(!alloc(b,n))return 20;Sine53BatchX50X67Direct direct(sine53_engine_eval);uint64_t mx=0;double wx=0,wo=0,wi=0;
    for(size_t j=0;j<b.x.size();++j){std::vector<double>o(n),p(n);direct.run(o.data(),b.x[j],n);IppStatus st=ippsSin_64f_A53((const Ipp64f*)b.x[j],(Ipp64f*)p.data(),(int)n);if(st!=ippStsNoErr)return 21;for(size_t i=0;i<n;++i){uint64_t u=ulp(o[i],p[i]);if(u>mx){mx=u;wx=b.x[j][i];wo=o[i];wi=p[i];}}}
    std::printf("SINEIPP_PAIR engine=%s n=%zu pair_max_ulp=%llu x=%.17g ours=%.17g ipp=%.17g\n",SINE53_ENGINE_WIDE?"wide":"unit",n,(unsigned long long)mx,wx,wo,wi);return 0;
}

static int sde(const std::string&stack,size_t n){
    pin0();B b;if(!alloc(b,n))return 30;Sine53BatchX50X67Direct direct(sine53_engine_eval);if(stack!="noop"){if(stack=="ours")ours_call(direct,b);else ipp_call(b);}sine53_profile_start();if(stack=="noop"){for(size_t j=0;j<b.x.size();++j)asm volatile(""::"r"(b.x[j]),"r"(b.y[j]),"r"(b.n):"memory");}else if(stack=="ours")ours_call(direct,b);else ipp_call(b);sine53_profile_stop();if(stack!="noop")g_sink+=b.y[0][(n*5u/13u)%n];std::printf("SINEIPP_SDE engine=%s stack=%s n=%zu cases=%zu sink=%.17g\n",SINE53_ENGINE_WIDE?"wide":"unit",stack.c_str(),n,b.x.size(),(double)g_sink);return 0;
}

int main(int argc,char**argv){if(argc<3||argc>4)return 2;if(!sine53_engine_init())return 3;std::string mode=argv[1];size_t n=(size_t)std::strtoull(argv[argc-1],nullptr,10);int rc=2;if(mode=="native"&&argc==4)rc=native(argv[2],n);else if(mode=="sde"&&argc==4)rc=sde(argv[2],n);else if(mode=="pair"&&argc==3)rc=paircheck(n);sine53_engine_cleanup();return rc;}

// clean-X67 rebenchmark trigger 2026-09-12
