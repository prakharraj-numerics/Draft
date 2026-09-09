#define _GNU_SOURCE
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <time.h>
#include <sys/resource.h>
#include <vector>
#include <mpfr.h>

extern "C" int sine53_engine_init(void);
extern "C" void sine53_engine_eval(double *, const double *, size_t);
extern "C" void sine53_engine_cleanup(void);
extern "C" void svml_sin_high(const double *, double *, size_t);

#ifndef SINE53_ENGINE_WIDE
#define SINE53_ENGINE_WIDE 0
#endif

static volatile double g_sink=0.0;
extern "C" __attribute__((noinline)) void sine53_profile_start(void){asm volatile("":::"memory");}
extern "C" __attribute__((noinline)) void sine53_profile_stop(void){asm volatile("":::"memory");}

static uint64_t mix64(uint64_t x){x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
static double u01(uint64_t x){return ((double)(mix64(x)>>11)+0.5)*0x1p-53;}
static void* al64(size_t b){void*p=nullptr;return posix_memalign(&p,64,b)==0?p:nullptr;}
static double clk(clockid_t id){timespec t;clock_gettime(id,&t);return(double)t.tv_sec*1e9+t.tv_nsec;}
static double med5(std::vector<double>&v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t reps_for(size_t n,size_t cases){size_t r=4000000/(n*cases);if(r<1)r=1;if(r>20000)r=20000;return r;}

static uint64_t ord(double x){uint64_t u;std::memcpy(&u,&x,8);return (u>>63)?~u:(u|UINT64_C(0x8000000000000000));}
static uint64_t ulp(double a,double b){if(std::isnan(a)||std::isnan(b))return UINT64_MAX;if(a==b)return 0;uint64_t A=ord(a),B=ord(b);return A>B?A-B:B-A;}

struct B{std::vector<double*>x,y;size_t n=0;int c0=0,c1=0;~B(){for(auto*p:x)free(p);for(auto*p:y)free(p);}};
static void fill(double*x,size_t n,int c){const double e=0x1p-20;double lo,hi;if(c<2){lo=e;hi=1.0-e;}else if(c<4){lo=1.0+e;hi=500.0-e;}else{lo=1000.0+e;hi=10000.0-e;}uint64_t seed=UINT64_C(0xd1b54a32d192ed03)^((uint64_t)n*UINT64_C(0x94d049bb133111eb))^((uint64_t)c<<58);for(size_t i=0;i<n;i++){double q=u01(seed+i*UINT64_C(0x9e3779b97f4a7c15));double v=lo+(hi-lo)*q;x[i]=(c&1)?-v:v;}}
static bool alloc(B&b,size_t n){b.n=n;b.c0=SINE53_ENGINE_WIDE?2:0;b.c1=SINE53_ENGINE_WIDE?6:2;int nc=b.c1-b.c0;b.x.resize(nc);b.y.resize(nc);for(int j=0;j<nc;j++){b.x[j]=(double*)al64(n*8);b.y[j]=(double*)al64(n*8);if(!b.x[j]||!b.y[j])return false;fill(b.x[j],n,b.c0+j);}return true;}

static void call(const std::string&s,double*out,const double*in,size_t n){if(s=="ours")sine53_engine_eval(out,in,n);else svml_sin_high(in,out,n);}
static void once(const std::string&s,B&b){for(size_t j=0;j<b.x.size();j++)call(s,b.y[j],b.x[j],b.n);}

static int native(const std::string&s,size_t n){B b;if(!alloc(b,n))return 10;for(int w=0;w<3;w++)once(s,b);size_t reps=reps_for(n,b.x.size());std::vector<double>wv,cv;for(int t=0;t<5;t++){double w0=clk(CLOCK_MONOTONIC_RAW),c0=clk(CLOCK_PROCESS_CPUTIME_ID);for(size_t r=0;r<reps;r++)once(s,b);double c1=clk(CLOCK_PROCESS_CPUTIME_ID),w1=clk(CLOCK_MONOTONIC_RAW);double d=(double)reps*n*b.x.size();wv.push_back((w1-w0)/d);cv.push_back((c1-c0)/d);g_sink+=b.y[t%b.y.size()][(n*7/11)%n];}rusage ru{};getrusage(RUSAGE_SELF,&ru);double mw=med5(wv),mc=med5(cv);std::printf("SINESVML_NATIVE engine=%s stack=%s n=%zu cases=%zu reps=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f maxrss_kib=%ld\n",SINE53_ENGINE_WIDE?"wide":"unit",s.c_str(),n,b.x.size(),reps,mw,mc,mw?mc/mw:0.0,ru.ru_maxrss);return 0;}

static int sde(const std::string&s,size_t n){B b;if(!alloc(b,n))return 20;if(s!="noop")once(s,b);sine53_profile_start();if(s=="noop"){for(size_t j=0;j<b.x.size();j++)asm volatile(""::"r"(b.x[j]),"r"(b.y[j]),"r"(b.n):"memory");}else once(s,b);sine53_profile_stop();if(s!="noop")g_sink+=b.y[0][(n*5/13)%n];std::printf("SINESVML_SDE engine=%s stack=%s n=%zu cases=%zu sink=%.17g\n",SINE53_ENGINE_WIDE?"wide":"unit",s.c_str(),n,b.x.size(),(double)g_sink);return 0;}

static int accuracy_band(const char*name,double lo,double hi,uint64_t seed){const size_t half=20000,N=40000;std::vector<double>x(N),o(N),v(N);std::mt19937_64 rng(seed);std::uniform_real_distribution<double>d(lo,hi);for(size_t i=0;i<half;i++){double a=d(rng);x[i]=a;x[i+half]=-a;}sine53_engine_eval(o.data(),x.data(),N);svml_sin_high(x.data(),v.data(),N);mpfr_t z;mpfr_init2(z,256);uint64_t mo=0,mv=0;size_t go=0,gv=0;double wox=0,wvx=0,wor=0,wvr=0,wog=0,wvg=0;for(size_t i=0;i<N;i++){mpfr_set_d(z,x[i],MPFR_RNDN);mpfr_sin(z,z,MPFR_RNDN);double r=mpfr_get_d(z,MPFR_RNDN);uint64_t uo=ulp(o[i],r),uv=ulp(v[i],r);if(uo>mo){mo=uo;wox=x[i];wor=r;wog=o[i];}if(uv>mv){mv=uv;wvx=x[i];wvr=r;wvg=v[i];}if(uo>1)go++;if(uv>1)gv++;}mpfr_clear(z);std::printf("SINESVML_ACC band=%s N=%zu OURS_MAX_ULP=%llu SVML_MAX_ULP=%llu OURS_GT1=%zu SVML_GT1=%zu\n",name,N,(unsigned long long)mo,(unsigned long long)mv,go,gv);std::printf("OURS_WORST band=%s x=%.17g got=%.17g ref=%.17g ulp=%llu\n",name,wox,wog,wor,(unsigned long long)mo);std::printf("SVML_WORST band=%s x=%.17g got=%.17g ref=%.17g ulp=%llu\n",name,wvx,wvg,wvr,(unsigned long long)mv);return 0;}

int main(int argc,char**argv){if(argc<2)return 2;if(!sine53_engine_init())return 3;std::string m=argv[1];int rc=2;if(m=="native"&&argc==4)rc=native(argv[2],strtoull(argv[3],nullptr,10));else if(m=="sde"&&argc==4)rc=sde(argv[2],strtoull(argv[3],nullptr,10));else if(m=="accuracy"){if(SINE53_ENGINE_WIDE){rc=accuracy_band("1_500",1.0+0x1p-20,500.0-0x1p-20,UINT64_C(0x5A17C0DE2234));if(!rc)rc=accuracy_band("1k_10k",1000.0+0x1p-20,10000.0-0x1p-20,UINT64_C(0x5A17C0DE3234));}else rc=accuracy_band("0_1",0x1p-20,1.0-0x1p-20,UINT64_C(0x5A17C0DE1234));}sine53_engine_cleanup();return rc;}
