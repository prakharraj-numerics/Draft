#define _GNU_SOURCE
#include <mkl.h>
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
#include "sine53_batch_production.hpp"

extern "C" int sine53_engine_init(void);
extern "C" void sine53_engine_eval(double *, const double *, size_t);
extern "C" void sine53_engine_cleanup(void);

#ifndef SINE53_ENGINE_WIDE
#define SINE53_ENGINE_WIDE 0
#endif

static volatile double g_sink = 0.0;
extern "C" __attribute__((noinline)) void sine53_profile_start(void){ asm volatile("" ::: "memory"); }
extern "C" __attribute__((noinline)) void sine53_profile_stop(void){ asm volatile("" ::: "memory"); }

static uint64_t mix64(uint64_t x){x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);} 
static double u01(uint64_t x){return ((double)(mix64(x)>>11)+0.5)*0x1p-53;}
static void unpin_current(){cpu_set_t set;CPU_ZERO(&set);long n=sysconf(_SC_NPROCESSORS_ONLN);for(int i=0;i<n;++i)CPU_SET(i,&set);(void)pthread_setaffinity_np(pthread_self(),sizeof(set),&set);} 
static void* al64(size_t bytes){void*p=nullptr;if(posix_memalign(&p,64,bytes)!=0)return nullptr;return p;}
static double clock_ns(clockid_t id){timespec t;clock_gettime(id,&t);return(double)t.tv_sec*1e9+(double)t.tv_nsec;}
static void fill(double*x,size_t n,int c){const double edge=0x1p-20;double lo,hi;if(c<2){lo=edge;hi=1.0-edge;}else if(c<4){lo=1.0+edge;hi=500.0-edge;}else{lo=1000.0+edge;hi=10000.0-edge;}uint64_t seed=UINT64_C(0xd1b54a32d192ed03)^((uint64_t)n*UINT64_C(0x94d049bb133111eb))^((uint64_t)c<<58);for(size_t i=0;i<n;++i){double q=u01(seed+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));double v=lo+(hi-lo)*q;x[i]=(c&1)?-v:v;}}

struct B{std::vector<double*>x,y;size_t n=0;int c0=0,c1=0;~B(){for(auto*p:x)free(p);for(auto*p:y)free(p);}};
static bool alloc(B&b,size_t n){b.n=n;b.c0=SINE53_ENGINE_WIDE?2:0;b.c1=SINE53_ENGINE_WIDE?6:2;int nc=b.c1-b.c0;b.x.resize(nc);b.y.resize(nc);for(int j=0;j<nc;++j){b.x[j]=(double*)al64(n*sizeof(double));b.y[j]=(double*)al64(n*sizeof(double));if(!b.x[j]||!b.y[j])return false;fill(b.x[j],n,b.c0+j);}return true;}
static size_t reps_for(size_t n,size_t cases){const size_t target=3000000;size_t r=target/(n*cases);if(r<1)r=1;if(r>20000)r=20000;return r;}
static double median11(std::vector<double>&v){std::sort(v.begin(),v.end());return v[v.size()/2];}

static void run_once(const std::string&stack,Sine53BatchProductionFrozen*prod,B&b){for(size_t j=0;j<b.x.size();++j){if(stack=="intel")vmdSin((MKL_INT)b.n,b.x[j],b.y[j],VML_HA);else prod->run(b.y[j],b.x[j],b.n);}}

static int native_mode(const std::string&stack,size_t n){
  unpin_current(); mkl_set_num_threads_local(0);
  B b;if(!alloc(b,n))return 10; size_t cases=b.x.size(),reps=reps_for(n,cases);
  Sine53BatchProductionFrozen*prod=(stack=="ours")?new Sine53BatchProductionFrozen(sine53_engine_eval):nullptr;
  for(int w=0;w<3;++w)run_once(stack,prod,b);
  std::vector<double> wall,cpu; wall.reserve(11);cpu.reserve(11);
  for(int t=0;t<11;++t){double w0=clock_ns(CLOCK_MONOTONIC_RAW),c0=clock_ns(CLOCK_PROCESS_CPUTIME_ID);for(size_t r=0;r<reps;++r)run_once(stack,prod,b);double c1=clock_ns(CLOCK_PROCESS_CPUTIME_ID),w1=clock_ns(CLOCK_MONOTONIC_RAW);double d=(double)reps*n*cases;wall.push_back((w1-w0)/d);cpu.push_back((c1-c0)/d);g_sink+=b.y[t%cases][(n*7u/11u)%n];}
  rusage ru{};getrusage(RUSAGE_SELF,&ru);double mw=median11(wall),mc=median11(cpu);
  std::printf("SINEDEF_NATIVE engine=%s stack=%s n=%zu cases=%zu reps=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f maxrss_kib=%ld\n",SINE53_ENGINE_WIDE?"wide":"unit",stack.c_str(),n,cases,reps,mw,mc,mw?mc/mw:0.0,ru.ru_maxrss);
  delete prod;return 0;
}

static int sde_mode(const std::string&stack,size_t n){
  unpin_current();mkl_set_num_threads_local(0);
  B b;if(!alloc(b,n))return 20;
  Sine53BatchProductionFrozen*prod=(stack=="ours")?new Sine53BatchProductionFrozen(sine53_engine_eval):nullptr;
  run_once(stack,prod,b); // warm/init before measured region
  sine53_profile_start();
  if(stack=="noop") { for(size_t j=0;j<b.x.size();++j) asm volatile("" : : "r"(b.x[j]),"r"(b.y[j]),"r"(b.n) : "memory"); }
  else run_once(stack,prod,b);
  sine53_profile_stop();
  if(stack!="noop")g_sink+=b.y[0][(n*5u/13u)%n];
  std::printf("SINEDEF_SDE engine=%s stack=%s n=%zu cases=%zu sink=%.17g\n",SINE53_ENGINE_WIDE?"wide":"unit",stack.c_str(),n,b.x.size(),(double)g_sink);
  delete prod;return 0;
}

int main(int argc,char**argv){if(argc!=4)return 2;if(!sine53_engine_init())return 3;std::string mode=argv[1],stack=argv[2];size_t n=(size_t)std::strtoull(argv[3],nullptr,10);int rc=2;if(mode=="native"&&(stack=="ours"||stack=="intel"))rc=native_mode(stack,n);else if(mode=="sde"&&(stack=="ours"||stack=="intel"||stack=="noop"))rc=sde_mode(stack,n);sine53_engine_cleanup();return rc;}
