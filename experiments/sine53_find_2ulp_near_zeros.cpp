#include <mpfr.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct Engine { void* h; int(*init)(); void(*eval)(double*,const double*,size_t); void(*cleanup)(); };
static Engine load(const char* p){ Engine e{}; e.h=dlopen(p,RTLD_NOW|RTLD_LOCAL); if(!e.h){std::fprintf(stderr,"dlopen: %s\n",dlerror());std::exit(2);} e.init=(int(*)())dlsym(e.h,"sine53_engine_init"); e.eval=(void(*)(double*,const double*,size_t))dlsym(e.h,"sine53_engine_eval"); e.cleanup=(void(*)())dlsym(e.h,"sine53_engine_cleanup"); if(!e.init||!e.eval||!e.cleanup||!e.init()) std::exit(3); return e; }
static uint64_t ord(double x){ uint64_t u; std::memcpy(&u,&x,8); return (u>>63)?~u:(u|UINT64_C(0x8000000000000000)); }
static uint64_t ulpd(double a,double b){ if(a==b)return 0; uint64_t x=ord(a),y=ord(b); return x>y?x-y:y-x; }
static uint64_t bits(double x){ uint64_t u; std::memcpy(&u,&x,8); return u; }

int main(int argc,char**argv){
  if(argc!=2){std::fprintf(stderr,"usage: scan engine.so\n");return 2;}
  Engine e=load(argv[1]);
  const int OFF=512;                 // per side per zero
  const double R=0.025;              // radians around k*pi; offender is ~0.0153 from zero
  const size_t CH=8192;
  std::vector<double> x; x.reserve(CH); std::vector<double> y(CH);
  mpfr_t a,s; mpfr_init2(a,256); mpfr_init2(s,256);
  uint64_t total=0, n2=0, n3=0, mx=0; int printed=0; const int CAP=200;
  auto flush=[&](){
    if(x.empty()) return;
    y.resize(x.size()); e.eval(y.data(),x.data(),x.size());
    for(size_t i=0;i<x.size();++i){
      mpfr_set_d(a,x[i],MPFR_RNDN); mpfr_sin(s,a,MPFR_RNDN); double ref=mpfr_get_d(s,MPFR_RNDN);
      uint64_t u=ulpd(y[i],ref); mx=std::max(mx,u); n2 += (u>=2); n3 += (u>=3); total++;
      if(u>=2 && printed<CAP){
        double q=std::nearbyint(x[i]/M_PI); double phase=x[i]-q*M_PI;
        std::printf("BAD ulp=%llu x=%.17g hex=%a out=%.17g ref=%.17g k=%.0f phase=%.17g bits=0x%016llx\n",
          (unsigned long long)u,x[i],x[i],y[i],ref,q,phase,(unsigned long long)bits(x[i])); printed++;
      }
    }
    x.clear();
  };
  // Positive domain [1,10000]. For each zero k*pi, sample both sides with a dense linear grid.
  int kmax=(int)std::floor(10000.0/M_PI);
  for(int k=1;k<=kmax;k++){
    double c=(double)k*M_PI;
    for(int side=-1;side<=1;side+=2){
      for(int j=0;j<OFF;j++){
        double r=R*((double)j+0.5)/(double)OFF;
        double v=c+(double)side*r;
        if(v>=1.0 && v<=10000.0) x.push_back(v);
        if(x.size()>=CH) flush();
      }
    }
  }
  flush();
  std::printf("SUMMARY total=%llu ge2=%llu ge3=%llu maxulp=%llu printed=%d kmax=%d radius=%.17g per_side=%d\n",
    (unsigned long long)total,(unsigned long long)n2,(unsigned long long)n3,(unsigned long long)mx,printed,kmax,R,OFF);
  mpfr_clear(s); mpfr_clear(a); e.cleanup(); dlclose(e.h); return 0;
}
