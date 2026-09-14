#include <mpfr.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
extern "C" int sine53_engine_init(void);
extern "C" void sine53_engine_eval(double*,const double*,size_t);
extern "C" void sine53_engine_cleanup(void);
static uint64_t ord(double x){uint64_t u;std::memcpy(&u,&x,8);return(u>>63)?~u:(u|UINT64_C(0x8000000000000000));}
static uint64_t ulpd(double a,double b){if(a==b)return 0;auto x=ord(a),y=ord(b);return x>y?x-y:y-x;}
struct R{double d,x; int q,j,sg; uint64_t u;};
static void stats(const char*name,std::vector<R> v){
  if(v.empty()){printf("DSTATS j=%s count=0\n",name);return;}
  std::sort(v.begin(),v.end(),[](const R&a,const R&b){return a.d<b.d;});
  size_t neg=0,pos=0,zero=0; double amin=INFINITY,amax=0;
  for(auto&r:v){if(r.d<0)neg++;else if(r.d>0)pos++;else zero++;double a=fabs(r.d);amin=std::min(amin,a);amax=std::max(amax,a);}
  auto Q=[&](double p)->double{size_t k=(size_t)llround(p*(double)(v.size()-1));return v[k].d;};
  std::vector<double> av;av.reserve(v.size());for(auto&r:v)av.push_back(fabs(r.d));std::sort(av.begin(),av.end());
  auto AQ=[&](double p)->double{size_t k=(size_t)llround(p*(double)(av.size()-1));return av[k];};
  printf("DSTATS j=%s count=%zu neg=%zu zero=%zu pos=%zu dmin=%.17g dmax=%.17g absmin=%.17g absmax=%.17g d_q00=%.17g d_q10=%.17g d_q25=%.17g d_q50=%.17g d_q75=%.17g d_q90=%.17g d_q100=%.17g abs_q10=%.17g abs_q25=%.17g abs_q50=%.17g abs_q75=%.17g abs_q90=%.17g\n",name,v.size(),neg,zero,pos,v.front().d,v.back().d,amin,amax,Q(0),Q(.10),Q(.25),Q(.50),Q(.75),Q(.90),Q(1),AQ(.10),AQ(.25),AQ(.50),AQ(.75),AQ(.90));
  printf("DLOW j=%s",name);for(size_t i=0;i<std::min<size_t>(8,v.size());i++)printf(" %.17g",v[i].d);printf("\n");
  printf("DHIGH j=%s",name);for(size_t z=std::min<size_t>(8,v.size());z>0;--z)printf(" %.17g",v[v.size()-z].d);printf("\n");
}
int main(){
 if(!sine53_engine_init())return 3;
 const double VINV=0x1.45f306dc9c883p+7, HHI=0x1.921fb54442d18p-8, HLO=0x1.1a62633145c07p-62;
 mpfr_t a,s;mpfr_init2(a,256);mpfr_init2(s,256);
 const int OFF=512; const double RAD=.025; const size_t CH=8192; std::vector<double>x,y; x.reserve(CH);
 std::vector<R> j2,j510,other; uint64_t total=0,ge2=0;
 auto flush=[&](){if(x.empty())return;y.resize(x.size());sine53_engine_eval(y.data(),x.data(),x.size());for(size_t i=0;i<x.size();++i){mpfr_set_d(a,x[i],MPFR_RNDN);mpfr_sin(s,a,MPFR_RNDN);double ref=mpfr_get_d(s,MPFR_RNDN);auto u=ulpd(y[i],ref);total++;if(u>=2){ge2++;double z=x[i]*VINV;int q=(int)std::nearbyint(z);int j=q&511;double N=(double)q;double d=std::fma(-N,HHI,x[i]);d=std::fma(-N,HLO,d);int sg=(q>>9)&1;R r{d,x[i],q,j,sg,u};if(j==2)j2.push_back(r);else if(j==510)j510.push_back(r);else other.push_back(r);printf("DCASE j=%d q=%d sg=%d x=%.17g d=%.17g ulp=%llu\n",j,q,sg,x[i],d,(unsigned long long)u);}}x.clear();};
 int kmax=(int)std::floor(10000.0/3.141592653589793238462643383279502884);for(int k=1;k<=kmax;k++){double c=(double)k*3.141592653589793238462643383279502884;for(int side=-1;side<=1;side+=2)for(int j=0;j<OFF;j++){double rr=RAD*((double)j+.5)/(double)OFF;double v=c+side*rr;if(v>=1&&v<=10000)x.push_back(v);if(x.size()>=CH)flush();}}flush();
 printf("DSUM total=%llu ge2=%llu j2=%zu j510=%zu other=%zu\n",(unsigned long long)total,(unsigned long long)ge2,j2.size(),j510.size(),other.size());stats("2",j2);stats("510",j510);stats("other",other);
 sine53_engine_cleanup();mpfr_clear(a);mpfr_clear(s);return 0;
}
