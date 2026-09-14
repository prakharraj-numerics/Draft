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
int main(int argc,char**argv){if(argc!=2)return 2;const char*tag=argv[1];if(!sine53_engine_init())return 3;
 mpfr_t a,s;mpfr_init2(a,256);mpfr_init2(s,256);uint64_t tot=0,g2=0,g3=0,mx=0,exact=0,one=0;const int OFF=512;const double R=.025;const size_t CH=8192;std::vector<double>x;x.reserve(CH);std::vector<double>y;
 auto flush=[&](){if(x.empty())return;y.resize(x.size());sine53_engine_eval(y.data(),x.data(),x.size());for(size_t i=0;i<x.size();++i){mpfr_set_d(a,x[i],MPFR_RNDN);mpfr_sin(s,a,MPFR_RNDN);double r=mpfr_get_d(s,MPFR_RNDN);auto u=ulpd(y[i],r);mx=std::max(mx,u);g2+=u>=2;g3+=u>=3;exact+=u==0;one+=u==1;tot++;}x.clear();};
 int kmax=(int)std::floor(10000.0/3.141592653589793238462643383279502884);for(int k=1;k<=kmax;k++){double c=(double)k*3.141592653589793238462643383279502884;for(int side=-1;side<=1;side+=2)for(int j=0;j<OFF;j++){double rr=R*((double)j+.5)/(double)OFF;double v=c+side*rr;if(v>=1&&v<=10000)x.push_back(v);if(x.size()>=CH)flush();}}flush();
 double xi[32],yo[32];for(int i=0;i<32;i++)xi[i]=895.33860424636578;sine53_engine_eval(yo,xi,32);mpfr_set_d(a,xi[0],MPFR_RNDN);mpfr_sin(s,a,MPFR_RNDN);double rr=mpfr_get_d(s,MPFR_RNDN);auto off=ulpd(yo[0],rr);
 sine53_engine_cleanup();
 printf("SCAN tag=%s total=%llu exact=%llu one=%llu ge2=%llu ge3=%llu maxulp=%llu offender=%llu\n",tag,(unsigned long long)tot,(unsigned long long)exact,(unsigned long long)one,(unsigned long long)g2,(unsigned long long)g3,(unsigned long long)mx,(unsigned long long)off);return 0;}
