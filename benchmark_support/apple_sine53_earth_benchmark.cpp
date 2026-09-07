#include <Accelerate/Accelerate.h>
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

#include "apple_cos53_coeff_aos.h"
#include "apple_sine53_earth_prepare.hpp"

extern "C" void sine53_earth_raw(const double*, double*, size_t,
                                  const double*, const double*, const double*,
                                  const uint64_t*);

static volatile double g_sink = 0.0;
static uint64_t mix64(uint64_t x){x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);x^=x>>31;return x;}
static double unit52(uint64_t h){return ((double)(h>>12)+0.5)*0x1p-52;}
static uint64_t ticks(){return mach_absolute_time();}
static double ticks_ns(uint64_t t){static mach_timebase_info_data_t tb=[](){mach_timebase_info_data_t v{};mach_timebase_info(&v);return v;}();return (double)t*tb.numer/tb.denom;}
static uint64_t ordered_bits(double x){uint64_t u;std::memcpy(&u,&x,8);return (u>>63)?~u:(u|UINT64_C(0x8000000000000000));}
static uint64_t ulpd(double a,double b){if(a==b)return 0;uint64_t x=ordered_bits(a),y=ordered_bits(b);return x>y?x-y:y-x;}

struct B{size_t n;std::vector<double>x,y,apple,d,c0,c1;std::vector<uint64_t>sign;explicit B(size_t nn):n(nn),x(nn),y(nn),apple(nn),d(nn),c0(nn),c1(nn),sign(nn){}};
static apple_sine53_earth::PrepareTables tables(){return {opt_cos53_coeff_aos,OPT_COS53_LUTN};}
static void fill(B&b,uint64_t seed){for(size_t i=0;i<b.n;++i){const int band=(int)(i%3);const double lo[3]={0.0,1.0,1000.0},hi[3]={1.0,500.0,10000.0};double u=unit52(mix64(seed+i*UINT64_C(0x9e3779b97f4a7c15)));double v=std::fma(hi[band]-lo[band],u,lo[band]);b.x[i]=(i&1)?-v:v;}}
static size_t prep(B&b){return apple_sine53_earth::prepare(b.x.data(),b.n,tables(),b.d.data(),b.c0.data(),b.c1.data(),b.sign.data());}
static void earth(B&b){sine53_earth_raw(b.x.data(),b.y.data(),b.n,b.d.data(),b.c0.data(),b.c1.data(),b.sign.data());}
static void apple(B&b){int n=(int)b.n;vvsin(b.apple.data(),b.x.data(),&n);}

static int validate(){
    B b(9600);fill(b,UINT64_C(202609070053));size_t repairs=prep(b);earth(b);apple(b);
    mpfr_t z,r;mpfr_init2(z,256);mpfr_init2(r,256);
    uint64_t earth_mpfr_max=0,apple_mpfr_max=0,earth_apple_max=0;size_t e_exact=0,e_le1=0,e_le2=0,e_le3=0,a_le3=0;size_t worst=0;double worst_ref=0.0;
    for(size_t i=0;i<b.n;++i){
        mpfr_set_d(z,b.x[i],MPFR_RNDN);mpfr_sin(r,z,MPFR_RNDN);double ref=mpfr_get_d(r,MPFR_RNDN);
        uint64_t ue=ulpd(b.y[i],ref),ua=ulpd(b.apple[i],ref),uv=ulpd(b.y[i],b.apple[i]);
        if(ue>earth_mpfr_max){earth_mpfr_max=ue;worst=i;worst_ref=ref;}
        apple_mpfr_max=std::max(apple_mpfr_max,ua);earth_apple_max=std::max(earth_apple_max,uv);
        e_exact+=(ue==0);e_le1+=(ue<=1);e_le2+=(ue<=2);e_le3+=(ue<=3);a_le3+=(ua<=3);
    }
    mpfr_clear(r);mpfr_clear(z);
    std::printf("SINE53_EARTH_VALIDATE cases=%zu repairs=%zu earth_exact=%zu earth_le1=%zu earth_le2=%zu earth_le3=%zu earth_mpfr_maxulp=%llu apple_vvsin_mpfr_maxulp=%llu earth_vs_vvsin_maxulp=%llu apple_le3=%zu worst_i=%zu worst_x=%.17g worst_out=%.17g worst_ref=%.17g reference=MPFR256\n",b.n,repairs,e_exact,e_le1,e_le2,e_le3,(unsigned long long)earth_mpfr_max,(unsigned long long)apple_mpfr_max,(unsigned long long)earth_apple_max,a_le3,worst,b.x[worst],b.y[worst],worst_ref);
    return earth_mpfr_max<=3?0:5;
}

static double median7(double x[7]){std::sort(x,x+7);return x[3];}
static int bench(size_t n){B b(n);fill(b,UINT64_C(202609070054)+n);size_t repairs=prep(b);for(int w=0;w<20;++w)earth(b);size_t reps=12000000/(n?n:1);if(reps<3)reps=3;if(reps>20000)reps=20000;double e[7],full[7],av[7];for(int t=0;t<7;++t){uint64_t a=ticks();for(size_t r=0;r<reps;++r)earth(b);uint64_t c=ticks();e[t]=ticks_ns(c-a)/(double)(reps*n);a=ticks();for(size_t r=0;r<reps;++r){prep(b);earth(b);}c=ticks();full[t]=ticks_ns(c-a)/(double)(reps*n);a=ticks();for(size_t r=0;r<reps;++r)apple(b);c=ticks();av[t]=ticks_ns(c-a)/(double)(reps*n);g_sink+=b.y[(t*97)%n];}double em=median7(e),fm=median7(full),am=median7(av);std::printf("SINE53_EARTH_BENCH n=%zu repairs=%zu earth_ns_el=%.9f prep_plus_earth_ns_el=%.9f prep_est_ns_el=%.9f vvsin_ns_el=%.9f earth_speedup_vs_vvsin=%.6f full_speedup_vs_vvsin=%.6f reps=%zu sink=%.17g\n",n,repairs,em,fm,fm-em,am,am/em,am/fm,reps,(double)g_sink);return 0;}

int main(int argc,char**argv){if(argc==2&&std::string(argv[1])=="validate")return validate();if(argc==3&&std::string(argv[1])=="bench")return bench((size_t)std::strtoull(argv[2],nullptr,10));return 2;}
