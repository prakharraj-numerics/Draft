#include <mpfr.h>
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#pragma STDC FENV_ACCESS ON

static constexpr double VINV = 0x1.45f306dc9c883p+7;
static constexpr double HHI  = 0x1.921fb54442d18p-8;
static constexpr double HLO  = 0x1.1a62633145c07p-62;
static constexpr double C0_BASE = 0x1.921d1fcdec784p-7;
static constexpr double C1_BASE = 0x1.fff62169b92dbp-1;
static constexpr double C24 = 1.0/24.0;
static constexpr double C120 = 1.0/120.0;
static constexpr double MH = -0.5;
static constexpr double M6 = -1.0/6.0;

struct Pt { double d, ref; int ji; bool sg; };

static uint64_t ord(double x){
    uint64_t u; std::memcpy(&u,&x,8);
    return (u>>63)?~u:(u|UINT64_C(0x8000000000000000));
}
static uint64_t ulpd(double a,double b){
    if(a==b) return 0;
    uint64_t x=ord(a),y=ord(b); return x>y?x-y:y-x;
}
static double step(double x,int n){
    if(n>0) while(n--) x=std::nextafter(x,INFINITY);
    else while(n++) x=std::nextafter(x,-INFINITY);
    return x;
}
static inline double eval_pt(const Pt&p,double c0,double c1mag){
    double c1 = p.ji==510 ? -c1mag : c1mag;
    double z= p.d*p.d;
    double ec=std::fma(z,C24,MH);
    double oc=std::fma(z,C120,M6);
    double cd=c1*p.d;
    double base=std::fma(c1,p.d,c0);
    double ep=c0*ec;
    double inner=std::fma(cd,oc,ep);
    double y=std::fma(z,inner,base);
    return p.sg ? -y : y;
}

struct Score { uint64_t ge2=0, ge3=0, maxu=0, sumu=0; int a=0,b=0; };
static bool better(const Score&a,const Score&b){
    if(a.ge2!=b.ge2) return a.ge2<b.ge2;
    if(a.ge3!=b.ge3) return a.ge3<b.ge3;
    if(a.maxu!=b.maxu) return a.maxu<b.maxu;
    if(a.sumu!=b.sumu) return a.sumu<b.sumu;
    return std::abs(a.a)+std::abs(a.b) < std::abs(b.a)+std::abs(b.b);
}

int main(){
    std::fesetround(FE_TONEAREST);
    mpfr_t a,s; mpfr_init2(a,256); mpfr_init2(s,256);
    const int OFF=512; const double R=.025;
    const double PI=3.141592653589793238462643383279502884;
    const int kmax=(int)std::floor(10000.0/PI);
    std::vector<Pt> pts; pts.reserve(900000);
    uint64_t all=0, relevant=0, base_ge2=0,base_ge3=0,base_max=0;
    for(int k=1;k<=kmax;k++){
        double c=(double)k*PI;
        for(int side=-1;side<=1;side+=2) for(int j=0;j<OFF;j++){
            double rr=R*((double)j+.5)/(double)OFF;
            double x=c+side*rr;
            if(x<1.0 || x>10000.0) continue;
            ++all;
            double qd=std::nearbyint(x*VINV);
            int32_t q=(int32_t)qd;
            int ji=q&511;
            if(ji!=2 && ji!=510) continue;
            double d=std::fma(-qd,HHI,x);
            d=std::fma(-qd,HLO,d);
            mpfr_set_d(a,x,MPFR_RNDN); mpfr_sin(s,a,MPFR_RNDN);
            double ref=mpfr_get_d(s,MPFR_RNDN);
            Pt p{d,ref,ji,((q>>9)&1)!=0};
            uint64_t u=ulpd(eval_pt(p,C0_BASE,C1_BASE),ref);
            base_ge2+=u>=2; base_ge3+=u>=3; base_max=std::max(base_max,u);
            pts.push_back(p); ++relevant;
        }
    }
    mpfr_clear(s); mpfr_clear(a);
    std::printf("COEFF_SEARCH_BASE all=%llu relevant=%llu ge2=%llu ge3=%llu maxulp=%llu\n",
        (unsigned long long)all,(unsigned long long)relevant,
        (unsigned long long)base_ge2,(unsigned long long)base_ge3,(unsigned long long)base_max);
    if(base_ge2!=52 || base_ge3!=0 || base_max!=2){
        std::fprintf(stderr,"baseline reproduction failed\n"); return 4;
    }

    std::vector<Score> top;
    Score best; best.ge2=UINT64_MAX; best.ge3=UINT64_MAX; best.maxu=UINT64_MAX; best.sumu=UINT64_MAX;
    const int LIM=24;
    for(int da=-LIM;da<=LIM;da++){
        double c0=step(C0_BASE,da);
        for(int db=-LIM;db<=LIM;db++){
            double c1=step(C1_BASE,db);
            Score sc; sc.a=da; sc.b=db;
            for(const Pt&p:pts){
                uint64_t u=ulpd(eval_pt(p,c0,c1),p.ref);
                sc.sumu += std::min<uint64_t>(u,1000000);
                sc.maxu=std::max(sc.maxu,u); sc.ge2+=u>=2; sc.ge3+=u>=3;
                if(sc.ge2>best.ge2+32) break;
            }
            if(better(sc,best)){
                best=sc;
                std::printf("COEFF_SEARCH_BEST dc0=%d dc1=%d ge2=%llu ge3=%llu maxulp=%llu sumulp=%llu c0=%a c1mag=%a\n",
                    da,db,(unsigned long long)sc.ge2,(unsigned long long)sc.ge3,
                    (unsigned long long)sc.maxu,(unsigned long long)sc.sumu,c0,c1);
                std::fflush(stdout);
            }
            top.push_back(sc);
        }
    }
    std::sort(top.begin(),top.end(),better);
    if(top.size()>12) top.resize(12);
    std::puts("COEFF_SEARCH_TOP_BEGIN");
    for(const auto&sc:top){
        double c0=step(C0_BASE,sc.a),c1=step(C1_BASE,sc.b);
        // Re-score without early exit for final reporting.
        Score full; full.a=sc.a; full.b=sc.b;
        for(const Pt&p:pts){ uint64_t u=ulpd(eval_pt(p,c0,c1),p.ref); full.sumu+=u; full.maxu=std::max(full.maxu,u); full.ge2+=u>=2; full.ge3+=u>=3; }
        std::printf("COEFF_SEARCH_TOP dc0=%d dc1=%d ge2=%llu ge3=%llu maxulp=%llu sumulp=%llu c0=%a c1mag=%a\n",
            full.a,full.b,(unsigned long long)full.ge2,(unsigned long long)full.ge3,
            (unsigned long long)full.maxu,(unsigned long long)full.sumu,c0,c1);
    }
    std::puts("COEFF_SEARCH_TOP_END");
    return 0;
}
