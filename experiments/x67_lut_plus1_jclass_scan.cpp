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
static int lut_j(double x){
    const double VINV=0x1.45f306dc9c883p+7;
    long long q=llrint(std::fabs(x)*VINV);
    return (int)(q & 511LL);
}

int main(){
    if(!sine53_engine_init()) return 3;
    mpfr_t a,s; mpfr_init2(a,256); mpfr_init2(s,256);
    const int OFF=512; const double R=.025; const size_t CH=8192;
    std::vector<double>x; x.reserve(CH); std::vector<double>y;
    uint64_t total=0,exact=0,one=0,ge2=0,ge3=0,maxu=0;
    uint64_t ge2_j2=0,ge2_j510=0,ge2_other=0;
    uint64_t one_j2=0,one_j510=0,one_other=0;
    auto flush=[&](){
        if(x.empty()) return;
        y.resize(x.size()); sine53_engine_eval(y.data(),x.data(),x.size());
        for(size_t i=0;i<x.size();++i){
            mpfr_set_d(a,x[i],MPFR_RNDN); mpfr_sin(s,a,MPFR_RNDN); double r=mpfr_get_d(s,MPFR_RNDN);
            uint64_t u=ulpd(y[i],r); int j=lut_j(x[i]);
            ++total; exact+=u==0; one+=u==1; ge2+=u>=2; ge3+=u>=3; maxu=std::max(maxu,u);
            if(u==1){ if(j==2)++one_j2; else if(j==510)++one_j510; else ++one_other; }
            if(u>=2){ if(j==2)++ge2_j2; else if(j==510)++ge2_j510; else ++ge2_other; }
        }
        x.clear();
    };
    int kmax=(int)std::floor(10000.0/3.141592653589793238462643383279502884);
    for(int k=1;k<=kmax;k++){
        double c=(double)k*3.141592653589793238462643383279502884;
        for(int side=-1;side<=1;side+=2) for(int j=0;j<OFF;j++){
            double rr=R*((double)j+.5)/(double)OFF; double v=c+side*rr;
            if(v>=1&&v<=10000) x.push_back(v);
            if(x.size()>=CH) flush();
        }
    }
    flush();
    mpfr_clear(a); mpfr_clear(s); sine53_engine_cleanup();
    std::printf("JCLASS total=%llu exact=%llu one=%llu ge2=%llu ge3=%llu maxulp=%llu ge2_j2=%llu ge2_j510=%llu ge2_other=%llu one_j2=%llu one_j510=%llu one_other=%llu\n",
        (unsigned long long)total,(unsigned long long)exact,(unsigned long long)one,(unsigned long long)ge2,(unsigned long long)ge3,(unsigned long long)maxu,
        (unsigned long long)ge2_j2,(unsigned long long)ge2_j510,(unsigned long long)ge2_other,
        (unsigned long long)one_j2,(unsigned long long)one_j510,(unsigned long long)one_other);
    return 0;
}
