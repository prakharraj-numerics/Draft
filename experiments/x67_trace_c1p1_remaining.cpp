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
struct Pair{double x;double r;};
static uint64_t ord(double x){uint64_t u;std::memcpy(&u,&x,8);return(u>>63)?~u:(u|UINT64_C(0x8000000000000000));}
static uint64_t ulpd(double a,double b){auto x=ord(a),y=ord(b);return x>y?x-y:y-x;}
static long long sulp(double a,double b){return (long long)ord(a)-(long long)ord(b);}
static int make_refs(const char*path){FILE*f=std::fopen(path,"wb");if(!f)return 10;mpfr_t a,s;mpfr_init2(a,256);mpfr_init2(s,256);const int OFF=512;const double R=.025;int kmax=(int)std::floor(10000.0/M_PI);for(int k=1;k<=kmax;k++){double c=(double)k*M_PI;for(int side=-1;side<=1;side+=2)for(int j=0;j<OFF;j++){double rr=R*((double)j+.5)/(double)OFF;double x=c+side*rr;if(x<1||x>10000)continue;mpfr_set_d(a,x,MPFR_RNDN);mpfr_sin(s,a,MPFR_RNDN);Pair p{x,mpfr_get_d(s,MPFR_RNDN)};if(std::fwrite(&p,sizeof(p),1,f)!=1)return 11;}}std::fclose(f);mpfr_clear(a);mpfr_clear(s);return 0;}
int main(int argc,char**argv){if(argc!=2)return 2;const char*path=argv[1];FILE*f=std::fopen(path,"rb");if(!f){if(make_refs(path))return 10;f=std::fopen(path,"rb");}std::fseek(f,0,SEEK_END);long nb=std::ftell(f);std::rewind(f);std::vector<Pair> refs((size_t)nb/sizeof(Pair));if(std::fread(refs.data(),sizeof(Pair),refs.size(),f)!=refs.size())return 11;std::fclose(f);if(!sine53_engine_init())return 3;const size_t CH=8192;std::vector<double>x(CH),y(CH);uint64_t nbad=0,j2=0,j510=0;const double VINV=0x1.45f306dc9c883p+7,HHI=0x1.921fb54442d18p-8,HLO=0x1.1a62633145c07p-62;for(size_t b=0;b<refs.size();b+=CH){size_t n=std::min(CH,refs.size()-b);for(size_t i=0;i<n;i++)x[i]=refs[b+i].x;sine53_engine_eval(y.data(),x.data(),n);for(size_t i=0;i<n;i++){uint64_t u=ulpd(y[i],refs[b+i].r);if(u<2)continue;double xx=x[i];long long q=llrint(xx*VINV);int j=(int)(q&511);double N=(double)q;double d=std::fma(-N,HHI,xx);d=std::fma(-N,HLO,d);int sg=(int)((q>>9)&1);long long se=sulp(y[i],refs[b+i].r);printf("BAD x=%.17g ref=%.17g got=%.17g ulp=%llu signed=%lld q=%lld j=%d d=%.17g sg=%d q1024=%lld q2048=%lld\n",xx,refs[b+i].r,y[i],(unsigned long long)u,se,q,j,d,sg,q&1023,q&2047);nbad++;j2+=j==2;j510+=j==510;}}
sine53_engine_cleanup();printf("SUMMARY bad=%llu j2=%llu j510=%llu\n",(unsigned long long)nbad,(unsigned long long)j2,(unsigned long long)j510);return 0;}
