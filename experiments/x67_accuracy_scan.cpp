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
static uint64_t ulpd(double a,double b){if(a==b)return 0;auto x=ord(a),y=ord(b);return x>y?x-y:y-x;}

static int make_refs(const char*path){
 FILE*f=std::fopen(path,"wb");if(!f)return 10;
 mpfr_t a,s;mpfr_init2(a,256);mpfr_init2(s,256);
 const int OFF=512;const double R=.025;uint64_t n=0;
 int kmax=(int)std::floor(10000.0/3.141592653589793238462643383279502884);
 for(int k=1;k<=kmax;k++){double c=(double)k*3.141592653589793238462643383279502884;for(int side=-1;side<=1;side+=2)for(int j=0;j<OFF;j++){double rr=R*((double)j+.5)/(double)OFF;double x=c+side*rr;if(x<1||x>10000)continue;mpfr_set_d(a,x,MPFR_RNDN);mpfr_sin(s,a,MPFR_RNDN);Pair p{x,mpfr_get_d(s,MPFR_RNDN)};if(std::fwrite(&p,sizeof(p),1,f)!=1)return 11;++n;}}
 std::fclose(f);mpfr_clear(a);mpfr_clear(s);std::printf("REFCACHE total=%llu path=%s\n",(unsigned long long)n,path);return 0;
}

int main(int argc,char**argv){
 if(argc!=3)return 2;const char*tag=argv[1];const char*refpath=argv[2];
 FILE*rf=std::fopen(refpath,"rb");if(!rf){int e=make_refs(refpath);if(e)return e;rf=std::fopen(refpath,"rb");if(!rf)return 12;}
 std::fseek(rf,0,SEEK_END);long nb=std::ftell(rf);std::rewind(rf);if(nb<0||nb%(long)sizeof(Pair))return 13;
 std::vector<Pair> refs((size_t)nb/sizeof(Pair));if(!refs.empty()&&std::fread(refs.data(),sizeof(Pair),refs.size(),rf)!=refs.size())return 14;std::fclose(rf);
 if(!sine53_engine_init())return 3;
 uint64_t tot=0,g2=0,g3=0,mx=0,exact=0,one=0;const size_t CH=8192;std::vector<double>x(CH),y(CH);
 for(size_t b=0;b<refs.size();b+=CH){size_t n=std::min(CH,refs.size()-b);for(size_t i=0;i<n;i++)x[i]=refs[b+i].x;sine53_engine_eval(y.data(),x.data(),n);for(size_t i=0;i<n;i++){auto u=ulpd(y[i],refs[b+i].r);mx=std::max(mx,u);g2+=u>=2;g3+=u>=3;exact+=u==0;one+=u==1;tot++;}}
 double xi[32],yo[32];for(int i=0;i<32;i++)xi[i]=895.33860424636578;sine53_engine_eval(yo,xi,32);
 mpfr_t a,s;mpfr_init2(a,256);mpfr_init2(s,256);mpfr_set_d(a,xi[0],MPFR_RNDN);mpfr_sin(s,a,MPFR_RNDN);double rr=mpfr_get_d(s,MPFR_RNDN);auto off=ulpd(yo[0],rr);mpfr_clear(a);mpfr_clear(s);
 sine53_engine_cleanup();
 std::printf("SCAN tag=%s total=%llu exact=%llu one=%llu ge2=%llu ge3=%llu maxulp=%llu offender=%llu\n",tag,(unsigned long long)tot,(unsigned long long)exact,(unsigned long long)one,(unsigned long long)g2,(unsigned long long)g3,(unsigned long long)mx,(unsigned long long)off);return 0;
}
