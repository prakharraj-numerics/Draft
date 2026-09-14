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

static int make_refs(const char*path){
 FILE*f=std::fopen(path,"wb");if(!f)return 10;
 mpfr_t a,s;mpfr_init2(a,256);mpfr_init2(s,256);
 const int OFF=512;const double R=.025;uint64_t n=0;
 int kmax=(int)std::floor(10000.0/3.141592653589793238462643383279502884);
 for(int k=1;k<=kmax;k++){
   double c=(double)k*3.141592653589793238462643383279502884;
   for(int side=-1;side<=1;side+=2)for(int j=0;j<OFF;j++){
     double rr=R*((double)j+.5)/(double)OFF;double x=c+side*rr;
     if(x<1||x>10000)continue;
     mpfr_set_d(a,x,MPFR_RNDN);mpfr_sin(s,a,MPFR_RNDN);
     Pair p{x,mpfr_get_d(s,MPFR_RNDN)};
     if(std::fwrite(&p,sizeof(p),1,f)!=1)return 11;++n;
   }
 }
 std::fclose(f);mpfr_clear(a);mpfr_clear(s);
 std::printf("REFCACHE total=%llu path=%s\n",(unsigned long long)n,path);return 0;
}

struct H { uint64_t m3=0,m2=0,m1=0,z=0,p1=0,p2=0,p3=0; };
static void add(H&h,double got,double ref){
 uint64_t u=ord(got),v=ord(ref);
 if(u==v){h.z++;return;}
 uint64_t d=u>v?u-v:v-u;
 if(u>v){if(d==1)h.p1++;else if(d==2)h.p2++;else h.p3++;}
 else {if(d==1)h.m1++;else if(d==2)h.m2++;else h.m3++;}
}
static void show(const char*tag,const H&h){
 std::printf("SIGNED tag=%s mge3=%llu m2=%llu m1=%llu exact=%llu p1=%llu p2=%llu pge3=%llu total=%llu\n",
 tag,(unsigned long long)h.m3,(unsigned long long)h.m2,(unsigned long long)h.m1,
 (unsigned long long)h.z,(unsigned long long)h.p1,(unsigned long long)h.p2,
 (unsigned long long)h.p3,(unsigned long long)(h.m3+h.m2+h.m1+h.z+h.p1+h.p2+h.p3));
}

int main(int argc,char**argv){
 if(argc!=2)return 2;const char*refpath=argv[1];
 FILE*rf=std::fopen(refpath,"rb");if(!rf){int e=make_refs(refpath);if(e)return e;rf=std::fopen(refpath,"rb");if(!rf)return 12;}
 std::fseek(rf,0,SEEK_END);long nb=std::ftell(rf);std::rewind(rf);if(nb<0||nb%(long)sizeof(Pair))return 13;
 std::vector<Pair> refs((size_t)nb/sizeof(Pair));if(!refs.empty()&&std::fread(refs.data(),sizeof(Pair),refs.size(),rf)!=refs.size())return 14;std::fclose(rf);
 if(!sine53_engine_init())return 3;
 H all,posref,negref,j2,j510;
 const size_t CH=8192;std::vector<double>x(CH),y(CH);
 const double VINV=0x1.45f306dc9c883p+7;
 for(size_t b=0;b<refs.size();b+=CH){
   size_t n=std::min(CH,refs.size()-b);
   for(size_t i=0;i<n;i++)x[i]=refs[b+i].x;
   sine53_engine_eval(y.data(),x.data(),n);
   for(size_t i=0;i<n;i++){
     const Pair&p=refs[b+i]; add(all,y[i],p.r); add(p.r>=0?posref:negref,y[i],p.r);
     long long q=llrint(p.x*VINV); int j=(int)(q&511LL);
     if(j==2)add(j2,y[i],p.r); else if(j==510)add(j510,y[i],p.r);
   }
 }
 sine53_engine_cleanup();
 show("all",all);show("ref_positive",posref);show("ref_negative",negref);show("j2",j2);show("j510",j510);
 return 0;
}
