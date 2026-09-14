#include <mpfr.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

struct Engine { void* h{}; int(*init)(){}; void(*eval)(double*,const double*,size_t){}; void(*cleanup)(){}; };
static Engine load_engine(const char* path){ Engine e; e.h=dlopen(path,RTLD_NOW|RTLD_LOCAL); if(!e.h){std::fprintf(stderr,"%s\n",dlerror());std::exit(2);} e.init=(int(*)())dlsym(e.h,"sine53_engine_init"); e.eval=(void(*)(double*,const double*,size_t))dlsym(e.h,"sine53_engine_eval"); e.cleanup=(void(*)())dlsym(e.h,"sine53_engine_cleanup"); if(!e.init||!e.eval||!e.cleanup||!e.init())std::exit(3); return e; }
static uint64_t bits(double x){uint64_t u;std::memcpy(&u,&x,8);return u;}
static uint64_t ord(double x){uint64_t u=bits(x);return (u>>63)?~u:(u|UINT64_C(0x8000000000000000));}
static uint64_t ulpd(double a,double b){if(a==b)return 0;uint64_t x=ord(a),y=ord(b);return x>y?x-y:y-x;}

int main(){
    const double tests[]={
      895.33860424636578,
      94.262843084256289,
      273.33298957324951,
      1042.9944787652489,
      1083.8341578712912,
      1957.1969155692536,
      2522.7136225122917,
      4058.922596133325,
      8774.48251487473,
      9437.329756188425,
      9814.350366806702
    };
    Engine pre=load_engine("/tmp/x67check/x67_pre.so");
    Engine post=load_engine("/tmp/x67check/x67_post.so");
    mpfr_t a,s; mpfr_init2(a,1024); mpfr_init2(s,1024);
    for(size_t t=0;t<sizeof(tests)/sizeof(tests[0]);++t){
      double x[32],ypre[32],ypost[32]; for(int i=0;i<32;i++)x[i]=tests[t];
      pre.eval(ypre,x,32); post.eval(ypost,x,32);
      mpfr_set_d(a,tests[t],MPFR_RNDN); mpfr_sin(s,a,MPFR_RNDN); double ref=mpfr_get_d(s,MPFR_RNDN);
      std::printf("CASE i=%zu x=%.17g ref=%.17g pre=%.17g pre_ulp=%llu post=%.17g post_ulp=%llu pre_eq_post=%d bits=0x%016llx\n",
        t,tests[t],ref,ypre[0],(unsigned long long)ulpd(ypre[0],ref),ypost[0],(unsigned long long)ulpd(ypost[0],ref),bits(ypre[0])==bits(ypost[0]),(unsigned long long)bits(tests[t]));
    }
    pre.cleanup(); post.cleanup(); mpfr_clear(s); mpfr_clear(a); return 0;
}
