#include <mpfr.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

struct Engine {
    void* h{};
    int (*init)(){};
    void (*eval)(double*, const double*, size_t){};
    void (*cleanup)(){};
};

static Engine load_engine(const char* path) {
    Engine e;
    e.h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!e.h) { std::fprintf(stderr, "%s\n", dlerror()); std::exit(2); }
    e.init = (int(*)())dlsym(e.h, "sine53_engine_init");
    e.eval = (void(*)(double*, const double*, size_t))dlsym(e.h, "sine53_engine_eval");
    e.cleanup = (void(*)())dlsym(e.h, "sine53_engine_cleanup");
    if (!e.init || !e.eval || !e.cleanup || !e.init()) std::exit(3);
    return e;
}

static uint64_t bits(double x) { uint64_t u; std::memcpy(&u, &x, 8); return u; }
static uint64_t ord(double x) { uint64_t u=bits(x); return (u>>63) ? ~u : (u | UINT64_C(0x8000000000000000)); }
static uint64_t ulpd(double a,double b) { if(a==b) return 0; uint64_t x=ord(a),y=ord(b); return x>y?x-y:y-x; }

int main() {
    const double xbad = 895.33860424636578;
    double x[32], ypre[32], ypost[32];
    for (int i=0;i<32;i++) x[i]=xbad;

    Engine pre=load_engine("/tmp/x67check/x67_pre.so");
    Engine post=load_engine("/tmp/x67check/x67_post.so");
    pre.eval(ypre,x,32);
    post.eval(ypost,x,32);

    mpfr_t a,s;
    mpfr_init2(a,1024); mpfr_init2(s,1024);
    mpfr_set_d(a,xbad,MPFR_RNDN); mpfr_sin(s,a,MPFR_RNDN);
    double ref=mpfr_get_d(s,MPFR_RNDN);

    std::printf("INPUT dec=%.17g hex=%a bits=0x%016llx batch=32 path=raw_X67\n", xbad,xbad,(unsigned long long)bits(xbad));
    mpfr_printf("MPFR1024 exact=%.80RNf\n",s);
    std::printf("MPFR1024 rounded dec=%.17g hex=%a bits=0x%016llx\n",ref,ref,(unsigned long long)bits(ref));
    std::printf("PRE_CLEAN commit=670bb969604e6acd994cba8033eb5171467e0e38 out_dec=%.17g out_hex=%a bits=0x%016llx ulp=%llu\n",ypre[0],ypre[0],(unsigned long long)bits(ypre[0]),(unsigned long long)ulpd(ypre[0],ref));
    std::printf("POST_CLEAN commit=e5803aaa88c1ff37f3f52e45f2abf2f29b0de87a out_dec=%.17g out_hex=%a bits=0x%016llx ulp=%llu\n",ypost[0],ypost[0],(unsigned long long)bits(ypost[0]),(unsigned long long)ulpd(ypost[0],ref));
    std::printf("PRE_EQ_POST=%d\n", bits(ypre[0])==bits(ypost[0]));

    pre.cleanup(); post.cleanup();
    mpfr_clear(s); mpfr_clear(a);
    return 0;
}
