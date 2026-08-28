#define _GNU_SOURCE
#include "sine_53_coeff_source.c"
#include <flint/arb.h>
#include <mkl.h>
#include <immintrin.h>
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CASES 150
#define STRESS 32768
#define TRIALS 9
#define ROUNDS 200000
#define LUTN SF_LUT_N
#define INVK (1.0/4096.0)

typedef struct { sine_fixed_ctx *ctx; int terms,deg; double *tab; } s53_kernel;
typedef struct { const s53_kernel *k; size_t count,padded; double *delta,*coef; } s53_plan;

static uint64_t now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec;}
static int cmpd(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return(x>y)-(x<y);}
static uint64_t mix64(uint64_t x){x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);x^=x>>31;return x;}
static int pin(void){cpu_set_t s,o;if(sched_getaffinity(0,sizeof(s),&s))return-1;int c=-1;for(int i=0;i<CPU_SETSIZE;i++)if(CPU_ISSET(i,&s)){c=i;break;}CPU_ZERO(&o);if(c>=0)CPU_SET(c,&o);return c<0||sched_setaffinity(0,sizeof(o),&o)?-1:c;}
static void *al64(size_t n){void*p=NULL;if(posix_memalign(&p,64,n?n:64))return NULL;return p;}

static double coeff_to_double(const mp_limb_t q[2],int neg){long double v=ldexpl((long double)q[1],64-103)+ldexpl((long double)q[0],-103);double d=(double)v;return neg?-d:d;}
static s53_kernel *kernel_create(int terms){if(terms<1||terms>3)return NULL;s53_kernel*k=calloc(1,sizeof(*k));if(!k)return NULL;k->ctx=s53_coeff_create_terms(terms);if(!k->ctx){free(k);return NULL;}k->terms=terms;k->deg=k->ctx->poly_deg;k->tab=al64((size_t)(k->deg+1)*LUTN*sizeof(double));if(!k->tab){s53_coeff_destroy(k->ctx);free(k);return NULL;}for(int a=0;a<LUTN;a++){size_t off=(size_t)a*(size_t)(k->deg+1);for(int j=0;j<=k->deg;j++)k->tab[(size_t)j*LUTN+(size_t)a]=coeff_to_double(k->ctx->coef+2*(off+(size_t)j),k->ctx->coef_sign[off+(size_t)j]!=0);}return k;}
static void kernel_destroy(s53_kernel*k){if(!k)return;free(k->tab);s53_coeff_destroy(k->ctx);free(k);}

static inline int anchor_of(double x){long j=(long)floor(x*4096.0+0.5);if(j<0)j=0;if(j>4096)j=4096;return(int)j;}
static inline double eval_scalar_one(const s53_kernel*k,double x){int a=anchor_of(x);double d=fma(-(double)a,INVK,x);double r=k->tab[(size_t)k->deg*LUTN+(size_t)a];for(int j=k->deg-1;j>=0;j--)r=fma(r,d,k->tab[(size_t)j*LUTN+(size_t)a]);return r;}

#if defined(__x86_64__) || defined(__i386__)
#define T512 __attribute__((target("avx512f,avx512dq,fma")))
T512 static void eval_e2e_avx512(const s53_kernel*k,const double*x,double*y,size_t n){const __m512d K=_mm512_set1_pd(4096.0),IK=_mm512_set1_pd(INVK);size_t i=0;for(;i+8<=n;i+=8){__m512d vx=_mm512_loadu_pd(x+i);__m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(vx,K),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);__m512i ji=_mm512_cvttpd_epi64(jd);__m512d d=_mm512_fnmadd_pd(jd,IK,vx);__m512d r=_mm512_i64gather_pd(ji,k->tab+(size_t)k->deg*LUTN,8);for(int j=k->deg-1;j>=0;j--){__m512d c=_mm512_i64gather_pd(ji,k->tab+(size_t)j*LUTN,8);r=_mm512_fmadd_pd(r,d,c);}_mm512_storeu_pd(y+i,r);}for(;i<n;i++)y[i]=eval_scalar_one(k,x[i]);}
T512 static void eval_hot_avx512(const s53_plan*p,double*y){size_t pd=p->padded;for(size_t i=0;i<pd;i+=8){__m512d d=_mm512_load_pd(p->delta+i);__m512d r=_mm512_load_pd(p->coef+(size_t)p->k->deg*pd+i);for(int j=p->k->deg-1;j>=0;j--)r=_mm512_fmadd_pd(r,d,_mm512_load_pd(p->coef+(size_t)j*pd+i));_mm512_store_pd(y+i,r);}}
#endif

static const char *backend(void){
#if defined(__x86_64__) || defined(__i386__)
if(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512dq")&&__builtin_cpu_supports("fma"))return "avx512-binary64-fma";
#endif
return "scalar-binary64-fma";}
static void eval_e2e(const s53_kernel*k,const double*x,double*y,size_t n){
#if defined(__x86_64__) || defined(__i386__)
if(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512dq")&&__builtin_cpu_supports("fma")){eval_e2e_avx512(k,x,y,n);return;}
#endif
for(size_t i=0;i<n;i++)y[i]=eval_scalar_one(k,x[i]);}

static s53_plan *plan_create(const s53_kernel*k,const double*x,size_t n){s53_plan*p=calloc(1,sizeof(*p));if(!p)return NULL;p->k=k;p->count=n;p->padded=(n+7U)&~(size_t)7U;size_t pd=p->padded;p->delta=al64(pd*sizeof(double));p->coef=al64((size_t)(k->deg+1)*pd*sizeof(double));if(!p->delta||!p->coef){free(p->coef);free(p->delta);free(p);return NULL;}for(size_t i=0;i<pd;i++){double xx=i<n?x[i]:0.0;int a=anchor_of(xx);p->delta[i]=fma(-(double)a,INVK,xx);for(int j=0;j<=k->deg;j++)p->coef[(size_t)j*pd+i]=k->tab[(size_t)j*LUTN+(size_t)a];}return p;}
static void plan_destroy(s53_plan*p){if(!p)return;free(p->coef);free(p->delta);free(p);}
static void eval_hot(const s53_plan*p,double*y){
#if defined(__x86_64__) || defined(__i386__)
if(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512dq")&&__builtin_cpu_supports("fma")){eval_hot_avx512(p,y);return;}
#endif
for(size_t i=0;i<p->count;i++){double r=p->coef[(size_t)p->k->deg*p->padded+i];for(int j=p->k->deg-1;j>=0;j--)r=fma(r,p->delta[i],p->coef[(size_t)j*p->padded+i]);y[i]=r;}}

static void make_stress(double*x){for(int i=0;i<STRESS;i++){uint64_t h=mix64(UINT64_C(2026082877)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));if(i<16384){int a=(int)(h&4095U);double b=((double)a+0.5)*INVK;switch(i&3){case 0:x[i]=b;break;case 1:x[i]=nextafter(b,0.0);break;case 2:x[i]=nextafter(b,1.0);break;default:{double z=(double)a*INVK;x[i]=nextafter(z,1.0);break;}}}else{uint64_t m=h&((UINT64_C(1)<<53)-1U);if(!m)m=1;x[i]=ldexp((double)m,-53);if(x[i]>=1.0)x[i]=nextafter(1.0,0.0);}}x[0]=0.0;x[1]=nextafter(0.0,1.0);x[2]=nextafter(1.0,0.0);x[3]=0.5;x[4]=INVK;x[5]=nextafter(INVK,0.0);x[6]=nextafter(INVK,1.0);}
static void make_bench(double*x){for(int i=0;i<CASES;i++){uint64_t h=mix64(UINT64_C(2026082879)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));uint64_t m=h&((UINT64_C(1)<<53)-1U);if(!m)m=1;x[i]=ldexp((double)m,-53);if(x[i]>=1.0)x[i]=nextafter(1.0,0.0);}}
static uint64_t dbits(double x){uint64_t u;memcpy(&u,&x,8);return u;}
static uint64_t ulpd(double a,double b){uint64_t x=dbits(a),y=dbits(b);return x>y?x-y:y-x;}

static int verify_profile(const s53_kernel*k,const double*x,double*ours,double*intel){eval_e2e(k,x,ours,STRESS);vmdSin(STRESS,x,intel,VML_HA);arb_t ax,ay;arf_t lo,hi;arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);uint64_t mo=0,mi=0;int eo=0,ei=0,o1=0,i1=0,unique=0;for(int i=0;i<STRESS;i++){arb_set_d(ax,x[i]);arb_sin(ay,ax,256);arb_get_lbound_arf(lo,ay,256);arb_get_ubound_arf(hi,ay,256);double rl=arf_get_d(lo,ARF_RND_NEAR),rh=arf_get_d(hi,ARF_RND_NEAR);if(dbits(rl)!=dbits(rh))continue;unique++;uint64_t uo=ulpd(ours[i],rl),ui=ulpd(intel[i],rl);if(uo>mo)mo=uo;if(ui>mi)mi=ui;if(!uo)eo++;if(!ui)ei++;if(uo<=1)o1++;if(ui<=1)i1++;}printf("S53_VERIFY terms=%d degree=%d cases=%d unique_ref=%d ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu reference=Arb256 binary64_inputs=1\n",k->terms,k->deg,STRESS,unique,eo,o1,(unsigned long)mo,ei,i1,(unsigned long)mi);arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);return unique==STRESS&&mo<=1;}

static uint64_t run_hot(const s53_plan*p,int rounds,volatile double*sink){double*y=al64(p->padded*sizeof(double));uint64_t t=now_ns();for(int r=0;r<rounds;r++)eval_hot(p,y);t=now_ns()-t;*sink+=y[CASES-1];free(y);return t;}
static uint64_t run_e2e(const s53_kernel*k,const double*x,int rounds,volatile double*sink){double y[CASES];uint64_t t=now_ns();for(int r=0;r<rounds;r++)eval_e2e(k,x,y,CASES);t=now_ns()-t;*sink+=y[CASES-1];return t;}
static uint64_t run_intel(const double*x,int rounds,volatile double*sink){double y[CASES];uint64_t t=now_ns();for(int r=0;r<rounds;r++)vmdSin(CASES,x,y,VML_HA);t=now_ns()-t;*sink+=y[CASES-1];return t;}

static int benchmark(const s53_kernel*k,const double*x){s53_plan*p=plan_create(k,x,CASES);if(!p)return 10;double h[CASES+8],e[CASES],ii[CASES];eval_hot(p,h);eval_e2e(k,x,e,CASES);vmdSin(CASES,x,ii,VML_HA);int same=0;for(int i=0;i<CASES;i++)if(dbits(h[i])==dbits(e[i]))same++;printf("S53_VECTOR_CHECK terms=%d degree=%d hot_vs_e2e_bit_identical=%d/%d backend=%s\n",k->terms,k->deg,same,CASES,backend());if(same!=CASES){plan_destroy(p);return 11;}if(strcmp(backend(),"avx512-binary64-fma")!=0){printf("S53_SKIP_TIMING reason=no_avx512 backend=%s\n",backend());plan_destroy(p);return 0;}volatile double sink=0;run_hot(p,5000,&sink);run_e2e(k,x,5000,&sink);run_intel(x,5000,&sink);double ht[TRIALS],et[TRIALS],it[TRIALS],calls=(double)ROUNDS*CASES;for(int t=0;t<TRIALS;t++){uint64_t a=run_hot(p,ROUNDS,&sink),b=run_e2e(k,x,ROUNDS,&sink),c=run_intel(x,ROUNDS,&sink);ht[t]=(double)a/calls;et[t]=(double)b/calls;it[t]=(double)c/calls;printf("S53_TRIAL trial=%d hot_ns=%.6f e2e_ns=%.6f intel_ha_ns=%.6f hot_over_intel=%.6fx e2e_over_intel=%.6fx\n",t+1,ht[t],et[t],it[t],ht[t]/it[t],et[t]/it[t]);}qsort(ht,TRIALS,sizeof(double),cmpd);qsort(et,TRIALS,sizeof(double),cmpd);qsort(it,TRIALS,sizeof(double),cmpd);double hm=ht[TRIALS/2],em=et[TRIALS/2],im=it[TRIALS/2];printf("S53_RESULT terms=%d degree=%d target=binary64_53bit accuracy_contract=le1ulp cases=%d hot_ns_per_input=%.6f e2e_ns_per_input=%.6f intel_ha_ns_per_input=%.6f hot_over_intel=%.6fx e2e_over_intel=%.6fx plan_build_excluded_hot=1 input_prepare_excluded_hot=1 e2e_includes_anchor_reduction=1 e2e_includes_coeff_gather=1 formula=unchanged_Mode5_secant_spine arithmetic=binary64_AVX512_FMA coefficient_source=Q103_secant_spine backend=%s sink=%.17g\n",k->terms,k->deg,CASES,hm,em,im,hm/im,em/im,backend(),(double)sink);plan_destroy(p);return 0;}

int main(void){int cpu=pin();mkl_set_num_threads_local(1);printf("S53_DOMAIN target=binary64_53bit domain=[0,1) vector_input=150 binary64_inputs=1 K=12 cpu_pin=%d intel=oneMKL_vmdSin_VML_HA backend=%s certification=offline_Arb256 formula=unchanged_Mode5_secant_spine\n",cpu,backend());double*x=al64(STRESS*sizeof(double)),*ours=al64(STRESS*sizeof(double)),*intel=al64(STRESS*sizeof(double));if(!x||!ours||!intel)return 2;make_stress(x);int win=0;for(int terms=1;terms<=3;terms++){s53_kernel*k=kernel_create(terms);if(!k)return 3;int ok=verify_profile(k,x,ours,intel);printf("S53_PROFILE terms=%d degree=%d accepted=%d criterion=ours_max_ulp_le_1\n",terms,k->deg,ok);kernel_destroy(k);if(ok&&!win)win=terms;}free(intel);free(ours);free(x);if(!win){fprintf(stderr,"no 53-bit Mode-5 FMA profile passed <=1 ULP verification\n");return 20;}printf("S53_WINNER terms=%d degree=%d selection=minimal_verified_profile\n",win,2*win+1);s53_kernel*k=kernel_create(win);double bx[CASES];make_bench(bx);int rc=benchmark(k,bx);kernel_destroy(k);flint_cleanup_master();return rc;}
