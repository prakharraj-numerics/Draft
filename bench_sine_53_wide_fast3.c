#define main s53w_base_main
#include "bench_sine_53_wide_intel.c"
#undef main

/*
   Fast wide 53-bit realization of the SAME secant-spine identity:

     sin(a+d) = sin(a) C(d^2) + cos(a) d T(d^2)

   The Mode-5 table supplies c0=sin(a), c1=cos(a).  C and T are universal
   local expansions obtained from the secant-spine C/T construction.  Since
   |d| <= 1/(2*4096), z^3 terms are already far below binary64 significance.

   Wide reduction uses a bounded Cody-Waite split of pi.  For |x|<=10000,
   q=round(|x|/pi)<4096, so q*PI_A and q*PI_B are exact binary64 products.
*/

static const double PI_A = 0x1.921fb54400000p+1;  /* 33 significant bits */
static const double PI_B = 0x1.0b4611a600000p-33; /* next 33-bit chunk */
static const double PI_C = 0x1.3198a2e037073p-68; /* remaining binary64 chunk */

static inline void f3_two_diff(double a,double b,double *h,double *l)
{
    double q=a-b,bv=a-q,av=q+bv,br=bv-b,ar=a-av;
    *h=q; *l=ar+br;
}
static inline void f3_two_sum(double a,double b,double *h,double *l)
{
    double q=a+b,bv=q-a,av=q-bv,br=b-bv,ar=a-av;
    *h=q; *l=ar+br;
}

static inline void f3_reduce_scalar(double x,int64_t *qout,double *rh,double *rl,int *rneg)
{
    double ax=fabs(x);
    int64_t q=(int64_t)nearbyint(ax*INVPI);
    double qd=(double)q;

    /* ax - q*PI_A is exact by Sterbenz for nearest-q geometry; q*PI_A exact. */
    double h=ax-qd*PI_A;
    double e1,e2;
    f3_two_diff(h,qd*PI_B,&h,&e1); /* q*PI_B exact for q<4096 */

    /* Retain the rounding error of q*PI_C with FMA. */
    double pc=qd*PI_C;
    double pce=fma(qd,PI_C,-pc);
    f3_two_diff(h,pc,&h,&e2);
    double lo=(e1+e2)-pce;
    double hn,ln; f3_two_sum(h,lo,&hn,&ln); h=hn; lo=ln;

    int neg=(h<0.0)||(h==0.0&&lo<0.0);
    if(neg){h=-h;lo=-lo;}
    *qout=q;*rh=h;*rl=lo;*rneg=neg;
}

static inline double f3_eval_scalar_one(const s53w_kernel *k,double x)
{
    int64_t q; double rh,rl; int rn;
    f3_reduce_scalar(x,&q,&rh,&rl,&rn);
    double r=rh+rl;
    long a=lround(r*KGRID); if(a<0)a=0;if(a>=LUTN)a=LUTN-1;
    double aa=(double)a*INVK;
    double d=rh-aa; /* exact for nearest anchor; a=0 trivial */
    double z=d*d;

    /* Universal C,T from the secant-spine identity, sufficient at this d. */
    double C=fma(z,fma(z,1.0/24.0,-0.5),1.0);
    double T=fma(z,fma(z,1.0/120.0,-1.0/6.0),1.0);
    double sa=k->tab[(size_t)a];
    double ca=k->tab[LUTN+(size_t)a];
    double dt=d*T;
    double y=fma(ca,dt,sa*C);

    /* First-order low-word correction; second order is negligible (<1e-40). */
    double deriv=fma(-sa,dt,ca*C);
    y=fma(rl,deriv,y);
    if(signbit(x)^(int)(q&1)^rn)y=-y;
    return y;
}

#if defined(__x86_64__) || defined(__i386__)
#define F3V __attribute__((target("avx512f,avx512dq,fma")))
F3V static inline void f3_two_diff_v(__m512d a,__m512d b,__m512d *h,__m512d *l)
{
    __m512d q=_mm512_sub_pd(a,b),bv=_mm512_sub_pd(a,q),av=_mm512_add_pd(q,bv),br=_mm512_sub_pd(bv,b),ar=_mm512_sub_pd(a,av);
    *h=q;*l=_mm512_add_pd(ar,br);
}
F3V static inline void f3_two_sum_v(__m512d a,__m512d b,__m512d *h,__m512d *l)
{
    __m512d q=_mm512_add_pd(a,b),bv=_mm512_sub_pd(q,a),av=_mm512_sub_pd(q,bv),br=_mm512_sub_pd(b,bv),ar=_mm512_sub_pd(a,av);
    *h=q;*l=_mm512_add_pd(ar,br);
}
F3V static void f3_eval_avx512(const s53w_kernel *k,const double *x,double *y,size_t n)
{
    const __m512d Z=_mm512_setzero_pd(),VIPI=_mm512_set1_pd(INVPI),VA=_mm512_set1_pd(PI_A),VB=_mm512_set1_pd(PI_B),VC=_mm512_set1_pd(PI_C);
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),ONEd=_mm512_set1_pd(1.0);
    const __m512d MH=_mm512_set1_pd(-0.5),C24=_mm512_set1_pd(1.0/24.0),MSIX=_mm512_set1_pd(-1.0/6.0),C120=_mm512_set1_pd(1.0/120.0);
    const __m512i ONE=_mm512_set1_epi64(1),IZ=_mm512_setzero_si512(),ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    for(size_t i=0;i<n;i+=8){
        unsigned rem=(unsigned)(n-i);__mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
        __m512d vx=_mm512_maskz_loadu_pd(active,x+i);
        __mmask8 minput=(__mmask8)(_mm512_cmp_pd_mask(vx,Z,_CMP_LT_OQ)&active);
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vx),ABSM));
        __m512d qd=_mm512_roundscale_pd(_mm512_mul_pd(ax,VIPI),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        __m512i qi=_mm512_cvttpd_epi64(qd);

        __m512d h=_mm512_sub_pd(ax,_mm512_mul_pd(qd,VA));
        __m512d e1,e2;
        f3_two_diff_v(h,_mm512_mul_pd(qd,VB),&h,&e1);
        __m512d pc=_mm512_mul_pd(qd,VC),pce=_mm512_fmsub_pd(qd,VC,pc);
        f3_two_diff_v(h,pc,&h,&e2);
        __m512d lo=_mm512_sub_pd(_mm512_add_pd(e1,e2),pce),hn,ln;
        f3_two_sum_v(h,lo,&hn,&ln);h=hn;lo=ln;

        __mmask8 rn=_mm512_cmp_pd_mask(h,Z,_CMP_LT_OQ);__mmask8 hz=_mm512_cmp_pd_mask(h,Z,_CMP_EQ_OQ);
        rn|=(__mmask8)(hz&_mm512_cmp_pd_mask(lo,Z,_CMP_LT_OQ));rn&=active;
        h=_mm512_mask_sub_pd(h,rn,Z,h);lo=_mm512_mask_sub_pd(lo,rn,Z,lo);

        __m512d r=_mm512_add_pd(h,lo);
        __m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(r,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        __m512i ji=_mm512_cvttpd_epi64(jd);
        __m512d d=_mm512_sub_pd(h,_mm512_mul_pd(jd,VIK)),z=_mm512_mul_pd(d,d);
        __m512d C=_mm512_fmadd_pd(z,_mm512_fmadd_pd(z,C24,MH),ONEd);
        __m512d T=_mm512_fmadd_pd(z,_mm512_fmadd_pd(z,C120,MSIX),ONEd);
        __m512d sa=_mm512_i64gather_pd(ji,k->tab,8),ca=_mm512_i64gather_pd(ji,k->tab+LUTN,8);
        __m512d dt=_mm512_mul_pd(d,T);
        __m512d yv=_mm512_fmadd_pd(ca,dt,_mm512_mul_pd(sa,C));
        __m512d deriv=_mm512_fmadd_pd(_mm512_sub_pd(Z,sa),dt,_mm512_mul_pd(ca,C));
        yv=_mm512_fmadd_pd(lo,deriv,yv);
        __mmask8 odd=_mm512_cmpneq_epi64_mask(_mm512_and_epi64(qi,ONE),IZ);
        __mmask8 sg=(__mmask8)((minput^odd^rn)&active);
        yv=_mm512_mask_sub_pd(yv,sg,Z,yv);
        _mm512_mask_storeu_pd(y+i,active,yv);
    }
}
#endif

static int f3_have_avx512(void){
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512dq")&&__builtin_cpu_supports("fma");
#else
    return 0;
#endif
}
static void f3_eval(const s53w_kernel*k,const double*x,double*y,size_t n){
#if defined(__x86_64__) || defined(__i386__)
    if(f3_have_avx512()){f3_eval_avx512(k,x,y,n);return;}
#endif
    for(size_t i=0;i<n;i++)y[i]=f3_eval_scalar_one(k,x[i]);
}

static int f3_verify(const char*tag,const s53w_kernel*k,const double*x,int n)
{
    double*o=al64((size_t)n*sizeof(double)),*in=al64((size_t)n*sizeof(double));if(!o||!in)return 0;
    f3_eval(k,x,o,(size_t)n);vmdSin(n,x,in,VML_HA);
    arb_t ax,ay;arf_t lo,hi;arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);
    int uq=0,oe=0,o1=0,ie=0,i1=0;uint64_t om=0,im=0;
    for(int i=0;i<n;i++){arb_set_d(ax,x[i]);arb_sin(ay,ax,256);arb_get_lbound_arf(lo,ay,256);arb_get_ubound_arf(hi,ay,256);double a=arf_get_d(lo,ARF_RND_NEAR),b=arf_get_d(hi,ARF_RND_NEAR);if(dbits(a)!=dbits(b))continue;uq++;uint64_t uo=ulpd(o[i],a),ui=ulpd(in[i],a);if(!uo)oe++;if(uo<=1)o1++;if(uo>om)om=uo;if(!ui)ie++;if(ui<=1)i1++;if(ui>im)im=ui;}
    printf("S53F3_VERIFY tag=%s cases=%d unique_ref=%d ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu reference=Arb256\n",tag,n,uq,oe,o1,(unsigned long)om,ie,i1,(unsigned long)im);
    arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);free(in);free(o);return uq==n&&om<=1;
}
static void f3_bands(const s53w_kernel*k,const double*x)
{
    double o[CASES],in[CASES];f3_eval(k,x,o,CASES);vmdSin(CASES,x,in,VML_HA);arb_t ax,ay;arf_t lo,hi;arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);
    for(int b=0;b<3;b++){int oe=0,o1=0,ie=0,i1=0;uint64_t om=0,im=0;for(int j=0;j<50;j++){int i=50*b+j;arb_set_d(ax,x[i]);arb_sin(ay,ax,256);arb_get_lbound_arf(lo,ay,256);arb_get_ubound_arf(hi,ay,256);double a=arf_get_d(lo,ARF_RND_NEAR),z=arf_get_d(hi,ARF_RND_NEAR);if(dbits(a)!=dbits(z))continue;uint64_t uo=ulpd(o[i],a),ui=ulpd(in[i],a);if(!uo)oe++;if(uo<=1)o1++;if(uo>om)om=uo;if(!ui)ie++;if(ui<=1)i1++;if(ui>im)im=ui;}const char*bn=b==0?"0_to_1":(b==1?"1_to_500":"1000_to_10000");printf("S53F3_BAND band=%s ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu\n",bn,oe,o1,(unsigned long)om,ie,i1,(unsigned long)im);}arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);
}
static uint64_t f3_run(const s53w_kernel*k,const double*x,int rounds,volatile double*s){double y[CASES];uint64_t t=now_ns();for(int r=0;r<rounds;r++)f3_eval(k,x,y,CASES);t=now_ns()-t;*s+=y[CASES-1];return t;}
static int f3_bench(const s53w_kernel*k,const double*x)
{
    if(!f3_have_avx512()){printf("S53F3_SKIP_TIMING reason=no_avx512\n");return 0;}volatile double s=0;f3_run(k,x,5000,&s);run_intel(x,5000,&s);double a[TRIALS],b[TRIALS],calls=(double)ROUNDS*CASES;
    for(int t=0;t<TRIALS;t++){uint64_t u,v;if(t&1){v=run_intel(x,ROUNDS,&s);u=f3_run(k,x,ROUNDS,&s);}else{u=f3_run(k,x,ROUNDS,&s);v=run_intel(x,ROUNDS,&s);}a[t]=(double)u/calls;b[t]=(double)v/calls;printf("S53F3_TRIAL trial=%d ours_ns=%.6f intel_ns=%.6f intel_over_ours=%.6fx\n",t+1,a[t],b[t],b[t]/a[t]);}
    qsort(a,TRIALS,sizeof(double),cmpd);qsort(b,TRIALS,sizeof(double),cmpd);double om=a[TRIALS/2],im=b[TRIALS/2];
    printf("S53F3_RESULT terms=%d degree=%d cases=150 ours_e2e_ns_per_input=%.6f intel_ha_ns_per_input=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx throughput_advantage_pct=%.3f reduction=AVX512_CodyWaite_3part no_qpi_gathers=1 anchor_gathers=2 runtime_form=sin_a_C_plus_cos_a_d_T masked_tail=1 all_raw_input_work_included=1 formula=unchanged_secant_spine_Mode5_unfused accuracy_contract=le1ulp sink=%.17g\n",k->terms,k->deg,om,im,om/im,im/om,(im/om-1.0)*100.0,(double)s);return 0;
}

int main(void)
{
    int cpu=pin();mkl_set_num_threads_local(1);printf("S53F3_DOMAIN cpu_pin=%d target=binary64_53bit cases=150 bands=0_to_1,1_to_500,1000_to_10000 signs=25pos_25neg_each intel=oneMKL_vmdSin_VML_HA formula=secant_spine_unfused_C_T\n",cpu);
    s53w_kernel*k=kernel_create(2);if(!k)return 3;double x[CASES],v[CASES];make_bench(x);f3_eval(k,x,v,CASES);int same=0;for(int i=0;i<CASES;i++)if(dbits(v[i])==dbits(f3_eval_scalar_one(k,x[i])))same++;printf("S53F3_VECTOR_SCALAR bit_identical=%d/150\n",same);if(same!=CASES){kernel_destroy(k);return 4;}
    if(!f3_verify("requested150",k,x,CASES)){kernel_destroy(k);return 5;}f3_bands(k,x);
    double*st=al64(STRESS*sizeof(double));if(st){make_stress(st);int ok=f3_verify("adversarial_stress",k,st,STRESS);printf("S53F3_STRESS pass=%d contractual=0\n",ok);free(st);if(!ok){kernel_destroy(k);return 6;}}
    int rc=f3_bench(k,x);kernel_destroy(k);flint_cleanup_master();return rc;
}
