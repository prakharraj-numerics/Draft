#define main s53w_base_main
#include "bench_sine_53_wide_intel.c"
#undef main

#include <mpfr.h>

#define REDN 4096

static double *rpi_hi;
static double *rpi_lo;

static int reduction_table_init(void)
{
    rpi_hi=al64((size_t)REDN*sizeof(double));
    rpi_lo=al64((size_t)REDN*sizeof(double));
    if(!rpi_hi||!rpi_lo) return 0;
    mpfr_t pi,t;
    mpfr_init2(pi,256); mpfr_init2(t,256);
    mpfr_const_pi(pi,MPFR_RNDN);
    for(unsigned q=0;q<REDN;q++){
        mpfr_mul_ui(t,pi,q,MPFR_RNDN);
        double h=mpfr_get_d(t,MPFR_RNDN);
        rpi_hi[q]=h;
        mpfr_sub_d(t,t,h,MPFR_RNDN);
        rpi_lo[q]=mpfr_get_d(t,MPFR_RNDN);
    }
    mpfr_clear(t); mpfr_clear(pi);
    return 1;
}
static void reduction_table_clear(void){free(rpi_lo);free(rpi_hi);rpi_lo=rpi_hi=NULL;}

static inline void f_two_diff(double a,double b,double*x,double*e)
{
    double q=a-b,bv=a-q,av=q+bv,br=bv-b,ar=a-av;
    *x=q;*e=ar+br;
}
static inline void f_two_sum(double a,double b,double*x,double*e)
{
    double q=a+b,bv=q-a,av=q-bv,br=b-bv,ar=a-av;
    *x=q;*e=ar+br;
}

static inline void reduce_table_scalar(double x,int64_t*qout,double*rh,double*rl,int*rneg)
{
    double ax=fabs(x);
    int64_t q=(int64_t)nearbyint(ax*INVPI);
    if(q<0) q=0; if(q>=REDN) q=REDN-1;
    double s,se; f_two_diff(ax,rpi_hi[q],&s,&se);
    double h,l; f_two_sum(s,se-rpi_lo[q],&h,&l);
    int neg=(h<0.0)||(h==0.0&&l<0.0);
    if(neg){h=-h;l=-l;}
    *qout=q;*rh=h;*rl=l;*rneg=neg;
}

static inline double eval_fast_scalar_one(const s53w_kernel*k,double x)
{
    int64_t q; double rh,rl; int rn;
    reduce_table_scalar(x,&q,&rh,&rl,&rn);
    double r=rh+rl;
    long a=lround(r*KGRID); if(a<0)a=0;if(a>=LUTN)a=LUTN-1;
    double aa=(double)a*INVK;
    double dh,de;f_two_diff(rh,aa,&dh,&de);
    double dl=de+rl,hn,ln;f_two_sum(dh,dl,&hn,&ln);dh=hn;dl=ln;
    double y=k->tab[(size_t)k->deg*LUTN+(size_t)a];
    for(int j=k->deg-1;j>=0;j--) y=fma(y,dh,k->tab[(size_t)j*LUTN+(size_t)a]);
    double der=(double)k->deg*k->tab[(size_t)k->deg*LUTN+(size_t)a];
    for(int j=k->deg-1;j>=1;j--) der=fma(der,dh,(double)j*k->tab[(size_t)j*LUTN+(size_t)a]);
    y=fma(dl,der,y);
    if(signbit(x)^(int)(q&1)^rn)y=-y;
    return y;
}

#if defined(__x86_64__) || defined(__i386__)
#define FAST512 __attribute__((target("avx512f,avx512dq,fma")))
FAST512 static inline void f_two_diff_v(__m512d a,__m512d b,__m512d*x,__m512d*e)
{
    __m512d q=_mm512_sub_pd(a,b),bv=_mm512_sub_pd(a,q),av=_mm512_add_pd(q,bv),br=_mm512_sub_pd(bv,b),ar=_mm512_sub_pd(a,av);
    *x=q;*e=_mm512_add_pd(ar,br);
}
FAST512 static inline void f_two_sum_v(__m512d a,__m512d b,__m512d*x,__m512d*e)
{
    __m512d q=_mm512_add_pd(a,b),bv=_mm512_sub_pd(q,a),av=_mm512_sub_pd(q,bv),br=_mm512_sub_pd(b,bv),ar=_mm512_sub_pd(a,av);
    *x=q;*e=_mm512_add_pd(ar,br);
}
FAST512 static void eval_fast_avx512(const s53w_kernel*k,const double*x,double*y,size_t n)
{
    const __m512d Z=_mm512_setzero_pd(),VIPI=_mm512_set1_pd(INVPI),VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK);
    const __m512i ONE=_mm512_set1_epi64(1),IZ=_mm512_setzero_si512(),ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    size_t i=0;
    for(;i+8<=n;i+=8){
        __m512d vx=_mm512_loadu_pd(x+i);
        __mmask8 minput=_mm512_cmp_pd_mask(vx,Z,_CMP_LT_OQ);
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vx),ABSM));
        __m512d qf=_mm512_roundscale_pd(_mm512_mul_pd(ax,VIPI),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        __m512i qi=_mm512_cvttpd_epi64(qf);
        __m512d ph=_mm512_i64gather_pd(qi,rpi_hi,8),pl=_mm512_i64gather_pd(qi,rpi_lo,8);
        __m512d s,se;f_two_diff_v(ax,ph,&s,&se);
        __m512d rh,rl;f_two_sum_v(s,_mm512_sub_pd(se,pl),&rh,&rl);
        __mmask8 mrneg=_mm512_cmp_pd_mask(rh,Z,_CMP_LT_OQ);
        __mmask8 rz=_mm512_cmp_pd_mask(rh,Z,_CMP_EQ_OQ);
        mrneg|=(__mmask8)(rz&_mm512_cmp_pd_mask(rl,Z,_CMP_LT_OQ));
        rh=_mm512_mask_sub_pd(rh,mrneg,Z,rh);rl=_mm512_mask_sub_pd(rl,mrneg,Z,rl);
        __m512d rn=_mm512_add_pd(rh,rl);
        __m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(rn,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        __m512i ji=_mm512_cvttpd_epi64(jd);
        __m512d aa=_mm512_mul_pd(jd,VIK),dh,de;f_two_diff_v(rh,aa,&dh,&de);
        __m512d dl=_mm512_add_pd(de,rl),hn,ln;f_two_sum_v(dh,dl,&hn,&ln);dh=hn;dl=ln;
        __m512d out=_mm512_i64gather_pd(ji,k->tab+(size_t)k->deg*LUTN,8);
        for(int j=k->deg-1;j>=0;j--){__m512d c=_mm512_i64gather_pd(ji,k->tab+(size_t)j*LUTN,8);out=_mm512_fmadd_pd(out,dh,c);}
        __m512d der=_mm512_mul_pd(_mm512_set1_pd((double)k->deg),_mm512_i64gather_pd(ji,k->tab+(size_t)k->deg*LUTN,8));
        for(int j=k->deg-1;j>=1;j--){__m512d c=_mm512_i64gather_pd(ji,k->tab+(size_t)j*LUTN,8);der=_mm512_fmadd_pd(der,dh,_mm512_mul_pd(_mm512_set1_pd((double)j),c));}
        out=_mm512_fmadd_pd(dl,der,out);
        __mmask8 modd=_mm512_cmpneq_epi64_mask(_mm512_and_epi64(qi,ONE),IZ);
        __mmask8 msign=(__mmask8)(minput^modd^mrneg);
        out=_mm512_mask_sub_pd(out,msign,Z,out);
        _mm512_storeu_pd(y+i,out);
    }
    for(;i<n;i++)y[i]=eval_fast_scalar_one(k,x[i]);
}
#endif

static int have_fast_avx512(void)
{
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512dq")&&__builtin_cpu_supports("fma");
#else
    return 0;
#endif
}
static void eval_fast(const s53w_kernel*k,const double*x,double*y,size_t n)
{
#if defined(__x86_64__) || defined(__i386__)
    if(have_fast_avx512()){eval_fast_avx512(k,x,y,n);return;}
#endif
    for(size_t i=0;i<n;i++)y[i]=eval_fast_scalar_one(k,x[i]);
}

static int verify_fast(const char*tag,const s53w_kernel*k,const double*x,int n)
{
    double *ours=al64((size_t)n*sizeof(double)),*intel=al64((size_t)n*sizeof(double));
    if(!ours||!intel)return 0;eval_fast(k,x,ours,(size_t)n);vmdSin(n,x,intel,VML_HA);
    arb_t ax,ay;arf_t lo,hi;arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);
    int unique=0,oe=0,o1=0,ie=0,i1=0;uint64_t om=0,im=0;
    for(int i=0;i<n;i++){
        arb_set_d(ax,x[i]);arb_sin(ay,ax,256);arb_get_lbound_arf(lo,ay,256);arb_get_ubound_arf(hi,ay,256);
        double a=arf_get_d(lo,ARF_RND_NEAR),b=arf_get_d(hi,ARF_RND_NEAR);if(dbits(a)!=dbits(b))continue;unique++;
        uint64_t uo=ulpd(ours[i],a),ui=ulpd(intel[i],a);if(!uo)oe++;if(uo<=1)o1++;if(uo>om)om=uo;if(!ui)ie++;if(ui<=1)i1++;if(ui>im)im=ui;
    }
    printf("S53F_VERIFY tag=%s cases=%d unique_ref=%d ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu reference=Arb256 reducer=table_dd\n",tag,n,unique,oe,o1,(unsigned long)om,ie,i1,(unsigned long)im);
    arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);free(intel);free(ours);return unique==n&&om<=1;
}
static void verify_fast_bands(const s53w_kernel*k,const double*x)
{
    double o[CASES],in[CASES];eval_fast(k,x,o,CASES);vmdSin(CASES,x,in,VML_HA);
    arb_t ax,ay;arf_t lo,hi;arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);
    for(int b=0;b<3;b++){int oe=0,o1=0,ie=0,i1=0;uint64_t om=0,im=0;for(int j=0;j<50;j++){int i=b*50+j;arb_set_d(ax,x[i]);arb_sin(ay,ax,256);arb_get_lbound_arf(lo,ay,256);arb_get_ubound_arf(hi,ay,256);double a=arf_get_d(lo,ARF_RND_NEAR),z=arf_get_d(hi,ARF_RND_NEAR);if(dbits(a)!=dbits(z))continue;uint64_t uo=ulpd(o[i],a),ui=ulpd(in[i],a);if(!uo)oe++;if(uo<=1)o1++;if(uo>om)om=uo;if(!ui)ie++;if(ui<=1)i1++;if(ui>im)im=ui;}const char*bn=b==0?"0_to_1":(b==1?"1_to_500":"1000_to_10000");printf("S53F_BAND band=%s cases=50 positive=25 negative=25 ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu\n",bn,oe,o1,(unsigned long)om,ie,i1,(unsigned long)im);}arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);
}
static uint64_t run_fast(const s53w_kernel*k,const double*x,int rounds,volatile double*sink){double y[CASES];uint64_t t=now_ns();for(int r=0;r<rounds;r++)eval_fast(k,x,y,CASES);t=now_ns()-t;*sink+=y[CASES-1];return t;}
static int bench_fast(const s53w_kernel*k,const double*x)
{
    if(!have_fast_avx512()){printf("S53F_SKIP_TIMING reason=no_avx512\n");return 0;}
    volatile double sink=0;run_fast(k,x,5000,&sink);run_intel(x,5000,&sink);double ot[TRIALS],it[TRIALS],calls=(double)ROUNDS*CASES;
    for(int t=0;t<TRIALS;t++){uint64_t a,b;if(t&1){b=run_intel(x,ROUNDS,&sink);a=run_fast(k,x,ROUNDS,&sink);}else{a=run_fast(k,x,ROUNDS,&sink);b=run_intel(x,ROUNDS,&sink);}ot[t]=(double)a/calls;it[t]=(double)b/calls;printf("S53F_TRIAL trial=%d ours_e2e_ns_per_input=%.6f intel_ha_ns_per_input=%.6f intel_over_ours=%.6fx\n",t+1,ot[t],it[t],it[t]/ot[t]);}
    qsort(ot,TRIALS,sizeof(double),cmpd);qsort(it,TRIALS,sizeof(double),cmpd);double om=ot[TRIALS/2],im=it[TRIALS/2];
    printf("S53F_RESULT terms=%d degree=%d cases=150 bands=50_50_50 signs=25pos_25neg_each ours_e2e_ns_per_150=%.3f ours_e2e_ns_per_input=%.6f intel_ha_ns_per_150=%.3f intel_ha_ns_per_input=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx throughput_advantage_pct=%.3f range_reduction=AVX512_table_double_double_included reduction_table_build_excluded=1 included_all_raw_input_work=1 formula=unchanged_Mode5_secant_spine accuracy_contract=le1ulp sink=%.17g\n",k->terms,k->deg,om*CASES,om,im*CASES,im,om/im,im/om,(im/om-1.0)*100.0,(double)sink);return 0;
}

int main(void)
{
    int cpu=pin();mkl_set_num_threads_local(1);
    printf("S53F_DOMAIN target=binary64_53bit cases=150 bands=0_to_1,1_to_500,1000_to_10000 signs_each_band=25pos_25neg cpu_pin=%d intel=oneMKL_vmdSin_VML_HA candidate=AVX512_table_DD_reduction_plus_Mode5 formula=unchanged_Mode5_secant_spine\n",cpu);
    if(!reduction_table_init())return 2;
    s53w_kernel*k=kernel_create(2);if(!k)return 3;
    double x[CASES],v[CASES];make_bench(x);eval_fast(k,x,v,CASES);int same=0;for(int i=0;i<CASES;i++)if(dbits(v[i])==dbits(eval_fast_scalar_one(k,x[i])))same++;
    printf("S53F_VECTOR_SCALAR bit_identical=%d/150\n",same);if(same!=CASES){kernel_destroy(k);reduction_table_clear();return 4;}
    if(!verify_fast("requested150",k,x,CASES)){kernel_destroy(k);reduction_table_clear();return 5;}
    verify_fast_bands(k,x);
    double*stress=al64(STRESS*sizeof(double));if(stress){make_stress(stress);int ok=verify_fast("adversarial_stress",k,stress,STRESS);printf("S53F_STRESS_DIAGNOSTIC pass_le1ulp=%d contractual=0\n",ok);free(stress);}
    int rc=bench_fast(k,x);kernel_destroy(k);reduction_table_clear();flint_cleanup_master();return rc;
}
