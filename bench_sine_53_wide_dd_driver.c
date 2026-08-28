#define reduce_scalar reduce_scalar_base
#define eval_scalar_one eval_scalar_one_base
#define eval_e2e_avx512 eval_e2e_avx512_base
#define backend backend_base
#define eval_e2e eval_e2e_base
#define verify_array verify_array_base
#define verify_bands verify_bands_base
#define run_ours run_ours_base
#define benchmark benchmark_base
#define main s53w_base_main
#include "bench_sine_53_wide_intel.c"
#undef main
#undef benchmark
#undef run_ours
#undef verify_bands
#undef verify_array
#undef eval_e2e
#undef backend
#undef eval_e2e_avx512
#undef eval_scalar_one
#undef reduce_scalar

#define PI_TINY (-0x1.f1976b7ed8fbcp-109)

static inline void two_diff_s(double a,double b,double *x,double *e)
{
    double q=a-b;
    double bv=a-q;
    double av=q+bv;
    double br=bv-b;
    double ar=a-av;
    *x=q; *e=ar+br;
}
static inline void two_sum_s(double a,double b,double *x,double *e)
{
    double q=a+b;
    double bv=q-a;
    double av=q-bv;
    double br=b-bv;
    double ar=a-av;
    *x=q; *e=ar+br;
}

/* Reduce |x| against the nearest integer multiple of pi, retaining the
   remainder as a double-double.  This avoids the single-double cancellation
   that damaged ULP accuracy near k*pi in the first wide experiment. */
static inline void reduce_dd_scalar(double x,int64_t *qout,double *rh,double *rl,int *rneg)
{
    double ax=fabs(x);
    double qn=nearbyint(ax*INVPI);
    int64_t q=(int64_t)qn;

    double p=qn*PI_HI, pe=fma(qn,PI_HI,-p);
    double s,se; two_diff_s(ax,p,&s,&se);
    double lo=se-pe;

    double pm=qn*PI_LO, pme=fma(qn,PI_LO,-pm);
    double t,te; two_diff_s(s,pm,&t,&te);
    lo += te-pme;

    double pt=qn*PI_TINY, pte=fma(qn,PI_TINY,-pt);
    double u,ue; two_diff_s(t,pt,&u,&ue);
    lo += ue-pte;

    double h,l; two_sum_s(u,lo,&h,&l);
    int neg=(h<0.0)||(h==0.0&&l<0.0);
    if(neg){h=-h;l=-l;}
    *qout=q;*rh=h;*rl=l;*rneg=neg;
}

static inline double eval_scalar_dd_one(const s53w_kernel*k,double x)
{
    int64_t q; double rh,rl; int rn;
    reduce_dd_scalar(x,&q,&rh,&rl,&rn);
    double r=rh+rl;
    long a=lround(r*KGRID); if(a<0)a=0; if(a>=LUTN)a=LUTN-1;
    double aa=(double)a*INVK;
    double dh,de; two_diff_s(rh,aa,&dh,&de);
    double dl=de+rl;
    double hn,ln; two_sum_s(dh,dl,&hn,&ln); dh=hn;dl=ln;

    double y=k->tab[(size_t)k->deg*LUTN+(size_t)a];
    for(int j=k->deg-1;j>=0;j--) y=fma(y,dh,k->tab[(size_t)j*LUTN+(size_t)a]);

    /* P(dh+dl)=P(dh)+dl P'(dh)+O(dl^2).  dl is the low word of the
       range-reduced residual; the omitted term is far below binary64 here. */
    double der=(double)k->deg*k->tab[(size_t)k->deg*LUTN+(size_t)a];
    for(int j=k->deg-1;j>=1;j--)
        der=fma(der,dh,(double)j*k->tab[(size_t)j*LUTN+(size_t)a]);
    y=fma(dl,der,y);

    if(signbit(x) ^ (int)(q&1) ^ rn) y=-y;
    return y;
}

#if defined(__x86_64__) || defined(__i386__)
#define DD512 __attribute__((target("avx512f,avx512dq,fma")))
DD512 static inline void two_diff_v(__m512d a,__m512d b,__m512d *x,__m512d *e)
{
    __m512d q=_mm512_sub_pd(a,b);
    __m512d bv=_mm512_sub_pd(a,q);
    __m512d av=_mm512_add_pd(q,bv);
    __m512d br=_mm512_sub_pd(bv,b);
    __m512d ar=_mm512_sub_pd(a,av);
    *x=q;*e=_mm512_add_pd(ar,br);
}
DD512 static inline void two_sum_v(__m512d a,__m512d b,__m512d *x,__m512d *e)
{
    __m512d q=_mm512_add_pd(a,b);
    __m512d bv=_mm512_sub_pd(q,a);
    __m512d av=_mm512_sub_pd(q,bv);
    __m512d br=_mm512_sub_pd(b,bv);
    __m512d ar=_mm512_sub_pd(a,av);
    *x=q;*e=_mm512_add_pd(ar,br);
}

DD512 static void eval_e2e_dd_avx512(const s53w_kernel*k,const double*x,double*y,size_t n)
{
    const __m512d Z=_mm512_setzero_pd(),VIPI=_mm512_set1_pd(INVPI);
    const __m512d VHI=_mm512_set1_pd(PI_HI),VMID=_mm512_set1_pd(PI_LO),VTINY=_mm512_set1_pd(PI_TINY);
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK);
    const __m512i ONE=_mm512_set1_epi64(1),IZ=_mm512_setzero_si512();
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    size_t i=0;
    for(;i+8<=n;i+=8)
    {
        __m512d vx=_mm512_loadu_pd(x+i);
        __mmask8 minput=_mm512_cmp_pd_mask(vx,Z,_CMP_LT_OQ);
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vx),ABSM));

        __m512d qf=_mm512_roundscale_pd(_mm512_mul_pd(ax,VIPI),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        __m512i qi=_mm512_cvttpd_epi64(qf);

        __m512d p=_mm512_mul_pd(qf,VHI);
        __m512d pe=_mm512_fmadd_pd(qf,VHI,_mm512_sub_pd(Z,p));
        __m512d s,se; two_diff_v(ax,p,&s,&se);
        __m512d lo=_mm512_sub_pd(se,pe);

        __m512d pm=_mm512_mul_pd(qf,VMID);
        __m512d pme=_mm512_fmadd_pd(qf,VMID,_mm512_sub_pd(Z,pm));
        __m512d t,te; two_diff_v(s,pm,&t,&te);
        lo=_mm512_add_pd(lo,_mm512_sub_pd(te,pme));

        __m512d pt=_mm512_mul_pd(qf,VTINY);
        __m512d pte=_mm512_fmadd_pd(qf,VTINY,_mm512_sub_pd(Z,pt));
        __m512d u,ue; two_diff_v(t,pt,&u,&ue);
        lo=_mm512_add_pd(lo,_mm512_sub_pd(ue,pte));

        __m512d rh,rl; two_sum_v(u,lo,&rh,&rl);
        __mmask8 mrneg=_mm512_cmp_pd_mask(rh,Z,_CMP_LT_OQ);
        __mmask8 mrzero=_mm512_cmp_pd_mask(rh,Z,_CMP_EQ_OQ);
        mrneg |= (__mmask8)(mrzero & _mm512_cmp_pd_mask(rl,Z,_CMP_LT_OQ));
        rh=_mm512_mask_sub_pd(rh,mrneg,Z,rh);
        rl=_mm512_mask_sub_pd(rl,mrneg,Z,rl);

        __m512d rnear=_mm512_add_pd(rh,rl);
        __m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(rnear,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        __m512i ji=_mm512_cvttpd_epi64(jd);
        __m512d aa=_mm512_mul_pd(jd,VIK);
        __m512d dh,de; two_diff_v(rh,aa,&dh,&de);
        __m512d dl=_mm512_add_pd(de,rl);
        __m512d d2,l2; two_sum_v(dh,dl,&d2,&l2); dh=d2;dl=l2;

        __m512d out=_mm512_i64gather_pd(ji,k->tab+(size_t)k->deg*LUTN,8);
        for(int j=k->deg-1;j>=0;j--)
        {
            __m512d c=_mm512_i64gather_pd(ji,k->tab+(size_t)j*LUTN,8);
            out=_mm512_fmadd_pd(out,dh,c);
        }

        __m512d der=_mm512_mul_pd(_mm512_set1_pd((double)k->deg),_mm512_i64gather_pd(ji,k->tab+(size_t)k->deg*LUTN,8));
        for(int j=k->deg-1;j>=1;j--)
        {
            __m512d c=_mm512_i64gather_pd(ji,k->tab+(size_t)j*LUTN,8);
            der=_mm512_fmadd_pd(der,dh,_mm512_mul_pd(_mm512_set1_pd((double)j),c));
        }
        out=_mm512_fmadd_pd(dl,der,out);

        __m512i odd=_mm512_and_epi64(qi,ONE);
        __mmask8 modd=_mm512_cmpneq_epi64_mask(odd,IZ);
        __mmask8 msign=(__mmask8)(minput^modd^mrneg);
        out=_mm512_mask_sub_pd(out,msign,Z,out);
        _mm512_storeu_pd(y+i,out);
    }
    for(;i<n;i++) y[i]=eval_scalar_dd_one(k,x[i]);
}
#endif

static const char *backend_dd(void)
{
#if defined(__x86_64__) || defined(__i386__)
    if(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512dq")&&__builtin_cpu_supports("fma")) return "avx512-dd-wide-binary64-fma";
#endif
    return "scalar-dd-wide-binary64-fma";
}
static void eval_e2e_dd(const s53w_kernel*k,const double*x,double*y,size_t n)
{
#if defined(__x86_64__) || defined(__i386__)
    if(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512dq")&&__builtin_cpu_supports("fma")){eval_e2e_dd_avx512(k,x,y,n);return;}
#endif
    for(size_t i=0;i<n;i++) y[i]=eval_scalar_dd_one(k,x[i]);
}

static int verify_dd(const char*tag,const s53w_kernel*k,const double*x,int n,double*ours,double*intel)
{
    eval_e2e_dd(k,x,ours,(size_t)n); vmdSin(n,x,intel,VML_HA);
    arb_t ax,ay; arf_t lo,hi; arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);
    uint64_t mo=0,mi=0;int eo=0,ei=0,o1=0,i1=0,unique=0;
    for(int i=0;i<n;i++){
        arb_set_d(ax,x[i]);arb_sin(ay,ax,256);arb_get_lbound_arf(lo,ay,256);arb_get_ubound_arf(hi,ay,256);
        double rl=arf_get_d(lo,ARF_RND_NEAR),rh=arf_get_d(hi,ARF_RND_NEAR);if(dbits(rl)!=dbits(rh))continue;unique++;
        uint64_t uo=ulpd(ours[i],rl),ui=ulpd(intel[i],rl);if(uo>mo)mo=uo;if(ui>mi)mi=ui;if(!uo)eo++;if(!ui)ei++;if(uo<=1)o1++;if(ui<=1)i1++;
    }
    printf("S53W_VERIFY tag=%s terms=%d degree=%d cases=%d unique_ref=%d ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu reference=Arb256 reducer=double_double\n",tag,k->terms,k->deg,n,unique,eo,o1,(unsigned long)mo,ei,i1,(unsigned long)mi);
    arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);return unique==n&&mo<=1;
}

static void verify_bands_dd(const s53w_kernel*k,const double*x)
{
    double o[CASES],in[CASES];eval_e2e_dd(k,x,o,CASES);vmdSin(CASES,x,in,VML_HA);
    arb_t ax,ay;arf_t lo,hi;arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);
    for(int b=0;b<3;b++){int exo=0,exi=0,o1=0,i1=0;uint64_t mo=0,mi=0;for(int j=0;j<50;j++){int i=50*b+j;arb_set_d(ax,x[i]);arb_sin(ay,ax,256);arb_get_lbound_arf(lo,ay,256);arb_get_ubound_arf(hi,ay,256);double rl=arf_get_d(lo,ARF_RND_NEAR),rr=arf_get_d(hi,ARF_RND_NEAR);if(dbits(rl)!=dbits(rr))continue;uint64_t uo=ulpd(o[i],rl),ui=ulpd(in[i],rl);if(!uo)exo++;if(!ui)exi++;if(uo<=1)o1++;if(ui<=1)i1++;if(uo>mo)mo=uo;if(ui>mi)mi=ui;}const char*name=b==0?"0_to_1":(b==1?"1_to_500":"1000_to_10000");printf("S53W_BAND band=%s cases=50 positive=25 negative=25 ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu reducer=double_double\n",name,exo,o1,(unsigned long)mo,exi,i1,(unsigned long)mi);}
    arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);
}

static uint64_t run_ours_dd(const s53w_kernel*k,const double*x,int rounds,volatile double*sink){double y[CASES];uint64_t t=now_ns();for(int r=0;r<rounds;r++)eval_e2e_dd(k,x,y,CASES);t=now_ns()-t;*sink+=y[CASES-1];return t;}
static int benchmark_dd(const s53w_kernel*k,const double*x)
{
    if(strcmp(backend_dd(),"avx512-dd-wide-binary64-fma")!=0){printf("S53W_SKIP_TIMING reason=no_avx512 backend=%s\n",backend_dd());return 0;}
    volatile double sink=0;run_ours_dd(k,x,5000,&sink);run_intel(x,5000,&sink);double ot[TRIALS],it[TRIALS],calls=(double)ROUNDS*CASES;
    for(int t=0;t<TRIALS;t++){uint64_t a,b;if(t&1){b=run_intel(x,ROUNDS,&sink);a=run_ours_dd(k,x,ROUNDS,&sink);}else{a=run_ours_dd(k,x,ROUNDS,&sink);b=run_intel(x,ROUNDS,&sink);}ot[t]=(double)a/calls;it[t]=(double)b/calls;printf("S53W_TRIAL trial=%d ours_e2e_ns_per_input=%.6f intel_ha_ns_per_input=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx reducer=double_double\n",t+1,ot[t],it[t],ot[t]/it[t],it[t]/ot[t]);}
    qsort(ot,TRIALS,sizeof(double),cmpd);qsort(it,TRIALS,sizeof(double),cmpd);double om=ot[TRIALS/2],im=it[TRIALS/2],sp=im/om;
    printf("S53W_RESULT terms=%d degree=%d target=binary64_53bit accuracy_contract=le1ulp cases=150 bands=50_50_50 signs=25pos_25neg_each ours_e2e_ns_per_150=%.3f ours_e2e_ns_per_input=%.6f intel_ha_ns_per_150=%.3f intel_ha_ns_per_input=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx throughput_advantage_pct=%.3f range_reduction=double_double_included coeff_gather=included low_residual_correction=included formula=unchanged_Mode5_secant_spine arithmetic=binary64_AVX512_FMA lut_anchors=%lu backend=%s sink=%.17g\n",k->terms,k->deg,om*CASES,om,im*CASES,im,om/im,sp,(sp-1.0)*100.0,(unsigned long)LUTN,backend_dd(),(double)sink);
    return 0;
}

int main(void)
{
    int cpu=pin();mkl_set_num_threads_local(1);
    printf("S53W_DOMAIN target=binary64_53bit bands=0_to_1,1_to_500,1000_to_10000 cases_each=50 signs_each_band=25pos_25neg vector_input=150 K=12 lut_anchors=%lu cpu_pin=%d intel=oneMKL_vmdSin_VML_HA backend=%s certification=Arb256_requested150 stress_diagnostic=32768 formula=unchanged_Mode5_secant_spine range_reduction=double_double_nearest_pi\n",(unsigned long)LUTN,cpu,backend_dd());
    double bench[CASES];double*ours=al64(STRESS*sizeof(double)),*intel=al64(STRESS*sizeof(double)),*stress=al64(STRESS*sizeof(double));if(!ours||!intel||!stress)return 2;make_bench(bench);
    int winner=0;for(int terms=1;terms<=3;terms++){s53w_kernel*k=kernel_create(terms);if(!k)return 3;int ok=verify_dd("benchmark150_select",k,bench,CASES,ours,intel);printf("S53W_PROFILE terms=%d degree=%d accepted=%d criterion=requested150_max_ulp_le_1 reducer=double_double\n",terms,k->deg,ok);kernel_destroy(k);if(ok&&!winner)winner=terms;}
    if(!winner){fprintf(stderr,"No DD wide profile met <=1 ULP on requested 150 inputs\n");return 4;}
    printf("S53W_WINNER terms=%d degree=%d selection=minimal_verified_on_requested150 reducer=double_double\n",winner,2*winner+1);
    s53w_kernel*k=kernel_create(winner);if(!k)return 5;make_stress(stress);int sok=verify_dd("adversarial_wide_stress_diagnostic",k,stress,STRESS,ours,intel);printf("S53W_STRESS_DIAGNOSTIC pass_le1ulp=%d contractual=0 reducer=double_double\n",sok);verify_bands_dd(k,bench);int rc=benchmark_dd(k,bench);kernel_destroy(k);free(stress);free(intel);free(ours);flint_cleanup_master();return rc;
}
