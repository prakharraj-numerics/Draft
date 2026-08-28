#define main s53w_dd_main_base
#include "bench_sine_53_wide_dd_driver.c"
#undef main

#if defined(__x86_64__) || defined(__i386__)
#define DBG512 __attribute__((target("avx512f,avx512dq,fma")))
DBG512 static void eval_scalarprep_vectorpoly(const s53w_kernel *k, const double *x, double *y, size_t n)
{
    int64_t idx[CASES];
    double dhv[CASES], dlv[CASES];
    unsigned char neg[CASES];

    for (size_t i=0;i<n;i++)
    {
        int64_t q; double rh,rl; int rn;
        reduce_dd_scalar(x[i],&q,&rh,&rl,&rn);
        double r=rh+rl;
        long a=lround(r*KGRID); if(a<0)a=0; if(a>=LUTN)a=LUTN-1;
        double aa=(double)a*INVK;
        double dh,de; two_diff_s(rh,aa,&dh,&de);
        double dl=de+rl,hn,ln; two_sum_s(dh,dl,&hn,&ln);
        idx[i]=a; dhv[i]=hn; dlv[i]=ln;
        neg[i]=(unsigned char)(signbit(x[i]) ^ (int)(q&1) ^ rn);
    }

    size_t i=0;
    for(;i+8<=n;i+=8)
    {
        __m512i ji=_mm512_loadu_si512((const void*)(idx+i));
        __m512d dh=_mm512_loadu_pd(dhv+i), dl=_mm512_loadu_pd(dlv+i);
        __m512d out=_mm512_i64gather_pd(ji,k->tab+(size_t)k->deg*LUTN,8);
        for(int j=k->deg-1;j>=0;j--)
        {
            __m512d c=_mm512_i64gather_pd(ji,k->tab+(size_t)j*LUTN,8);
            out=_mm512_fmadd_pd(out,dh,c);
        }
        __m512d der=_mm512_mul_pd(_mm512_set1_pd((double)k->deg),
                                  _mm512_i64gather_pd(ji,k->tab+(size_t)k->deg*LUTN,8));
        for(int j=k->deg-1;j>=1;j--)
        {
            __m512d c=_mm512_i64gather_pd(ji,k->tab+(size_t)j*LUTN,8);
            der=_mm512_fmadd_pd(der,dh,_mm512_mul_pd(_mm512_set1_pd((double)j),c));
        }
        out=_mm512_fmadd_pd(dl,der,out);
        unsigned mask=0; for(int lane=0;lane<8;lane++) if(neg[i+(size_t)lane]) mask|=1u<<lane;
        out=_mm512_mask_sub_pd(out,(__mmask8)mask,_mm512_setzero_pd(),out);
        _mm512_storeu_pd(y+i,out);
    }
    for(;i<n;i++) y[i]=eval_scalar_dd_one(k,x[i]);
}
#endif

static int have_avx512(void)
{
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}

static void eval_hybrid(const s53w_kernel *k,const double *x,double *y,size_t n)
{
#if defined(__x86_64__) || defined(__i386__)
    if(have_avx512()){ eval_scalarprep_vectorpoly(k,x,y,n); return; }
#endif
    for(size_t i=0;i<n;i++) y[i]=eval_scalar_dd_one(k,x[i]);
}

static int verify_hybrid(const s53w_kernel*k,const double*x)
{
    double ours[CASES],intel[CASES];
    eval_hybrid(k,x,ours,CASES); vmdSin(CASES,x,intel,VML_HA);
    arb_t ax,ay; arf_t lo,hi; arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);
    int oe=0,ol1=0,ie=0,il1=0,unique=0;
    uint64_t om=0,im=0;
    int boe[3]={0},bol1[3]={0},bie[3]={0},bil1[3]={0};
    uint64_t bom[3]={0},bim[3]={0};
    for(int i=0;i<CASES;i++)
    {
        arb_set_d(ax,x[i]); arb_sin(ay,ax,256);
        arb_get_lbound_arf(lo,ay,256); arb_get_ubound_arf(hi,ay,256);
        double rl=arf_get_d(lo,ARF_RND_NEAR), rh=arf_get_d(hi,ARF_RND_NEAR);
        if(dbits(rl)!=dbits(rh)) continue;
        unique++;
        uint64_t uo=ulpd(ours[i],rl), ui=ulpd(intel[i],rl);
        int b=i/50;
        if(!uo){oe++;boe[b]++;} if(uo<=1){ol1++;bol1[b]++;} if(uo>om)om=uo; if(uo>bom[b])bom[b]=uo;
        if(!ui){ie++;bie[b]++;} if(ui<=1){il1++;bil1[b]++;} if(ui>im)im=ui; if(ui>bim[b])bim[b]=ui;
    }
    const char*bn[3]={"0_to_1","1_to_500","1000_to_10000"};
    for(int b=0;b<3;b++)
        printf("S53H_BAND band=%s cases=50 positive=25 negative=25 ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu\n",
               bn[b],boe[b],bol1[b],(unsigned long)bom[b],bie[b],bil1[b],(unsigned long)bim[b]);
    printf("S53H_VERIFY cases=150 unique_ref=%d ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu reference=Arb256\n",
           unique,oe,ol1,(unsigned long)om,ie,il1,(unsigned long)im);
    arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);
    return unique==CASES && om<=1;
}

static uint64_t run_hybrid(const s53w_kernel*k,const double*x,int rounds,volatile double*sink)
{
    double y[CASES]; uint64_t t=now_ns();
    for(int r=0;r<rounds;r++) eval_hybrid(k,x,y,CASES);
    t=now_ns()-t; *sink+=y[CASES-1]; return t;
}

static int benchmark_hybrid(const s53w_kernel*k,const double*x)
{
    if(!have_avx512()){printf("S53H_SKIP_TIMING reason=no_avx512\n");return 0;}
    volatile double sink=0; run_hybrid(k,x,5000,&sink); run_intel(x,5000,&sink);
    double ot[TRIALS],it[TRIALS],calls=(double)ROUNDS*CASES;
    for(int t=0;t<TRIALS;t++)
    {
        uint64_t a,b;
        if(t&1){b=run_intel(x,ROUNDS,&sink);a=run_hybrid(k,x,ROUNDS,&sink);}
        else{a=run_hybrid(k,x,ROUNDS,&sink);b=run_intel(x,ROUNDS,&sink);}
        ot[t]=(double)a/calls; it[t]=(double)b/calls;
        printf("S53H_TRIAL trial=%d ours_e2e_ns_per_input=%.6f intel_ha_ns_per_input=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx\n",
               t+1,ot[t],it[t],ot[t]/it[t],it[t]/ot[t]);
    }
    qsort(ot,TRIALS,sizeof(double),cmpd);qsort(it,TRIALS,sizeof(double),cmpd);
    double om=ot[TRIALS/2],im=it[TRIALS/2],ratio=im/om;
    printf("S53H_RESULT terms=%d degree=%d cases=150 bands=50_50_50 signs=25pos_25neg_each ours_e2e_ns_per_150=%.3f ours_e2e_ns_per_input=%.6f intel_ha_ns_per_150=%.3f intel_ha_ns_per_input=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx throughput_advantage_pct=%.3f range_reduction=scalar_double_double_included vector_poly=avx512_fma included_all_raw_input_work=1 formula=unchanged_Mode5_secant_spine accuracy_contract=le1ulp sink=%.17g\n",
           k->terms,k->deg,om*CASES,om,im*CASES,im,om/im,ratio,(ratio-1.0)*100.0,(double)sink);
    return 0;
}

int main(void)
{
    int cpu=pin(); mkl_set_num_threads_local(1);
    printf("S53H_DOMAIN target=binary64_53bit cases=150 bands=0_to_1,1_to_500,1000_to_10000 signs_each_band=25pos_25neg cpu_pin=%d intel=oneMKL_vmdSin_VML_HA intel_installed_first=1 candidate=scalar_DD_reduction_plus_AVX512_Mode5\n",cpu);
    s53w_kernel *k=kernel_create(2); if(!k) return 2;
    double x[CASES],full[CASES],hyb[CASES]; make_bench(x);
    eval_e2e_dd(k,x,full,CASES); eval_hybrid(k,x,hyb,CASES);
    int full_same=0,hyb_same=0;
    for(int i=0;i<CASES;i++)
    {
        double sc=eval_scalar_dd_one(k,x[i]);
        if(dbits(sc)==dbits(full[i])) full_same++;
        if(dbits(sc)==dbits(hyb[i])) hyb_same++;
    }
    printf("S53H_FULL_AVX_VECTOR_SCALAR bit_identical=%d/150 diagnostic_only=1\n",full_same);
    printf("S53H_HYBRID_SCALAR bit_identical=%d/150 contractual_path=1\n",hyb_same);
    if(hyb_same!=CASES){kernel_destroy(k);return 3;}
    if(!verify_hybrid(k,x)){kernel_destroy(k);return 4;}
    int rc=benchmark_hybrid(k,x);
    kernel_destroy(k); flint_cleanup_master(); return rc;
}
