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

int main(void)
{
    int cpu=pin(); mkl_set_num_threads_local(1);
    printf("S53W_DEBUG backend=%s cpu_pin=%d\n",backend_dd(),cpu);
    s53w_kernel *k=kernel_create(2); if(!k) return 2;
    double x[CASES],full[CASES],spv[CASES]; make_bench(x);
    eval_e2e_dd(k,x,full,CASES);
#if defined(__x86_64__) || defined(__i386__)
    if(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512dq")&&__builtin_cpu_supports("fma"))
        eval_scalarprep_vectorpoly(k,x,spv,CASES);
    else
#endif
        for(int i=0;i<CASES;i++) spv[i]=eval_scalar_dd_one(k,x[i]);

    int full_same=0,spv_same=0,full_m=0,spv_m=0;
    for(int i=0;i<CASES;i++)
    {
        double sc=eval_scalar_dd_one(k,x[i]);
        if(dbits(sc)==dbits(full[i])) full_same++;
        else if(++full_m<=12)
        {
            int64_t q;double rh,rl;int rn;reduce_dd_scalar(x[i],&q,&rh,&rl,&rn);
            double r=rh+rl;long a=lround(r*KGRID);double aa=(double)a*INVK,dh,de;two_diff_s(rh,aa,&dh,&de);double dl=de+rl,hn,ln;two_sum_s(dh,dl,&hn,&ln);
            printf("S53W_FULL_MISMATCH i=%d x=%.17g scalar=%.17g full=%.17g ulp=%lu q=%ld rneg=%d rh=%.17g rl=%.17g anchor=%ld dh=%.17g dl=%.17g\n",
                   i,x[i],sc,full[i],(unsigned long)ulpd(sc,full[i]),(long)q,rn,rh,rl,a,hn,ln);
        }
        if(dbits(sc)==dbits(spv[i])) spv_same++;
        else if(++spv_m<=12)
            printf("S53W_POLY_MISMATCH i=%d x=%.17g scalar=%.17g scalarprep_vectorpoly=%.17g ulp=%lu\n",
                   i,x[i],sc,spv[i],(unsigned long)ulpd(sc,spv[i]));
    }
    printf("S53W_FULL_VECTOR_SCALAR bit_identical=%d/%d mismatches=%d\n",full_same,CASES,full_m);
    printf("S53W_SCALARPREP_VECTORPOLY bit_identical=%d/%d mismatches=%d\n",spv_same,CASES,spv_m);
    kernel_destroy(k); flint_cleanup_master();
    return spv_m?2:0;
}
