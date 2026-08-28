from pathlib import Path
import runpy

# v18: exact-baseline-coefficient experiment.
# The Mode-5 builder gives, mathematically,
#   c0=s*C0=s, c2=s*C1=-s/2, c4=s*C2=s/24
#   c1=c*T0=c, c3=c*T1=-c/6, c5=c*T2=c/120.
# Gather c0,c1 only, reconstruct c2..c5 in binary64, then apply tiny signed
# raw-bit corrections packed four int16 values per anchor into ONE uint64
# gather. After correction, c0..c5 must be bit-for-bit the baseline v12 table.
# Horner and all reducer mathematics remain unchanged.
runpy.run_path('make_sine_53_xeon_v12_batch.py', run_name='__main__')
p=Path('bench_sine_53_xeon_v12_build.c')
s=p.read_text()

start=s.index('OVEC static inline __m512d mode5_poly_x11')
end=s.index('OVEC static inline void twodiff_cw',start)
helpers=r'''
static uint64_t v18_corr[LUTN] __attribute__((aligned(64)));
static int v18_corr_ready=0;
static inline double v18_round_mul(double a,double b){ volatile double z=a*b; return z; }
static int v18_corr_init(const s53w_kernel *k)
{
    if(v18_corr_ready)return 1;
    if(!k || k->deg!=5)return 0;
    long long maxabs[4]={0,0,0,0};
    const double mult[4]={-0.5,-1.0/6.0,1.0/24.0,1.0/120.0};
    const int srcplane[4]={0,1,0,1},dstplane[4]={2,3,4,5};
    for(int a=0;a<(int)LUTN;a++){
        uint64_t pack=0;
        for(int z=0;z<4;z++){
            double src=k->tab[(size_t)srcplane[z]*LUTN+(size_t)a];
            double r=v18_round_mul(src,mult[z]);
            double t=k->tab[(size_t)dstplane[z]*LUTN+(size_t)a];
            __int128 d=(__int128)(unsigned long long)dbits(t)-(__int128)(unsigned long long)dbits(r);
            if(d < -32768 || d > 32767){
                printf("S53V18_CORR_FAIL anchor=%d plane=%d raw_delta=%lld\n",a,dstplane[z],(long long)d);
                return 0;
            }
            long long ad=(long long)(d<0?-d:d); if(ad>maxabs[z])maxabs[z]=ad;
            uint16_t u=(uint16_t)(int16_t)(long long)d;
            pack |= ((uint64_t)u) << (16*z);
        }
        v18_corr[a]=pack;
    }
    /* Setup-time exact reconstruction self-check. */
    for(int a=0;a<(int)LUTN;a++){
        uint64_t pack=v18_corr[a];
        for(int z=0;z<4;z++){
            double src=k->tab[(size_t)srcplane[z]*LUTN+(size_t)a];
            double r=v18_round_mul(src,mult[z]);
            int16_t d=(int16_t)((pack>>(16*z))&UINT64_C(0xffff));
            uint64_t ub=dbits(r)+(int64_t)d,ut=dbits(k->tab[(size_t)dstplane[z]*LUTN+(size_t)a]);
            if(ub!=ut){printf("S53V18_CORR_SELFCHECK_FAIL anchor=%d plane=%d\n",a,dstplane[z]);return 0;}
        }
    }
    v18_corr_ready=1;
    printf("S53V18_CORR_PASS maxraw_c2=%lld maxraw_c3=%lld maxraw_c4=%lld maxraw_c5=%lld bytes=%zu exact_baseline_coeff_bits=1\n",
           maxabs[0],maxabs[1],maxabs[2],maxabs[3],sizeof(v18_corr));
    return 1;
}

OVEC static inline void v18_coeffs(const s53w_kernel *k,__m256i ji,
                                   __m512d *c0,__m512d *c1,__m512d *c2,
                                   __m512d *c3,__m512d *c4,__m512d *c5)
{
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0);
    const __m512d C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    *c0=_mm512_i32gather_pd(ji,tab+0*LUTN,8);
    *c1=_mm512_i32gather_pd(ji,tab+1*LUTN,8);
    *c2=_mm512_mul_pd(*c0,MH);
    *c3=_mm512_mul_pd(*c1,M6);
    *c4=_mm512_mul_pd(*c0,C24);
    *c5=_mm512_mul_pd(*c1,C120);
    __m512i q=_mm512_i32gather_epi64(ji,(const long long *)v18_corr,8);
    __m512i d2=_mm512_srai_epi64(_mm512_slli_epi64(q,48),48);
    __m512i d3=_mm512_srai_epi64(_mm512_slli_epi64(q,32),48);
    __m512i d4=_mm512_srai_epi64(_mm512_slli_epi64(q,16),48);
    __m512i d5=_mm512_srai_epi64(q,48);
    *c2=_mm512_castsi512_pd(_mm512_add_epi64(_mm512_castpd_si512(*c2),d2));
    *c3=_mm512_castsi512_pd(_mm512_add_epi64(_mm512_castpd_si512(*c3),d3));
    *c4=_mm512_castsi512_pd(_mm512_add_epi64(_mm512_castpd_si512(*c4),d4));
    *c5=_mm512_castsi512_pd(_mm512_add_epi64(_mm512_castpd_si512(*c5),d5));
}

OVEC static inline __m512d mode5_poly_x11(const s53w_kernel *k,__m512d y,
                                          __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    __m512d sy=_mm512_mul_pd(y,VK);
    __m256i ji=_mm512_cvt_roundpd_epi32(sy,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd=_mm512_cvtepi32_pd(ji);
    __m512d d=_mm512_fnmadd_pd(jd,VIK,y);
    __m512d c0,c1,c2,c3,c4,c5;v18_coeffs(k,ji,&c0,&c1,&c2,&c3,&c4,&c5);
    __m512d p=_mm512_fmadd_pd(c5,d,c4);
    p=_mm512_fmadd_pd(p,d,c3);p=_mm512_fmadd_pd(p,d,c2);
    p=_mm512_fmadd_pd(p,d,c1);p=_mm512_fmadd_pd(p,d,c0);
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

OVEC static inline __m512d mode5_poly_low_x11(const s53w_kernel *k,
                                               __m512d yh,__m512d yl,
                                               __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    __m512d ya=_mm512_add_pd(yh,yl),sy=_mm512_mul_pd(ya,VK);
    __m256i ji=_mm512_cvt_roundpd_epi32(sy,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd=_mm512_cvtepi32_pd(ji);
    __m512d d=_mm512_sub_pd(yh,_mm512_mul_pd(jd,VIK));d=_mm512_add_pd(d,yl);
    __m512d c0,c1,c2,c3,c4,c5;v18_coeffs(k,ji,&c0,&c1,&c2,&c3,&c4,&c5);
    __m512d p=_mm512_fmadd_pd(c5,d,c4);
    p=_mm512_fmadd_pd(p,d,c3);p=_mm512_fmadd_pd(p,d,c2);
    p=_mm512_fmadd_pd(p,d,c1);p=_mm512_fmadd_pd(p,d,c0);
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

'''
s=s[:start]+helpers+s[end:]
# Initialize the packed corrections immediately after the production degree-5
# kernel is created. The generated main has this literal once.
needle='s53w_kernel *k=kernel_create(2);'
if needle not in s: raise SystemExit('production kernel_create(2) marker missing')
s=s.replace(needle,needle+'if(!k||!v18_corr_init(k))return 18;',1)
s=s.replace('S53X12_','S53V18_')
s=s.replace('xeon_v12_tiled_two_stage_batch','xeon_v18_two_double_plus_one_packed_gather_exact_coeffbits')
Path('bench_sine_53_xeon_v18_exact3g_build.c').write_text(s)
print('S53V18_BUILD_PASS same_secant_Mode5_polynomial=1 exact_baseline_coeff_bits_target=1 double_gathers=2 packed_u64_gathers=1 same_anchor_rule=1 same_delta=1 same_Horner_FMA_order=1 correction_setup_untimed=1 no_refit=1 no_SVML_inside=1')
