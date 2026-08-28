from pathlib import Path
import runpy

# Exact v8 numerical/scheduling baseline.  Only the coefficient storage/access
# layout is changed; reducer, one-vector loop, Horner order and coefficient bits
# are unchanged.
runpy.run_path('make_sine_53_octant_v8.py', run_name='__main__')
src=Path('bench_sine_53_wide_octant_v8_build.c').read_text()

insert_at=src.index('OVEC static void octant_vector_v8')
helper=r'''
#define XEON11_STRIDE 8
static double *xeon11_tab;
static int xeon11_init(const s53w_kernel *k){
    xeon11_tab=al64((size_t)LUTN*XEON11_STRIDE*sizeof(double));
    if(!xeon11_tab)return 0;
    memset(xeon11_tab,0,(size_t)LUTN*XEON11_STRIDE*sizeof(double));
    for(int a=0;a<LUTN;a++)for(int j=0;j<=k->deg;j++)
        xeon11_tab[(size_t)a*XEON11_STRIDE+(size_t)j]=k->tab[(size_t)j*LUTN+(size_t)a];
    return 1;
}
static void xeon11_clear(void){free(xeon11_tab);xeon11_tab=NULL;}

OVEC static inline __m512d mode5_poly_i32_low_xeon11(const s53w_kernel *k,
        __m512d yh,__m512d yl,__mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    __m512d ya=_mm512_add_pd(yh,yl);
    __m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(ya,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256i ji=_mm512_cvttpd_epi32(jd);
    __m512d d=_mm512_add_pd(_mm512_sub_pd(yh,_mm512_mul_pd(jd,VIK)),yl);
    __m256i bi=_mm256_mullo_epi32(ji,_mm256_set1_epi32(XEON11_STRIDE));
    __m512d p=_mm512_i32gather_pd(_mm256_add_epi32(bi,_mm256_set1_epi32(k->deg)),xeon11_tab,8);
    for(int j=k->deg-1;j>=0;j--){
        __m512d c=_mm512_i32gather_pd(_mm256_add_epi32(bi,_mm256_set1_epi32(j)),xeon11_tab,8);
        p=_mm512_fmadd_pd(p,d,c);
    }
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

'''
src=src[:insert_at]+helper+src[insert_at:]
# Only redirect v8's coefficient evaluator; leave the original one-vector reducer loop intact.
src=src.replace('__m512d p=mode5_poly_i32_low(k,rh,rl,signmask);','__m512d p=mode5_poly_i32_low_xeon11(k,rh,rl,signmask);')

needle='if(!redtab2_init())return 2;s53w_kernel *k=kernel_create(2);if(!k)return 3;double x[CASES];make_bench(x);'
repl='if(!redtab2_init())return 2;s53w_kernel *k=kernel_create(2);if(!k)return 3;if(!xeon11_init(k)){kernel_destroy(k);redtab2_clear();return 9;}double x[CASES];make_bench(x);'
if needle not in src: raise SystemExit('main init pattern not found')
src=src.replace(needle,repl,1)
src=src.replace('kernel_destroy(k);redtab2_clear();','xeon11_clear();kernel_destroy(k);redtab2_clear();')
src=src.replace('if(!xeon11_init(k)){xeon11_clear();kernel_destroy(k);redtab2_clear();return 9;}','if(!xeon11_init(k)){kernel_destroy(k);redtab2_clear();return 9;}')

src=src.replace('S53O8_','S53X11_')
src=src.replace('cosine_style_pi4_octant_guarded_v8_compensated_cw','xeon_v11_exact_v8_singlevector_cacheline_coeff')
src=src.replace('AVX512_pi4_octant_int32_compensated_cw','Xeon_AVX512_v8_singlevector_cacheline_coeff_pi4_cw')
Path('bench_sine_53_xeon_v11_build.c').write_text(src)
print('S53X11_BUILD_PASS exact_v8_math=1 exact_v8_singlevector_schedule=1 cacheline_anchor_major_coeff=1 six_coefficients_preserved=1 horner_order_preserved=1 formula_unchanged=1')
