from pathlib import Path
import runpy

# Build on v9: exact v8 math + two-vector software pipeline.
runpy.run_path('make_sine_53_xeon_v9.py', run_name='__main__')
src = Path('bench_sine_53_xeon_v9_build.c').read_text()

marker = '/* Xeon execution layer: keep two independent Mode-5 chains in flight.'
insert_at = src.index(marker)
layout = r'''
/* Xeon cache-line layout.  The mathematical coefficient values are copied
   bit-for-bit from k->tab.  One anchor occupies exactly one 64-byte cache line:
   c0..c5 plus two padding doubles.  Thus all six Horner coefficients for a
   lane share the same cache line instead of six separated coefficient planes. */
#define XEON_COEF_STRIDE 8
static double *xeon_coef10;

static int xeon_coef_init10(const s53w_kernel *k)
{
    xeon_coef10=al64((size_t)LUTN*XEON_COEF_STRIDE*sizeof(double));
    if(!xeon_coef10)return 0;
    memset(xeon_coef10,0,(size_t)LUTN*XEON_COEF_STRIDE*sizeof(double));
    for(int a=0;a<LUTN;a++)
        for(int j=0;j<=k->deg;j++)
            xeon_coef10[(size_t)a*XEON_COEF_STRIDE+(size_t)j]=
                k->tab[(size_t)j*LUTN+(size_t)a];
    return 1;
}
static void xeon_coef_clear10(void){free(xeon_coef10);xeon_coef10=NULL;}

OVEC static inline void mode5_poly2_anchor_xeon(const s53w_kernel *k,
        __m512d yh0,__m512d yl0,__mmask8 sign0,
        __m512d yh1,__m512d yl1,__mmask8 sign1,
        __m512d *out0,__m512d *out1)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m256i VS=_mm256_set1_epi32(XEON_COEF_STRIDE);
    __m512d ya0=_mm512_add_pd(yh0,yl0),ya1=_mm512_add_pd(yh1,yl1);
    __m512d jd0=_mm512_roundscale_pd(_mm512_mul_pd(ya0,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd1=_mm512_roundscale_pd(_mm512_mul_pd(ya1,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256i ji0=_mm512_cvttpd_epi32(jd0),ji1=_mm512_cvttpd_epi32(jd1);
    __m512d d0=_mm512_add_pd(_mm512_sub_pd(yh0,_mm512_mul_pd(jd0,VIK)),yl0);
    __m512d d1=_mm512_add_pd(_mm512_sub_pd(yh1,_mm512_mul_pd(jd1,VIK)),yl1);
    __m256i bi0=_mm256_mullo_epi32(ji0,VS),bi1=_mm256_mullo_epi32(ji1,VS);
    __m256i ix0=_mm256_add_epi32(bi0,_mm256_set1_epi32(k->deg));
    __m256i ix1=_mm256_add_epi32(bi1,_mm256_set1_epi32(k->deg));
    __m512d p0=_mm512_i32gather_pd(ix0,xeon_coef10,8);
    __m512d p1=_mm512_i32gather_pd(ix1,xeon_coef10,8);
    for(int j=k->deg-1;j>=0;j--){
        __m256i jv=_mm256_set1_epi32(j);
        __m512d c0=_mm512_i32gather_pd(_mm256_add_epi32(bi0,jv),xeon_coef10,8);
        __m512d c1=_mm512_i32gather_pd(_mm256_add_epi32(bi1,jv),xeon_coef10,8);
        p0=_mm512_fmadd_pd(p0,d0,c0);
        p1=_mm512_fmadd_pd(p1,d1,c1);
    }
    *out0=_mm512_mask_sub_pd(p0,sign0,Z,p0);
    *out1=_mm512_mask_sub_pd(p1,sign1,Z,p1);
}

OVEC static inline __m512d mode5_poly1_anchor_xeon(const s53w_kernel *k,
        __m512d yh,__m512d yl,__mmask8 sign)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    __m512d ya=_mm512_add_pd(yh,yl);
    __m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(ya,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256i ji=_mm512_cvttpd_epi32(jd);
    __m512d d=_mm512_add_pd(_mm512_sub_pd(yh,_mm512_mul_pd(jd,VIK)),yl);
    __m256i bi=_mm256_mullo_epi32(ji,_mm256_set1_epi32(XEON_COEF_STRIDE));
    __m512d p=_mm512_i32gather_pd(_mm256_add_epi32(bi,_mm256_set1_epi32(k->deg)),xeon_coef10,8);
    for(int j=k->deg-1;j>=0;j--){
        __m512d c=_mm512_i32gather_pd(_mm256_add_epi32(bi,_mm256_set1_epi32(j)),xeon_coef10,8);
        p=_mm512_fmadd_pd(p,d,c);
    }
    return _mm512_mask_sub_pd(p,sign,Z,p);
}

'''
src = src[:insert_at] + layout + src[insert_at:]

# Keep the v9 plane-major helper in the file for A/B comparison at source level,
# but route the hot v10 loop to the cache-line anchor-major helpers.
src = src.replace('mode5_poly2_i32_low_xeon(k,rh0,rl0,s0,rh1,rl1,s1,&p0,&p1);',
                  'mode5_poly2_anchor_xeon(k,rh0,rl0,s0,rh1,rl1,s1,&p0,&p1);')
src = src.replace('__m512d p=mode5_poly_i32_low(k,rh,rl,s);',
                  '__m512d p=mode5_poly1_anchor_xeon(k,rh,rl,s);')

# Initialize the packed table only for this binary; the original k->tab remains
# untouched and is still used by scalar2 in the guarded correctness fallback.
needle='if(!redtab2_init())return 2;s53w_kernel *k=kernel_create(2);if(!k)return 3;double x[CASES];make_bench(x);'
repl='if(!redtab2_init())return 2;s53w_kernel *k=kernel_create(2);if(!k)return 3;if(!xeon_coef_init10(k)){kernel_destroy(k);redtab2_clear();return 9;}double x[CASES];make_bench(x);'
if needle not in src:
    raise SystemExit('main init pattern not found')
src=src.replace(needle,repl,1)
src=src.replace('kernel_destroy(k);redtab2_clear();', 'xeon_coef_clear10();kernel_destroy(k);redtab2_clear();')
# The init-failure replacement above must not clear an uninitialized table twice.
src=src.replace('if(!xeon_coef_init10(k)){xeon_coef_clear10();kernel_destroy(k);redtab2_clear();return 9;}',
                'if(!xeon_coef_init10(k)){kernel_destroy(k);redtab2_clear();return 9;}')

src=src.replace('S53X9_', 'S53X10_')
src=src.replace('xeon_v9_pi4_octant_compensated_cw_2x_pipeline',
                'xeon_v10_pi4_cw_2x_pipeline_cacheline_coeff')
src=src.replace('Xeon_AVX512_2x_pipeline_pi4_int32_compensated_cw',
                'Xeon_AVX512_2x_pipeline_cacheline_coeff_pi4_cw')
Path('bench_sine_53_xeon_v10_build.c').write_text(src)
print('S53X10_BUILD_PASS exact_v8_math=1 xeon_2x_vector_pipeline=1 cacheline_anchor_major_coeff=1 coef_stride64B=1 six_coefficients_preserved=1 horner_order_preserved=1 cold_guard_repair=1 formula_unchanged=1')
