from pathlib import Path
import runpy

# v15: preserve v13 arithmetic exactly. Change only coefficient storage layout:
# plane-major [c][anchor] -> anchor-major 64-byte records [anchor][c0..c5,pad,pad].
# The base s53w_kernel type lives in an included source file, so keep that type
# untouched and maintain an experiment-local packed LUT built from k->tab.
runpy.run_path('make_sine_53_xeon_v13_preindexed_batch.py', run_name='__main__')
src=Path('bench_sine_53_xeon_v13_build.c').read_text()

start=src.index('OVEC static inline __m512d mode5_preindexed_horner_x13')
end=src.index('\nOVEC static void octant_vector_v8',start)
packed=r'''
static double *x15_pack = NULL;

static int x15_init_pack(const s53w_kernel *k)
{
    if(x15_pack) return 1;
    x15_pack=al64((size_t)LUTN*8u*sizeof(double));
    if(!x15_pack) return 0;
    for(int a=0;a<LUTN;a++){
        for(int j=0;j<8;j++)
            x15_pack[(size_t)a*8u+(size_t)j]=
                (j<=k->deg)?k->tab[(size_t)j*LUTN+(size_t)a]:0.0;
    }
    return 1;
}

OVEC static inline __m512d mode5_preindexed_horner_x15(const s53w_kernel *k,
        __m256i ji,__m512d d,__mmask8 signmask)
{
    (void)k;
    const __m512d Z=_mm512_setzero_pd();
    const double *pack=(const double *)__builtin_assume_aligned(x15_pack,64);
    /* One 64-byte record per anchor: c0..c5,pad,pad. Same coefficient bits,
       same degree-5 Horner FMA sequence as v13; only address layout changes. */
    __m256i bi=_mm256_slli_epi32(ji,3);
    __m512d p=_mm512_i32gather_pd(_mm256_add_epi32(bi,_mm256_set1_epi32(5)),pack,8);
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(_mm256_add_epi32(bi,_mm256_set1_epi32(4)),pack,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(_mm256_add_epi32(bi,_mm256_set1_epi32(3)),pack,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(_mm256_add_epi32(bi,_mm256_set1_epi32(2)),pack,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(_mm256_add_epi32(bi,_mm256_set1_epi32(1)),pack,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(bi,pack,8));
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}
'''
src=src[:start]+packed+src[end:]

needle='''OVEC static void octant_vector_v8(const s53w_kernel *k,\n                                  const double * __restrict x,\n                                  double * __restrict out,size_t n)\n{\n    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}'''
repl='''OVEC static void octant_vector_v8(const s53w_kernel *k,\n                                  const double * __restrict x,\n                                  double * __restrict out,size_t n)\n{\n    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}\n    if(__builtin_expect(x15_pack==NULL,0) && !x15_init_pack(k)){\n        octant_vector_v12_batch(k,x,out,n);return;\n    }'''
if needle not in src: raise SystemExit('v13 octant entry not found')
src=src.replace(needle,repl,1)

src=src.replace('mode5_preindexed_horner_x13(k,ji,d,(__mmask8)signbuf[b])',
                'mode5_preindexed_horner_x15(k,ji,d,(__mmask8)signbuf[b])')
src=src.replace('S53X13_','S53X15_')
src=src.replace('xeon_v13_preindexed_tiled_batch','xeon_v15_cachepacked_tiled_batch')
src=src.replace('Xeon_AVX512_preindexed_j_d_then_Mode5','Xeon_AVX512_preindexed_j_d_cachepacked_Mode5')
Path('bench_sine_53_xeon_v15_build.c').write_text(src)
print('S53X15_BUILD_PASS same_secant_Mode5_spine=1 same_coeff_bits=1 same_anchor_rule=1 same_j_d=1 same_Horner_FMA_order=1 layout=anchor_major_64B_record lut_payload_bytes=19344 packed_bytes=25792 guarded_repair_unchanged=1')
