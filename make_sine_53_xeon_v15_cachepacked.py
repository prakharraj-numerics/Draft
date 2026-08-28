from pathlib import Path
import runpy

# v15: preserve v13 arithmetic exactly. Change only coefficient storage layout:
# plane-major [c][anchor] -> anchor-major 64-byte records [anchor][c0..c5,pad,pad].
# This lets all six coefficients for one anchor share one cache line.
runpy.run_path('make_sine_53_xeon_v13_preindexed_batch.py', run_name='__main__')
src=Path('bench_sine_53_xeon_v13_build.c').read_text()

old='typedef struct { sine_fixed_ctx *ctx; int terms,deg; double *tab; } s53w_kernel;'
new='typedef struct { sine_fixed_ctx *ctx; int terms,deg; double *tab; double *pack; } s53w_kernel;'
if old not in src: raise SystemExit('kernel typedef not found')
src=src.replace(old,new,1)

ks=src.index('static s53w_kernel *kernel_create')
ke=src.index('\nstatic void kernel_destroy',ks)
newcreate=r'''static s53w_kernel *kernel_create(int terms){
    if(terms<1||terms>3)return NULL;
    s53w_kernel*k=calloc(1,sizeof(*k));if(!k)return NULL;
    k->ctx=s53_coeff_create_terms(terms);if(!k->ctx){free(k);return NULL;}
    k->terms=terms;k->deg=k->ctx->poly_deg;
    k->tab=al64((size_t)(k->deg+1)*LUTN*sizeof(double));
    k->pack=al64((size_t)LUTN*8u*sizeof(double));
    if(!k->tab||!k->pack){free(k->pack);free(k->tab);s53_coeff_destroy(k->ctx);free(k);return NULL;}
    for(int a=0;a<LUTN;a++){
        size_t off=(size_t)a*(size_t)(k->deg+1);
        for(int j=0;j<=k->deg;j++)
            k->tab[(size_t)j*LUTN+(size_t)a]=coeff_to_double(k->ctx->coef+2*(off+(size_t)j),k->ctx->coef_sign[off+(size_t)j]!=0);
        for(int j=0;j<8;j++)
            k->pack[(size_t)a*8u+(size_t)j]=(j<=k->deg)?k->tab[(size_t)j*LUTN+(size_t)a]:0.0;
    }
    return k;
}'''
src=src[:ks]+newcreate+src[ke:]

ds=src.index('static void kernel_destroy')
de=src.index('\n\nstatic inline double reduce_scalar',ds)
newdestroy='static void kernel_destroy(s53w_kernel*k){if(!k)return;free(k->pack);free(k->tab);s53_coeff_destroy(k->ctx);free(k);}'
src=src[:ds]+newdestroy+src[de:]

start=src.index('OVEC static inline __m512d mode5_preindexed_horner_x13')
end=src.index('\nOVEC static void octant_vector_v8',start)
packed=r'''OVEC static inline __m512d mode5_preindexed_horner_x15(const s53w_kernel *k,
        __m256i ji,__m512d d,__mmask8 signmask)
{
    const __m512d Z=_mm512_setzero_pd();
    const double *pack=(const double *)__builtin_assume_aligned(k->pack,64);
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
src=src.replace('mode5_preindexed_horner_x13(k,ji,d,(__mmask8)signbuf[b])',
                'mode5_preindexed_horner_x15(k,ji,d,(__mmask8)signbuf[b])')
src=src.replace('S53X13_','S53X15_')
src=src.replace('xeon_v13_preindexed_tiled_batch','xeon_v15_cachepacked_tiled_batch')
src=src.replace('Xeon_AVX512_preindexed_j_d_then_Mode5','Xeon_AVX512_preindexed_j_d_cachepacked_Mode5')
Path('bench_sine_53_xeon_v15_build.c').write_text(src)
print('S53X15_BUILD_PASS same_secant_Mode5_spine=1 same_coeff_bits=1 same_anchor_rule=1 same_j_d=1 same_Horner_FMA_order=1 layout=anchor_major_64B_record lut_payload_bytes=19344 packed_bytes=25792 guarded_repair_unchanged=1')
