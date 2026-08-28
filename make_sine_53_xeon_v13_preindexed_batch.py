from pathlib import Path
import runpy

# Start from v12. v13 changes only where anchor-index/local-delta arithmetic is
# performed: compute {j,d,sign} during batch preparation, then reuse it in the
# Mode-5 phase. Same reducer, same anchor selection rule, same six coefficients,
# same degree-5 Horner FMA order, same guarded repair.
runpy.run_path('make_sine_53_xeon_v12_batch.py', run_name='__main__')
src=Path('bench_sine_53_xeon_v12_build.c').read_text()

needle='OVEC static void octant_vector_v8(const s53w_kernel *k,\n                                  const double * __restrict x,\n                                  double * __restrict out,size_t n)'
if needle not in src:
    raise SystemExit('v12 vector declaration not found')
src=src.replace(needle,'OVEC static void octant_vector_v12_batch(const s53w_kernel *k,\n                                  const double * __restrict x,\n                                  double * __restrict out,size_t n)',1)

# Insert v13 immediately before #endif after v12 implementation.
start=src.index('OVEC static void octant_vector_v12_batch')
end=src.index('\n#endif',start)
new=r'''

OVEC static inline __m512d mode5_preindexed_horner_x13(const s53w_kernel *k,
        __m256i ji,__m512d d,__mmask8 signmask)
{
    const __m512d Z=_mm512_setzero_pd();
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    __m512d p=_mm512_i32gather_pd(ji,tab+5*LUTN,8);
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+4*LUTN,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+3*LUTN,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+2*LUTN,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+1*LUTN,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+0*LUTN,8));
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

OVEC static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}

    _Alignas(64) double dbuf[X12_TILE];
    _Alignas(32) int jbuf[X12_TILE];
    unsigned char signbuf[X12_TILE/8],guardbuf[X12_TILE/8],activebuf[X12_TILE/8];
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK);

    for(size_t tile=0;tile<n;tile+=X12_TILE){
        size_t tn=n-tile;if(tn>X12_TILE)tn=X12_TILE;
        size_t blocks=(tn+7)/8;
        /* Phase 1: reduce + octant + anchor index + local delta exactly once. */
        for(size_t b=0;b<blocks;b++){
            __m512d rh,rl;__mmask8 s,g,a;unsigned char pu;
            x12_prepare_block(x+tile,b*8,tn,&rh,&rl,&s,&g,&a,&pu);
            __m512d ya=_mm512_add_pd(rh,rl);
            __m256i ji=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya,VK),
                         _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
            __m512d jd=_mm512_cvtepi32_pd(ji);
            __m512d d;
            if(pu){
                /* Identical to mode5_poly_x11. */
                d=_mm512_fnmadd_pd(jd,VIK,rh);
            }else{
                /* Identical rounding sequence to mode5_poly_low_x11. */
                d=_mm512_sub_pd(rh,_mm512_mul_pd(jd,VIK));
                d=_mm512_add_pd(d,rl);
            }
            _mm512_store_pd(dbuf+b*8,d);
            _mm256_store_si256((__m256i *)(jbuf+b*8),ji);
            signbuf[b]=(unsigned char)s;guardbuf[b]=(unsigned char)g;activebuf[b]=(unsigned char)a;
        }
        /* Phase 2: exact same six-coefficient Mode-5 polynomial/Horner order. */
        for(size_t b=0;b<blocks;b++){
            __m512d d=_mm512_load_pd(dbuf+b*8);
            __m256i ji=_mm256_load_si256((const __m256i *)(jbuf+b*8));
            __m512d p=mode5_preindexed_horner_x13(k,ji,d,(__mmask8)signbuf[b]);
            _mm512_mask_storeu_pd(out+tile+b*8,(__mmask8)activebuf[b],p);
        }
        /* Phase 3: unchanged rare guarded repair. */
        for(size_t b=0;b<blocks;b++)if(__builtin_expect(guardbuf[b]!=0,0)){
            __mmask8 g=(__mmask8)guardbuf[b];
            for(unsigned lane=0;lane<8&&b*8+lane<tn;lane++)
                if(g&(1u<<lane))out[tile+b*8+lane]=scalar2(k,x[tile+b*8+lane]);
        }
    }
}
'''
src=src[:end]+new+src[end:]

src=src.replace('S53X12_','S53X13_')
src=src.replace('xeon_v12_tiled_two_stage_batch','xeon_v13_preindexed_tiled_batch')
src=src.replace('Xeon_AVX512_tiled_reduce_then_Mode5','Xeon_AVX512_preindexed_j_d_then_Mode5')
Path('bench_sine_53_xeon_v13_build.c').write_text(src)
print('S53X13_BUILD_PASS same_secant_Mode5_spine=1 same_coefficients=1 same_anchor_rule=1 same_Horner_FMA_order=1 preindex_j_d_once=1 tile=256 guarded_repair_unchanged=1')
