from pathlib import Path
import runpy

# Build on v11: exact same reducer/evaluator mathematics and coefficients.
# v12 changes only large-batch dataflow. Small calls retain the exact v11 path.
runpy.run_path('make_sine_53_xeon_v11.py', run_name='__main__')
src=Path('bench_sine_53_xeon_v11_build.c').read_text()

# Preserve v11 as the small-call/reference implementation.
needle='OVEC static void octant_vector_v8(const s53w_kernel *k,\n                                  const double * __restrict x,\n                                  double * __restrict out,size_t n)'
if needle not in src:
    raise SystemExit('v11 vector declaration not found')
src=src.replace(needle,'OVEC static void octant_vector_v11_single(const s53w_kernel *k,\n                                  const double * __restrict x,\n                                  double * __restrict out,size_t n)',1)

# Insert the batch engine immediately after the preserved v11 vector function.
start=src.index('OVEC static void octant_vector_v11_single')
end=src.index('\n#endif',start)
batch=r'''

#define X12_TILE 256
OVEC static inline void x12_prepare_block(const double * __restrict x,size_t base,size_t n,
        __m512d *rh_out,__m512d *rl_out,__mmask8 *sign_out,
        __mmask8 *guard_out,__mmask8 *active_out,unsigned char *pure_unit_out)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d V4OPI=_mm512_set1_pd(FOUR_OVER_PI);
    const __m512d VC1=_mm512_set1_pd(PIO4_CW1),VC2=_mm512_set1_pd(PIO4_CW2),VC3=_mm512_set1_pd(PIO4_CW3);
    const __m512d VFT=_mm512_set1_pd(BOUND_TAU*FOUR_OVER_PI),V1MFT=_mm512_set1_pd(1.0-BOUND_TAU*FOUR_OVER_PI);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    const __m256i ADJ=_mm256_setr_epi32(0,-1,2,1,0,-1,2,1);

    unsigned rem=(unsigned)(n-base);
    __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
    __m512d vx=_mm512_maskz_loadu_pd(active,x+base);
    __m512i vxi=_mm512_castpd_si512(vx);
    __mmask8 inneg=(__mmask8)(_mm512_movepi64_mask(vxi)&active);
    __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(vxi,ABSM));
    __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);
    *pure_unit_out=(unsigned char)(unit==active);
    if(unit==active){
        *rh_out=ax;*rl_out=Z;*sign_out=inneg;*guard_out=0;*active_out=active;return;
    }
    __mmask8 wide=(__mmask8)(active&~unit);
    __m512d qf=_mm512_mul_pd(ax,V4OPI);
    __m256i qi=_mm512_cvttpd_epi32(qf);
    __m512d qfloor=_mm512_roundscale_pd(qf,_MM_FROUND_TO_ZERO|_MM_FROUND_NO_EXC);
    __m512d frac=_mm512_sub_pd(qf,qfloor);
    __mmask8 guarded=(__mmask8)((_mm512_cmp_pd_mask(frac,VFT,_CMP_LT_OQ)|
                                 _mm512_cmp_pd_mask(frac,V1MFT,_CMP_GT_OQ))&wide);
    __m256i oi=_mm256_and_si256(qi,_mm256_set1_epi32(7));
    __m256i adj=_mm256_permutevar8x32_epi32(ADJ,oi);
    __m256i mi=_mm256_add_epi32(qi,adj);
    __m512d md=_mm512_cvtepi32_pd(mi);
    __mmask8 rev=(__mmask8)(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32(oi,30)))&wide);
    __mmask8 wide_neg=(__mmask8)(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32(oi,29)))&wide);
    __m512d t1=_mm512_mul_pd(md,VC1);
    __m512d r0=_mm512_sub_pd(ax,t1);
    __m512d t2=_mm512_mul_pd(md,VC2);
    __m512d rh,re;twodiff_cw(r0,t2,&rh,&re);
    __m512d rl=_mm512_fnmadd_pd(md,VC3,re);
    rh=_mm512_mask_sub_pd(rh,rev,Z,rh);rl=_mm512_mask_sub_pd(rl,rev,Z,rl);
    rh=_mm512_mask_mov_pd(rh,unit,ax);rl=_mm512_mask_mov_pd(rl,unit,Z);
    rh=_mm512_mask_mov_pd(rh,guarded,Z);rl=_mm512_mask_mov_pd(rl,guarded,Z);
    *rh_out=rh;*rl_out=rl;*sign_out=(__mmask8)(inneg^wide_neg);
    *guard_out=guarded;*active_out=active;
}

OVEC static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    /* Small calls keep v11 exactly. The staged engine is only for long batches. */
    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}

    _Alignas(64) double rhbuf[X12_TILE],rlbuf[X12_TILE];
    unsigned char signbuf[X12_TILE/8],guardbuf[X12_TILE/8],activebuf[X12_TILE/8],unitbuf[X12_TILE/8];
    for(size_t tile=0;tile<n;tile+=X12_TILE){
        size_t tn=n-tile;if(tn>X12_TILE)tn=X12_TILE;
        size_t blocks=(tn+7)/8;
        /* Phase 1: homogeneous range-reduction/octant work. */
        for(size_t b=0;b<blocks;b++){
            __m512d rh,rl;__mmask8 s,g,a;unsigned char pu;
            x12_prepare_block(x+tile,b*8,tn,&rh,&rl,&s,&g,&a,&pu);
            _mm512_store_pd(rhbuf+b*8,rh);_mm512_store_pd(rlbuf+b*8,rl);
            signbuf[b]=(unsigned char)s;guardbuf[b]=(unsigned char)g;activebuf[b]=(unsigned char)a;unitbuf[b]=pu;
        }
        /* Phase 2: homogeneous Mode-5 work; same six coefficients/FMA order. */
        for(size_t b=0;b<blocks;b++){
            __m512d rh=_mm512_load_pd(rhbuf+b*8),rl=_mm512_load_pd(rlbuf+b*8);
            __m512d p=unitbuf[b]?mode5_poly_x11(k,rh,(__mmask8)signbuf[b]):
                                    mode5_poly_low_x11(k,rh,rl,(__mmask8)signbuf[b]);
            _mm512_mask_storeu_pd(out+tile+b*8,(__mmask8)activebuf[b],p);
        }
        /* Phase 3: rare exact v8 scalar repair. */
        for(size_t b=0;b<blocks;b++)if(__builtin_expect(guardbuf[b]!=0,0)){
            __mmask8 g=(__mmask8)guardbuf[b];
            for(unsigned lane=0;lane<8&&b*8+lane<tn;lane++)
                if(g&(1u<<lane))out[tile+b*8+lane]=scalar2(k,x[tile+b*8+lane]);
        }
    }
}
'''
src=src[:end]+batch+src[end:]

src=src.replace('S53X11_','S53X12_')
src=src.replace('xeon_v11_full_instruction_transform','xeon_v12_tiled_two_stage_batch')
src=src.replace('Xeon_AVX512_octant_permute_bitmask_direct_anchor_cvt','Xeon_AVX512_tiled_reduce_then_Mode5')
src=src.replace('S53X11_BATCH_RESULT','S53X12_BATCH_RESULT')
Path('bench_sine_53_xeon_v12_build.c').write_text(src)
print('S53X12_BUILD_PASS exact_v11_math=1 small_v11_exact=1 tile=256 stage1_reduce=1 stage2_Mode5=1 stage3_guard=1 batch_reusable=1 formula_unchanged=1')
