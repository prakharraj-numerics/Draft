from pathlib import Path
import runpy,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: make_sine_53_xeon_x24_hybrid_group.py <G> <threshold_bits>')
G=int(sys.argv[1]); B=int(sys.argv[2])
if G not in (2,3,4,5,6,8): raise SystemExit('G must be 2,3,4,5,6,8')
if B not in (12,16,20): raise SystemExit('threshold bits must be 12,16,20')

# Start from the generalized one-pass two-gather Mode-5 scheduler, then replace
# only its reducer by the accuracy-gated nearest-pi 2-piece fast path + rare
# 3-piece repair. This lets us re-tune ILP width after the reducer got cheaper.
# x24 width-sweep trigger after workflow registration.
saved=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x21_batch_search.py','plane',str(G)]
    runpy.run_path('make_sine_53_xeon_x21_batch_search.py',run_name='__main__')
finally:
    sys.argv=saved
p=Path(f'bench_sine_53_xeon_x21_plane_g{G}_build.c')
s=p.read_text()
start=s.index('OVEC static inline void x12_prepare_block')
end=s.index('OVEC static void octant_vector_x21_tail',start)
helper=f'''OVEC static inline void x12_prepare_block(const double * __restrict x,size_t base,size_t n,
        __m512d *rh_out,__m512d *rl_out,__mmask8 *sign_out,
        __mmask8 *guard_out,__mmask8 *active_out,unsigned char *pure_unit_out)
{{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d VINVP=_mm512_set1_pd(0x1.45f306dc9c883p-2);
    const __m512d PIH=_mm512_set1_pd(0x1.921fb54442d18p+1);
    const __m512d PILN=_mm512_set1_pd(-0x1.1a62633145c07p-53);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    unsigned rem=(unsigned)(n-base);
    __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
    __m512d vx=_mm512_maskz_loadu_pd(active,x+base);
    __m512i vxi=_mm512_castpd_si512(vx);
    __mmask8 inneg=(__mmask8)(_mm512_movepi64_mask(vxi)&active);
    __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(vxi,ABSM));
    __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);
    *pure_unit_out=(unsigned char)(unit==active);
    if(unit==active){{*rh_out=ax;*rl_out=Z;*sign_out=inneg;*guard_out=0;*active_out=active;return;}}

    __m256i qi=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ax,VINVP),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d qd=_mm512_cvtepi32_pd(qi);
    __m512d rh=_mm512_fnmadd_pd(qd,PIH,ax);
    __m512d rl=_mm512_mul_pd(qd,PILN);

    __m512d rs_fast=_mm512_add_pd(rh,rl);
    __m512d ars=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs_fast),ABSM));
    __mmask8 repair=(__mmask8)(_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-{B}),_CMP_LT_OQ)&active&~unit);
    if(__builtin_expect(repair!=0,0)){{
        const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
        const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
        const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
        __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(qd,PI1));
        __m512d rh3,re3;twodiff_cw(r0,_mm512_mul_pd(qd,PI2),&rh3,&re3);
        __m512d rl3=_mm512_fnmadd_pd(qd,PI3,re3);
        rh=_mm512_mask_mov_pd(rh,repair,rh3);
        rl=_mm512_mask_mov_pd(rl,repair,rl3);
    }}
    __m512d rs=_mm512_add_pd(rh,rl);
    __mmask8 rneg=(__mmask8)(_mm512_movepi64_mask(_mm512_castpd_si512(rs))&active);
    rh=_mm512_mask_sub_pd(rh,rneg,Z,rh); rl=_mm512_mask_sub_pd(rl,rneg,Z,rl);
    rh=_mm512_mask_mov_pd(rh,unit,ax); rl=_mm512_mask_mov_pd(rl,unit,Z);
    __m256i parityv=_mm256_slli_epi32(_mm256_and_si256(qi,_mm256_set1_epi32(1)),31);
    __mmask8 parity=(__mmask8)(_mm256_movemask_ps(_mm256_castsi256_ps(parityv))&active);
    *rh_out=rh;*rl_out=rl;*sign_out=(__mmask8)((inneg^parity^rneg)&active);
    *guard_out=0;*active_out=active;
}}

'''
s=s[:start]+helper+s[end:]
s=s.replace(f'S53X21PLANEG{G}_',f'S53X24G{G}B{B}_')
s=s.replace(f'xeon_x21_plane_g{G}',f'xeon_x24_hybridpi_g{G}_b{B}')
s=s.replace(f'Xeon_AVX512_x21_plane_g{G}',f'Xeon_AVX512_hybridpi_g{G}_b{B}_two_gather')
out=Path(f'bench_sine_53_xeon_x24_g{G}_b{B}_build.c'); out.write_text(s)
print(f'X24_BUILD_PASS group={G} threshold_bits={B} onepass=1 two_gather=1 nearestpi_fast2piece=1 rare3piece_repair=1 same_secant_Mode5_spine=1 same_anchor_rule=1 same_delta=1 same_Horner_FMA_order=1 requires_Arb_regate=1')
