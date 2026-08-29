from pathlib import Path
import runpy

# X62 is a reducer-only experiment on top of X50.
# Preserve the X50 Mode5 evaluator, LUT, local delta, Horner/FMA order and
# cross-iteration gather schedule. Replace only x12_prepare_block's >1
# pi/4/octant frontend with nearest-pi Cody-Waite-style reduction.
runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x50_build.c')
s=p.read_text()
start=s.index('OVEC static inline void x12_prepare_block(')
end=s.index('\nOVEC ', start+10)

new=r'''OVEC static inline void x12_prepare_block(const double * __restrict x,size_t base,size_t n,
        __m512d *rh_out,__m512d *rl_out,__mmask8 *sign_out,
        __mmask8 *guard_out,__mmask8 *active_out,unsigned char *pure_unit_out)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d VINVP=_mm512_set1_pd(0x1.45f306dc9c883p-2);
    const __m512d PIH=_mm512_set1_pd(0x1.921fb54442d18p+1);
    const __m512d PIL=_mm512_set1_pd(-0x1.1a62633145c07p-53);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));

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

    /* Fast path: x = n*pi + r, n = nearest integer(x/pi).
       One FMA with binary64 pi plus the exact low correction keeps the common
       residual accurate without octant classification. */
    __m256i qi=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ax,VINVP),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d qd=_mm512_cvtepi32_pd(qi);
    __m512d rh=_mm512_fnmadd_pd(qd,PIH,ax);
    __m512d rl=_mm512_mul_pd(qd,PIL);
    __m512d rs_fast=_mm512_add_pd(rh,rl);

    /* Near an integer multiple of pi, cancellation is the accuracy-sensitive
       case. Use the existing compensated three-piece pi machinery only there. */
    __m512d ars=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs_fast),ABSM));
    __mmask8 wide=(__mmask8)(active&~unit);
    __mmask8 repair=(__mmask8)(_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-14),_CMP_LT_OQ)&wide);
    if(__builtin_expect(repair!=0,0)){
        const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
        const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
        const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
        __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(qd,PI1));
        __m512d rh3,re3;twodiff_cw(r0,_mm512_mul_pd(qd,PI2),&rh3,&re3);
        __m512d rl3=_mm512_fnmadd_pd(qd,PI3,re3);
        rh=_mm512_mask_mov_pd(rh,repair,rh3);
        rl=_mm512_mask_mov_pd(rl,repair,rl3);
    }

    /* Mode5 wants a nonnegative local argument. Fold r's sign into the final
       sine sign together with original-input sign and parity of n. */
    __m512d rs=_mm512_add_pd(rh,rl);
    __mmask8 rneg=(__mmask8)(_mm512_movepi64_mask(_mm512_castpd_si512(rs))&active);
    rh=_mm512_mask_sub_pd(rh,rneg,Z,rh);
    rl=_mm512_mask_sub_pd(rl,rneg,Z,rl);
    rh=_mm512_mask_mov_pd(rh,unit,ax);
    rl=_mm512_mask_mov_pd(rl,unit,Z);
    __m256i parityv=_mm256_slli_epi32(_mm256_and_si256(qi,_mm256_set1_epi32(1)),31);
    __mmask8 parity=(__mmask8)(_mm256_movemask_ps(_mm256_castsi256_ps(parityv))&active);

    *rh_out=rh;*rl_out=rl;
    *sign_out=(__mmask8)((inneg^parity^rneg)&active);
    *guard_out=0;*active_out=active;
}
'''
s=s[:start]+new+s[end:]
s=s.replace('S53X50_','S53X62_')
s=s.replace('xeon_x50_cross_iteration_lookahead_g4','xeon_x62_nearest_pi_fast_reducer_g4')
s=s.replace('Xeon_AVX512_X50_cross_iteration_lookahead','Xeon_AVX512_X62_nearest_pi_fast_reducer')
pout=Path('bench_sine_53_xeon_x62_build.c')
pout.write_text(s)
print('S53X62_BUILD_PASS parent=X50 reducer_only=1 nearest_pi=1 split_pi_common=2piece_fma near_zero=3piece_compensated parity_sign_xor=1 unit_path_unchanged=1 Mode5_unchanged=1 LUT_unchanged=1 Horner_unchanged=1 gather_schedule_unchanged=1 domain_target_abs_le_500000 requires_Arb_regate=1')
