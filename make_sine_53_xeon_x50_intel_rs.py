from pathlib import Path
import runpy

# X50 evaluator unchanged. Replace only the |x|>1 range-reduction frontend
# with the public Intel/SVML-family right-shifter scheme:
#   Y = |x|*(1/pi) + RS
#   N = Y - RS
# parity comes directly from the bit pattern of Y (no FP->int conversion),
# followed by FMA subtraction with Intel's public 3-piece pi split.
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
    const __m512d RS=_mm512_set1_pd(0x1.8p52);
    /* Public Intel/SVML-family FMA pi split. */
    const __m512d PI1=_mm512_castsi512_pd(_mm512_set1_epi64((long long)UINT64_C(0x400921fb54442d18)));
    const __m512d PI2=_mm512_castsi512_pd(_mm512_set1_epi64((long long)UINT64_C(0x3ca1a62633145c06)));
    const __m512d PI3=_mm512_castsi512_pd(_mm512_set1_epi64((long long)UINT64_C(0x395c1cd129024e09)));
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

    /* Intel-style right-shifter reduction. Adding RS rounds ax/pi to the
       nearest representable integer without an FP->integer conversion. The
       LSB of Y's bit pattern carries N parity, exactly as in public SVML code. */
    __m512d Y=_mm512_fmadd_pd(ax,VINVP,RS);
    __m512d N=_mm512_sub_pd(Y,RS);
    __m512i Ybits=_mm512_castpd_si512(Y);
    __m512i paritybits=_mm512_slli_epi64(Ybits,63);
    __mmask8 parity=(__mmask8)(_mm512_movepi64_mask(paritybits)&active);

    /* R = |x| - N*pi using the public Intel three-piece FMA split. */
    __m512d r=_mm512_fnmadd_pd(N,PI1,ax);
    r=_mm512_fnmadd_pd(N,PI2,r);
    r=_mm512_fnmadd_pd(N,PI3,r);

    /* The X50 Mode5/LUT machinery consumes a nonnegative reduced argument.
       Fold residual sign into the final sine sign, preserving the evaluator. */
    __mmask8 rneg=(__mmask8)(_mm512_movepi64_mask(_mm512_castpd_si512(r))&active);
    __m512d ar=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(r),ABSM));
    ar=_mm512_mask_mov_pd(ar,unit,ax);
    parity=(__mmask8)(parity&~unit);
    rneg=(__mmask8)(rneg&~unit);

    *rh_out=ar;*rl_out=Z;
    *sign_out=(__mmask8)((inneg^parity^rneg)&active);
    *guard_out=0;*active_out=active;
}
'''
s=s[:start]+new+s[end:]
s=s.replace('S53X50_','S53X50IRS_')
s=s.replace('xeon_x50_cross_iteration_lookahead_g4','xeon_x50_intel_rightshifter_g4')
s=s.replace('Xeon_AVX512_X50_cross_iteration_lookahead','Xeon_AVX512_X50_Intel_rightshifter')
out=Path('bench_sine_53_xeon_x50_intel_rs_build.c')
out.write_text(s)
print('S53X50IRS_BUILD_PASS parent=X50 reducer_only=1 source_arch=public_Intel_SVML_family right_shifter=0x1.8p52 parity_from_Y_bits=1 fp_to_int_quotient=0 pi_split=Intel_public_FMA_3piece Mode5_unchanged=1 LUT_unchanged=1 Horner_unchanged=1 gather_schedule_unchanged=1 target_abs_le_10000 requires_Arb_regate=1')
