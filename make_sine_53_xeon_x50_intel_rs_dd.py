from pathlib import Path
import runpy

# X50 evaluator/LUT/Horner/scheduling unchanged.
# Intel/SVML-style right-shifter quotient + parity from FP bit pattern,
# combined with X50's compensated (rh,rl) reduced-argument contract.
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
        *rh_out=ax; *rl_out=Z; *sign_out=inneg; *guard_out=0; *active_out=active; return;
    }

    /* Intel/SVML right-shifter architecture: nearest integer N=x/pi without
       vcvt*. Parity is encoded in the low bit of Y's significand. */
    __m512d Y=_mm512_fmadd_pd(ax,VINVP,RS);
    __m512d N=_mm512_sub_pd(Y,RS);
    __m512i Ybits=_mm512_castpd_si512(Y);
    __m512i paritybits=_mm512_slli_epi64(Ybits,63);
    __mmask8 parity=(__mmask8)(_mm512_movepi64_mask(paritybits)&active);

    /* Preserve X50's double-double residual contract. */
    __m512d rh=_mm512_fnmadd_pd(N,PIH,ax);
    __m512d rl=_mm512_mul_pd(N,PIL);
    __m512d rs=_mm512_add_pd(rh,rl);

    /* Rare cancellation repair, same compensated 3-piece machinery already
       used by the accurate nearest-pi bridge, but N came from right-shifter. */
    __m512d ars=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs),ABSM));
    __mmask8 wide=(__mmask8)(active&~unit);
    __mmask8 repair=(__mmask8)(_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-14),_CMP_LT_OQ)&wide);
    if(__builtin_expect(repair!=0,0)){
        const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
        const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
        const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
        __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(N,PI1));
        __m512d rh3,re3; twodiff_cw(r0,_mm512_mul_pd(N,PI2),&rh3,&re3);
        __m512d rl3=_mm512_fnmadd_pd(N,PI3,re3);
        rh=_mm512_mask_mov_pd(rh,repair,rh3);
        rl=_mm512_mask_mov_pd(rl,repair,rl3);
        rs=_mm512_add_pd(rh,rl);
    }

    __mmask8 rneg=(__mmask8)(_mm512_movepi64_mask(_mm512_castpd_si512(rs))&active);
    rh=_mm512_mask_sub_pd(rh,rneg,Z,rh);
    rl=_mm512_mask_sub_pd(rl,rneg,Z,rl);
    rh=_mm512_mask_mov_pd(rh,unit,ax);
    rl=_mm512_mask_mov_pd(rl,unit,Z);
    parity=(__mmask8)(parity&~unit);
    rneg=(__mmask8)(rneg&~unit);

    *rh_out=rh; *rl_out=rl;
    *sign_out=(__mmask8)((inneg^parity^rneg)&active);
    *guard_out=0; *active_out=active;
}
'''
s=s[:start]+new+s[end:]
s=s.replace('S53X50_','S53X50IRSDD_')
s=s.replace('xeon_x50_cross_iteration_lookahead_g4','xeon_x50_intel_rs_dd_g4')
s=s.replace('Xeon_AVX512_X50_cross_iteration_lookahead','Xeon_AVX512_X50_Intel_RS_DD')
Path('bench_sine_53_xeon_x50_intel_rs_dd_build.c').write_text(s)
print('S53X50IRSDD_BUILD_PASS parent=X50 reducer_only=1 Intel_style_right_shifter=1 fp_to_int_quotient=0 parity_from_Y_bits=1 compensated_rh_rl=1 rare_3piece_repair=1 Mode5_unchanged=1 LUT_unchanged=1 Horner_unchanged=1 gather_schedule_unchanged=1 target_abs_le_10000 requires_Arb_regate=1')
