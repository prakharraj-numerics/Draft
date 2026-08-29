from pathlib import Path
import runpy

# X55: keep X50's Mode5-derived LUT, degree-5 reconstruction, G4/32-wide
# scheduling and cross-iteration lookahead.  Specialize the prepare stage for
# full 8-lane blocks whose magnitudes are all >= 1, instead of carrying the
# generic mixed-domain unit-lane masking through the >1 hot path.
runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x50_build.c')
s=p.read_text()

start=s.index('OVEC static inline void x12_prepare_block(')
end=s.index('\nOVEC ', start+10)
old=s[start:end]
generic=old.replace('x12_prepare_block(', 'x55_prepare_generic(', 1)

wrapper=r'''OVEC static inline void x12_prepare_block(const double * __restrict x,size_t base,size_t n,
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

    /* Preserve X50's very cheap direct unit path. */
    if(__builtin_expect(unit==active,0)){
        *rh_out=ax; *rl_out=Z; *sign_out=inneg; *guard_out=0;
        *active_out=active; *pure_unit_out=1; return;
    }

    /* Dedicated full-width >1 path.  No unit-lane blends, no integer quotient
       conversion, no generic octant bookkeeping. */
    if(__builtin_expect(active==0xff && unit==0,1)){
        __m512d Y=_mm512_fmadd_pd(ax,VINVP,RS);
        __m512d N=_mm512_sub_pd(Y,RS);
        __m512i Ybits=_mm512_castpd_si512(Y);
        __m512i paritybits=_mm512_slli_epi64(Ybits,63);
        __mmask8 parity=_mm512_movepi64_mask(paritybits);

        __m512d rh=_mm512_fnmadd_pd(N,PIH,ax);
        __m512d rl=_mm512_mul_pd(N,PIL);
        __m512d rs=_mm512_add_pd(rh,rl);

        __m512d ars=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs),ABSM));
        __mmask8 repair=_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-14),_CMP_LT_OQ);
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

        __mmask8 rneg=_mm512_movepi64_mask(_mm512_castpd_si512(rs));
        rh=_mm512_mask_sub_pd(rh,rneg,Z,rh);
        rl=_mm512_mask_sub_pd(rl,rneg,Z,rl);
        *rh_out=rh; *rl_out=rl;
        *sign_out=(__mmask8)(inneg^parity^rneg);
        *guard_out=0; *active_out=0xff; *pure_unit_out=0; return;
    }

    /* Mixed/partial blocks retain the already-certified X50 behavior. */
    x55_prepare_generic(x,base,n,rh_out,rl_out,sign_out,guard_out,active_out,pure_unit_out);
}
'''

s=s[:start]+generic+'\n'+wrapper+s[end:]
s=s.replace('S53X50_','S53X55_')
s=s.replace('xeon_x50_cross_iteration_lookahead_g4','xeon_x55_wide_specialized_g4')
s=s.replace('Xeon_AVX512_X50_cross_iteration_lookahead','Xeon_AVX512_X55_wide_specialized')
Path('bench_sine_53_xeon_x55_build.c').write_text(s)
print('S53X55_BUILD_PASS parent=X50 dedicated_gt1_full8=1 right_shifter_quotient=1 parity_from_fp_bits=1 compensated_DD=1 rare_3piece_repair=1 mixed_fallback=X50 unit_direct=1 Mode5_unchanged=1 LUT_unchanged=1 Horner_unchanged=1 G4_schedule_unchanged=1 cross_iteration_lookahead=1 requires_Arb_regate=1')
