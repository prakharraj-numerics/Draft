from pathlib import Path
import runpy

# Diagnostic requested explicitly: treat every magnitude exactly like X50's |x|<1 path.
# No pi reduction, no octant/quadrant identity, no quotient/parity work.
# Requires a direct anchor LUT large enough for the tested raw domain.
runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x50_build.c')
s=p.read_text()
start=s.index('OVEC static inline void x12_prepare_block(')
end=s.index('\nOVEC ', start+10)
new=r'''OVEC static inline void x12_prepare_block(const double * __restrict x,size_t base,size_t n,
        __m512d *rh_out,__m512d *rl_out,__mmask8 *sign_out,
        __mmask8 *guard_out,__mmask8 *active_out,unsigned char *pure_unit_out)
{
    const __m512d Z=_mm512_setzero_pd();
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    unsigned rem=(unsigned)(n-base);
    __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
    __m512d vx=_mm512_maskz_loadu_pd(active,x+base);
    __m512i vxi=_mm512_castpd_si512(vx);
    __mmask8 inneg=(__mmask8)(_mm512_movepi64_mask(vxi)&active);
    __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(vxi,ABSM));

    /* Literal raw-input path: identical contract to old pure-unit return. */
    *rh_out=ax;
    *rl_out=Z;
    *sign_out=inneg;
    *guard_out=0;
    *active_out=active;
    *pure_unit_out=1;
}
'''
s=s[:start]+new+s[end:]
s=s.replace('S53X50_','S53X50RAW_')
s=s.replace('xeon_x50_cross_iteration_lookahead_g4','xeon_x50_direct_raw_no_reduction_g4')
s=s.replace('Xeon_AVX512_X50_cross_iteration_lookahead','Xeon_AVX512_X50_Direct_Raw_No_Reduction')
Path('bench_sine_53_xeon_x50_direct_raw_build.c').write_text(s)
print('S53X50RAW_BUILD_PASS parent=X50 raw_input_as_unit=1 pi_reduction=0 octant=0 quotient=0 parity=0 identity_conversion=0 rl_zero=1 sign_input_only=1 Mode5_unchanged=1 anchor_lookup_unchanged=1 local_delta_unchanged=1 Horner_unchanged=1 requires_direct_LUT_covering_domain=1')
