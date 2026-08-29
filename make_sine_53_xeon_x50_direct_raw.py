from pathlib import Path
import runpy

# Diagnostic requested explicitly: treat every magnitude exactly like X50's |x|<1 path.
# No pi reduction, no octant/quadrant identity, no quotient/parity work for |x|<=10000.
# Out-of-domain stress lanes are clamped only for safe gathering and then scalar-repaired.
runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x50_build.c')
s=p.read_text()
start=s.index('OVEC static inline void x12_prepare_block(')
end=s.index('\nOVEC ', start+10)
new=r'''OVEC static inline void x12_prepare_block(const double * __restrict x,size_t base,size_t n,
        __m512d *rh_out,__m512d *rl_out,__mmask8 *sign_out,
        __mmask8 *guard_out,__mmask8 *active_out,unsigned char *pure_unit_out)
{
    const __m512d Z=_mm512_setzero_pd(),MAXRAW=_mm512_set1_pd(10000.0);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    unsigned rem=(unsigned)(n-base);
    __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
    __m512d vx=_mm512_maskz_loadu_pd(active,x+base);
    __m512i vxi=_mm512_castpd_si512(vx);
    __mmask8 inneg=(__mmask8)(_mm512_movepi64_mask(vxi)&active);
    __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(vxi,ABSM));
    __mmask8 ood=(__mmask8)(_mm512_cmp_pd_mask(ax,MAXRAW,_CMP_GT_OQ)&active);

    /* Literal raw-input path over the benchmark contract |x|<=10000.  The only
       special handling is for stress inputs outside that diagnostic domain:
       make their gather index safe, then scalar2 repairs those lanes afterward. */
    __m512d safe=_mm512_mask_mov_pd(ax,ood,Z);
    *rh_out=safe;
    *rl_out=Z;
    *sign_out=inneg;
    *guard_out=ood;
    *active_out=active;
    *pure_unit_out=(unsigned char)(ood==0);
}
'''
s=s[:start]+new+s[end:]
s=s.replace('S53X50_','S53X50RAW_')
s=s.replace('xeon_x50_cross_iteration_lookahead_g4','xeon_x50_direct_raw_no_reduction_g4')
s=s.replace('Xeon_AVX512_X50_cross_iteration_lookahead','Xeon_AVX512_X50_Direct_Raw_No_Reduction')
Path('bench_sine_53_xeon_x50_direct_raw_build.c').write_text(s)
print('S53X50RAW_BUILD_PASS parent=X50 raw_input_as_unit=1 timed_domain_abs_le_10000=1 pi_reduction=0 octant=0 quotient=0 parity=0 identity_conversion=0 rl_zero=1 sign_input_only=1 Mode5_unchanged=1 anchor_lookup_unchanged=1 local_delta_unchanged=1 Horner_unchanged=1 ood_stress_scalar_guard=1 requires_direct_LUT_covering_domain=1')
