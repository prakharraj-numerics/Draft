from pathlib import Path
import runpy

# v17: same secant/Mode-5 formal polynomial, but exploit the builder identity
#   c0=s, c2=-s/2, c4=s/24
#   c1=c, c3=-c/6, c5=c/120
# to gather only c0,c1 and reconstruct c2..c5 in binary64.
# This changes coefficient *realization* (pre-fused high-precision rounded bits
# are not promised identical for c3/c4/c5), so it MUST be re-certified.
runpy.run_path('make_sine_53_xeon_v12_batch.py', run_name='__main__')
p=Path('bench_sine_53_xeon_v12_build.c')
s=p.read_text()

start=s.index('OVEC static inline __m512d mode5_poly_x11')
end=s.index('OVEC static inline void twodiff_cw',start)
helpers=r'''OVEC static inline __m512d mode5_poly_x11(const s53w_kernel *k,__m512d y,
                                          __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0);
    const __m512d C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    __m512d sy=_mm512_mul_pd(y,VK);
    __m256i ji=_mm512_cvt_roundpd_epi32(sy,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd=_mm512_cvtepi32_pd(ji);
    __m512d d=_mm512_fnmadd_pd(jd,VIK,y);
    __m512d c0=_mm512_i32gather_pd(ji,tab+0*LUTN,8);
    __m512d c1=_mm512_i32gather_pd(ji,tab+1*LUTN,8);
    __m512d c2=_mm512_mul_pd(c0,MH);
    __m512d c3=_mm512_mul_pd(c1,M6);
    __m512d c4=_mm512_mul_pd(c0,C24);
    __m512d c5=_mm512_mul_pd(c1,C120);
    __m512d p=_mm512_fmadd_pd(c5,d,c4);
    p=_mm512_fmadd_pd(p,d,c3);
    p=_mm512_fmadd_pd(p,d,c2);
    p=_mm512_fmadd_pd(p,d,c1);
    p=_mm512_fmadd_pd(p,d,c0);
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

OVEC static inline __m512d mode5_poly_low_x11(const s53w_kernel *k,
                                               __m512d yh,__m512d yl,
                                               __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0);
    const __m512d C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    __m512d ya=_mm512_add_pd(yh,yl);
    __m512d sy=_mm512_mul_pd(ya,VK);
    __m256i ji=_mm512_cvt_roundpd_epi32(sy,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd=_mm512_cvtepi32_pd(ji);
    __m512d d=_mm512_sub_pd(yh,_mm512_mul_pd(jd,VIK));
    d=_mm512_add_pd(d,yl);
    __m512d c0=_mm512_i32gather_pd(ji,tab+0*LUTN,8);
    __m512d c1=_mm512_i32gather_pd(ji,tab+1*LUTN,8);
    __m512d c2=_mm512_mul_pd(c0,MH);
    __m512d c3=_mm512_mul_pd(c1,M6);
    __m512d c4=_mm512_mul_pd(c0,C24);
    __m512d c5=_mm512_mul_pd(c1,C120);
    __m512d p=_mm512_fmadd_pd(c5,d,c4);
    p=_mm512_fmadd_pd(p,d,c3);
    p=_mm512_fmadd_pd(p,d,c2);
    p=_mm512_fmadd_pd(p,d,c1);
    p=_mm512_fmadd_pd(p,d,c0);
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

'''
s=s[:start]+helpers+s[end:]
s=s.replace('S53X12_','S53V17_')
s=s.replace('xeon_v12_tiled_two_stage_batch','xeon_v17_two_gather_reconstructed_Mode5')
Path('bench_sine_53_xeon_v17_twogather_build.c').write_text(s)
print('S53V17_BUILD_PASS same_secant_Mode5_formal_polynomial=1 c0_c1_gathers=2 reconstruct_c2_c3_c4_c5=1 same_anchor_rule=1 same_delta=1 same_Horner_FMA_order=1 fused_coeff_bits_not_guaranteed=1 requires_Arb_regate=1 no_refit=1 no_SVML_inside=1')
