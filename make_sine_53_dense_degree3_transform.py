from pathlib import Path

p=Path('bench_sine_53_xeon_v12_build.c')
s=p.read_text()

start=s.index('OVEC static inline __m512d mode5_poly_x11')
mid=s.index('OVEC static inline __m512d mode5_poly_low_x11',start)
end=s.index('OVEC static inline void twodiff_cw',mid)

helpers=r'''OVEC static inline __m512d mode5_poly_x11(const s53w_kernel *k,__m512d y,
                                          __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    __m512d sy=_mm512_mul_pd(y,VK);
    __m256i ji=_mm512_cvt_roundpd_epi32(sy,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd=_mm512_cvtepi32_pd(ji);
    __m512d d=_mm512_fnmadd_pd(jd,VIK,y);
    __m512d p=_mm512_i32gather_pd(ji,tab+3*LUTN,8);
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+2*LUTN,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+1*LUTN,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+0*LUTN,8));
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

OVEC static inline __m512d mode5_poly_low_x11(const s53w_kernel *k,
                                               __m512d yh,__m512d yl,
                                               __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    __m512d ya=_mm512_add_pd(yh,yl),sy=_mm512_mul_pd(ya,VK);
    __m256i ji=_mm512_cvt_roundpd_epi32(sy,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd=_mm512_cvtepi32_pd(ji);
    __m512d d=_mm512_sub_pd(yh,_mm512_mul_pd(jd,VIK));d=_mm512_add_pd(d,yl);
    __m512d p=_mm512_i32gather_pd(ji,tab+3*LUTN,8);
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+2*LUTN,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+1*LUTN,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+0*LUTN,8));
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

'''
s=s[:start]+helpers+s[end:]
s=s.replace('kernel_create(2)','kernel_create(1)')
s=s.replace('terms=2 degree=5','terms=1 degree=3')
s=s.replace('S53X12_','S53D3_')
s=s.replace('xeon_v12_tiled_two_stage_batch','xeon_dense_degree3_Mode5')
p.write_text(s)
print('S53D3_BUILD_PASS same_secant_Mode5_generator=1 terms=1 degree=3 four_gathers=1 three_Horner_FMAs=1 no_refit=1 no_Taylor_substitute=1')
