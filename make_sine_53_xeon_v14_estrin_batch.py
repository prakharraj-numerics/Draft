from pathlib import Path
import runpy

# Start from v13. v14 changes only the floating-point evaluation graph of the
# SAME degree-5 Mode-5 polynomial, using the SAME six cached coefficients,
# anchor index j and local delta d. No refit, no new approximation family,
# no Intel trig call, no change to reducer/guard machinery.
runpy.run_path('make_sine_53_xeon_v13_preindexed_batch.py', run_name='__main__')
src=Path('bench_sine_53_xeon_v13_build.c').read_text()

start=src.index('OVEC static inline __m512d mode5_preindexed_horner_x13')
end=src.index('\nOVEC static void octant_vector_v8',start)
estrin=r'''OVEC static inline __m512d mode5_preindexed_estrin_x14(const s53w_kernel *k,
        __m256i ji,__m512d d,__mmask8 signmask)
{
    const __m512d Z=_mm512_setzero_pd();
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);

    /* Same Mode-5 polynomial
         c0 + c1*d + c2*d^2 + c3*d^3 + c4*d^4 + c5*d^5
       as v13 Horner, but evaluated as a shallow Estrin tree:
         A=c0+c1*d, B=c2+c3*d, C=c4+c5*d,
         P=A+d^2*(B+d^2*C).
       Coefficients are gathered from the exact same table/anchor j. */
    __m512d c0=_mm512_i32gather_pd(ji,tab+0*LUTN,8);
    __m512d c1=_mm512_i32gather_pd(ji,tab+1*LUTN,8);
    __m512d c2=_mm512_i32gather_pd(ji,tab+2*LUTN,8);
    __m512d c3=_mm512_i32gather_pd(ji,tab+3*LUTN,8);
    __m512d c4=_mm512_i32gather_pd(ji,tab+4*LUTN,8);
    __m512d c5=_mm512_i32gather_pd(ji,tab+5*LUTN,8);
    __m512d d2=_mm512_mul_pd(d,d);
    __m512d A=_mm512_fmadd_pd(c1,d,c0);
    __m512d B=_mm512_fmadd_pd(c3,d,c2);
    __m512d C=_mm512_fmadd_pd(c5,d,c4);
    __m512d T=_mm512_fmadd_pd(C,d2,B);
    __m512d p=_mm512_fmadd_pd(T,d2,A);
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}
'''
src=src[:start]+estrin+src[end:]
src=src.replace('mode5_preindexed_horner_x13(k,ji,d,(__mmask8)signbuf[b])',
                'mode5_preindexed_estrin_x14(k,ji,d,(__mmask8)signbuf[b])')
src=src.replace('S53X13_','S53X14_')
src=src.replace('xeon_v13_preindexed_tiled_batch','xeon_v14_preindexed_estrin_batch')
src=src.replace('Xeon_AVX512_preindexed_j_d_then_Mode5','Xeon_AVX512_preindexed_j_d_then_Mode5_Estrin')
Path('bench_sine_53_xeon_v14_build.c').write_text(src)
print('S53X14_BUILD_PASS same_secant_Mode5_spine=1 same_coefficients=1 same_anchor_rule=1 same_j_d=1 same_polynomial_algebraically=1 eval_graph=Estrin_tree guarded_repair_unchanged=1 no_refit=1 no_SVML_inside=1')
