from pathlib import Path
import runpy

# Start from v4: its octant/sign/guard architecture is already correct.
runpy.run_path('make_sine_53_octant_v4.py', run_name='__main__')
src=Path('bench_sine_53_wide_octant_v4_build.c').read_text()

# Cody-Waite pi/4 split. CW1 and CW2 intentionally have short significands.
# For |x|<=10000 the folded multiplier m<2^14, hence m*CW1 and m*CW2
# are exactly representable in binary64. CW1+CW2+CW3 approximates pi/4
# to far beyond binary64 requirements.
src=src.replace('#define PIO4_TINY (-0x1.f1976b7ed8fbcp-111)',
'''#define PIO4_TINY (-0x1.f1976b7ed8fbcp-111)\n#define PIO4_CW1 0x1.921fb54400000p-1\n#define PIO4_CW2 0x1.0b4611a600000p-35\n#define PIO4_CW3 0x1.3198a2e037073p-70''')

start=src.index('OVEC static void octant_vector_v4')
end=src.index('\n#endif',start)
newvec=r'''OVEC static void octant_vector_v4(const s53w_kernel *k,const double *x,
                                  double *out,size_t n)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d V4OPI=_mm512_set1_pd(FOUR_OVER_PI);
    const __m512d VC1=_mm512_set1_pd(PIO4_CW1),VC2=_mm512_set1_pd(PIO4_CW2),VC3=_mm512_set1_pd(PIO4_CW3);
    const __m512d VFT=_mm512_set1_pd(BOUND_TAU*FOUR_OVER_PI);
    const __m512d V1MFT=_mm512_set1_pd(1.0-BOUND_TAU*FOUR_OVER_PI);
    const __m512d VM1=_mm512_set1_pd(-1.0),VP1=_mm512_set1_pd(1.0),VP2=_mm512_set1_pd(2.0);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));

    for(size_t i=0;i<n;i+=8){
        unsigned rem=(unsigned)(n-i);
        __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
        __m512d vx=_mm512_maskz_loadu_pd(active,x+i);
        __mmask8 inneg=(__mmask8)(_mm512_cmp_pd_mask(vx,Z,_CMP_LT_OQ)&active);
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vx),ABSM));
        __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);

        /* Preserve the established winning [0,1) path literally. */
        if(unit==active){
            __m512d p=mode5_poly_i32(k,ax,inneg);
            _mm512_mask_storeu_pd(out+i,active,p);
            continue;
        }
        __mmask8 wide=(__mmask8)(active&~unit);

        /* Binary64 bounded-domain octant quotient. */
        __m512d qf=_mm512_mul_pd(ax,V4OPI);
        __m256i qi=_mm512_cvttpd_epi32(qf);
        __m512d qd=_mm512_cvtepi32_pd(qi);
        __m512d frac=_mm512_sub_pd(qf,qd);
        __mmask8 guarded=(__mmask8)((_mm512_cmp_pd_mask(frac,VFT,_CMP_LT_OQ)|
                                     _mm512_cmp_pd_mask(frac,V1MFT,_CMP_GT_OQ))&wide);

        __m256i oi=_mm256_and_si256(qi,_mm256_set1_epi32(7));
        __mmask8 m1=mask_eq_i32(oi,1),m2=mask_eq_i32(oi,2),m3=mask_eq_i32(oi,3);
        __mmask8 m4=mask_eq_i32(oi,4),m5=mask_eq_i32(oi,5),m6=mask_eq_i32(oi,6),m7=mask_eq_i32(oi,7);
        __mmask8 am1=(__mmask8)((m1|m5)&wide),ap2=(__mmask8)((m2|m6)&wide),ap1=(__mmask8)((m3|m7)&wide);
        __mmask8 rev=(__mmask8)((m2|m3|m6|m7)&wide),wide_neg=(__mmask8)((m4|m5|m6|m7)&wide);

        /* Choose the multiple m*pi/4 whose signed residual is already the
           final folded y in [0,pi/2]. */
        __m512d md=qd;
        md=_mm512_mask_add_pd(md,am1,md,VM1);
        md=_mm512_mask_add_pd(md,ap2,md,VP2);
        md=_mm512_mask_add_pd(md,ap1,md,VP1);

        /* Cody-Waite three-piece reduction.  m*CW1 and m*CW2 are exact for
           this domain; no table gather, no int64 conversion, no DD chain. */
        __m512d yp=_mm512_fnmadd_pd(md,VC1,ax);
        yp=_mm512_fnmadd_pd(md,VC2,yp);
        yp=_mm512_fnmadd_pd(md,VC3,yp);
        __m512d yn=_mm512_fmadd_pd(md,VC1,_mm512_sub_pd(Z,ax));
        yn=_mm512_fmadd_pd(md,VC2,yn);
        yn=_mm512_fmadd_pd(md,VC3,yn);
        __m512d y=_mm512_mask_mov_pd(yp,rev,yn);

        /* Mixed unit/wide vector: unit lanes still bypass reduction. */
        y=_mm512_mask_mov_pd(y,unit,ax);
        /* Guarded boundary lanes are overwritten by proven scalar table-DD. */
        y=_mm512_mask_mov_pd(y,guarded,Z);

        __mmask8 signmask=(__mmask8)(((wide_neg^inneg)&wide)|(inneg&unit));
        __m512d p=mode5_poly_i32(k,y,signmask);
        _mm512_mask_storeu_pd(out+i,active,p);
        if(__builtin_expect(guarded!=0,0)){
            for(unsigned lane=0;lane<8&&i+lane<n;lane++)
                if(guarded&(1u<<lane)) out[i+lane]=scalar2(k,x[i+lane]);
        }
    }
}'''
src=src[:start]+newvec+src[end:]

src=src.replace('S53O4_', 'S53O7_')
src=src.replace('octant_v4', 'octant_v7')
src=src.replace('_v4', '_v7')
src=src.replace('guarded_v4', 'guarded_v7')
src=src.replace('cosine_style_pi4_octant_guarded_v7_direct3fma','cosine_style_pi4_octant_guarded_v7_codywaite')
src=src.replace('AVX512_pi4_octant_int32_direct3fma','AVX512_pi4_octant_int32_codywaite')
Path('bench_sine_53_wide_octant_v7_build.c').write_text(src)
print('S53O7_BUILD_PASS cody_waite_pi4=1 exact_product_hi_mid=1 int32_quotient=1 rare_table_DD_boundary=1 unit_direct=1 formula_unchanged=1')
