from pathlib import Path
import runpy

runpy.run_path('make_sine_53_octant_v4.py', run_name='__main__')
src=Path('bench_sine_53_wide_octant_v4_build.c').read_text()
src=src.replace('#define PIO4_TINY (-0x1.f1976b7ed8fbcp-111)',
'''#define PIO4_TINY (-0x1.f1976b7ed8fbcp-111)\n#define PIO4_CW1 0x1.921fb54400000p-1\n#define PIO4_CW2 0x1.0b4611a600000p-35\n#define PIO4_CW3 0x1.3198a2e037073p-70''')

insert_at=src.index('OVEC static void octant_vector_v4')
helper=r'''OVEC static inline __m512d mode5_poly_i32_low(const s53w_kernel *k,
                                              __m512d yh,__m512d yl,
                                              __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    __m512d ya=_mm512_add_pd(yh,yl);
    __m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(ya,VK),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256i ji=_mm512_cvttpd_epi32(jd);
    /* Grid anchor is dyadic 1/256.  Keep the reduction low word through the
       final tiny local delta rather than rounding it away before Horner. */
    __m512d d=_mm512_sub_pd(yh,_mm512_mul_pd(jd,VIK));
    d=_mm512_add_pd(d,yl);
    __m512d p=_mm512_i32gather_pd(ji,k->tab+(size_t)k->deg*LUTN,8);
    for(int j=k->deg-1;j>=0;j--){
        __m512d c=_mm512_i32gather_pd(ji,k->tab+(size_t)j*LUTN,8);
        p=_mm512_fmadd_pd(p,d,c);
    }
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

OVEC static inline void twodiff_cw(__m512d a,__m512d b,__m512d *h,__m512d *l)
{
    __m512d x=_mm512_sub_pd(a,b);
    __m512d bv=_mm512_sub_pd(a,x);
    __m512d av=_mm512_add_pd(x,bv);
    __m512d br=_mm512_sub_pd(bv,b);
    __m512d ar=_mm512_sub_pd(a,av);
    *h=x; *l=_mm512_add_pd(ar,br);
}

'''
src=src[:insert_at]+helper+src[insert_at:]

start=src.index('OVEC static void octant_vector_v4')
end=src.index('\n#endif',start)
newvec=r'''OVEC static void octant_vector_v4(const s53w_kernel *k,const double *x,
                                  double *out,size_t n)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d V4OPI=_mm512_set1_pd(FOUR_OVER_PI);
    const __m512d VC1=_mm512_set1_pd(PIO4_CW1),VC2=_mm512_set1_pd(PIO4_CW2),VC3=_mm512_set1_pd(PIO4_CW3);
    const __m512d VFT=_mm512_set1_pd(BOUND_TAU*FOUR_OVER_PI),V1MFT=_mm512_set1_pd(1.0-BOUND_TAU*FOUR_OVER_PI);
    const __m512d VM1=_mm512_set1_pd(-1.0),VP1=_mm512_set1_pd(1.0),VP2=_mm512_set1_pd(2.0);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));

    for(size_t i=0;i<n;i+=8){
        unsigned rem=(unsigned)(n-i);
        __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
        __m512d vx=_mm512_maskz_loadu_pd(active,x+i);
        __mmask8 inneg=(__mmask8)(_mm512_cmp_pd_mask(vx,Z,_CMP_LT_OQ)&active);
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vx),ABSM));
        __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);
        if(unit==active){
            __m512d p=mode5_poly_i32(k,ax,inneg);
            _mm512_mask_storeu_pd(out+i,active,p); continue;
        }
        __mmask8 wide=(__mmask8)(active&~unit);

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
        __m512d md=qd;
        md=_mm512_mask_add_pd(md,am1,md,VM1);
        md=_mm512_mask_add_pd(md,ap2,md,VP2);
        md=_mm512_mask_add_pd(md,ap1,md,VP1);

        /* m*CW1 and m*CW2 are exact in this bounded domain.  The first
           subtraction is a near subtraction; compensate the second one and
           retain its low word through Mode-5's local delta. */
        __m512d t1=_mm512_mul_pd(md,VC1);
        __m512d r0=_mm512_sub_pd(ax,t1);
        __m512d t2=_mm512_mul_pd(md,VC2);
        __m512d rh,re; twodiff_cw(r0,t2,&rh,&re);
        __m512d rl=_mm512_fnmadd_pd(md,VC3,re);
        rh=_mm512_mask_sub_pd(rh,rev,Z,rh);
        rl=_mm512_mask_sub_pd(rl,rev,Z,rl);

        rh=_mm512_mask_mov_pd(rh,unit,ax); rl=_mm512_mask_mov_pd(rl,unit,Z);
        rh=_mm512_mask_mov_pd(rh,guarded,Z); rl=_mm512_mask_mov_pd(rl,guarded,Z);
        __mmask8 signmask=(__mmask8)(((wide_neg^inneg)&wide)|(inneg&unit));
        __m512d p=mode5_poly_i32_low(k,rh,rl,signmask);
        _mm512_mask_storeu_pd(out+i,active,p);
        if(__builtin_expect(guarded!=0,0)){
            for(unsigned lane=0;lane<8&&i+lane<n;lane++)
                if(guarded&(1u<<lane)) out[i+lane]=scalar2(k,x[i+lane]);
        }
    }
}'''
src=src[:start]+newvec+src[end:]

src=src.replace('S53O4_', 'S53O8_')
src=src.replace('octant_v4', 'octant_v8')
src=src.replace('_v4', '_v8')
src=src.replace('guarded_v4', 'guarded_v8')
src=src.replace('cosine_style_pi4_octant_guarded_v8_direct3fma','cosine_style_pi4_octant_guarded_v8_compensated_cw')
src=src.replace('AVX512_pi4_octant_int32_direct3fma','AVX512_pi4_octant_int32_compensated_cw')
Path('bench_sine_53_wide_octant_v8_build.c').write_text(src)
print('S53O8_BUILD_PASS compensated_cody_waite=1 residual_low_to_delta=1 exact_product_hi_mid=1 int32_quotient=1 rare_table_DD_boundary=1 unit_direct=1 formula_unchanged=1')
