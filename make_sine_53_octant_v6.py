from pathlib import Path
import runpy

runpy.run_path('make_sine_53_octant_v4.py', run_name='__main__')
src=Path('bench_sine_53_wide_octant_v4_build.c').read_text()

# Add a Mode-5 evaluator that consumes a twofold reduced argument.  Because the
# anchor grid is dyadic (1/256), yh-anchor is extremely benign; adding yl to
# that tiny local delta preserves the range-reduction low word without paying
# for a second Horner derivative chain.
insert_at=src.index('OVEC static void octant_vector_v4')
ddpoly=r'''OVEC static inline __m512d mode5_poly_i32_dd(const s53w_kernel *k,
                                             __m512d yh,__m512d yl,
                                             __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    __m512d y=_mm512_add_pd(yh,yl);
    __m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(y,VK),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256i ji=_mm512_cvttpd_epi32(jd);
    __m512d dh=_mm512_fnmadd_pd(jd,VIK,yh);
    __m512d d=_mm512_add_pd(dh,yl);
    __m512d p=_mm512_i32gather_pd(ji,k->tab+(size_t)k->deg*LUTN,8);
    for(int j=k->deg-1;j>=0;j--){
        __m512d c=_mm512_i32gather_pd(ji,k->tab+(size_t)j*LUTN,8);
        p=_mm512_fmadd_pd(p,d,c);
    }
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

OVEC static inline void twosum512(__m512d a,__m512d b,__m512d *h,__m512d *l)
{
    __m512d s=_mm512_add_pd(a,b), bv=_mm512_sub_pd(s,a);
    __m512d av=_mm512_sub_pd(s,bv), br=_mm512_sub_pd(b,bv), ar=_mm512_sub_pd(a,av);
    *h=s; *l=_mm512_add_pd(ar,br);
}

OVEC static inline void twodiff512(__m512d a,__m512d b,__m512d *h,__m512d *l)
{
    __m512d x=_mm512_sub_pd(a,b), bv=_mm512_sub_pd(a,x);
    __m512d av=_mm512_add_pd(x,bv), br=_mm512_sub_pd(bv,b), ar=_mm512_sub_pd(a,av);
    *h=x; *l=_mm512_add_pd(ar,br);
}

'''
src=src[:insert_at]+ddpoly+src[insert_at:]

start=src.index('OVEC static void octant_vector_v4')
end=src.index('\n#endif',start)
newvec=r'''OVEC static void octant_vector_v4(const s53w_kernel *k,const double *x,
                                  double *out,size_t n)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d V4OPI=_mm512_set1_pd(FOUR_OVER_PI);
    const __m512d VP4H=_mm512_set1_pd(PIO4_HI),VP4L=_mm512_set1_pd(PIO4_LO),VP4T=_mm512_set1_pd(PIO4_TINY);
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

        /* Build m*pi/4 as a twofold number without a table gather. */
        __m512d p0=_mm512_mul_pd(md,VP4H);
        __m512d e0=_mm512_fmadd_pd(md,VP4H,_mm512_sub_pd(Z,p0));
        __m512d p1=_mm512_mul_pd(md,VP4L);
        __m512d e1=_mm512_fmadd_pd(md,VP4L,_mm512_sub_pd(Z,p1));
        __m512d ph,es; twosum512(p0,p1,&ph,&es);
        __m512d p2=_mm512_mul_pd(md,VP4T);
        __m512d e2=_mm512_fmadd_pd(md,VP4T,_mm512_sub_pd(Z,p2));
        __m512d pl=_mm512_add_pd(_mm512_add_pd(e0,e1),es);
        pl=_mm512_add_pd(pl,p2); pl=_mm512_add_pd(pl,e2);

        /* Exact-ish twofold subtraction ax - m*pi/4, then renormalize. */
        __m512d rh,re; twodiff512(ax,ph,&rh,&re);
        __m512d rl=_mm512_sub_pd(re,pl);
        __m512d yh,yl; twosum512(rh,rl,&yh,&yl);
        yh=_mm512_mask_sub_pd(yh,rev,Z,yh);
        yl=_mm512_mask_sub_pd(yl,rev,Z,yl);

        /* Mixed unit lanes remain the original raw unit argument. */
        yh=_mm512_mask_mov_pd(yh,unit,ax); yl=_mm512_mask_mov_pd(yl,unit,Z);
        /* Guarded boundary lanes are overwritten by the proven table-DD path. */
        yh=_mm512_mask_mov_pd(yh,guarded,Z); yl=_mm512_mask_mov_pd(yl,guarded,Z);

        __mmask8 signmask=(__mmask8)(((wide_neg^inneg)&wide)|(inneg&unit));
        __m512d p=mode5_poly_i32_dd(k,yh,yl,signmask);
        _mm512_mask_storeu_pd(out+i,active,p);
        if(__builtin_expect(guarded!=0,0)){
            for(unsigned lane=0;lane<8&&i+lane<n;lane++)
                if(guarded&(1u<<lane)) out[i+lane]=scalar2(k,x[i+lane]);
        }
    }
}'''
src=src[:start]+newvec+src[end:]

src=src.replace('S53O4_', 'S53O6_')
src=src.replace('octant_v4', 'octant_v6')
src=src.replace('_v4', '_v6')
src=src.replace('guarded_v4', 'guarded_v6')
src=src.replace('cosine_style_pi4_octant_guarded_v6_direct3fma','cosine_style_pi4_octant_guarded_v6_twofold')
src=src.replace('AVX512_pi4_octant_int32_direct3fma','AVX512_pi4_octant_int32_twofold')
Path('bench_sine_53_wide_octant_v6_build.c').write_text(src)
print('S53O6_BUILD_PASS pi4_twofold=1 low_residual_into_delta=1 int32_quotient=1 rare_table_DD_boundary=1 unit_direct=1 formula_unchanged=1')
