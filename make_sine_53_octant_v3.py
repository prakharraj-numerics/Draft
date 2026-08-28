from pathlib import Path

src = Path('bench_sine_53_wide_octant_v2.c').read_text()

start = src.index('OVEC static void octant_vector_v2')
end = src.index('\n#endif', start)
new_vec = r'''OVEC static void octant_vector_v2(const s53w_kernel *k,const double *x,
                                  double *out,size_t n)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d V4OPI=_mm512_set1_pd(FOUR_OVER_PI);
    const __m512d VP4H=_mm512_set1_pd(PIO4_HI),VP4L=_mm512_set1_pd(PIO4_LO);
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
            _mm512_mask_storeu_pd(out+i,active,p);
            continue;
        }

        __mmask8 wide=(__mmask8)(active&~unit);
        __m512d qf=_mm512_mul_pd(ax,V4OPI);
        __m256i ki=_mm512_cvttpd_epi32(qf);
        __m512d kd=_mm512_cvtepi32_pd(ki);

        /* Detect possible quotient/boundary ambiguity from the fractional
           coordinate itself.  At |x|<=10000 its rounding error is far below
           the 2^-32-radian guard converted to pi/4 units. */
        __m512d frac=_mm512_sub_pd(qf,kd);
        __mmask8 near0=_mm512_cmp_pd_mask(frac,VFT,_CMP_LT_OQ);
        __mmask8 near1=_mm512_cmp_pd_mask(frac,V1MFT,_CMP_GT_OQ);
        __mmask8 guarded=(__mmask8)((near0|near1)&wide);

        __m256i oi=_mm256_and_si256(ki,_mm256_set1_epi32(7));
        __mmask8 m1=mask_eq_i32(oi,1),m2=mask_eq_i32(oi,2),m3=mask_eq_i32(oi,3);
        __mmask8 m4=mask_eq_i32(oi,4),m5=mask_eq_i32(oi,5),m6=mask_eq_i32(oi,6),m7=mask_eq_i32(oi,7);
        __mmask8 am1=(__mmask8)((m1|m5)&wide);
        __mmask8 ap2=(__mmask8)((m2|m6)&wide);
        __mmask8 ap1=(__mmask8)((m3|m7)&wide);
        __mmask8 rev=(__mmask8)((m2|m3|m6|m7)&wide);
        __mmask8 wide_neg=(__mmask8)((m4|m5|m6|m7)&wide);

        /* Key cosine-derived improvement: do NOT form w and then subtract it
           from pi/4 or pi/2.  Select the appropriate multiple m*pi/4 and form
           the final folded angle directly.  This removes the second
           cancellation which cost up to ~9 ULP on the first AVX port. */
        __m512d md=kd;
        md=_mm512_mask_add_pd(md,am1,md,VM1);
        md=_mm512_mask_add_pd(md,ap2,md,VP2);
        md=_mm512_mask_add_pd(md,ap1,md,VP1);

        __m512d yp=_mm512_fnmadd_pd(md,VP4H,ax);
        yp=_mm512_fnmadd_pd(md,VP4L,yp);          /* ax - m*pi/4 */
        __m512d yn=_mm512_fmadd_pd(md,VP4H,_mm512_sub_pd(Z,ax));
        yn=_mm512_fmadd_pd(md,VP4L,yn);           /* m*pi/4 - ax */
        __m512d y=_mm512_mask_mov_pd(yp,rev,yn);
        y=_mm512_mask_mov_pd(y,unit,ax);

        /* Guarded lanes are overwritten by the DD fallback below.  Zero them
           before the gather so even a quotient exactly on the other side of a
           boundary can never create a negative/out-of-range LUT index. */
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
src = src[:start] + new_vec + src[end:]

# Make diagnostics count exactly the same rare path used by the vector code.
gs = src.index('static int guard_count_v2')
ge = src.index('\n}\n\nstatic int verify_v2', gs) + 2
new_guard = r'''static int guard_count_v2(const double *x,int n)
{
    int c=0; const double ft=BOUND_TAU*FOUR_OVER_PI;
    for(int i=0;i<n;i++){
        double ax=fabs(x[i]);if(ax<1.0)continue;
        double qf=ax*FOUR_OVER_PI;double q=trunc(qf);double frac=qf-q;
        if(frac<ft||frac>1.0-ft)c++;
    }
    return c;
}'''
src = src[:gs] + new_guard + src[ge:]

src = src.replace('S53O2_', 'S53O3_')
src = src.replace('octant_v2', 'octant_v3')
src = src.replace('_v2', '_v3')
src = src.replace('guarded_v2', 'guarded_v3')
src = src.replace('cosine_style_pi4_octant_guarded_v2', 'cosine_style_pi4_octant_guarded_v3_direct_multiple')
src = src.replace('AVX512_pi4_octant_int32_split', 'AVX512_pi4_octant_direct_multiple_int32_split')
Path('bench_sine_53_wide_octant_v3_build.c').write_text(src)
print('S53O3_BUILD_PASS direct_multiple_fold=1 rare_dd_boundary=1 unit_direct=1')
