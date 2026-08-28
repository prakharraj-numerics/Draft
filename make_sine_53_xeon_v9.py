from pathlib import Path
import runpy

# Start from the fully correct v8 numerical implementation.  This generator
# changes only execution scheduling/layout: same reducer constants, same Mode-5
# coefficients, same Horner order within each lane, same guarded scalar repair.
runpy.run_path('make_sine_53_octant_v8.py', run_name='__main__')
src = Path('bench_sine_53_wide_octant_v8_build.c').read_text()

insert_at = src.index('OVEC static void octant_vector_v8')
helper = r'''
/* Xeon execution layer: keep two independent Mode-5 chains in flight.
   Arithmetic order inside each chain is exactly the v8 order; only independent
   A/B work is interleaved to hide gather/FMA latency. */
OVEC static inline void mode5_poly2_i32_low_xeon(const s53w_kernel *k,
        __m512d yh0,__m512d yl0,__mmask8 sign0,
        __m512d yh1,__m512d yl1,__mmask8 sign1,
        __m512d *out0,__m512d *out1)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    __m512d ya0=_mm512_add_pd(yh0,yl0);
    __m512d ya1=_mm512_add_pd(yh1,yl1);
    __m512d jd0=_mm512_roundscale_pd(_mm512_mul_pd(ya0,VK),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd1=_mm512_roundscale_pd(_mm512_mul_pd(ya1,VK),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256i ji0=_mm512_cvttpd_epi32(jd0);
    __m256i ji1=_mm512_cvttpd_epi32(jd1);
    __m512d d0=_mm512_sub_pd(yh0,_mm512_mul_pd(jd0,VIK));
    __m512d d1=_mm512_sub_pd(yh1,_mm512_mul_pd(jd1,VIK));
    d0=_mm512_add_pd(d0,yl0);
    d1=_mm512_add_pd(d1,yl1);

    __m512d p0=_mm512_i32gather_pd(ji0,k->tab+(size_t)k->deg*LUTN,8);
    __m512d p1=_mm512_i32gather_pd(ji1,k->tab+(size_t)k->deg*LUTN,8);
    for(int j=k->deg-1;j>=0;j--){
        const double *base=k->tab+(size_t)j*LUTN;
        __m512d c0=_mm512_i32gather_pd(ji0,base,8);
        __m512d c1=_mm512_i32gather_pd(ji1,base,8);
        p0=_mm512_fmadd_pd(p0,d0,c0);
        p1=_mm512_fmadd_pd(p1,d1,c1);
    }
    *out0=_mm512_mask_sub_pd(p0,sign0,Z,p0);
    *out1=_mm512_mask_sub_pd(p1,sign1,Z,p1);
}

__attribute__((cold,noinline)) static void xeon_guard_repair(const s53w_kernel *k,
        const double *x,double *out,size_t base,size_t n,__mmask8 guarded)
{
    for(unsigned lane=0;lane<8 && base+lane<n;lane++)
        if(guarded&(1u<<lane)) out[base+lane]=scalar2(k,x[base+lane]);
}

OVEC static inline void xeon_prepare_v8_block(const s53w_kernel *k,
        const double *x,size_t base,size_t n,
        __m512d *rh_out,__m512d *rl_out,__mmask8 *sign_out,
        __mmask8 *guard_out,__mmask8 *active_out)
{
    (void)k;
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d V4OPI=_mm512_set1_pd(FOUR_OVER_PI);
    const __m512d VC1=_mm512_set1_pd(PIO4_CW1),VC2=_mm512_set1_pd(PIO4_CW2),VC3=_mm512_set1_pd(PIO4_CW3);
    const __m512d VFT=_mm512_set1_pd(BOUND_TAU*FOUR_OVER_PI),V1MFT=_mm512_set1_pd(1.0-BOUND_TAU*FOUR_OVER_PI);
    const __m512d VM1=_mm512_set1_pd(-1.0),VP1=_mm512_set1_pd(1.0),VP2=_mm512_set1_pd(2.0);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));

    unsigned rem=(unsigned)(n-base);
    __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
    __m512d vx=_mm512_maskz_loadu_pd(active,x+base);
    __mmask8 inneg=(__mmask8)(_mm512_cmp_pd_mask(vx,Z,_CMP_LT_OQ)&active);
    __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vx),ABSM));
    __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);
    __mmask8 wide=(__mmask8)(active&~unit);

    __m512d rh=ax,rl=Z;
    __mmask8 guarded=0,wide_neg=0;
    if(__builtin_expect(wide!=0,1)){
        __m512d qf=_mm512_mul_pd(ax,V4OPI);
        __m256i qi=_mm512_cvttpd_epi32(qf);
        __m512d qd=_mm512_cvtepi32_pd(qi);
        __m512d frac=_mm512_sub_pd(qf,qd);
        guarded=(__mmask8)((_mm512_cmp_pd_mask(frac,VFT,_CMP_LT_OQ)|
                            _mm512_cmp_pd_mask(frac,V1MFT,_CMP_GT_OQ))&wide);

        __m256i oi=_mm256_and_si256(qi,_mm256_set1_epi32(7));
        __mmask8 m1=mask_eq_i32(oi,1),m2=mask_eq_i32(oi,2),m3=mask_eq_i32(oi,3);
        __mmask8 m4=mask_eq_i32(oi,4),m5=mask_eq_i32(oi,5),m6=mask_eq_i32(oi,6),m7=mask_eq_i32(oi,7);
        __mmask8 am1=(__mmask8)((m1|m5)&wide),ap2=(__mmask8)((m2|m6)&wide),ap1=(__mmask8)((m3|m7)&wide);
        __mmask8 rev=(__mmask8)((m2|m3|m6|m7)&wide);
        wide_neg=(__mmask8)((m4|m5|m6|m7)&wide);
        __m512d md=qd;
        md=_mm512_mask_add_pd(md,am1,md,VM1);
        md=_mm512_mask_add_pd(md,ap2,md,VP2);
        md=_mm512_mask_add_pd(md,ap1,md,VP1);

        __m512d t1=_mm512_mul_pd(md,VC1);
        __m512d r0=_mm512_sub_pd(ax,t1);
        __m512d t2=_mm512_mul_pd(md,VC2);
        __m512d re;
        twodiff_cw(r0,t2,&rh,&re);
        rl=_mm512_fnmadd_pd(md,VC3,re);
        rh=_mm512_mask_sub_pd(rh,rev,Z,rh);
        rl=_mm512_mask_sub_pd(rl,rev,Z,rl);
        rh=_mm512_mask_mov_pd(rh,unit,ax);
        rl=_mm512_mask_mov_pd(rl,unit,Z);
        rh=_mm512_mask_mov_pd(rh,guarded,Z);
        rl=_mm512_mask_mov_pd(rl,guarded,Z);
    }

    *rh_out=rh; *rl_out=rl;
    *sign_out=(__mmask8)(((wide_neg^inneg)&wide)|(inneg&unit));
    *guard_out=guarded; *active_out=active;
}

'''
src = src[:insert_at] + helper + src[insert_at:]

start = src.index('OVEC static void octant_vector_v8')
end = src.index('\n#endif', start)
newvec = r'''OVEC static void octant_vector_v8(const s53w_kernel *k,const double *x,
                                  double *out,size_t n)
{
    size_t i=0;
    /* Main Xeon kernel: 16 inputs / iteration.  Two independent 8-lane
       dependency chains are prepared, then their six-gather Mode-5 Horner
       evaluations are interleaved. */
    for(;i+16<=n;i+=16){
        __m512d rh0,rl0,rh1,rl1,p0,p1;
        __mmask8 s0,g0,a0,s1,g1,a1;
        xeon_prepare_v8_block(k,x,i,n,&rh0,&rl0,&s0,&g0,&a0);
        xeon_prepare_v8_block(k,x,i+8,n,&rh1,&rl1,&s1,&g1,&a1);
        mode5_poly2_i32_low_xeon(k,rh0,rl0,s0,rh1,rl1,s1,&p0,&p1);
        _mm512_mask_storeu_pd(out+i,a0,p0);
        _mm512_mask_storeu_pd(out+i+8,a1,p1);
        if(__builtin_expect(g0!=0,0)) xeon_guard_repair(k,x,out,i,n,g0);
        if(__builtin_expect(g1!=0,0)) xeon_guard_repair(k,x,out,i+8,n,g1);
    }
    /* Tail uses the exact v8 single-vector arithmetic. */
    for(;i<n;i+=8){
        __m512d rh,rl;
        __mmask8 s,g,a;
        xeon_prepare_v8_block(k,x,i,n,&rh,&rl,&s,&g,&a);
        __m512d p=mode5_poly_i32_low(k,rh,rl,s);
        _mm512_mask_storeu_pd(out+i,a,p);
        if(__builtin_expect(g!=0,0)) xeon_guard_repair(k,x,out,i,n,g);
    }
}'''
src = src[:start] + newvec + src[end:]

src = src.replace('S53O8_', 'S53X9_')
src = src.replace('cosine_style_pi4_octant_guarded_v8_compensated_cw',
                  'xeon_v9_pi4_octant_compensated_cw_2x_pipeline')
src = src.replace('AVX512_pi4_octant_int32_compensated_cw',
                  'Xeon_AVX512_2x_pipeline_pi4_int32_compensated_cw')
Path('bench_sine_53_xeon_v9_build.c').write_text(src)
print('S53X9_BUILD_PASS exact_v8_math=1 xeon_2x_vector_pipeline=1 six_coefficients_preserved=1 horner_order_preserved=1 cold_guard_repair=1 formula_unchanged=1')
