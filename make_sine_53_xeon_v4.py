from pathlib import Path
import runpy

# Start from Xeon v3: exact v8 numerical path + gather-scheduled degree-5 core.
runpy.run_path('make_sine_53_xeon_v3.py', run_name='__main__')
src = Path('bench_sine_53_xeon_v3_build.c').read_text()

# Add a two-vector polynomial helper before the vector kernel.
insert_at = src.index('OVEC static void octant_vector_v8')
pair = r'''
/* Two-vector Xeon software pipeline.  For two independent AVX-512 vectors,
   compute both anchor indices/deltas, issue coefficient gathers interleaved,
   then interleave the identical degree-5 Horner chains.  Per-vector arithmetic
   order is unchanged from v3/v8. */
OVEC static inline void mode5_x4_pair(const s53w_kernel *k,
        __m512d yh0,__m512d yl0,__mmask8 sg0,
        __m512d yh1,__m512d yl1,__mmask8 sg1,
        __m512d *out0,__m512d *out1)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    __m512d ya0=_mm512_add_pd(yh0,yl0);
    __m512d ya1=_mm512_add_pd(yh1,yl1);
    __m512d jd0=_mm512_roundscale_pd(_mm512_mul_pd(ya0,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd1=_mm512_roundscale_pd(_mm512_mul_pd(ya1,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256i ji0=_mm512_cvttpd_epi32(jd0), ji1=_mm512_cvttpd_epi32(jd1);
    __m512d d0=_mm512_sub_pd(yh0,_mm512_mul_pd(jd0,VIK)); d0=_mm512_add_pd(d0,yl0);
    __m512d d1=_mm512_sub_pd(yh1,_mm512_mul_pd(jd1,VIK)); d1=_mm512_add_pd(d1,yl1);
    const double *tab=k->tab;
    __m512d a5=_mm512_i32gather_pd(ji0,tab+(size_t)5*LUTN,8);
    __m512d b5=_mm512_i32gather_pd(ji1,tab+(size_t)5*LUTN,8);
    __m512d a4=_mm512_i32gather_pd(ji0,tab+(size_t)4*LUTN,8);
    __m512d b4=_mm512_i32gather_pd(ji1,tab+(size_t)4*LUTN,8);
    __m512d a3=_mm512_i32gather_pd(ji0,tab+(size_t)3*LUTN,8);
    __m512d b3=_mm512_i32gather_pd(ji1,tab+(size_t)3*LUTN,8);
    __m512d a2=_mm512_i32gather_pd(ji0,tab+(size_t)2*LUTN,8);
    __m512d b2=_mm512_i32gather_pd(ji1,tab+(size_t)2*LUTN,8);
    __m512d a1=_mm512_i32gather_pd(ji0,tab+(size_t)1*LUTN,8);
    __m512d b1=_mm512_i32gather_pd(ji1,tab+(size_t)1*LUTN,8);
    __m512d a0=_mm512_i32gather_pd(ji0,tab,8);
    __m512d b0=_mm512_i32gather_pd(ji1,tab,8);
    __m512d p0=_mm512_fmadd_pd(a5,d0,a4);
    __m512d p1=_mm512_fmadd_pd(b5,d1,b4);
    p0=_mm512_fmadd_pd(p0,d0,a3); p1=_mm512_fmadd_pd(p1,d1,b3);
    p0=_mm512_fmadd_pd(p0,d0,a2); p1=_mm512_fmadd_pd(p1,d1,b2);
    p0=_mm512_fmadd_pd(p0,d0,a1); p1=_mm512_fmadd_pd(p1,d1,b1);
    p0=_mm512_fmadd_pd(p0,d0,a0); p1=_mm512_fmadd_pd(p1,d1,b0);
    *out0=_mm512_mask_sub_pd(p0,sg0,Z,p0);
    *out1=_mm512_mask_sub_pd(p1,sg1,Z,p1);
}

OVEC static inline void x4_reduce_block(const double *x,size_t i,size_t n,
        __m512d *yh,__m512d *yl,__mmask8 *sg,__mmask8 *active_out,__mmask8 *guard_out)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d V4OPI=_mm512_set1_pd(FOUR_OVER_PI);
    const __m512d VC1=_mm512_set1_pd(PIO4_CW1),VC2=_mm512_set1_pd(PIO4_CW2),VC3=_mm512_set1_pd(PIO4_CW3);
    const __m512d VFT=_mm512_set1_pd(BOUND_TAU*FOUR_OVER_PI),V1MFT=_mm512_set1_pd(1.0-BOUND_TAU*FOUR_OVER_PI);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    unsigned rem=(unsigned)(n-i);
    __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
    __m512d vx=_mm512_maskz_loadu_pd(active,x+i);
    __mmask8 inneg=(__mmask8)(_mm512_cmp_pd_mask(vx,Z,_CMP_LT_OQ)&active);
    __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vx),ABSM));
    __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);
    if(unit==active){*yh=ax;*yl=Z;*sg=inneg;*active_out=active;*guard_out=0;return;}
    __mmask8 wide=(__mmask8)(active&~unit);
    __m512d qf=_mm512_mul_pd(ax,V4OPI);
    __m256i qi=_mm512_cvttpd_epi32(qf);
    __m512d qd=_mm512_cvtepi32_pd(qi);
    __m512d frac=_mm512_sub_pd(qf,qd);
    __mmask8 guarded=(__mmask8)((_mm512_cmp_pd_mask(frac,VFT,_CMP_LT_OQ)|_mm512_cmp_pd_mask(frac,V1MFT,_CMP_GT_OQ))&wide);
    __m256i oi=_mm256_and_si256(qi,_mm256_set1_epi32(7));
    __m256i rr=_mm256_and_si256(oi,_mm256_set1_epi32(3));
    __m256i b1=_mm256_and_si256(rr,_mm256_set1_epi32(1));
    __m256i b2=_mm256_and_si256(rr,_mm256_set1_epi32(2));
    __m256i mi=_mm256_add_epi32(qi,_mm256_sub_epi32(b2,b1));
    __m512d md=_mm512_cvtepi32_pd(mi);
    __mmask8 rev=(__mmask8)((~mask_eq_i32(b2,0))&wide);
    __m256i bit4=_mm256_and_si256(oi,_mm256_set1_epi32(4));
    __mmask8 wide_neg=(__mmask8)((~mask_eq_i32(bit4,0))&wide);
    __m512d t1=_mm512_mul_pd(md,VC1), r0=_mm512_sub_pd(ax,t1), t2=_mm512_mul_pd(md,VC2);
    __m512d rh,re; twodiff_cw(r0,t2,&rh,&re);
    __m512d rl=_mm512_fnmadd_pd(md,VC3,re);
    rh=_mm512_mask_sub_pd(rh,rev,Z,rh); rl=_mm512_mask_sub_pd(rl,rev,Z,rl);
    rh=_mm512_mask_mov_pd(rh,unit,ax); rl=_mm512_mask_mov_pd(rl,unit,Z);
    rh=_mm512_mask_mov_pd(rh,guarded,Z); rl=_mm512_mask_mov_pd(rl,guarded,Z);
    *yh=rh;*yl=rl;*sg=(__mmask8)(((wide_neg^inneg)&wide)|(inneg&unit));
    *active_out=active;*guard_out=guarded;
}

'''
src = src[:insert_at] + pair + src[insert_at:]

start=src.index('OVEC static void octant_vector_v8')
end=src.index('\n#endif',start)
new_vec=r'''OVEC static void octant_vector_v8(const s53w_kernel *k,const double *x,double *out,size_t n)
{
    const __m512d Z=_mm512_setzero_pd();
    for(size_t i=0;i<n;i+=16){
        __m512d h0,l0,h1=Z,l1=Z,p0,p1; __mmask8 s0,a0,g0,s1=0,a1=0,g1=0;
        x4_reduce_block(x,i,n,&h0,&l0,&s0,&a0,&g0);
        if(i+8<n) x4_reduce_block(x,i+8,n,&h1,&l1,&s1,&a1,&g1);
        mode5_x4_pair(k,h0,l0,s0,h1,l1,s1,&p0,&p1);
        _mm512_mask_storeu_pd(out+i,a0,p0);
        if(a1)_mm512_mask_storeu_pd(out+i+8,a1,p1);
        if(__builtin_expect(g0!=0,0)) for(unsigned lane=0;lane<8&&i+lane<n;lane++) if(g0&(1u<<lane)) out[i+lane]=scalar2(k,x[i+lane]);
        if(__builtin_expect(g1!=0,0)) for(unsigned lane=0;lane<8&&i+8+lane<n;lane++) if(g1&(1u<<lane)) out[i+8+lane]=scalar2(k,x[i+8+lane]);
    }
}'''
src=src[:start]+new_vec+src[end:]
src=src.replace('S53X3_','S53X4_')
src=src.replace('reduction=cosine_style_pi4_octant_guarded_x3_xeon_sched','reduction=cosine_style_pi4_octant_guarded_x4_pairpipe')
src=src.replace('AVX512_pi4_octant_int32_compensated_cw_deg5_gather_sched','AVX512_pi4_octant_int32_compensated_cw_deg5_pairpipe')
Path('bench_sine_53_xeon_v4_build.c').write_text(src)
print('S53X4_BUILD_PASS v3_exact_math=1 two_vector_pipeline=1 interleaved_12_gathers=1 interleaved_horner=1 edge_fallback_unchanged=1')
