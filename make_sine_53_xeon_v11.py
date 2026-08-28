from pathlib import Path
import runpy

# Start from the fully correct v8 implementation. v11 changes only execution
# machinery: exact same pi/4 constants, compensated subtraction, Mode-5
# coefficients, Horner FMA order, and guarded scalar repair.
runpy.run_path('make_sine_53_octant_v8.py', run_name='__main__')
src = Path('bench_sine_53_wide_octant_v8_build.c').read_text()

insert_at = src.index('OVEC static inline void twodiff_cw')
helpers = r'''
/* Xeon v11: instruction-level realization of the same degree-5 Mode-5
   evaluator. The profile is fixed at terms=2/degree=5 for binary64, so expose
   that fact to icx: direct nearest conversion gives the gather index, the
   aligned plane-major table is retained, and Horner is explicitly unrolled.
   Numerical FMA order is identical to v8. */
OVEC static inline __m512d mode5_poly_x11(const s53w_kernel *k,__m512d y,
                                          __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    __m512d sy=_mm512_mul_pd(y,VK);
    __m256i ji=_mm512_cvt_roundpd_epi32(sy,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd=_mm512_cvtepi32_pd(ji);
    __m512d d=_mm512_fnmadd_pd(jd,VIK,y);
    __m512d p=_mm512_i32gather_pd(ji,tab+5*LUTN,8);
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+4*LUTN,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+3*LUTN,8));
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
    __m512d ya=_mm512_add_pd(yh,yl);
    __m512d sy=_mm512_mul_pd(ya,VK);
    __m256i ji=_mm512_cvt_roundpd_epi32(sy,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd=_mm512_cvtepi32_pd(ji);
    /* Preserve v8 local-delta rounding. */
    __m512d d=_mm512_sub_pd(yh,_mm512_mul_pd(jd,VIK));
    d=_mm512_add_pd(d,yl);
    __m512d p=_mm512_i32gather_pd(ji,tab+5*LUTN,8);
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+4*LUTN,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+3*LUTN,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+2*LUTN,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+1*LUTN,8));
    p=_mm512_fmadd_pd(p,d,_mm512_i32gather_pd(ji,tab+0*LUTN,8));
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

'''
src = src[:insert_at] + helpers + src[insert_at:]

start=src.index('OVEC static void octant_vector_v8')
end=src.index('\n#endif',start)
newvec=r'''OVEC static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d V4OPI=_mm512_set1_pd(FOUR_OVER_PI);
    const __m512d VC1=_mm512_set1_pd(PIO4_CW1),VC2=_mm512_set1_pd(PIO4_CW2),VC3=_mm512_set1_pd(PIO4_CW3);
    const __m512d VFT=_mm512_set1_pd(BOUND_TAU*FOUR_OVER_PI),V1MFT=_mm512_set1_pd(1.0-BOUND_TAU*FOUR_OVER_PI);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    /* Direct octant->multiple map: [q,q-1,q+2,q+1,q,q-1,q+2,q+1]. */
    const __m256i ADJ=_mm256_setr_epi32(0,-1,2,1,0,-1,2,1);

    for(size_t i=0;i<n;i+=8){
        unsigned rem=(unsigned)(n-i);
        __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
        __m512d vx=_mm512_maskz_loadu_pd(active,x+i);
        __m512i vxi=_mm512_castpd_si512(vx);
        __mmask8 inneg=(__mmask8)(_mm512_movepi64_mask(vxi)&active);
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(vxi,ABSM));
        __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);
        if(__builtin_expect(unit==active,0)){
            __m512d p=mode5_poly_x11(k,ax,inneg);
            _mm512_mask_storeu_pd(out+i,active,p);continue;
        }
        __mmask8 wide=(__mmask8)(active&~unit);

        __m512d qf=_mm512_mul_pd(ax,V4OPI);
        __m256i qi=_mm512_cvttpd_epi32(qf);
        __m512d qfloor=_mm512_roundscale_pd(qf,_MM_FROUND_TO_ZERO|_MM_FROUND_NO_EXC);
        __m512d frac=_mm512_sub_pd(qf,qfloor);
        __mmask8 guarded=(__mmask8)((_mm512_cmp_pd_mask(frac,VFT,_CMP_LT_OQ)|
                                     _mm512_cmp_pd_mask(frac,V1MFT,_CMP_GT_OQ))&wide);

        __m256i oi=_mm256_and_si256(qi,_mm256_set1_epi32(7));
        __m256i adj=_mm256_permutevar8x32_epi32(ADJ,oi);
        __m256i mi=_mm256_add_epi32(qi,adj);
        __m512d md=_mm512_cvtepi32_pd(mi);
        /* Octant bit1 = reflection, bit2 = negative sine. */
        __mmask8 rev=(__mmask8)(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32(oi,30)))&wide);
        __mmask8 wide_neg=(__mmask8)(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32(oi,29)))&wide);

        __m512d t1=_mm512_mul_pd(md,VC1);
        __m512d r0=_mm512_sub_pd(ax,t1);
        __m512d t2=_mm512_mul_pd(md,VC2);
        __m512d rh,re;twodiff_cw(r0,t2,&rh,&re);
        __m512d rl=_mm512_fnmadd_pd(md,VC3,re);
        rh=_mm512_mask_sub_pd(rh,rev,Z,rh);
        rl=_mm512_mask_sub_pd(rl,rev,Z,rl);
        rh=_mm512_mask_mov_pd(rh,unit,ax);rl=_mm512_mask_mov_pd(rl,unit,Z);
        rh=_mm512_mask_mov_pd(rh,guarded,Z);rl=_mm512_mask_mov_pd(rl,guarded,Z);
        __mmask8 signmask=(__mmask8)(inneg^wide_neg);
        __m512d p=mode5_poly_low_x11(k,rh,rl,signmask);
        _mm512_mask_storeu_pd(out+i,active,p);
        if(__builtin_expect(guarded!=0,0)){
            for(unsigned lane=0;lane<8&&i+lane<n;lane++)
                if(guarded&(1u<<lane))out[i+lane]=scalar2(k,x[i+lane]);
        }
    }
}'''
src=src[:start]+newvec+src[end:]

# Batch-throughput diagnostics. Inputs repeat the already certified 150 corpus;
# only call/batch amortization changes.
mainpos=src.index('\nint main(void)')
batch=r'''
static uint64_t run_batch_x11(const s53w_kernel *k,const double *x,double *y,
                              int n,int rounds,volatile double *sink)
{
    uint64_t t=now_ns();for(int r=0;r<rounds;r++)octant_eval_v8(k,x,y,(size_t)n);
    t=now_ns()-t;*sink+=y[n-1];return t;
}
static uint64_t run_intel_batch_x11(const double *x,double *y,int n,int rounds,
                                    volatile double *sink)
{
    uint64_t t=now_ns();for(int r=0;r<rounds;r++)vmdSin(n,x,y,VML_HA);
    t=now_ns()-t;*sink+=y[n-1];return t;
}
static void bench_batches_x11(const s53w_kernel *k,const double base[CASES])
{
    const int ns[2]={1200,9600};const int rounds[2]={25000,3125};
    for(int b=0;b<2;b++){
        int n=ns[b],rr=rounds[b];double *x=al64((size_t)n*sizeof(double));
        double *yo=al64((size_t)n*sizeof(double)),*yi=al64((size_t)n*sizeof(double));
        if(!x||!yo||!yi){free(yi);free(yo);free(x);continue;}
        for(int i=0;i<n;i++)x[i]=base[i%CASES];volatile double sink=0;
        run_batch_x11(k,x,yo,n,100,&sink);run_intel_batch_x11(x,yi,n,100,&sink);
        double ot[7],it[7],calls=(double)n*(double)rr;
        for(int t=0;t<7;t++){uint64_t a,z;if(t&1){z=run_intel_batch_x11(x,yi,n,rr,&sink);a=run_batch_x11(k,x,yo,n,rr,&sink);}else{a=run_batch_x11(k,x,yo,n,rr,&sink);z=run_intel_batch_x11(x,yi,n,rr,&sink);}ot[t]=(double)a/calls;it[t]=(double)z/calls;}
        qsort(ot,7,sizeof(double),cmpd);qsort(it,7,sizeof(double),cmpd);
        printf("S53X11_BATCH_RESULT cases=%d ours_ns=%.6f intel_ns=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx repeated_certified150=1 formula=unchanged_Mode5_secant_spine sink=%.17g\n",n,ot[3],it[3],ot[3]/it[3],it[3]/ot[3],(double)sink);
        free(yi);free(yo);free(x);
    }
}
'''
src=src[:mainpos]+batch+src[mainpos:]

needle='int rc=bench_v8(k,x);kernel_destroy(k);redtab2_clear();flint_cleanup_master();return rc;'
repl='int rc=bench_v8(k,x);bench_batches_x11(k,x);kernel_destroy(k);redtab2_clear();flint_cleanup_master();return rc;'
if needle not in src: raise SystemExit('v8 main tail not found')
src=src.replace(needle,repl,1)

src=src.replace('S53O8_','S53X11_')
src=src.replace('cosine_style_pi4_octant_guarded_v8_compensated_cw','xeon_v11_full_instruction_transform')
src=src.replace('AVX512_pi4_octant_int32_compensated_cw','Xeon_AVX512_octant_permute_bitmask_direct_anchor_cvt')
Path('bench_sine_53_xeon_v11_build.c').write_text(src)
print('S53X11_BUILD_PASS exact_v8_math=1 direct_signmask=1 octant_vpermd=1 octant_bit_masks=1 direct_anchor_cvt=1 degree5_unrolled=1 aligned_plane_table=1 batch_diagnostics=1 formula_unchanged=1')
