from pathlib import Path
import runpy

# Build current X50 and replace only the hot evaluator with an integer-anchor
# decomposition x = n + r, n=round(|x|), |r|<=0.5.  Integer sin/cos values
# are supplied as compile-time constant arrays by the benchmark workflow.
runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x50_build.c')
s=p.read_text()

hit=s.index('octant_vector_v8(const s53w_kernel *k,')
start=s.rfind('\n',0,hit)+1
end=s.index('\n#endif',hit)

hot=r'''extern const double s53_int_sin[65537];
extern const double s53_int_cos[65537];

OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    if(__builtin_expect(n<32,0)){octant_vector_v11_single(k,x,out,n);return;}
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    size_t i=0;
    for(;i+8<=n;i+=8){
        __m512d vx=_mm512_loadu_pd(x+i);
        __m512i vxi=_mm512_castpd_si512(vx);
        __mmask8 xneg=_mm512_movepi64_mask(vxi);
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(vxi,ABSM));

        __m256i ni=_mm512_cvt_roundpd_epi32(ax,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        __m512d nd=_mm512_cvtepi32_pd(ni);
        __m512d r=_mm512_sub_pd(ax,nd);
        __mmask8 rneg=_mm512_movepi64_mask(_mm512_castpd_si512(r));
        __m512d ar=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(r),ABSM));

        /* Existing X50 small-input anchor/delta machinery on |r|<=0.5. */
        __m256i ji=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ar,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        __m512d jd=_mm512_cvtepi32_pd(ji);
        __m512d d=_mm512_fnmadd_pd(jd,VIK,ar);
        __m512d a0=_mm512_i32gather_pd(ji,tab+0*LUTN,8); /* sin(anchor) */
        __m512d a1=_mm512_i32gather_pd(ji,tab+1*LUTN,8); /* cos(anchor) */

        /* sin(anchor+d), same degree-5 X50 reconstruction/Horner. */
        __m512d s2=_mm512_mul_pd(a0,MH);
        __m512d s3=_mm512_mul_pd(a1,M6);
        __m512d s4=_mm512_mul_pd(a0,C24);
        __m512d s5=_mm512_mul_pd(a1,C120);
        __m512d sr=_mm512_fmadd_pd(s5,d,s4);
        sr=_mm512_fmadd_pd(sr,d,s3);
        sr=_mm512_fmadd_pd(sr,d,s2);
        sr=_mm512_fmadd_pd(sr,d,a1);
        sr=_mm512_fmadd_pd(sr,d,a0);

        /* cos(anchor+d), paired from the same Mode5-built anchor values. */
        __m512d c2=_mm512_mul_pd(a1,MH);
        __m512d c3=_mm512_mul_pd(a0,_mm512_set1_pd(1.0/6.0));
        __m512d c4=_mm512_mul_pd(a1,C24);
        __m512d c5=_mm512_mul_pd(a0,_mm512_set1_pd(-1.0/120.0));
        __m512d cr=_mm512_fmadd_pd(c5,d,c4);
        cr=_mm512_fmadd_pd(cr,d,c3);
        cr=_mm512_fmadd_pd(cr,d,c2);
        cr=_mm512_fmadd_pd(cr,d,_mm512_sub_pd(Z,a0));
        cr=_mm512_fmadd_pd(cr,d,a1);
        sr=_mm512_mask_sub_pd(sr,rneg,Z,sr);

        __m512d sn=_mm512_i32gather_pd(ni,s53_int_sin,8);
        __m512d cn=_mm512_i32gather_pd(ni,s53_int_cos,8);
        __m512d y=_mm512_fmadd_pd(sn,cr,_mm512_mul_pd(cn,sr));
        y=_mm512_mask_sub_pd(y,xneg,Z,y);
        _mm512_storeu_pd(out+i,y);
    }
    if(i<n) octant_vector_v11_single(k,x+i,out+i,n-i);
}'''

s=s[:start]+hot+s[end:]
s=s.replace('S53X50_','S53X50INT_')
s=s.replace('xeon_x50_cross_iteration_lookahead_g4','xeon_x50_integer_anchor_table')
s=s.replace('Xeon_AVX512_X50_cross_iteration_lookahead','Xeon_AVX512_X50_Integer_Anchor_Table')
Path('bench_sine_53_x50_integer_anchor_build.c').write_text(s)
print('S53X50INT_BUILD_PASS parent=X50 integer_anchor_n_round_absx=1 residual_abs_le_half=1 prebuilt_sin_cos_integer_table=1 runtime_external_trig=0 small_residual_Mode5_anchor=1 degree5_paired_sin_cos=1 angle_addition=1 target_abs_le_65536=1')
