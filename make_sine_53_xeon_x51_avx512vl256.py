from pathlib import Path
import runpy, sys

# X51: keep X49's exact 512-bit range reduction, but split each 8-lane stream
# into two 4-lane halves for coefficient gathers + local polynomial work.
# This tests narrower hardware gather/FMA behavior without changing the math.
saved=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x49_production.py']
    runpy.run_path('make_sine_53_xeon_x49_production.py',run_name='__main__')
finally:
    sys.argv=saved
p=Path('bench_sine_53_xeon_x49_build.c')
s=p.read_text()
start=s.index('OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8')
end=s.index('\n#endif\n\nstatic inline double unit_scalar_v8',start)
helper=r'''OVEC static inline __m256d x51_eval256(__m256d c0,__m256d c1,__m128i ji,__m256d rh,__m256d rl)
{
    const __m256d VIK=_mm256_set1_pd(INVK);
    const __m256d MH=_mm256_set1_pd(-0.5),M6=_mm256_set1_pd(-1.0/6.0),C24=_mm256_set1_pd(1.0/24.0),C120=_mm256_set1_pd(1.0/120.0);
    __m256d jd=_mm256_cvtepi32_pd(ji);
    __m256d d=_mm256_add_pd(_mm256_fnmadd_pd(jd,VIK,rh),rl);
    __m256d c2=_mm256_mul_pd(c0,MH),c3=_mm256_mul_pd(c1,M6),c4=_mm256_mul_pd(c0,C24),c5=_mm256_mul_pd(c1,C120);
    __m256d p=_mm256_fmadd_pd(c5,d,c4);
    p=_mm256_fmadd_pd(p,d,c3); p=_mm256_fmadd_pd(p,d,c2); p=_mm256_fmadd_pd(p,d,c1); p=_mm256_fmadd_pd(p,d,c0);
    return p;
}

OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}
    const __m512d VK=_mm512_set1_pd(KGRID),Z=_mm512_setzero_pd();
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    size_t i=0;
    for(;i+32<=n;i+=32){
        __m512d rh0,rl0,rh1,rl1,rh2,rl2,rh3,rl3; __m256i ji0,ji1,ji2,ji3;
        __mmask8 s0,g0,a0,s1,g1,a1,s2,g2,a2,s3,g3,a3; unsigned char pu0,pu1,pu2,pu3;
        __m128i j0l,j0h,j1l,j1h,j2l,j2h,j3l,j3h;
        __m256d c00l,c10l,c00h,c10h,c01l,c11l,c01h,c11h;
        __m256d c02l,c12l,c02h,c12h,c03l,c13l,c03h,c13h;

        /* Two-stream rolling pipeline: narrow gathers for stream 0 are in
           flight while stream 1 reduction/index executes. */
        x12_prepare_block(x+i,0,32,&rh0,&rl0,&s0,&g0,&a0,&pu0);
        ji0=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh0,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        j0l=_mm256_castsi256_si128(ji0); j0h=_mm256_extracti128_si256(ji0,1);
        c00l=_mm256_i32gather_pd(tab+0*LUTN,j0l,8); c10l=_mm256_i32gather_pd(tab+1*LUTN,j0l,8);
        c00h=_mm256_i32gather_pd(tab+0*LUTN,j0h,8); c10h=_mm256_i32gather_pd(tab+1*LUTN,j0h,8);

        x12_prepare_block(x+i,8,32,&rh1,&rl1,&s1,&g1,&a1,&pu1);
        ji1=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh1,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        j1l=_mm256_castsi256_si128(ji1); j1h=_mm256_extracti128_si256(ji1,1);
        c01l=_mm256_i32gather_pd(tab+0*LUTN,j1l,8); c11l=_mm256_i32gather_pd(tab+1*LUTN,j1l,8);
        c01h=_mm256_i32gather_pd(tab+0*LUTN,j1h,8); c11h=_mm256_i32gather_pd(tab+1*LUTN,j1h,8);

        __m256d p0l=x51_eval256(c00l,c10l,j0l,_mm512_castpd512_pd256(rh0),_mm512_castpd512_pd256(rl0));
        __m256d p0h=x51_eval256(c00h,c10h,j0h,_mm512_extractf64x4_pd(rh0,1),_mm512_extractf64x4_pd(rl0,1));
        __m512d p0=_mm512_castpd256_pd512(p0l); p0=_mm512_insertf64x4(p0,p0h,1); p0=_mm512_mask_sub_pd(p0,s0,Z,p0); _mm512_storeu_pd(out+i+0,p0);

        x12_prepare_block(x+i,16,32,&rh2,&rl2,&s2,&g2,&a2,&pu2);
        ji2=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh2,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        j2l=_mm256_castsi256_si128(ji2); j2h=_mm256_extracti128_si256(ji2,1);
        c02l=_mm256_i32gather_pd(tab+0*LUTN,j2l,8); c12l=_mm256_i32gather_pd(tab+1*LUTN,j2l,8);
        c02h=_mm256_i32gather_pd(tab+0*LUTN,j2h,8); c12h=_mm256_i32gather_pd(tab+1*LUTN,j2h,8);

        __m256d p1l=x51_eval256(c01l,c11l,j1l,_mm512_castpd512_pd256(rh1),_mm512_castpd512_pd256(rl1));
        __m256d p1h=x51_eval256(c01h,c11h,j1h,_mm512_extractf64x4_pd(rh1,1),_mm512_extractf64x4_pd(rl1,1));
        __m512d p1=_mm512_castpd256_pd512(p1l); p1=_mm512_insertf64x4(p1,p1h,1); p1=_mm512_mask_sub_pd(p1,s1,Z,p1); _mm512_storeu_pd(out+i+8,p1);

        x12_prepare_block(x+i,24,32,&rh3,&rl3,&s3,&g3,&a3,&pu3);
        ji3=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh3,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        j3l=_mm256_castsi256_si128(ji3); j3h=_mm256_extracti128_si256(ji3,1);
        c03l=_mm256_i32gather_pd(tab+0*LUTN,j3l,8); c13l=_mm256_i32gather_pd(tab+1*LUTN,j3l,8);
        c03h=_mm256_i32gather_pd(tab+0*LUTN,j3h,8); c13h=_mm256_i32gather_pd(tab+1*LUTN,j3h,8);

        __m256d p2l=x51_eval256(c02l,c12l,j2l,_mm512_castpd512_pd256(rh2),_mm512_castpd512_pd256(rl2));
        __m256d p2h=x51_eval256(c02h,c12h,j2h,_mm512_extractf64x4_pd(rh2,1),_mm512_extractf64x4_pd(rl2,1));
        __m512d p2=_mm512_castpd256_pd512(p2l); p2=_mm512_insertf64x4(p2,p2h,1); p2=_mm512_mask_sub_pd(p2,s2,Z,p2); _mm512_storeu_pd(out+i+16,p2);

        __m256d p3l=x51_eval256(c03l,c13l,j3l,_mm512_castpd512_pd256(rh3),_mm512_castpd512_pd256(rl3));
        __m256d p3h=x51_eval256(c03h,c13h,j3h,_mm512_extractf64x4_pd(rh3,1),_mm512_extractf64x4_pd(rl3,1));
        __m512d p3=_mm512_castpd256_pd512(p3l); p3=_mm512_insertf64x4(p3,p3h,1); p3=_mm512_mask_sub_pd(p3,s3,Z,p3); _mm512_storeu_pd(out+i+24,p3);

        if(__builtin_expect(g0!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g0&(1u<<lane)) out[i+0+lane]=scalar2(k,x[i+0+lane]);
        if(__builtin_expect(g1!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g1&(1u<<lane)) out[i+8+lane]=scalar2(k,x[i+8+lane]);
        if(__builtin_expect(g2!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g2&(1u<<lane)) out[i+16+lane]=scalar2(k,x[i+16+lane]);
        if(__builtin_expect(g3!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g3&(1u<<lane)) out[i+24+lane]=scalar2(k,x[i+24+lane]);
    }
    if(i<n) octant_vector_x20_tail(k,x+i,out+i,n-i);
}'''
s=s[:start]+helper+s[end:]
s=s.replace('S53X49_','S53X51_')
s=s.replace('Xeon_AVX512_X49_production_hw_schedule','Xeon_AVX512_X51_512reduce_256gather_poly')
s=s.replace('xeon_x49_production_hw_schedule_g4','xeon_x51_512reduce_256gather_poly')
out=Path('bench_sine_53_xeon_x51_build.c')
out.write_text(s)
print('X51_BUILD_PASS parent=X49 math_identical=1 reducer_512bit=1 gather_256bit=1 polynomial_256bit=1 four_lane_halves=1 G4_total32=1 requires_Arb_regate=1')
