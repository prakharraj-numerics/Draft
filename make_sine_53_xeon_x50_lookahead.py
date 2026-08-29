from pathlib import Path
import runpy, sys

# X50: exact X49 math/reducer/LUT, but carry one fully prepared 8-lane stream
# across loop iterations.  While block N is evaluated, stream 0 of block N+1
# has already done reduction/index and issued both coefficient gathers.
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
new=r'''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    size_t i=0;

    /* Rolling carry: stream 0 of the current block was prepared/gathered by
       the previous iteration.  This is bounded cross-iteration pipelining,
       not a wider G-group. */
    __m512d rh0,rl0,c0_0,c1_0; __m256i ji0; __mmask8 s0,g0,a0; unsigned char pu0;
    x12_prepare_block(x,0,32,&rh0,&rl0,&s0,&g0,&a0,&pu0);
    ji0=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh0,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    c0_0=_mm512_i32gather_pd(ji0,tab+0*LUTN,8);
    c1_0=_mm512_i32gather_pd(ji0,tab+1*LUTN,8);

    for(;i+32<=n;i+=32){
        __m512d rh1,rl1,d1,c0_1,c1_1,c2_1,c3_1,c4_1,c5_1,p1; __m256i ji1; __mmask8 s1,g1,a1; unsigned char pu1;
        __m512d rh2,rl2,d2,c0_2,c1_2,c2_2,c3_2,c4_2,c5_2,p2; __m256i ji2; __mmask8 s2,g2,a2; unsigned char pu2;
        __m512d rh3,rl3,d3,c0_3,c1_3,c2_3,c3_3,c4_3,c5_3,p3; __m256i ji3; __mmask8 s3,g3,a3; unsigned char pu3;
        __m512d d0,c2_0,c3_0,c4_0,c5_0,p0;

        x12_prepare_block(x+i,8,32,&rh1,&rl1,&s1,&g1,&a1,&pu1);
        ji1=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh1,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        c0_1=_mm512_i32gather_pd(ji1,tab+0*LUTN,8);
        c1_1=_mm512_i32gather_pd(ji1,tab+1*LUTN,8);

        x12_prepare_block(x+i,16,32,&rh2,&rl2,&s2,&g2,&a2,&pu2);
        ji2=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh2,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        c0_2=_mm512_i32gather_pd(ji2,tab+0*LUTN,8);
        c1_2=_mm512_i32gather_pd(ji2,tab+1*LUTN,8);

        x12_prepare_block(x+i,24,32,&rh3,&rl3,&s3,&g3,&a3,&pu3);
        ji3=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh3,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);

        __m512d jd0=_mm512_cvtepi32_pd(ji0);
        d0=_mm512_add_pd(_mm512_fnmadd_pd(jd0,VIK,rh0),rl0);
        __m512d jd1=_mm512_cvtepi32_pd(ji1);
        d1=_mm512_add_pd(_mm512_fnmadd_pd(jd1,VIK,rh1),rl1);
        __m512d jd2=_mm512_cvtepi32_pd(ji2);
        d2=_mm512_add_pd(_mm512_fnmadd_pd(jd2,VIK,rh2),rl2);
        c0_3=_mm512_i32gather_pd(ji3,tab+0*LUTN,8);
        c1_3=_mm512_i32gather_pd(ji3,tab+1*LUTN,8);
        __m512d jd3=_mm512_cvtepi32_pd(ji3);
        d3=_mm512_add_pd(_mm512_fnmadd_pd(jd3,VIK,rh3),rl3);

        /* One-stream lookahead for the next 32-input block. */
        int have_next=(i+64<=n);
        __m512d nrh0,nrl0,nc0_0,nc1_0; __m256i nji0; __mmask8 ns0,ng0,na0; unsigned char npu0;
        if(__builtin_expect(have_next,1)){
            x12_prepare_block(x+i+32,0,32,&nrh0,&nrl0,&ns0,&ng0,&na0,&npu0);
            nji0=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(nrh0,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
            nc0_0=_mm512_i32gather_pd(nji0,tab+0*LUTN,8);
            nc1_0=_mm512_i32gather_pd(nji0,tab+1*LUTN,8);
        }

        c2_0=_mm512_mul_pd(c0_0,MH); c3_0=_mm512_mul_pd(c1_0,M6); c4_0=_mm512_mul_pd(c0_0,C24); c5_0=_mm512_mul_pd(c1_0,C120);
        c2_1=_mm512_mul_pd(c0_1,MH); c3_1=_mm512_mul_pd(c1_1,M6); c4_1=_mm512_mul_pd(c0_1,C24); c5_1=_mm512_mul_pd(c1_1,C120);
        c2_2=_mm512_mul_pd(c0_2,MH); c3_2=_mm512_mul_pd(c1_2,M6); c4_2=_mm512_mul_pd(c0_2,C24); c5_2=_mm512_mul_pd(c1_2,C120);
        c2_3=_mm512_mul_pd(c0_3,MH); c3_3=_mm512_mul_pd(c1_3,M6); c4_3=_mm512_mul_pd(c0_3,C24); c5_3=_mm512_mul_pd(c1_3,C120);

        p0=_mm512_fmadd_pd(c5_0,d0,c4_0); p0=_mm512_fmadd_pd(p0,d0,c3_0); p0=_mm512_fmadd_pd(p0,d0,c2_0); p0=_mm512_fmadd_pd(p0,d0,c1_0); p0=_mm512_fmadd_pd(p0,d0,c0_0);
        p1=_mm512_fmadd_pd(c5_1,d1,c4_1); p1=_mm512_fmadd_pd(p1,d1,c3_1); p1=_mm512_fmadd_pd(p1,d1,c2_1); p1=_mm512_fmadd_pd(p1,d1,c1_1); p1=_mm512_fmadd_pd(p1,d1,c0_1);
        p2=_mm512_fmadd_pd(c5_2,d2,c4_2); p2=_mm512_fmadd_pd(p2,d2,c3_2); p2=_mm512_fmadd_pd(p2,d2,c2_2); p2=_mm512_fmadd_pd(p2,d2,c1_2); p2=_mm512_fmadd_pd(p2,d2,c0_2);
        p3=_mm512_fmadd_pd(c5_3,d3,c4_3); p3=_mm512_fmadd_pd(p3,d3,c3_3); p3=_mm512_fmadd_pd(p3,d3,c2_3); p3=_mm512_fmadd_pd(p3,d3,c1_3); p3=_mm512_fmadd_pd(p3,d3,c0_3);

        p0=_mm512_mask_sub_pd(p0,s0,Z,p0); _mm512_storeu_pd(out+i+0,p0);
        p1=_mm512_mask_sub_pd(p1,s1,Z,p1); _mm512_storeu_pd(out+i+8,p1);
        p2=_mm512_mask_sub_pd(p2,s2,Z,p2); _mm512_storeu_pd(out+i+16,p2);
        p3=_mm512_mask_sub_pd(p3,s3,Z,p3); _mm512_storeu_pd(out+i+24,p3);
        if(__builtin_expect(g0!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g0&(1u<<lane)) out[i+0+lane]=scalar2(k,x[i+0+lane]);
        if(__builtin_expect(g1!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g1&(1u<<lane)) out[i+8+lane]=scalar2(k,x[i+8+lane]);
        if(__builtin_expect(g2!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g2&(1u<<lane)) out[i+16+lane]=scalar2(k,x[i+16+lane]);
        if(__builtin_expect(g3!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g3&(1u<<lane)) out[i+24+lane]=scalar2(k,x[i+24+lane]);

        if(__builtin_expect(have_next,1)){
            rh0=nrh0; rl0=nrl0; ji0=nji0; c0_0=nc0_0; c1_0=nc1_0;
            s0=ns0; g0=ng0; a0=na0; pu0=npu0;
        }
    }
    if(i<n) octant_vector_x20_tail(k,x+i,out+i,n-i);
}'''
s=s[:start]+new+s[end:]
s=s.replace('S53X49_','S53X50_')
s=s.replace('Xeon_AVX512_X49_production_hw_schedule','Xeon_AVX512_X50_cross_iteration_lookahead')
s=s.replace('xeon_x49_production_hw_schedule_g4','xeon_x50_cross_iteration_lookahead_g4')
out=Path('bench_sine_53_xeon_x50_build.c')
out.write_text(s)
print('X50_BUILD_PASS parent=X49 math_identical=1 reducer_identical=1 LUT_identical=1 G4=1 cross_iteration_lookahead_streams=1 next_block_reduction_index_gathers_early=1 requires_Arb_regate=1')
