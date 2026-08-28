from pathlib import Path
import runpy, sys

# X30: preserve X23's cheap all-unit G4 path, but otherwise use the X29-U
# branchless delta/reduction structure.  The key is to classify all four AVX-512
# streams before making exactly one scalar control decision.
saved = sys.argv[:]
try:
    sys.argv = ['make_sine_53_xeon_x23_hybridpi.py', '14']
    runpy.run_path('make_sine_53_xeon_x23_hybridpi.py', run_name='__main__')
finally:
    sys.argv = saved

p = Path('bench_sine_53_xeon_x23_h14_build.c')
s = p.read_text()

start = s.index('OVEC static void octant_vector_v8(const s53w_kernel *k,')
end = s.index('\n#endif', start)

helper = r'''
OVEC static inline void x30_classify_block(const double * __restrict x,size_t base,size_t n,
        __m512d *ax_out,__mmask8 *inneg_out,__mmask8 *unit_out,__mmask8 *active_out)
{
    const __m512d ONE=_mm512_set1_pd(1.0);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    unsigned rem=(unsigned)(n-base);
    __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
    __m512d vx=_mm512_maskz_loadu_pd(active,x+base);
    __m512i vxi=_mm512_castpd_si512(vx);
    __mmask8 inneg=(__mmask8)(_mm512_movepi64_mask(vxi)&active);
    __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(vxi,ABSM));
    __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);
    *ax_out=ax; *inneg_out=inneg; *unit_out=unit; *active_out=active;
}

OVEC static inline void x30_reduce_loaded(__m512d ax,__mmask8 inneg,__mmask8 unit,__mmask8 active,
        __m512d *rh_out,__m512d *rl_out,__mmask8 *sign_out,__mmask8 *guard_out)
{
    const __m512d Z=_mm512_setzero_pd();
    const __m512d VINVP=_mm512_set1_pd(0x1.45f306dc9c883p-2);
    const __m512d PIH=_mm512_set1_pd(0x1.921fb54442d18p+1);

    __m256i qi=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ax,VINVP),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d qd=_mm512_cvtepi32_pd(qi);
    __m512d rh=_mm512_fnmadd_pd(qd,PIH,ax);
    __m512d rl=_mm512_mul_pd(qd,_mm512_set1_pd(-0x1.1a62633145c07p-53));

    /* X23-H14 rare near-zero three-piece repair: keep its scalar branch. */
    __m512d rs_fast=_mm512_add_pd(rh,rl);
    __m512i absm=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    __m512d ars=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs_fast),absm));
    __mmask8 repair=(__mmask8)(_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-14),_CMP_LT_OQ)&active&~unit);
    if(__builtin_expect(repair!=0,0)){
        const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
        const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
        const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
        __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(qd,PI1));
        __m512d rh3,re3;twodiff_cw(r0,_mm512_mul_pd(qd,PI2),&rh3,&re3);
        __m512d rl3=_mm512_fnmadd_pd(qd,PI3,re3);
        rh=_mm512_mask_mov_pd(rh,repair,rh3);
        rl=_mm512_mask_mov_pd(rl,repair,rl3);
    }

    __m512d rs=_mm512_add_pd(rh,rl);
    __mmask8 rneg=(__mmask8)(_mm512_movepi64_mask(_mm512_castpd_si512(rs))&active);
    rh=_mm512_mask_sub_pd(rh,rneg,Z,rh);
    rl=_mm512_mask_sub_pd(rl,rneg,Z,rl);
    rh=_mm512_mask_mov_pd(rh,unit,ax);
    rl=_mm512_mask_mov_pd(rl,unit,Z);

    __m256i parityv=_mm256_slli_epi32(_mm256_and_si256(qi,_mm256_set1_epi32(1)),31);
    __mmask8 parity=(__mmask8)(_mm256_movemask_ps(_mm256_castsi256_ps(parityv))&active);
    *rh_out=rh; *rl_out=rl;
    *sign_out=(__mmask8)((inneg^parity^rneg)&active);
    *guard_out=0;
}

OVEC static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    size_t i=0;
    for(;i+32<=n;i+=32){
        __m512d ax0,ax1,ax2,ax3,rh0,rl0,rh1,rl1,rh2,rl2,rh3,rl3;
        __mmask8 in0,in1,in2,in3,u0,u1,u2,u3,a0,a1,a2,a3,s0,s1,s2,s3,g0,g1,g2,g3;
        x30_classify_block(x+i,0,32,&ax0,&in0,&u0,&a0);
        x30_classify_block(x+i,8,32,&ax1,&in1,&u1,&a1);
        x30_classify_block(x+i,16,32,&ax2,&in2,&u2,&a2);
        x30_classify_block(x+i,24,32,&ax3,&in3,&u3,&a3);

        /* Exactly one scalar decision for the four-vector group. */
        unsigned all_unit=(unsigned)((u0==a0)&(u1==a1)&(u2==a2)&(u3==a3));
        if(__builtin_expect(all_unit,0)){
            rh0=ax0; rl0=Z; s0=in0; g0=0;
            rh1=ax1; rl1=Z; s1=in1; g1=0;
            rh2=ax2; rl2=Z; s2=in2; g2=0;
            rh3=ax3; rl3=Z; s3=in3; g3=0;
        } else {
            x30_reduce_loaded(ax0,in0,u0,a0,&rh0,&rl0,&s0,&g0);
            x30_reduce_loaded(ax1,in1,u1,a1,&rh1,&rl1,&s1,&g1);
            x30_reduce_loaded(ax2,in2,u2,a2,&rh2,&rl2,&s2,&g2);
            x30_reduce_loaded(ax3,in3,u3,a3,&rh3,&rl3,&s3,&g3);
        }

        __m512d ya0=_mm512_add_pd(rh0,rl0),ya1=_mm512_add_pd(rh1,rl1),ya2=_mm512_add_pd(rh2,rl2),ya3=_mm512_add_pd(rh3,rl3);
        __m256i ji0=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya0,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        __m256i ji1=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya1,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        __m256i ji2=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya2,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        __m256i ji3=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya3,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        __m512d jd0=_mm512_cvtepi32_pd(ji0),jd1=_mm512_cvtepi32_pd(ji1),jd2=_mm512_cvtepi32_pd(ji2),jd3=_mm512_cvtepi32_pd(ji3);
        __m512d d0=_mm512_fnmadd_pd(jd0,VIK,rh0),d1=_mm512_fnmadd_pd(jd1,VIK,rh1),d2=_mm512_fnmadd_pd(jd2,VIK,rh2),d3=_mm512_fnmadd_pd(jd3,VIK,rh3);
        if(__builtin_expect(!all_unit,1)){
            d0=_mm512_add_pd(d0,rl0); d1=_mm512_add_pd(d1,rl1);
            d2=_mm512_add_pd(d2,rl2); d3=_mm512_add_pd(d3,rl3);
        }

        __m512d c0_0=_mm512_i32gather_pd(ji0,tab+0*LUTN,8);
        __m512d c0_1=_mm512_i32gather_pd(ji1,tab+0*LUTN,8);
        __m512d c0_2=_mm512_i32gather_pd(ji2,tab+0*LUTN,8);
        __m512d c0_3=_mm512_i32gather_pd(ji3,tab+0*LUTN,8);
        __m512d c1_0=_mm512_i32gather_pd(ji0,tab+1*LUTN,8);
        __m512d c1_1=_mm512_i32gather_pd(ji1,tab+1*LUTN,8);
        __m512d c1_2=_mm512_i32gather_pd(ji2,tab+1*LUTN,8);
        __m512d c1_3=_mm512_i32gather_pd(ji3,tab+1*LUTN,8);

        __m512d c2_0=_mm512_mul_pd(c0_0,MH),c3_0=_mm512_mul_pd(c1_0,M6),c4_0=_mm512_mul_pd(c0_0,C24),c5_0=_mm512_mul_pd(c1_0,C120);
        __m512d c2_1=_mm512_mul_pd(c0_1,MH),c3_1=_mm512_mul_pd(c1_1,M6),c4_1=_mm512_mul_pd(c0_1,C24),c5_1=_mm512_mul_pd(c1_1,C120);
        __m512d c2_2=_mm512_mul_pd(c0_2,MH),c3_2=_mm512_mul_pd(c1_2,M6),c4_2=_mm512_mul_pd(c0_2,C24),c5_2=_mm512_mul_pd(c1_2,C120);
        __m512d c2_3=_mm512_mul_pd(c0_3,MH),c3_3=_mm512_mul_pd(c1_3,M6),c4_3=_mm512_mul_pd(c0_3,C24),c5_3=_mm512_mul_pd(c1_3,C120);

        __m512d p0=_mm512_fmadd_pd(c5_0,d0,c4_0),p1=_mm512_fmadd_pd(c5_1,d1,c4_1),p2=_mm512_fmadd_pd(c5_2,d2,c4_2),p3=_mm512_fmadd_pd(c5_3,d3,c4_3);
        p0=_mm512_fmadd_pd(p0,d0,c3_0); p1=_mm512_fmadd_pd(p1,d1,c3_1); p2=_mm512_fmadd_pd(p2,d2,c3_2); p3=_mm512_fmadd_pd(p3,d3,c3_3);
        p0=_mm512_fmadd_pd(p0,d0,c2_0); p1=_mm512_fmadd_pd(p1,d1,c2_1); p2=_mm512_fmadd_pd(p2,d2,c2_2); p3=_mm512_fmadd_pd(p3,d3,c2_3);
        p0=_mm512_fmadd_pd(p0,d0,c1_0); p1=_mm512_fmadd_pd(p1,d1,c1_1); p2=_mm512_fmadd_pd(p2,d2,c1_2); p3=_mm512_fmadd_pd(p3,d3,c1_3);
        p0=_mm512_fmadd_pd(p0,d0,c0_0); p1=_mm512_fmadd_pd(p1,d1,c0_1); p2=_mm512_fmadd_pd(p2,d2,c0_2); p3=_mm512_fmadd_pd(p3,d3,c0_3);

        p0=_mm512_mask_sub_pd(p0,s0,Z,p0); p1=_mm512_mask_sub_pd(p1,s1,Z,p1); p2=_mm512_mask_sub_pd(p2,s2,Z,p2); p3=_mm512_mask_sub_pd(p3,s3,Z,p3);
        _mm512_storeu_pd(out+i,p0); _mm512_storeu_pd(out+i+8,p1); _mm512_storeu_pd(out+i+16,p2); _mm512_storeu_pd(out+i+24,p3);
        if(__builtin_expect(g0!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g0&(1u<<lane)) out[i+lane]=scalar2(k,x[i+lane]);
        if(__builtin_expect(g1!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g1&(1u<<lane)) out[i+8+lane]=scalar2(k,x[i+8+lane]);
        if(__builtin_expect(g2!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g2&(1u<<lane)) out[i+16+lane]=scalar2(k,x[i+16+lane]);
        if(__builtin_expect(g3!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g3&(1u<<lane)) out[i+24+lane]=scalar2(k,x[i+24+lane]);
    }
    if(i<n) octant_vector_x20_tail(k,x+i,out+i,n-i);
}
'''

s = s[:start] + helper + s[end:]
s = s.replace('S53X23H14_', 'S53X30_')
s = s.replace('xeon_x23_nearestpi_2f_hybrid_b14_g4_2g', 'xeon_x30_g4_aggregate_unit_hybrid')
s = s.replace('Xeon_AVX512_G4_nearestpi_2f_hybrid_b14_two_gather_Mode5', 'Xeon_AVX512_X30_G4_aggregate_unit_hybrid')
out = Path('bench_sine_53_xeon_x30_build.c')
out.write_text(s)
print('X30_BUILD_PASS parent=X23H14 G4=1 two_gather=1 aggregate_unit_decision=1 scalar_unit_branches_per_stream=0 all_unit_fastpath_preserved=1 nonunit_x29u_branchless_delta=1 rare_B14_branch_preserved=1 formula_unchanged=1 requires_Arb_regate=1')
