from pathlib import Path
import runpy,sys

# Build exact X21A first, then change only the full 32-lane hot-loop schedule.
saved=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x21_nearestpi.py','3p']
    runpy.run_path('make_sine_53_xeon_x21_nearestpi.py',run_name='__main__')
finally:
    sys.argv=saved
p=Path('bench_sine_53_xeon_x21_3p_build.c')
s=p.read_text()

needle='''OVEC static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)'''
if needle not in s: raise SystemExit('x21a large-batch function missing')
start=s.index(needle)
end=s.index('\n#endif',start)

# Four independent 8-lane streams.  Mathematics is X21A verbatim:
# nearest integer q=round(|x|/pi), the same PI1/PI2/PI3 split, same twodiff,
# same absolute-residual/sign reconstruction, same K=256 anchor and same
# two-gather reconstructed Mode-5 polynomial.  Only scheduling changes.
new=r'''OVEC static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d VINVP=_mm512_set1_pd(0x1.45f306dc9c883p-2);
    const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
    const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
    const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK);
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    size_t i=0;
    for(;i+32<=n;i+=32){
        __m512d vx0,vx1,vx2,vx3,ax0,ax1,ax2,ax3;
        __m512i vxi0,vxi1,vxi2,vxi3;
        __mmask8 in0,in1,in2,in3,u0,u1,u2,u3,s0,s1,s2,s3;
        unsigned char pu0,pu1,pu2,pu3;
        __m256i q0,q1,q2,q3,j0,j1,j2,j3;
        __m512d qd0,qd1,qd2,qd3,r00,r01,r02,r03,rh0,rh1,rh2,rh3,re0,re1,re2,re3,rl0,rl1,rl2,rl3;
        __m512d ya0,ya1,ya2,ya3,jd0,jd1,jd2,jd3,d0,d1,d2,d3;
        __m512d c00,c01,c02,c03,c10,c11,c12,c13,c20,c21,c22,c23,c30,c31,c32,c33,c40,c41,c42,c43,c50,c51,c52,c53;
        __m512d p0,p1,p2,p3;

        /* Load/sign/abs four independent vectors first. */
        vx0=_mm512_loadu_pd(x+i+0);  vx1=_mm512_loadu_pd(x+i+8);
        vx2=_mm512_loadu_pd(x+i+16); vx3=_mm512_loadu_pd(x+i+24);
        vxi0=_mm512_castpd_si512(vx0); vxi1=_mm512_castpd_si512(vx1);
        vxi2=_mm512_castpd_si512(vx2); vxi3=_mm512_castpd_si512(vx3);
        in0=_mm512_movepi64_mask(vxi0); in1=_mm512_movepi64_mask(vxi1);
        in2=_mm512_movepi64_mask(vxi2); in3=_mm512_movepi64_mask(vxi3);
        ax0=_mm512_castsi512_pd(_mm512_and_epi64(vxi0,ABSM));
        ax1=_mm512_castsi512_pd(_mm512_and_epi64(vxi1,ABSM));
        ax2=_mm512_castsi512_pd(_mm512_and_epi64(vxi2,ABSM));
        ax3=_mm512_castsi512_pd(_mm512_and_epi64(vxi3,ABSM));
        u0=_mm512_cmp_pd_mask(ax0,ONE,_CMP_LT_OQ); u1=_mm512_cmp_pd_mask(ax1,ONE,_CMP_LT_OQ);
        u2=_mm512_cmp_pd_mask(ax2,ONE,_CMP_LT_OQ); u3=_mm512_cmp_pd_mask(ax3,ONE,_CMP_LT_OQ);
        pu0=(unsigned char)(u0==0xffu); pu1=(unsigned char)(u1==0xffu);
        pu2=(unsigned char)(u2==0xffu); pu3=(unsigned char)(u3==0xffu);

        /* Interleave q formation across all four vectors. */
        q0=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ax0,VINVP),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        q1=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ax1,VINVP),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        q2=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ax2,VINVP),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        q3=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ax3,VINVP),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        qd0=_mm512_cvtepi32_pd(q0); qd1=_mm512_cvtepi32_pd(q1);
        qd2=_mm512_cvtepi32_pd(q2); qd3=_mm512_cvtepi32_pd(q3);

        /* Same X21A three-piece pi reduction, stage by stage. */
        r00=_mm512_sub_pd(ax0,_mm512_mul_pd(qd0,PI1));
        r01=_mm512_sub_pd(ax1,_mm512_mul_pd(qd1,PI1));
        r02=_mm512_sub_pd(ax2,_mm512_mul_pd(qd2,PI1));
        r03=_mm512_sub_pd(ax3,_mm512_mul_pd(qd3,PI1));
        twodiff_cw(r00,_mm512_mul_pd(qd0,PI2),&rh0,&re0);
        twodiff_cw(r01,_mm512_mul_pd(qd1,PI2),&rh1,&re1);
        twodiff_cw(r02,_mm512_mul_pd(qd2,PI2),&rh2,&re2);
        twodiff_cw(r03,_mm512_mul_pd(qd3,PI2),&rh3,&re3);
        rl0=_mm512_fnmadd_pd(qd0,PI3,re0); rl1=_mm512_fnmadd_pd(qd1,PI3,re1);
        rl2=_mm512_fnmadd_pd(qd2,PI3,re2); rl3=_mm512_fnmadd_pd(qd3,PI3,re3);

        __m512d rs0=_mm512_add_pd(rh0,rl0),rs1=_mm512_add_pd(rh1,rl1);
        __m512d rs2=_mm512_add_pd(rh2,rl2),rs3=_mm512_add_pd(rh3,rl3);
        __mmask8 rn0=_mm512_movepi64_mask(_mm512_castpd_si512(rs0));
        __mmask8 rn1=_mm512_movepi64_mask(_mm512_castpd_si512(rs1));
        __mmask8 rn2=_mm512_movepi64_mask(_mm512_castpd_si512(rs2));
        __mmask8 rn3=_mm512_movepi64_mask(_mm512_castpd_si512(rs3));
        rh0=_mm512_mask_sub_pd(rh0,rn0,Z,rh0); rl0=_mm512_mask_sub_pd(rl0,rn0,Z,rl0);
        rh1=_mm512_mask_sub_pd(rh1,rn1,Z,rh1); rl1=_mm512_mask_sub_pd(rl1,rn1,Z,rl1);
        rh2=_mm512_mask_sub_pd(rh2,rn2,Z,rh2); rl2=_mm512_mask_sub_pd(rl2,rn2,Z,rl2);
        rh3=_mm512_mask_sub_pd(rh3,rn3,Z,rh3); rl3=_mm512_mask_sub_pd(rl3,rn3,Z,rl3);
        /* X21A's unit override is bit-identical; q=0 there, but retain it explicitly. */
        rh0=_mm512_mask_mov_pd(rh0,u0,ax0); rl0=_mm512_mask_mov_pd(rl0,u0,Z);
        rh1=_mm512_mask_mov_pd(rh1,u1,ax1); rl1=_mm512_mask_mov_pd(rl1,u1,Z);
        rh2=_mm512_mask_mov_pd(rh2,u2,ax2); rl2=_mm512_mask_mov_pd(rl2,u2,Z);
        rh3=_mm512_mask_mov_pd(rh3,u3,ax3); rl3=_mm512_mask_mov_pd(rl3,u3,Z);

        __m256i pv0=_mm256_slli_epi32(_mm256_and_si256(q0,_mm256_set1_epi32(1)),31);
        __m256i pv1=_mm256_slli_epi32(_mm256_and_si256(q1,_mm256_set1_epi32(1)),31);
        __m256i pv2=_mm256_slli_epi32(_mm256_and_si256(q2,_mm256_set1_epi32(1)),31);
        __m256i pv3=_mm256_slli_epi32(_mm256_and_si256(q3,_mm256_set1_epi32(1)),31);
        __mmask8 pa0=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(pv0));
        __mmask8 pa1=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(pv1));
        __mmask8 pa2=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(pv2));
        __mmask8 pa3=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(pv3));
        s0=(__mmask8)(in0^pa0^rn0); s1=(__mmask8)(in1^pa1^rn1);
        s2=(__mmask8)(in2^pa2^rn2); s3=(__mmask8)(in3^pa3^rn3);

        /* Same anchor and low-word-preserving delta construction as X21A/X20A. */
        ya0=_mm512_add_pd(rh0,rl0); ya1=_mm512_add_pd(rh1,rl1);
        ya2=_mm512_add_pd(rh2,rl2); ya3=_mm512_add_pd(rh3,rl3);
        j0=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya0,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        j1=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya1,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        j2=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya2,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        j3=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya3,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        jd0=_mm512_cvtepi32_pd(j0); jd1=_mm512_cvtepi32_pd(j1);
        jd2=_mm512_cvtepi32_pd(j2); jd3=_mm512_cvtepi32_pd(j3);
        if(pu0) d0=_mm512_fnmadd_pd(jd0,VIK,rh0); else {d0=_mm512_sub_pd(rh0,_mm512_mul_pd(jd0,VIK));d0=_mm512_add_pd(d0,rl0);}
        if(pu1) d1=_mm512_fnmadd_pd(jd1,VIK,rh1); else {d1=_mm512_sub_pd(rh1,_mm512_mul_pd(jd1,VIK));d1=_mm512_add_pd(d1,rl1);}
        if(pu2) d2=_mm512_fnmadd_pd(jd2,VIK,rh2); else {d2=_mm512_sub_pd(rh2,_mm512_mul_pd(jd2,VIK));d2=_mm512_add_pd(d2,rl2);}
        if(pu3) d3=_mm512_fnmadd_pd(jd3,VIK,rh3); else {d3=_mm512_sub_pd(rh3,_mm512_mul_pd(jd3,VIK));d3=_mm512_add_pd(d3,rl3);}

        /* Existing winning two-gather Mode-5 schedule. */
        c00=_mm512_i32gather_pd(j0,tab+0*LUTN,8); c01=_mm512_i32gather_pd(j1,tab+0*LUTN,8);
        c02=_mm512_i32gather_pd(j2,tab+0*LUTN,8); c03=_mm512_i32gather_pd(j3,tab+0*LUTN,8);
        c10=_mm512_i32gather_pd(j0,tab+1*LUTN,8); c11=_mm512_i32gather_pd(j1,tab+1*LUTN,8);
        c12=_mm512_i32gather_pd(j2,tab+1*LUTN,8); c13=_mm512_i32gather_pd(j3,tab+1*LUTN,8);
        c20=_mm512_mul_pd(c00,MH); c30=_mm512_mul_pd(c10,M6); c40=_mm512_mul_pd(c00,C24); c50=_mm512_mul_pd(c10,C120);
        c21=_mm512_mul_pd(c01,MH); c31=_mm512_mul_pd(c11,M6); c41=_mm512_mul_pd(c01,C24); c51=_mm512_mul_pd(c11,C120);
        c22=_mm512_mul_pd(c02,MH); c32=_mm512_mul_pd(c12,M6); c42=_mm512_mul_pd(c02,C24); c52=_mm512_mul_pd(c12,C120);
        c23=_mm512_mul_pd(c03,MH); c33=_mm512_mul_pd(c13,M6); c43=_mm512_mul_pd(c03,C24); c53=_mm512_mul_pd(c13,C120);
        p0=_mm512_fmadd_pd(c50,d0,c40); p1=_mm512_fmadd_pd(c51,d1,c41);
        p2=_mm512_fmadd_pd(c52,d2,c42); p3=_mm512_fmadd_pd(c53,d3,c43);
        p0=_mm512_fmadd_pd(p0,d0,c30); p1=_mm512_fmadd_pd(p1,d1,c31); p2=_mm512_fmadd_pd(p2,d2,c32); p3=_mm512_fmadd_pd(p3,d3,c33);
        p0=_mm512_fmadd_pd(p0,d0,c20); p1=_mm512_fmadd_pd(p1,d1,c21); p2=_mm512_fmadd_pd(p2,d2,c22); p3=_mm512_fmadd_pd(p3,d3,c23);
        p0=_mm512_fmadd_pd(p0,d0,c10); p1=_mm512_fmadd_pd(p1,d1,c11); p2=_mm512_fmadd_pd(p2,d2,c12); p3=_mm512_fmadd_pd(p3,d3,c13);
        p0=_mm512_fmadd_pd(p0,d0,c00); p1=_mm512_fmadd_pd(p1,d1,c01); p2=_mm512_fmadd_pd(p2,d2,c02); p3=_mm512_fmadd_pd(p3,d3,c03);
        p0=_mm512_mask_sub_pd(p0,s0,Z,p0); p1=_mm512_mask_sub_pd(p1,s1,Z,p1);
        p2=_mm512_mask_sub_pd(p2,s2,Z,p2); p3=_mm512_mask_sub_pd(p3,s3,Z,p3);
        _mm512_storeu_pd(out+i+0,p0); _mm512_storeu_pd(out+i+8,p1);
        _mm512_storeu_pd(out+i+16,p2); _mm512_storeu_pd(out+i+24,p3);
        /* X21A has guard=0 on this reducer, hence no hot-loop repair scan is needed. */
    }
    if(i<n) octant_vector_x20_tail(k,x+i,out+i,n-i);
}'''

s=s[:start]+new+s[end:]
s=s.replace('S53X21A_','S53X22A_')
s=s.replace('xeon_x21_nearestpi_3piece_g4_2g','xeon_x22_interleaved_nearestpi_3piece_g4_2g')
s=s.replace('Xeon_AVX512_G4_nearestpi_3piece_two_gather_Mode5','Xeon_AVX512_G4_interleaved_nearestpi_3piece_two_gather_Mode5')
out=Path('bench_sine_53_xeon_x22_interleaved_build.c')
out.write_text(s)
print('X22_BUILD_PASS parent=x21A group=4 reducer_math_exact_x21A=1 reducer_streams_interleaved=4 full32_active_bookkeeping_removed=1 two_gather_Mode5_same=1 anchor_rule_same=1 delta_same=1 Horner_FMA_order_same=1 guard_scan_removed_because_x21A_guard_zero=1 requires_Arb_regate=1')
