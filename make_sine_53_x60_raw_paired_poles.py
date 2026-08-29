from pathlib import Path
import runpy

# X60: direct raw-x secant partial-fraction derivative path.
# No pi/modulo/quadrant/parity/Cody-Waite reduction.  Raw x enters
# S(x)=sec(x) partial fractions; B=S'(x); sin(x)=B/S^2.
# 2048 paired pole blocks are evaluated directly in AVX-512, with a
# 64th-degree analytically precomputed infinite-tail correction.
runpy.run_path('make_sine_53_xeon_x56_wide_specialized_cold.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x56_build.c')
s=p.read_text()
hit=s.index('octant_vector_v8(const s53w_kernel *k,')
start=s.rfind('\n',0,hit)+1
end=s.index('\n#endif',hit)
orig=s[start:end]
general=orig.replace('octant_vector_v8(', 'octant_vector_v8_x56_general(', 1)

U='''
0x1.ffffff800000ap-15,0x1.000ffebfe8045p-14,0x1.001ffdbf90105p-14,0x1.002ffcbed8286p-14,
0x1.003ffbbda0507p-14,0x1.004ffabbc88cap-14,0x1.005ff9b930e10p-14,0x1.006ff8b5b951ap-14,
0x1.007ff7b141e2ap-14,0x1.008ff6abaa983p-14,0x1.009ff5a4d3768p-14,0x1.00aff49c9c81dp-14,
0x1.00bff392e5be6p-14,0x1.00cff2878f308p-14,0x1.00dff17a78dc9p-14,0x1.00eff06b90fdbp-14,
0x1.00ffef5a9d22ep-14,0x1.010fee478995fp-14,0x1.011fed3236555p-14,0x1.012fec1a835fbp-14,
0x1.013feb0050b46p-14,0x1.014fe9e37e534p-14,0x1.015fe8c3ec3cbp-14,0x1.016fe7a17a71ep-14,
0x1.017fe67c08f49p-14,0x1.018fe55377c75p-14,0x1.019fe427a6ed4p-14,0x1.01afe2f8766a3p-14,
0x1.01bfe1c5c6429p-14,0x1.01cfe08f767b5p-14,0x1.01dfdf55671a2p-14,0x1.01efde1778251p-14,
0x1.01ffdcd589a29p-14,0x1.020fdb8f7b99cp-14,0x1.021fda452e11fp-14,0x1.022fd8f68112fp-14,
0x1.023fd7a354a4bp-14,0x1.024fd64b88cfbp-14,0x1.025fd4eefd9c7p-14,0x1.026fd38d9313dp-14,
0x1.027fd227293eep-14,0x1.028fd0bba026ep-14,0x1.029fcf4ad7d53p-14,0x1.02afcdd4b0536p-14,
0x1.02bfcc5909ab2p-14,0x1.02cfcad7c3e64p-14,0x1.02dfc950bf0ebp-14,0x1.02efc7c3db2e7p-14,
0x1.02ffc630f84fcp-14,0x1.030fc497f67cdp-14,0x1.031fc2f8b5c01p-14,0x1.032fc1531623dp-14,
0x1.033fbfa6f7b2dp-14,0x1.034fbdf43a77ap-14,0x1.035fbc3abe7d1p-14,0x1.036fba7a63ce2p-14,
0x1.037fb8b30a75cp-14,0x1.038fb6e4927f2p-14,0x1.039fb50edbf5ap-14,0x1.03afb331c6e4bp-14,
0x1.03bfb14d3357fp-14,0x1.03cfaf61015b2p-14,0x1.03dfad6d10fa3p-14,0x1.03efab7142414p-14,
0x1.03ffa96d753cap-14'''.strip()

raw=f'''static const double x60_tailU[65]={{ {U} }};
OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawpoles(const s53w_kernel *k,
                                  const double * __restrict x,double * __restrict out,size_t n)
{{
    (void)k;
    const __m512d C4PI=_mm512_set1_pd(0x1.45f306dc9c883p+0); /* 4/pi */
    const __m512d C4PI2=_mm512_set1_pd(0x1.9f9c02660badfp-2); /* 4/pi^2 */
    const __m512d C8PI2=_mm512_set1_pd(0x1.9f9c02660badfp-1); /* 8/pi^2 */
    const __m512d ONE=_mm512_set1_pd(1.0);
    const __m512d R0=_mm512_set1_pd(67125249.0); /* (4*2048+1)^2 */
    size_t i=0;
    for(;i+8<=n;i+=8){{
        __m512d vx=_mm512_loadu_pd(x+i);
        __m512d a=_mm512_mul_pd(_mm512_mul_pd(vx,vx),C4PI2);
        __m512d s0=_mm512_setzero_pd(),s1=s0,s2=s0,s3=s0;
        __m512d d0=s0,d1=s0,d2=s0,d3=s0;
        for(int kk=0;kk<2048;kk++){{
            double ps=4.0*(double)kk+1.0, qs=ps+2.0;
            __m512d p=_mm512_set1_pd(ps),q=_mm512_set1_pd(qs);
            __m512d dp=_mm512_sub_pd(_mm512_set1_pd(ps*ps),a);
            __m512d dq=_mm512_sub_pd(_mm512_set1_pd(qs*qs),a);
            __m512d ip=_mm512_div_pd(ONE,dp), iq=_mm512_div_pd(ONE,dq);
            __m512d sp=_mm512_sub_pd(_mm512_mul_pd(p,ip),_mm512_mul_pd(q,iq));
            __m512d dd=_mm512_sub_pd(_mm512_mul_pd(p,_mm512_mul_pd(ip,ip)),_mm512_mul_pd(q,_mm512_mul_pd(iq,iq)));
            switch(kk&3){{case 0:s0=_mm512_add_pd(s0,sp);d0=_mm512_add_pd(d0,dd);break;
                         case 1:s1=_mm512_add_pd(s1,sp);d1=_mm512_add_pd(d1,dd);break;
                         case 2:s2=_mm512_add_pd(s2,sp);d2=_mm512_add_pd(d2,dd);break;
                         default:s3=_mm512_add_pd(s3,sp);d3=_mm512_add_pd(d3,dd);break;}}
        }}
        __m512d S=_mm512_mul_pd(C4PI,_mm512_add_pd(_mm512_add_pd(s0,s1),_mm512_add_pd(s2,s3)));
        __m512d dS=_mm512_mul_pd(C4PI,_mm512_add_pd(_mm512_add_pd(d0,d1),_mm512_add_pd(d2,d3)));
        __m512d rho=_mm512_div_pd(a,R0);
        __m512d P=_mm512_set1_pd(x60_tailU[64]),DP=_mm512_setzero_pd();
        for(int j=63;j>=0;j--){{ DP=_mm512_fmadd_pd(DP,rho,P); P=_mm512_fmadd_pd(P,rho,_mm512_set1_pd(x60_tailU[j])); }}
        S=_mm512_fmadd_pd(C4PI,P,S);
        dS=_mm512_fmadd_pd(_mm512_div_pd(C4PI,R0),DP,dS);
        __m512d B=_mm512_mul_pd(_mm512_mul_pd(C8PI2,vx),dS);
        __m512d y=_mm512_div_pd(B,_mm512_mul_pd(S,S));
        _mm512_storeu_pd(out+i,y);
    }}
    if(i<n) octant_vector_v8_x56_general(k,x+i,out+i,n-i);
}}
'''

dispatch=r'''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,double * __restrict out,size_t n)
{
    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}
    const __m512d ONE=_mm512_set1_pd(1.0);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    size_t j=0;
    for(;j+8<=n;j+=8){__m512i xi=_mm512_castpd_si512(_mm512_loadu_pd(x+j));
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(xi,ABSM));
        if(__builtin_expect(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)!=0,0)){octant_vector_v8_x56_general(k,x,out,n);return;}}
    for(;j<n;j++){uint64_t u;memcpy(&u,x+j,sizeof(u));u&=UINT64_C(0x7fffffffffffffff);double aa;memcpy(&aa,&u,sizeof(aa));
        if(aa<1.0){octant_vector_v8_x56_general(k,x,out,n);return;}}
    octant_vector_v8_rawpoles(k,x,out,n);
}
'''

s=s[:start]+general+'\n'+raw+'\n'+dispatch+s[end:]
s=s.replace('S53X56_','S53X60_')
s=s.replace('xeon_x56_wide_specialized_cold_g4','xeon_x60_raw_paired_poles')
s=s.replace('Xeon_AVX512_X56_wide_specialized_cold','Xeon_AVX512_X60_raw_paired_poles')
Path('bench_sine_53_xeon_x60_build.c').write_text(s)
print('S53X60_BUILD_PASS parent=X56 raw_paired_poles=1 pole_pairs=2048 tail_degree=64 direct_secant_derivative=1 sin_equals_B_over_S2=1 pi_reduction=0 modulo=0 quadrant=0 parity=0 cody_waite=0 AVX512_lanes=8 multiaccum=4 requires_Arb_regate=1')
