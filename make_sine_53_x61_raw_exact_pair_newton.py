from pathlib import Path
import runpy

# X61: keep raw x as raw x.  No range reduction, no phase/quadrant/octant
# lookup, no LUT/anchor machinery in the >1 hot branch.  This fixes X60's
# biggest algebraic mistake: each adjacent pole pair is evaluated as ONE exact
# rational block rather than two reciprocals followed by subtraction.
runpy.run_path('make_sine_53_x60_raw_paired_poles.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x60_build.c')
s=p.read_text()

start=s.index('OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawpoles')
end=s.index('\nOVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(', start)

raw=r'''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawpoles(const s53w_kernel *k,
                                  const double * __restrict x,double * __restrict out,size_t n)
{
    (void)k;
    const __m512d C4PI=_mm512_set1_pd(0x1.45f306dc9c883p+0);  /* 4/pi */
    const __m512d C4PI2=_mm512_set1_pd(0x1.9f9c02660badfp-2); /* 4/pi^2 */
    const __m512d C8PI2=_mm512_set1_pd(0x1.9f9c02660badfp-1); /* 8/pi^2 */
    const __m512d TWO=_mm512_set1_pd(2.0);
    const __m512d FOUR=_mm512_set1_pd(4.0);
    const __m512d EIGHT=_mm512_set1_pd(8.0);
    const __m512d SIXTEEN=_mm512_set1_pd(16.0);
    const __m512d R0=_mm512_set1_pd(67125249.0); /* (4*2048+1)^2 */
    size_t i=0;
    for(;i+8<=n;i+=8){
        const __m512d vx=_mm512_loadu_pd(x+i);
        const __m512d a=_mm512_mul_pd(_mm512_mul_pd(vx,vx),C4PI2);
        __m512d s0=_mm512_setzero_pd(),s1=s0,s2=s0,s3=s0;
        __m512d d0=s0,d1=s0,d2=s0,d3=s0;

        /* p=4k+1, q=p+2.  Maintain p,p^2,q^2,pq recursively so the hot
           loop has no conversions, no gathers, and no coefficient table. */
        __m512d p=_mm512_set1_pd(1.0);
        __m512d p2=_mm512_set1_pd(1.0);
        __m512d q2=_mm512_set1_pd(9.0);
        __m512d pq=_mm512_set1_pd(3.0);

        for(int kk=0;kk<2048;kk++){
            const __m512d dp=_mm512_sub_pd(p2,a);
            const __m512d dq=_mm512_sub_pd(q2,a);
            const __m512d D=_mm512_mul_pd(dp,dq);
            const __m512d N=_mm512_add_pd(pq,a);

            /* AVX-512 rcp14 + two Newton steps: r <- r(2-Dr). */
            __m512d r=_mm512_rcp14_pd(D);
            r=_mm512_mul_pd(r,_mm512_fnmadd_pd(D,r,TWO));
            r=_mm512_mul_pd(r,_mm512_fnmadd_pd(D,r,TWO));

            const __m512d F=_mm512_mul_pd(TWO,_mm512_mul_pd(N,r));
            const __m512d Dp=_mm512_sub_pd(_mm512_add_pd(a,a),_mm512_add_pd(p2,q2));
            const __m512d Nd=_mm512_fnmadd_pd(N,Dp,D); /* D - N*D' */
            const __m512d dF=_mm512_mul_pd(TWO,_mm512_mul_pd(Nd,_mm512_mul_pd(r,r)));

            switch(kk&3){
              case 0:s0=_mm512_add_pd(s0,F);d0=_mm512_add_pd(d0,dF);break;
              case 1:s1=_mm512_add_pd(s1,F);d1=_mm512_add_pd(d1,dF);break;
              case 2:s2=_mm512_add_pd(s2,F);d2=_mm512_add_pd(d2,dF);break;
              default:s3=_mm512_add_pd(s3,F);d3=_mm512_add_pd(d3,dF);break;
            }

            /* Advance p -> p+4 exactly in binary64 for this integer range.
               (p+4)^2=p^2+8p+16; q^2=(p+2)^2; pq=p(p+2). */
            p2=_mm512_add_pd(p2,_mm512_fmadd_pd(EIGHT,p,SIXTEEN));
            p=_mm512_add_pd(p,FOUR);
            q2=_mm512_fmadd_pd(_mm512_add_pd(p,TWO),_mm512_add_pd(p,TWO),_mm512_setzero_pd());
            pq=_mm512_mul_pd(p,_mm512_add_pd(p,TWO));
        }

        __m512d PS=_mm512_add_pd(_mm512_add_pd(s0,s1),_mm512_add_pd(s2,s3));
        __m512d PD=_mm512_add_pd(_mm512_add_pd(d0,d1),_mm512_add_pd(d2,d3));

        /* Same convergent analytic tail as X60; first omitted pole is 8193
           and max |x|=10000 gives rho<0.604. */
        const __m512d rho=_mm512_div_pd(a,R0);
        __m512d P=_mm512_set1_pd(x60_tailU[64]),DP=_mm512_setzero_pd();
        for(int j=63;j>=0;j--){
            DP=_mm512_fmadd_pd(DP,rho,P);
            P=_mm512_fmadd_pd(P,rho,_mm512_set1_pd(x60_tailU[j]));
        }
        PS=_mm512_add_pd(PS,P);
        PD=_mm512_fmadd_pd(_mm512_div_pd(_mm512_set1_pd(1.0),R0),DP,PD);

        const __m512d S=_mm512_mul_pd(C4PI,PS);
        const __m512d dS=_mm512_mul_pd(C4PI,PD);
        const __m512d B=_mm512_mul_pd(_mm512_mul_pd(C8PI2,vx),dS);
        const __m512d y=_mm512_div_pd(B,_mm512_mul_pd(S,S));
        _mm512_storeu_pd(out+i,y);
    }
    if(i<n){
        /* scalar-sized tail still receives raw x; no reducer. */
        double tmpx[8]={0},tmpy[8]; size_t r=n-i;
        for(size_t j=0;j<r;j++) tmpx[j]=x[i+j];
        for(size_t j=r;j<8;j++) tmpx[j]=tmpx[r-1];
        octant_vector_v8_rawpoles(k,tmpx,tmpy,8);
        for(size_t j=0;j<r;j++) out[i+j]=tmpy[j];
    }
}
'''

s=s[:start]+raw+s[end:]
# Remove the small-n local fallback from the dispatcher so homogeneous >1
# batches of any size reach the direct raw branch.
old='    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}\n'
s=s.replace(old,'',1)
s=s.replace('S53X60_','S53X61_')
s=s.replace('xeon_x60_raw_paired_poles','xeon_x61_raw_exact_pair_newton')
s=s.replace('Xeon_AVX512_X60_raw_paired_poles','Xeon_AVX512_X61_raw_exact_pair_newton')
Path('bench_sine_53_xeon_x61_build.c').write_text(s)
print('S53X61_BUILD_PASS raw_gt1_direct=1 input_stays_input=1 LUT=0 anchors=0 range_reduction=0 phase_mapping=0 exact_adjacent_pair_block=1 one_reciprocal_per_pair=1 rcp14_newton2=1 derivative_reuses_denominator=1 AVX512_lanes=8 multiaccum=4 tail_degree=64 requires_Arb_regate=1')
