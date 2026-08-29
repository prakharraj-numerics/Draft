from pathlib import Path
import runpy, math
import mpmath as mp
import numpy as np

# X63: keep user's secant pole structure exact through first 192 poles,
# approximate only the smooth remaining secant tail on |x|<=500 by Chebyshev.
# Raw x is never reduced/remapped/anchored. G4 AVX-512, paired poles,
# rcp14 + 2 Newton steps, shared broadcasts, fused derivative.
runpy.run_path('make_sine_53_xeon_x56_wide_specialized_cold.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x56_build.c')
s=p.read_text()
hit=s.index('octant_vector_v8(const s53w_kernel *k,')
start=s.rfind('\n',0,hit)+1
end=s.index('\n#endif',hit)
orig=s[start:end]
general=orig.replace('octant_vector_v8(', 'octant_vector_v8_x56_general(', 1)

M=192
PAIRS=M//2
DEG=24
XMAX=500.0
mp.mp.dps=90

def tail_sec(x):
    xx=mp.mpf(x); z=xx*xx
    h=mp.mpf('0')
    for m in range(M):
        A=mp.mpf(4)*((-1)**m)/(mp.mpf(2*m+1)*mp.pi)
        q=mp.mpf(4)/(mp.pi**2*mp.mpf(2*m+1)**2)
        h += A/(1-q*z)
    return 1/mp.cos(xx)-h

# Fit only the smooth tail in t=2*x^2/XMAX^2-1.
nfit=900
ks=np.arange(nfit)
t=np.cos(np.pi*(ks+0.5)/nfit)
xs=XMAX*np.sqrt((t+1.0)/2.0)
y=np.array([float(tail_sec(float(v))) for v in xs])
cc=np.polynomial.chebyshev.chebfit(t,y,DEG)
dc=np.polynomial.chebyshev.chebder(cc)

def arrhex(a): return ','.join(float(v).hex() for v in a)
p2=[];q2=[];pq=[]
for k in range(PAIRS):
    pp=4.0*k+1.0; qq=pp+2.0
    p2.append(pp*pp);q2.append(qq*qq);pq.append(pp*qq)
raw=f'''static const double x63_p2[{PAIRS}] __attribute__((aligned(64)))={{ {arrhex(p2)} }};
static const double x63_q2[{PAIRS}] __attribute__((aligned(64)))={{ {arrhex(q2)} }};
static const double x63_pq[{PAIRS}] __attribute__((aligned(64)))={{ {arrhex(pq)} }};
static const double x63_tc[{DEG+1}] __attribute__((aligned(64)))={{ {arrhex(cc)} }};
static const double x63_tdc[{DEG}] __attribute__((aligned(64)))={{ {arrhex(dc)} }};
static inline __m512d x63_recip(__m512d d){{
    const __m512d TWO=_mm512_set1_pd(2.0);
    __m512d r=_mm512_rcp14_pd(d);
    r=_mm512_mul_pd(r,_mm512_fnmadd_pd(d,r,TWO));
    r=_mm512_mul_pd(r,_mm512_fnmadd_pd(d,r,TWO));
    return r;
}}
static inline void x63_tail(__m512d x,__m512d z,__m512d *sv,__m512d *bv){{
    const __m512d ONE=_mm512_set1_pd(1.0),TWO=_mm512_set1_pd(2.0);
    const __m512d SC=_mm512_set1_pd(2.0/(500.0*500.0));
    __m512d t=_mm512_fmadd_pd(z,SC,_mm512_set1_pd(-1.0));
    __m512d b1=_mm512_setzero_pd(),b2=b1;
    for(int j={DEG};j>=1;j--){{__m512d bj=_mm512_fmadd_pd(_mm512_mul_pd(TWO,t),b1,_mm512_sub_pd(_mm512_set1_pd(x63_tc[j]),b2));b2=b1;b1=bj;}}
    *sv=_mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(t,b1),b2),_mm512_set1_pd(x63_tc[0]));
    b1=_mm512_setzero_pd();b2=b1;
    for(int j={DEG-1};j>=1;j--){{__m512d bj=_mm512_fmadd_pd(_mm512_mul_pd(TWO,t),b1,_mm512_sub_pd(_mm512_set1_pd(x63_tdc[j]),b2));b2=b1;b1=bj;}}
    __m512d ddt=_mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(t,b1),b2),_mm512_set1_pd(x63_tdc[0]));
    *bv=_mm512_mul_pd(ddt,_mm512_mul_pd(x,_mm512_set1_pd(4.0/(500.0*500.0))));
}}
OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawx63(const s53w_kernel *k,const double * __restrict x,double * __restrict out,size_t n){{
    (void)k; const __m512d C8PI=_mm512_set1_pd(8.0/M_PI), C4PI2=_mm512_set1_pd(4.0/(M_PI*M_PI));
    const __m512d C8PI2=_mm512_set1_pd(8.0/(M_PI*M_PI)), TWO=_mm512_set1_pd(2.0);
    size_t i=0;
    for(;i+32<=n;i+=32){{
        __m512d vx[4],a[4],S[4],Da[4];
        for(int g=0;g<4;g++){{vx[g]=_mm512_loadu_pd(x+i+8*g);a[g]=_mm512_mul_pd(_mm512_mul_pd(vx[g],vx[g]),C4PI2);S[g]=_mm512_setzero_pd();Da[g]=_mm512_setzero_pd();}}
        for(int kk=0;kk<{PAIRS};kk++){{
            __m512d P2=_mm512_set1_pd(x63_p2[kk]),Q2=_mm512_set1_pd(x63_q2[kk]),PQ=_mm512_set1_pd(x63_pq[kk]);
            __m512d SUM2=_mm512_add_pd(P2,Q2);
            for(int g=0;g<4;g++){{
                __m512d dp=_mm512_sub_pd(P2,a[g]),dq=_mm512_sub_pd(Q2,a[g]);
                __m512d D=_mm512_mul_pd(dp,dq), r=x63_recip(D), N=_mm512_add_pd(PQ,a[g]);
                __m512d Dp=_mm512_sub_pd(_mm512_add_pd(a[g],a[g]),SUM2);
                S[g]=_mm512_fmadd_pd(N,r,S[g]);
                __m512d num=_mm512_fnmadd_pd(N,Dp,D);
                Da[g]=_mm512_fmadd_pd(num,_mm512_mul_pd(r,r),Da[g]);
            }}
        }}
        for(int g=0;g<4;g++){{
            S[g]=_mm512_mul_pd(C8PI,S[g]); Da[g]=_mm512_mul_pd(C8PI,Da[g]);
            __m512d ts,tb; x63_tail(vx[g],_mm512_mul_pd(vx[g],vx[g]),&ts,&tb);
            S[g]=_mm512_add_pd(S[g],ts);
            __m512d B=_mm512_add_pd(_mm512_mul_pd(_mm512_mul_pd(C8PI2,vx[g]),Da[g]),tb);
            __m512d rS=x63_recip(S[g]);
            _mm512_storeu_pd(out+i+8*g,_mm512_mul_pd(B,_mm512_mul_pd(rS,rS)));
        }}
    }}
    if(i<n) octant_vector_v8_x56_general(k,x+i,out+i,n-i);
}}
'''
dispatch=r'''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,const double * __restrict x,double * __restrict out,size_t n){
    if(n>=32){size_t j=0;for(;j<n;j++){double a=fabs(x[j]);if(a<1.0||a>500.0){octant_vector_v8_x56_general(k,x,out,n);return;}}octant_vector_v8_rawx63(k,x,out,n);return;}
    octant_vector_v8_x56_general(k,x,out,n);
}
'''
s=s[:start]+general+'\n'+raw+'\n'+dispatch+s[end:]
s=s.replace('S53X56_','S53X63_').replace('xeon_x56_wide_specialized_cold_g4','xeon_x63_secant192_tailcheb').replace('Xeon_AVX512_X56_wide_specialized_cold','Xeon_AVX512_X63_secant192_tailcheb')
Path('bench_sine_53_xeon_x63_build.c').write_text(s)
print('S53X63_BUILD_PASS secant_poles=192 pairs=96 tail_cheb_degree=24 raw_x=1 G4=1 rcp14_NR2=1 derivative_fused=1 range_reduction=0 anchor=0 LUT=0')
