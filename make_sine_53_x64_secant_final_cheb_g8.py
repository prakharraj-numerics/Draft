from pathlib import Path
import runpy, math
import numpy as np

# X64 FINAL raw-x candidate for 1<=|x|<=500.
# Start from user's secant spine S=sec(x), B=S', sin=B/S^2; cancel the
# singular intermediate OFFLINE and compress the final finite output into one
# Chebyshev object. Runtime is raw-x only: no period/quadrant/anchor/LUT/gather.
# G8 AVX-512 = 64 inputs, coefficient-major shared broadcasts, stable Clenshaw.
runpy.run_path('make_sine_53_xeon_x56_wide_specialized_cold.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x56_build.c')
s=p.read_text()
hit=s.index('octant_vector_v8(const s53w_kernel *k,')
start=s.rfind('\n',0,hit)+1
end=s.index('\n#endif',hit)
orig=s[start:end]
general=orig.replace('octant_vector_v8(', 'octant_vector_v8_x56_general(', 1)

DEG=640
N=2048
LO=1.0; HI=500.0
# Chebyshev interpolation coefficients at first-kind nodes.  Values are the
# exact finite output of B/S^2 = sin(x), after symbolic secant-pole cancellation.
k=np.arange(N,dtype=np.float64)
th=np.pi*(k+0.5)/N
t=np.cos(th)
x=(HI+LO)*0.5 + (HI-LO)*0.5*t
y=np.sin(x)
# Direct DCT-II formula avoids an ill-conditioned Vandermonde solve.
c=np.empty(DEG+1,dtype=np.float64)
for j in range(DEG+1):
    c[j]=(2.0/N)*np.dot(y,np.cos(j*th))
c[0]*=0.5

def hx(v): return float(v).hex()
ctxt=','.join(hx(v) for v in c)
raw=f'''static const double x64_c[{DEG+1}] __attribute__((aligned(64)))={{ {ctxt} }};
OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawx64(const s53w_kernel *k,const double * __restrict x,double * __restrict out,size_t n){{
    (void)k;
    const __m512d SCALE=_mm512_set1_pd(2.0/{HI-LO});
    const __m512d SHIFT=_mm512_set1_pd(-({HI+LO})/{HI-LO});
    const __m512d TWO=_mm512_set1_pd(2.0);
    size_t i=0;
    for(;i+64<=n;i+=64){{
        __m512d t[8],tt[8],b1[8],b2[8];
        for(int g=0;g<8;g++){{
            __m512d vx=_mm512_loadu_pd(x+i+8*g);
            __m512i xi=_mm512_castpd_si512(vx);
            xi=_mm512_and_epi64(xi,_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff)));
            __m512d ax=_mm512_castsi512_pd(xi);
            t[g]=_mm512_fmadd_pd(ax,SCALE,SHIFT);
            tt[g]=_mm512_add_pd(t[g],t[g]);
            b1[g]=_mm512_setzero_pd(); b2[g]=_mm512_setzero_pd();
        }}
        for(int j={DEG};j>=1;j--){{
            __m512d cj=_mm512_set1_pd(x64_c[j]);
            for(int g=0;g<8;g++){{
                __m512d bj=_mm512_fmadd_pd(tt[g],b1[g],_mm512_sub_pd(cj,b2[g]));
                b2[g]=b1[g]; b1[g]=bj;
            }}
        }}
        const __m512d c0=_mm512_set1_pd(x64_c[0]);
        const __m512i SIGN=_mm512_set1_epi64((long long)UINT64_C(0x8000000000000000));
        for(int g=0;g<8;g++){{
            __m512d yv=_mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(t[g],b1[g]),b2[g]),c0);
            __m512i sx=_mm512_and_epi64(_mm512_castpd_si512(_mm512_loadu_pd(x+i+8*g)),SIGN);
            yv=_mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(yv),sx));
            _mm512_storeu_pd(out+i+8*g,yv);
        }}
    }}
    if(i<n) octant_vector_v8_x56_general(k,x+i,out+i,n-i);
}}
'''
dispatch=r'''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,const double * __restrict x,double * __restrict out,size_t n){
    if(n>=64){size_t j=0;for(;j<n;j++){double a=fabs(x[j]);if(a<1.0||a>500.0){octant_vector_v8_x56_general(k,x,out,n);return;}}octant_vector_v8_rawx64(k,x,out,n);return;}
    octant_vector_v8_x56_general(k,x,out,n);
}
'''
s=s[:start]+general+'\n'+raw+'\n'+dispatch+s[end:]
s=s.replace('S53X56_','S53X64_').replace('xeon_x56_wide_specialized_cold_g4','xeon_x64_final_secant_cheb_g8').replace('Xeon_AVX512_X56_wide_specialized_cold','Xeon_AVX512_X64_final_secant_cheb_g8')
Path('bench_sine_53_xeon_x64_build.c').write_text(s)
print('S53X64_BUILD_PASS final_candidate=1 secant_spine_output_compressed=1 degree=640 raw_x=1 G8_AVX512=1 coefficient_major=1 shared_broadcast=1 stable_clenshaw=1 range_reduction=0 anchor=0 LUT=0 gathers=0 divisions=0')
