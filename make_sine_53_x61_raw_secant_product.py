from pathlib import Path
import runpy, math

# X61: direct raw-x implementation derived from the user's secant spine.
# sec(x) poles are inverted algebraically into the cosine canonical product;
# sine is propagated as the derivative of that same product.
# RAW x is never reduced, remapped, indexed, anchored, or looked up.
# Runtime uses 4096 product factors, G4 AVX-512 (32 inputs), no LUT/gathers/divides.
runpy.run_path('make_sine_53_xeon_x56_wide_specialized_cold.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x56_build.c')
s=p.read_text()
hit=s.index('octant_vector_v8(const s53w_kernel *k,')
start=s.rfind('\n',0,hit)+1
end=s.index('\n#endif',hit)
orig=s[start:end]
general=orig.replace('octant_vector_v8(', 'octant_vector_v8_x56_general(', 1)

M=4096
qs=[4.0/(math.pi*math.pi*(2*m+1)*(2*m+1)) for m in range(M)]
qtxt=',\n'.join(','.join(float(q).hex() for q in qs[i:i+8]) for i in range(0,M,8))
raw=f'''static const double x61_q[{M}] __attribute__((aligned(64)))={{\n{qtxt}\n}};
OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawproduct(const s53w_kernel *k,
                                  const double * __restrict x,double * __restrict out,size_t n)
{{
    (void)k;
    const __m512d ONE=_mm512_set1_pd(1.0), TWO=_mm512_set1_pd(2.0);
    size_t i=0;
    for(;i+32<=n;i+=32){{
        __m512d x0=_mm512_loadu_pd(x+i+0), x1=_mm512_loadu_pd(x+i+8);
        __m512d x2=_mm512_loadu_pd(x+i+16),x3=_mm512_loadu_pd(x+i+24);
        __m512d z0=_mm512_mul_pd(x0,x0),z1=_mm512_mul_pd(x1,x1),z2=_mm512_mul_pd(x2,x2),z3=_mm512_mul_pd(x3,x3);
        __m512d c0=ONE,c1=ONE,c2=ONE,c3=ONE;
        __m512d d0=_mm512_setzero_pd(),d1=d0,d2=d0,d3=d0;
        for(int m=0;m<{M};m++){{
            __m512d q=_mm512_set1_pd(x61_q[m]);
            __m512d f0=_mm512_fnmadd_pd(q,z0,ONE),f1=_mm512_fnmadd_pd(q,z1,ONE);
            __m512d f2=_mm512_fnmadd_pd(q,z2,ONE),f3=_mm512_fnmadd_pd(q,z3,ONE);
            __m512d nd0=_mm512_fnmadd_pd(q,c0,_mm512_mul_pd(d0,f0));
            __m512d nd1=_mm512_fnmadd_pd(q,c1,_mm512_mul_pd(d1,f1));
            __m512d nd2=_mm512_fnmadd_pd(q,c2,_mm512_mul_pd(d2,f2));
            __m512d nd3=_mm512_fnmadd_pd(q,c3,_mm512_mul_pd(d3,f3));
            c0=_mm512_mul_pd(c0,f0); c1=_mm512_mul_pd(c1,f1);
            c2=_mm512_mul_pd(c2,f2); c3=_mm512_mul_pd(c3,f3);
            d0=nd0; d1=nd1; d2=nd2; d3=nd3;
        }}
        _mm512_storeu_pd(out+i+0,_mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(),TWO),x0),d0));
        _mm512_storeu_pd(out+i+8,_mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(),TWO),x1),d1));
        _mm512_storeu_pd(out+i+16,_mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(),TWO),x2),d2));
        _mm512_storeu_pd(out+i+24,_mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(),TWO),x3),d3));
    }}
    if(i<n){{
        size_t j=i;
        for(;j+8<=n;j+=8){{
            __m512d vx=_mm512_loadu_pd(x+j),z=_mm512_mul_pd(vx,vx),c=ONE,d=_mm512_setzero_pd();
            for(int m=0;m<{M};m++){{__m512d q=_mm512_set1_pd(x61_q[m]);__m512d f=_mm512_fnmadd_pd(q,z,ONE);d=_mm512_fnmadd_pd(q,c,_mm512_mul_pd(d,f));c=_mm512_mul_pd(c,f);}}
            _mm512_storeu_pd(out+j,_mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(),TWO),vx),d));
        }}
        if(j<n) octant_vector_v8_x56_general(k,x+j,out+j,n-j);
    }}
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
    octant_vector_v8_rawproduct(k,x,out,n);
}
'''

s=s[:start]+general+'\n'+raw+'\n'+dispatch+s[end:]
s=s.replace('S53X56_','S53X61_')
s=s.replace('xeon_x56_wide_specialized_cold_g4','xeon_x61_raw_secant_product')
s=s.replace('Xeon_AVX512_X56_wide_specialized_cold','Xeon_AVX512_X61_raw_secant_product')
Path('bench_sine_53_xeon_x61_build.c').write_text(s)
print('S53X61_BUILD_PASS parent=X56 user_secant_spine=1 secant_poles_inverted_to_cos_product=1 derivative_product_sine=1 factors=4096 raw_x_unchanged=1 pi_reduction=0 modulo=0 quadrant=0 parity=0 anchor=0 LUT_runtime=0 gathers=0 divisions=0 G4_AVX512=1 requires_Arb_regate=1')
