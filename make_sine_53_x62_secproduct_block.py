from pathlib import Path
import runpy, math, sys

# X62: hardware-synchronized raw-x form derived from the user's secant spine.
# The same cosine canonical product used in X61 is algebraically fused OFFLINE
# into fixed polynomial blocks. Runtime is coefficient-major G4 AVX-512:
# 32 raw inputs, shared coefficient broadcasts, FMA Horner for F and F',
# derivative/product propagated together, no range/LUT/anchor/gather/div/cvt.
# Usage: python3 make_sine_53_x62_secproduct_block.py BLOCK (2,4,8,16)
B=int(sys.argv[1]) if len(sys.argv)>1 else 8
if B not in (2,4,8,16): raise SystemExit('BLOCK must be 2,4,8,16')
M=4096
assert M%B==0
runpy.run_path('make_sine_53_xeon_x56_wide_specialized_cold.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x56_build.c')
s=p.read_text()
hit=s.index('octant_vector_v8(const s53w_kernel *k,')
start=s.rfind('\n',0,hit)+1
end=s.index('\n#endif',hit)
orig=s[start:end]
general=orig.replace('octant_vector_v8(', 'octant_vector_v8_x56_general(', 1)

qs=[4.0/(math.pi*math.pi*(2*m+1)*(2*m+1)) for m in range(M)]
blocks=[]
for j in range(0,M,B):
    a=[1.0]
    for q in qs[j:j+B]:
        b=[0.0]*(len(a)+1)
        for k,v in enumerate(a):
            b[k]+=v; b[k+1]-=v*q
        a=b
    da=[(k+1)*a[k+1] for k in range(B)]
    blocks.append((a,da))
flat=[]
for a,da in blocks: flat.extend(a); flat.extend(da)
txt=',\n'.join(','.join(float(v).hex() for v in flat[i:i+8]) for i in range(0,len(flat),8))
stride=2*B+1

# Emit lockstep coefficient-major Horner. One broadcast feeds all 4 streams.
lines=[]
lines.append(f'            const double *cb=x62_coef + (size_t)bi*{stride};')
lines.append(f'            __m512d p0=_mm512_set1_pd(cb[{B}]),p1=p0,p2=p0,p3=p0;')
for k in range(B-1,-1,-1):
    lines.append(f'            __m512d a{k}=_mm512_set1_pd(cb[{k}]);')
    for r in range(4): lines.append(f'            p{r}=_mm512_fmadd_pd(p{r},z{r},a{k});')
lines.append(f'            __m512d h0=_mm512_set1_pd(cb[{stride-1}]),h1=h0,h2=h0,h3=h0;')
for k in range(B-2,-1,-1):
    idx=(B+1)+k
    lines.append(f'            __m512d b{k}=_mm512_set1_pd(cb[{idx}]);')
    for r in range(4): lines.append(f'            h{r}=_mm512_fmadd_pd(h{r},z{r},b{k});')
for r in range(4):
    lines.append(f'            d{r}=_mm512_fmadd_pd(d{r},p{r},_mm512_mul_pd(c{r},h{r}));')
    lines.append(f'            c{r}=_mm512_mul_pd(c{r},p{r});')
body='\n'.join(lines)

raw=f'''static const double x62_coef[{len(flat)}] __attribute__((aligned(64)))={{\n{txt}\n}};
OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawproduct(const s53w_kernel *k,
                                  const double * __restrict x,double * __restrict out,size_t n)
{{
    (void)k;
    const __m512d ONE=_mm512_set1_pd(1.0), MTWO=_mm512_set1_pd(-2.0), Z=_mm512_setzero_pd();
    size_t i=0;
    for(;i+32<=n;i+=32){{
        __m512d x0=_mm512_loadu_pd(x+i+0),x1=_mm512_loadu_pd(x+i+8),x2=_mm512_loadu_pd(x+i+16),x3=_mm512_loadu_pd(x+i+24);
        __m512d z0=_mm512_mul_pd(x0,x0),z1=_mm512_mul_pd(x1,x1),z2=_mm512_mul_pd(x2,x2),z3=_mm512_mul_pd(x3,x3);
        __m512d c0=ONE,c1=ONE,c2=ONE,c3=ONE,d0=Z,d1=Z,d2=Z,d3=Z;
        for(int bi=0;bi<{M//B};bi++){{
{body}
        }}
        _mm512_storeu_pd(out+i+0,_mm512_mul_pd(_mm512_mul_pd(MTWO,x0),d0));
        _mm512_storeu_pd(out+i+8,_mm512_mul_pd(_mm512_mul_pd(MTWO,x1),d1));
        _mm512_storeu_pd(out+i+16,_mm512_mul_pd(_mm512_mul_pd(MTWO,x2),d2));
        _mm512_storeu_pd(out+i+24,_mm512_mul_pd(_mm512_mul_pd(MTWO,x3),d3));
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
    octant_vector_v8_rawproduct(k,x,out,n);
}
'''

s=s[:start]+general+'\n'+raw+'\n'+dispatch+s[end:]
s=s.replace('S53X56_','S53X62_')
s=s.replace('xeon_x56_wide_specialized_cold_g4',f'xeon_x62_secproduct_block{B}')
s=s.replace('Xeon_AVX512_X56_wide_specialized_cold',f'Xeon_AVX512_X62_secproduct_block{B}')
out=f'bench_sine_53_xeon_x62_b{B}_build.c'
Path(out).write_text(s)
print(f'S53X62_BUILD_PASS block={B} parent=X56 user_secant_spine=1 fused_product_blocks=1 factors={M} raw_x_unchanged=1 coefficient_major=1 shared_broadcasts=1 G4_AVX512=1 FMA_Horner=1 derivative_fused=1 pi_reduction=0 modulo=0 quadrant=0 parity=0 anchor=0 LUT_runtime=0 gathers=0 divisions=0 cvt=0 requires_Arb_regate=1')
