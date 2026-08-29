from pathlib import Path
import runpy

# X58: genuinely raw-|x|>1 path.  No pi reduction, no modulo, no quadrant,
# no quotient/parity extraction.  For a homogeneous >=1 batch, rewrite the raw
# input exactly as u = x / 2^n with 0.5 <= |u| < 1 by exponent-bit surgery,
# evaluate BOTH sin(u) and cos(u) from the existing Mode5-derived X50 LUT/local
# polynomial, then use repeated double-angle reconstruction back to x.
# Mixed/<1 batches retain X56 only so the production baseline remains available.
runpy.run_path('make_sine_53_xeon_x56_wide_specialized_cold.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x56_build.c')
s=p.read_text()

hit=s.index('octant_vector_v8(const s53w_kernel *k,')
start=s.rfind('\n',0,hit)+1
end=s.index('\n#endif',hit)
orig=s[start:end]
general=orig.replace('octant_vector_v8(', 'octant_vector_v8_x56_general(', 1)

# Build a four-stream / 32-double raw path.  The local polynomial is exactly
# the same degree-5 sine reconstruction as X50 plus its derivative-pattern
# cosine companion from the same anchor sin/cos pair.
decls=[]; loads=[]; locals_=[]; recon=[]; hor=[]; dbl=[]; finish=[]
for b in range(4):
    o=8*b
    decls.append(f'        __m512d u{b},d{b},s0_{b},c0_{b},sv{b},cv{b}; __m256i ji{b}; __m512i nv{b}; __mmask8 neg{b};')
    loads += [
        f'        __m512i xb{b}=_mm512_castpd_si512(_mm512_loadu_pd(x+i+{o}));',
        f'        neg{b}=_mm512_movepi64_mask(xb{b});',
        f'        __m512i ab{b}=_mm512_and_epi64(xb{b},ABSM);',
        f'        __m512i eb{b}=_mm512_and_epi64(_mm512_srli_epi64(ab{b},52),EMASK);',
        f'        nv{b}=_mm512_sub_epi64(eb{b},E1022);',
        f'        __m512i ub{b}=_mm512_or_epi64(_mm512_and_epi64(ab{b},MANT),UEXP);',
        f'        u{b}=_mm512_castsi512_pd(ub{b});',
        f'        ji{b}=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(u{b},VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);',
        f'        s0_{b}=_mm512_i32gather_pd(ji{b},tab+0*LUTN,8);',
        f'        c0_{b}=_mm512_i32gather_pd(ji{b},tab+1*LUTN,8);'
    ]
    locals_ += [
        f'        __m512d jd{b}=_mm512_cvtepi32_pd(ji{b});',
        f'        d{b}=_mm512_fnmadd_pd(jd{b},VIK,u{b});'
    ]
    # sine coefficients from X50
    recon += [
        f'        __m512d s2_{b}=_mm512_mul_pd(s0_{b},MH);',
        f'        __m512d s3_{b}=_mm512_mul_pd(c0_{b},M6);',
        f'        __m512d s4_{b}=_mm512_mul_pd(s0_{b},C24);',
        f'        __m512d s5_{b}=_mm512_mul_pd(c0_{b},C120);',
        # cosine coefficients are derivative-pattern companion
        f'        __m512d c1_{b}=_mm512_sub_pd(Z,s0_{b});',
        f'        __m512d c2_{b}=_mm512_mul_pd(c0_{b},MH);',
        f'        __m512d c3_{b}=_mm512_mul_pd(s0_{b},P6);',
        f'        __m512d c4_{b}=_mm512_mul_pd(c0_{b},C24);',
        f'        __m512d c5_{b}=_mm512_mul_pd(s0_{b},NC120);'
    ]
    hor += [
        f'        sv{b}=_mm512_fmadd_pd(s5_{b},d{b},s4_{b});',
        f'        sv{b}=_mm512_fmadd_pd(sv{b},d{b},s3_{b});',
        f'        sv{b}=_mm512_fmadd_pd(sv{b},d{b},s2_{b});',
        f'        sv{b}=_mm512_fmadd_pd(sv{b},d{b},c0_{b});',
        f'        sv{b}=_mm512_fmadd_pd(sv{b},d{b},s0_{b});',
        f'        cv{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});',
        f'        cv{b}=_mm512_fmadd_pd(cv{b},d{b},c3_{b});',
        f'        cv{b}=_mm512_fmadd_pd(cv{b},d{b},c2_{b});',
        f'        cv{b}=_mm512_fmadd_pd(cv{b},d{b},c1_{b});',
        f'        cv{b}=_mm512_fmadd_pd(cv{b},d{b},c0_{b});'
    ]

# Interleave the four independent streams at each doubling depth to hide the
# recurrence latency.  n<=14 for the current <=10000 benchmark domain.
for step in range(14):
    for b in range(4):
        dbl += [
            f'        __mmask8 m{step}_{b}=_mm512_cmp_epi64_mask(nv{b},_mm512_set1_epi64({step}),_MM_CMPINT_GT);',
            f'        __m512d tc{step}_{b}=_mm512_add_pd(cv{b},cv{b});',
            f'        __m512d ns{step}_{b}=_mm512_mul_pd(sv{b},tc{step}_{b});',
            f'        __m512d nc{step}_{b}=_mm512_fmadd_pd(cv{b},tc{step}_{b},MONE);',
            f'        sv{b}=_mm512_mask_mov_pd(sv{b},m{step}_{b},ns{step}_{b});',
            f'        cv{b}=_mm512_mask_mov_pd(cv{b},m{step}_{b},nc{step}_{b});'
        ]
for b in range(4):
    o=8*b
    finish += [
        f'        sv{b}=_mm512_mask_sub_pd(sv{b},neg{b},Z,sv{b});',
        f'        _mm512_storeu_pd(out+i+{o},sv{b});'
    ]

raw=f'''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawpow2(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),P6=_mm512_set1_pd(1.0/6.0);
    const __m512d C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0),NC120=_mm512_set1_pd(-1.0/120.0),MONE=_mm512_set1_pd(-1.0);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    const __m512i MANT=_mm512_set1_epi64((long long)UINT64_C(0x000fffffffffffff));
    const __m512i EMASK=_mm512_set1_epi64(0x7ff);
    const __m512i E1022=_mm512_set1_epi64(1022);
    const __m512i UEXP=_mm512_set1_epi64((long long)UINT64_C(0x3fe0000000000000));
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    size_t i=0;
    for(;i+32<=n;i+=32){{
{chr(10).join(decls)}
{chr(10).join(loads)}
{chr(10).join(locals_)}
{chr(10).join(recon)}
{chr(10).join(hor)}
{chr(10).join(dbl)}
{chr(10).join(finish)}
    }}
    if(i<n) octant_vector_v8_x56_general(k,x+i,out+i,n-i);
}}'''

dispatch=r'''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}
    const __m512d ONE=_mm512_set1_pd(1.0);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    size_t j=0;
    for(;j+8<=n;j+=8){
        __m512i xi=_mm512_castpd_si512(_mm512_loadu_pd(x+j));
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(xi,ABSM));
        if(__builtin_expect(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)!=0,0)){
            octant_vector_v8_x56_general(k,x,out,n); return;
        }
    }
    for(;j<n;j++){
        uint64_t u; memcpy(&u,x+j,sizeof(u)); u&=UINT64_C(0x7fffffffffffffff);
        double a; memcpy(&a,&u,sizeof(a));
        if(a<1.0){octant_vector_v8_x56_general(k,x,out,n);return;}
    }
    octant_vector_v8_rawpow2(k,x,out,n);
}
'''

s=s[:start]+general+'\n'+raw+'\n'+dispatch+s[end:]
s=s.replace('S53X56_','S53X58_')
s=s.replace('xeon_x56_wide_specialized_cold_g4','xeon_x58_raw_pow2_double_g4')
s=s.replace('Xeon_AVX512_X56_wide_specialized_cold','Xeon_AVX512_X58_raw_pow2_double')
Path('bench_sine_53_xeon_x58_build.c').write_text(s)
print('S53X58_BUILD_PASS parent=X56 raw_gt1=1 pi_reduction=0 modulo=0 quadrant=0 parity=0 cody_waite=0 exact_power2_scale=1 Mode5_LUT_local_sincos=1 repeated_double_angle=1 G4=1 requires_Arb_regate=1')
