from pathlib import Path
import runpy

# Build the already-certified Intel-style compensated reducer candidate first,
# then change only the full 32-input hot-loop scheduling.  The Mode5/LUT/
# Horner evaluator is kept identical.  Four 8-lane vectors are reduced as one
# lock-step G4 pipeline instead of four separate x12_prepare_block calls.
runpy.run_path('make_sine_53_xeon_x50_intel_rs_dd.py', run_name='__main__')
s=Path('bench_sine_53_xeon_x50_intel_rs_dd_build.c').read_text()

def replace_hot(src, newhot):
    hit=src.index('octant_vector_v8(const s53w_kernel *k,')
    start=src.rfind('\n',0,hit)+1
    end=src.index('\n#endif',hit)
    return src[:start]+newhot+src[end:]

G=4
decl=[]
load=[]
unitfast=[]
y=[]; nn=[]; parity=[]; rr=[]; repair=[]; fix=[]; sign=[]
anchor=[]; gather=[]; delta=[]; recon=[]; hor=[]; finish=[]
for b in range(G):
    decl.append(f'        __m512d vx{b},ax{b},rh{b},rl{b},d{b},c0_{b},c1_{b},c2_{b},c3_{b},c4_{b},c5_{b},p{b}; __m256i ji{b}; __mmask8 in{b},u{b},sg{b};')
    load += [
        f'        vx{b}=_mm512_loadu_pd(x+i+{8*b});',
        f'        __m512i vi{b}=_mm512_castpd_si512(vx{b});',
        f'        in{b}=_mm512_movepi64_mask(vi{b});',
        f'        ax{b}=_mm512_castsi512_pd(_mm512_and_epi64(vi{b},ABSM));',
        f'        u{b}=_mm512_cmp_pd_mask(ax{b},ONE,_CMP_LT_OQ);'
    ]
    unitfast += [f'            rh{b}=ax{b}; rl{b}=Z; sg{b}=in{b};']
    y.append(f'            __m512d Y{b}=_mm512_fmadd_pd(ax{b},VINVP,RS);')
    nn.append(f'            __m512d N{b}=_mm512_sub_pd(Y{b},RS);')
    parity += [
        f'            __m512i pb{b}=_mm512_slli_epi64(_mm512_castpd_si512(Y{b}),63);',
        f'            __mmask8 pa{b}=_mm512_movepi64_mask(pb{b});'
    ]
    rr += [
        f'            rh{b}=_mm512_fnmadd_pd(N{b},PIH,ax{b});',
        f'            rl{b}=_mm512_mul_pd(N{b},PIL);',
        f'            __m512d rs{b}=_mm512_add_pd(rh{b},rl{b});',
        f'            __m512d ars{b}=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs{b}),ABSM));',
        f'            __mmask8 rep{b}=(_mm512_cmp_pd_mask(ars{b},B14,_CMP_LT_OQ)&(__mmask8)~u{b});'
    ]
    fix += [
        f'                __m512d t0_{b}=_mm512_sub_pd(ax{b},_mm512_mul_pd(N{b},PI1));',
        f'                __m512d rh3_{b},re3_{b}; twodiff_cw(t0_{b},_mm512_mul_pd(N{b},PI2),&rh3_{b},&re3_{b});',
        f'                __m512d rl3_{b}=_mm512_fnmadd_pd(N{b},PI3,re3_{b});',
        f'                rh{b}=_mm512_mask_mov_pd(rh{b},rep{b},rh3_{b});',
        f'                rl{b}=_mm512_mask_mov_pd(rl{b},rep{b},rl3_{b});'
    ]
    sign += [
        f'            rs{b}=_mm512_add_pd(rh{b},rl{b});',
        f'            __mmask8 rn{b}=_mm512_movepi64_mask(_mm512_castpd_si512(rs{b}));',
        f'            rh{b}=_mm512_mask_sub_pd(rh{b},rn{b},Z,rh{b});',
        f'            rl{b}=_mm512_mask_sub_pd(rl{b},rn{b},Z,rl{b});',
        f'            rh{b}=_mm512_mask_mov_pd(rh{b},u{b},ax{b});',
        f'            rl{b}=_mm512_mask_mov_pd(rl{b},u{b},Z);',
        f'            pa{b}=(__mmask8)(pa{b}&~u{b}); rn{b}= (__mmask8)(rn{b}&~u{b});',
        f'            sg{b}=(__mmask8)(in{b}^pa{b}^rn{b});'
    ]
    anchor.append(f'        ji{b}=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh{b},VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);')
    gather += [
        f'        c0_{b}=_mm512_i32gather_pd(ji{b},tab+0*LUTN,8);',
        f'        c1_{b}=_mm512_i32gather_pd(ji{b},tab+1*LUTN,8);'
    ]
    delta += [
        f'        __m512d jd{b}=_mm512_cvtepi32_pd(ji{b});',
        f'        d{b}=_mm512_fnmadd_pd(jd{b},VIK,rh{b});',
        f'        d{b}=_mm512_add_pd(d{b},rl{b});'
    ]
    recon.append(f'        c2_{b}=_mm512_mul_pd(c0_{b},MH); c3_{b}=_mm512_mul_pd(c1_{b},M6); c4_{b}=_mm512_mul_pd(c0_{b},C24); c5_{b}=_mm512_mul_pd(c1_{b},C120);')
    hor.append(f'        p{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});')
    for cp in (3,2,1,0):
        hor.append(f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c{cp}_{b});')
    finish += [
        f'        p{b}=_mm512_mask_sub_pd(p{b},sg{b},Z,p{b});',
        f'        _mm512_storeu_pd(out+i+{8*b},p{b});'
    ]

hot=f'''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{{
    if(__builtin_expect(n<X12_TILE,0)){{octant_vector_v11_single(k,x,out,n);return;}}
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d VINVP=_mm512_set1_pd(0x1.45f306dc9c883p-2),RS=_mm512_set1_pd(0x1.8p52);
    const __m512d PIH=_mm512_set1_pd(0x1.921fb54442d18p+1),PIL=_mm512_set1_pd(-0x1.1a62633145c07p-53);
    const __m512d B14=_mm512_set1_pd(0x1p-14);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK);
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    size_t i=0;
    for(;i+32<=n;i+=32){{
{chr(10).join(decl)}
{chr(10).join(load)}
        unsigned all_unit=(unsigned)(u0==0xff)&(unsigned)(u1==0xff)&(unsigned)(u2==0xff)&(unsigned)(u3==0xff);
        if(__builtin_expect(all_unit,0)){{
{chr(10).join(unitfast)}
        }} else {{
            /* One 32-input reduction pipeline: same operation on all four ZMM
               streams before advancing the dependency chain. */
{chr(10).join(y)}
{chr(10).join(nn)}
{chr(10).join(parity)}
{chr(10).join(rr)}
            __mmask8 rep_any=(__mmask8)(rep0|rep1|rep2|rep3);
            if(__builtin_expect(rep_any!=0,0)){{
                const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
                const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
                const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
{chr(10).join(fix)}
            }}
{chr(10).join(sign)}
        }}
        /* X50 Mode5 anchor selection and evaluator are unchanged. */
{chr(10).join(anchor)}
{chr(10).join(gather)}
{chr(10).join(delta)}
{chr(10).join(recon)}
{chr(10).join(hor)}
{chr(10).join(finish)}
    }}
    if(i<n) octant_vector_x20_tail(k,x+i,out+i,n-i);
}}'''

s=replace_hot(s,hot)
s=s.replace('S53X50IRSDD_','S53X50IRSG4_')
s=s.replace('Xeon_AVX512_X50_Intel_RS_DD','Xeon_AVX512_X50_Intel_RS_G4_32wide')
s=s.replace('xeon_x50_intel_rs_dd_g4','xeon_x50_intel_rs_g4_32wide')
Path('bench_sine_53_xeon_x50_intel_rs_g4_build.c').write_text(s)
print('S53X50IRSG4_BUILD_PASS parent=Intel_RS_DD hot_full32=1 four_separate_prepare_calls=0 lockstep_G4_reduction=1 one_allunit_test_per32=1 one_repair_branch_per32=1 fp_to_int_quotient=0 compensated_rh_rl=1 Mode5_unchanged=1 LUT_unchanged=1 Horner_unchanged=1 target_abs_le_10000 requires_Arb_regate=1')
