from pathlib import Path
import runpy, sys

if len(sys.argv)!=2 or sys.argv[1] not in ('2g','3g'):
    raise SystemExit('usage: make_sine_53_xeon_x20_fused_batch.py {2g|3g}')
mode=sys.argv[1]
G=4
if mode=='2g':
    runpy.run_path('make_sine_53_xeon_v17_twogather.py',run_name='__main__')
    p=Path('bench_sine_53_xeon_v17_twogather_build.c')
    prefix='S53X20A_'; arch='xeon_x20_2g_g4_onepass'; label='Xeon_AVX512_G4_two_gather_Mode5'
else:
    runpy.run_path('make_sine_53_xeon_v18_exact3g.py',run_name='__main__')
    p=Path('bench_sine_53_xeon_v18_exact3g_build.c')
    prefix='S53X20B_'; arch='xeon_x20_exact3g_g4_onepass'; label='Xeon_AVX512_G4_exact3g_Mode5'
s=p.read_text()

needle='''OVEC static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)'''
if needle not in s: raise SystemExit('large batch declaration not found')
s=s.replace(needle,'''OVEC static void octant_vector_x20_tail(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)''',1)
start=s.index('OVEC static void octant_vector_x20_tail')
end=s.index('\n#endif',start)

decl=[]; prep=[]; g0=[]; g1=[]; corr=[]; recon=[]; hor=[]; finish=[]; repair=[]
for b in range(G):
    decl.append(f'        __m512d rh{b},rl{b},d{b},c0_{b},c1_{b},c2_{b},c3_{b},c4_{b},c5_{b},p{b}; __m256i ji{b}; __mmask8 s{b},g{b},a{b}; unsigned char pu{b};')
    prep += [
      f'        x12_prepare_block(x+i,{8*b},32,&rh{b},&rl{b},&s{b},&g{b},&a{b},&pu{b});',
      f'        __m512d ya{b}=_mm512_add_pd(rh{b},rl{b});',
      f'        ji{b}=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya{b},VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);',
      f'        __m512d jd{b}=_mm512_cvtepi32_pd(ji{b});',
      f'        if(pu{b}) d{b}=_mm512_fnmadd_pd(jd{b},VIK,rh{b}); else {{d{b}=_mm512_sub_pd(rh{b},_mm512_mul_pd(jd{b},VIK));d{b}=_mm512_add_pd(d{b},rl{b});}}'
    ]
    g0.append(f'        c0_{b}=_mm512_i32gather_pd(ji{b},tab+0*LUTN,8);')
    g1.append(f'        c1_{b}=_mm512_i32gather_pd(ji{b},tab+1*LUTN,8);')
    if mode=='3g':
      corr.append(f'        __m512i q{b}=_mm512_i32gather_epi64(ji{b},(const long long *)v18_corr,8);')
      recon += [
       f'        c2_{b}=_mm512_mul_pd(c0_{b},MH); c3_{b}=_mm512_mul_pd(c1_{b},M6); c4_{b}=_mm512_mul_pd(c0_{b},C24); c5_{b}=_mm512_mul_pd(c1_{b},C120);',
       f'        __m512i d2_{b}=_mm512_srai_epi64(_mm512_slli_epi64(q{b},48),48),d3_{b}=_mm512_srai_epi64(_mm512_slli_epi64(q{b},32),48);',
       f'        __m512i d4_{b}=_mm512_srai_epi64(_mm512_slli_epi64(q{b},16),48),d5_{b}=_mm512_srai_epi64(q{b},48);',
       f'        c2_{b}=_mm512_castsi512_pd(_mm512_add_epi64(_mm512_castpd_si512(c2_{b}),d2_{b})); c3_{b}=_mm512_castsi512_pd(_mm512_add_epi64(_mm512_castpd_si512(c3_{b}),d3_{b}));',
       f'        c4_{b}=_mm512_castsi512_pd(_mm512_add_epi64(_mm512_castpd_si512(c4_{b}),d4_{b})); c5_{b}=_mm512_castsi512_pd(_mm512_add_epi64(_mm512_castpd_si512(c5_{b}),d5_{b}));'
      ]
    else:
      recon.append(f'        c2_{b}=_mm512_mul_pd(c0_{b},MH); c3_{b}=_mm512_mul_pd(c1_{b},M6); c4_{b}=_mm512_mul_pd(c0_{b},C24); c5_{b}=_mm512_mul_pd(c1_{b},C120);')
    hor.append(f'        p{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});')
    for cp in (3,2,1,0): hor.append(f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c{cp}_{b});')
    finish += [f'        p{b}=_mm512_mask_sub_pd(p{b},s{b},Z,p{b});',f'        _mm512_storeu_pd(out+i+{8*b},p{b});']
    repair += [f'        if(__builtin_expect(g{b}!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g{b}&(1u<<lane)) out[i+{8*b}+lane]=scalar2(k,x[i+{8*b}+lane]);']

new=f'''
OVEC static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{{
    if(__builtin_expect(n<X12_TILE,0)){{octant_vector_v11_single(k,x,out,n);return;}}
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    size_t i=0;
    for(;i+32<=n;i+=32){{
{chr(10).join(decl)}
{chr(10).join(prep)}
        /* Four independent anchor streams: issue each memory plane together. */
{chr(10).join(g0)}
{chr(10).join(g1)}
{chr(10).join(corr)}
        /* Reconstruct Mode-5 coefficients while gather latency is exposed. */
{chr(10).join(recon)}
        /* Preserve the five Horner FMA operations per lane. */
{chr(10).join(hor)}
{chr(10).join(finish)}
{chr(10).join(repair)}
    }}
    if(i<n) octant_vector_x20_tail(k,x+i,out+i,n-i);
}}
'''
s=s[:end]+new+s[end:]
# Replace whatever experiment prefix the parent generator installed.
s=s.replace('S53V17_',prefix).replace('S53V18_',prefix)
s=s.replace('xeon_v17_two_gather_reconstructed_Mode5',arch).replace('xeon_v18_two_double_plus_one_packed_gather_exact_coeffbits',arch)
s=s.replace('Xeon_AVX512_tiled_reduce_then_Mode5',label)
out=Path(f'bench_sine_53_xeon_x20_{mode}_build.c')
out.write_text(s)
print(f'X20_BUILD_PASS mode={mode} group=4 onepass=1 exact_v12_reducer=1 same_secant_Mode5_spine=1 same_anchor_rule=1 same_delta=1 same_Horner_FMA_order=1 double_gathers=2 packed_correction_gathers={1 if mode=="3g" else 0} exact_baseline_coeff_bits={1 if mode=="3g" else 0}')
