from pathlib import Path
import runpy,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: make_sine_53_xeon_x22_rsqrt_batch.py <newton_steps> <group>')
steps=int(sys.argv[1]); G=int(sys.argv[2])
if steps not in (1,2,3) or G not in (3,4,5,6,8): raise SystemExit('steps 1..3; G 3,4,5,6,8')

# Start from the Arb-regated two-gather Mode-5 realization. Keep c0 as the one
# exact anchor gather. Reconstruct c1=cos(anchor)=sqrt(1-c0^2) using AVX-512
# rsqrt14 plus Newton refinement, exploiting the sine/cos derivative structure
# of this specific Mode-5 spine. c2..c5 then follow from c0/c1 as before.
runpy.run_path('make_sine_53_xeon_v17_twogather.py',run_name='__main__')
p=Path('bench_sine_53_xeon_v17_twogather_build.c'); s=p.read_text()
needle='''OVEC static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)'''
if needle not in s: raise SystemExit('batch declaration missing')
s=s.replace(needle,'''OVEC static void octant_vector_x22_tail(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)''',1)
start=s.index('OVEC static void octant_vector_x22_tail'); end=s.index('\n#endif',start)
W=8*G
decl=[];prep=[];g0=[];recon_c1=[];derive=[];hor=[];finish=[];repair=[]
for b in range(G):
    decl.append(f'        __m512d rh{b},rl{b},d{b},c0_{b},c1_{b},c2_{b},c3_{b},c4_{b},c5_{b},p{b}; __m256i ji{b}; __mmask8 s{b},g{b},a{b}; unsigned char pu{b};')
    prep += [
      f'        x12_prepare_block(x+i,{8*b},{W},&rh{b},&rl{b},&s{b},&g{b},&a{b},&pu{b});',
      f'        __m512d ya{b}=_mm512_add_pd(rh{b},rl{b});',
      f'        ji{b}=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya{b},VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);',
      f'        __m512d jd{b}=_mm512_cvtepi32_pd(ji{b});',
      f'        if(pu{b}) d{b}=_mm512_fnmadd_pd(jd{b},VIK,rh{b}); else {{d{b}=_mm512_sub_pd(rh{b},_mm512_mul_pd(jd{b},VIK));d{b}=_mm512_add_pd(d{b},rl{b});}}'
    ]
    g0.append(f'        c0_{b}=_mm512_i32gather_pd(ji{b},tab+0*LUTN,8);')
    recon_c1 += [f'        __m512d z{b}=_mm512_max_pd(TINY,_mm512_fnmadd_pd(c0_{b},c0_{b},ONE));',
                   f'        __m512d y{b}=_mm512_rsqrt14_pd(z{b});']
    for r in range(steps):
        recon_c1.append(f'        y{b}=_mm512_mul_pd(y{b},_mm512_fnmadd_pd(_mm512_mul_pd(HALF,z{b}),_mm512_mul_pd(y{b},y{b}),THREEHALF));')
    recon_c1.append(f'        c1_{b}=_mm512_mul_pd(z{b},y{b});')
    derive.append(f'        c2_{b}=_mm512_mul_pd(c0_{b},MH); c3_{b}=_mm512_mul_pd(c1_{b},M6); c4_{b}=_mm512_mul_pd(c0_{b},C24); c5_{b}=_mm512_mul_pd(c1_{b},C120);')
    hor.append(f'        p{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});')
    for cp in (3,2,1,0): hor.append(f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c{cp}_{b});')
    finish += [f'        p{b}=_mm512_mask_sub_pd(p{b},s{b},Z,p{b});',f'        _mm512_storeu_pd(out+i+{8*b},p{b});']
    repair.append(f'        if(__builtin_expect(g{b}!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g{b}&(1u<<lane)) out[i+{8*b}+lane]=scalar2(k,x[i+{8*b}+lane]);')
new=f'''
OVEC static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{{
    if(__builtin_expect(n<X12_TILE,0)){{octant_vector_v11_single(k,x,out,n);return;}}
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d HALF=_mm512_set1_pd(0.5),THREEHALF=_mm512_set1_pd(1.5),TINY=_mm512_set1_pd(0x1p-1022);
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    size_t i=0;
    for(;i+{W}<=n;i+={W}){{
{chr(10).join(decl)}
{chr(10).join(prep)}
        /* One random anchor gather per vector. */
{chr(10).join(g0)}
        /* c1=sqrt(1-c0^2), via rsqrt14 + Newton; independent streams interleaved. */
{chr(10).join(recon_c1)}
{chr(10).join(derive)}
{chr(10).join(hor)}
{chr(10).join(finish)}
{chr(10).join(repair)}
    }}
    if(i<n) octant_vector_x22_tail(k,x+i,out+i,n-i);
}}
'''
s=s[:end]+new+s[end:]
pref=f'S53X22R{steps}G{G}_'
s=s.replace('S53V17_',pref)
s=s.replace('xeon_v17_two_gather_reconstructed_Mode5',f'xeon_x22_rsqrt{steps}_g{G}')
s=s.replace('Xeon_AVX512_tiled_reduce_then_Mode5',f'Xeon_AVX512_x22_rsqrt{steps}_g{G}')
out=Path(f'bench_sine_53_xeon_x22_r{steps}_g{G}_build.c'); out.write_text(s)
print(f'X22_BUILD_PASS newton_steps={steps} group={G} width={W} one_anchor_gather=1 rsqrt14=1 same_secant_Mode5_spine=1 same_reducer=1 same_anchor_rule=1 same_delta=1 same_Horner_FMA_order=1 requires_Arb_regate=1')
