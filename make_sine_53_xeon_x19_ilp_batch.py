from pathlib import Path
import runpy, sys

if len(sys.argv) != 2:
    raise SystemExit("usage: make_sine_53_xeon_x19_ilp_batch.py <group>")
G = int(sys.argv[1])
if G not in (2,4,6,8):
    raise SystemExit("group must be one of 2,4,6,8")

# x19: keep v12's exact reducer, Mode-5 coefficient bits, anchor rule,
# local-delta rounding and five-Horner-FMA order, but remove the 256-lane
# scratch round-trip. Prepare G independent AVX-512 blocks in registers, then
# interleave the six coefficient gathers / five Horner FMAs plane-by-plane.
# This is real batch scheduling tailored to the Mode-5 spine: independent
# anchor-indexed coefficient streams are kept concurrently in flight.
runpy.run_path('make_sine_53_xeon_v12_batch.py', run_name='__main__')
p=Path('bench_sine_53_xeon_v12_build.c')
s=p.read_text()

needle='''OVEC static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)'''
if needle not in s:
    raise SystemExit('v12 vector declaration not found')
s=s.replace(needle,'''OVEC static void octant_vector_v12_batch(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)''',1)

start=s.index('OVEC static void octant_vector_v12_batch')
end=s.index('\n#endif',start)

decl=[]
prep=[]
gather5=[]
planes={j:[] for j in range(4,-1,-1)}
finish=[]
repair=[]
for b in range(G):
    decl.append(f'        __m512d rh{b},rl{b},d{b},p{b}; __m256i ji{b}; __mmask8 s{b},g{b},a{b}; unsigned char pu{b};')
    prep += [
        f'        x12_prepare_block(x+i,{b*8},{G*8},&rh{b},&rl{b},&s{b},&g{b},&a{b},&pu{b});',
        f'        __m512d ya{b}=_mm512_add_pd(rh{b},rl{b});',
        f'        ji{b}=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya{b},VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);',
        f'        __m512d jd{b}=_mm512_cvtepi32_pd(ji{b});',
        f'        if(pu{b}) d{b}=_mm512_fnmadd_pd(jd{b},VIK,rh{b});',
        f'        else {{ d{b}=_mm512_sub_pd(rh{b},_mm512_mul_pd(jd{b},VIK)); d{b}=_mm512_add_pd(d{b},rl{b}); }}'
    ]
    gather5.append(f'        p{b}=_mm512_i32gather_pd(ji{b},tab+5*LUTN,8);')
    for j in range(4,-1,-1):
        planes[j].append(f'        p{b}=_mm512_fmadd_pd(p{b},d{b},_mm512_i32gather_pd(ji{b},tab+{j}*LUTN,8));')
    finish += [
        f'        p{b}=_mm512_mask_sub_pd(p{b},s{b},Z,p{b});',
        f'        _mm512_mask_storeu_pd(out+i+{b*8},a{b},p{b});'
    ]
    repair += [
        f'        if(__builtin_expect(g{b}!=0,0))',
        f'            for(unsigned lane=0;lane<8;lane++) if(g{b}&(1u<<lane)) out[i+{b*8}+lane]=scalar2(k,x[i+{b*8}+lane]);'
    ]

new = f'''
OVEC static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{{
    if(__builtin_expect(n<X12_TILE,0)){{octant_vector_v11_single(k,x,out,n);return;}}
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    size_t i=0;
    for(;i+{G*8}<=n;i+={G*8}){{
{chr(10).join(decl)}
        /* Prepare independent reduction/anchor states in registers. */
{chr(10).join(prep)}
        /* Interleave independent gather chains plane-by-plane to expose MLP/ILP. */
{chr(10).join(gather5)}
{chr(10).join(sum((planes[j] for j in range(4,-1,-1)), []))}
{chr(10).join(finish)}
        /* Same rare exact scalar repair as v12. */
{chr(10).join(repair)}
    }}
    if(i<n) octant_vector_v12_batch(k,x+i,out+i,n-i);
}}
'''
s=s[:end]+new+s[end:]
s=s.replace('S53X12_',f'S53X19G{G}_')
s=s.replace('xeon_v12_tiled_two_stage_batch',f'xeon_x19_onepass_ilp_g{G}')
s=s.replace('Xeon_AVX512_tiled_reduce_then_Mode5',f'Xeon_AVX512_onepass_interleaved_Mode5_g{G}')
out=Path(f'bench_sine_53_xeon_x19_g{G}_build.c')
out.write_text(s)
print(f'S53X19_BUILD_PASS group={G} exact_v12_reducer=1 same_secant_Mode5_spine=1 same_coeff_bits=1 same_anchor_rule=1 same_delta_rounding=1 same_Horner_FMA_order=1 no_tile_scratch_roundtrip=1 interleaved_independent_gather_chains={G} tail=v12')
