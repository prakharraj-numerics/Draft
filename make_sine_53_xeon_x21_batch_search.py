from pathlib import Path
import runpy,sys

if len(sys.argv)!=3 or sys.argv[1] not in ('plane','pair','sqrt'):
    raise SystemExit('usage: make_sine_53_xeon_x21_batch_search.py {plane|pair|sqrt} G')
mode=sys.argv[1]; G=int(sys.argv[2])
if G not in (1,2,3,4,5,6,8): raise SystemExit('G must be 1,2,3,4,5,6,8')

# Start from the certified v17 realization: c0/c1 are the two anchor values and
# c2..c5 follow directly from the Mode-5 sine/cos derivative identities.
runpy.run_path('make_sine_53_xeon_v17_twogather.py',run_name='__main__')
p=Path('bench_sine_53_xeon_v17_twogather_build.c')
s=p.read_text()

# Optional 16-byte anchor-pair table: [c0,c1] per anchor. This lets the second
# gather reuse the exact cache lines touched by the first gather.
if mode=='pair':
    insert=s.index('OVEC static inline __m512d mode5_poly_x11')
    helper=r'''
static double x21_pair[2*LUTN] __attribute__((aligned(64)));
static int x21_pair_ready=0;
static int x21_init_pair(const s53w_kernel *k){
    if(x21_pair_ready)return 1;
    if(!k||k->deg!=5)return 0;
    for(int a=0;a<(int)LUTN;a++){
        x21_pair[2*a+0]=k->tab[0*LUTN+(size_t)a];
        x21_pair[2*a+1]=k->tab[1*LUTN+(size_t)a];
    }
    x21_pair_ready=1;return 1;
}

'''
    s=s[:insert]+helper+s[insert:]
    needle='s53w_kernel *k=kernel_create(2);'
    if needle not in s: raise SystemExit('kernel marker missing')
    s=s.replace(needle,needle+'if(!k||!x21_init_pair(k))return 21;',1)

needle='''OVEC static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)'''
if needle not in s: raise SystemExit('batch declaration missing')
s=s.replace(needle,'''OVEC static void octant_vector_x21_tail(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)''',1)
start=s.index('OVEC static void octant_vector_x21_tail')
end=s.index('\n#endif',start)

W=8*G
decl=[];prep=[];g0=[];g1=[];derive=[];hor=[];finish=[];repair=[]
for b in range(G):
    decl.append(f'        __m512d rh{b},rl{b},d{b},c0_{b},c1_{b},c2_{b},c3_{b},c4_{b},c5_{b},p{b}; __m256i ji{b}; __mmask8 s{b},g{b},a{b}; unsigned char pu{b};')
    prep += [
      f'        x12_prepare_block(x+i,{8*b},{W},&rh{b},&rl{b},&s{b},&g{b},&a{b},&pu{b});',
      f'        __m512d ya{b}=_mm512_add_pd(rh{b},rl{b});',
      f'        ji{b}=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya{b},VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);',
      f'        __m512d jd{b}=_mm512_cvtepi32_pd(ji{b});',
      f'        if(pu{b}) d{b}=_mm512_fnmadd_pd(jd{b},VIK,rh{b}); else {{d{b}=_mm512_sub_pd(rh{b},_mm512_mul_pd(jd{b},VIK));d{b}=_mm512_add_pd(d{b},rl{b});}}'
    ]
    if mode=='pair':
        g0.append(f'        __m256i bi{b}=_mm256_slli_epi32(ji{b},1); c0_{b}=_mm512_i32gather_pd(bi{b},x21_pair,8);')
        g1.append(f'        c1_{b}=_mm512_i32gather_pd(_mm256_add_epi32(bi{b},_mm256_set1_epi32(1)),x21_pair,8);')
    else:
        g0.append(f'        c0_{b}=_mm512_i32gather_pd(ji{b},tab+0*LUTN,8);')
        if mode=='plane': g1.append(f'        c1_{b}=_mm512_i32gather_pd(ji{b},tab+1*LUTN,8);')
        else: g1.append(f'        c1_{b}=_mm512_sqrt_pd(_mm512_max_pd(Z,_mm512_fnmadd_pd(c0_{b},c0_{b},ONE)));')
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
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    size_t i=0;
    for(;i+{W}<=n;i+={W}){{
{chr(10).join(decl)}
{chr(10).join(prep)}
        /* Interleave independent anchor streams to expose gather / arithmetic latency. */
{chr(10).join(g0)}
{chr(10).join(g1)}
{chr(10).join(derive)}
{chr(10).join(hor)}
{chr(10).join(finish)}
{chr(10).join(repair)}
    }}
    if(i<n) octant_vector_x21_tail(k,x+i,out+i,n-i);
}}
'''
s=s[:end]+new+s[end:]
pref=f'S53X21{mode.upper()}G{G}_'
s=s.replace('S53V17_',pref)
s=s.replace('xeon_v17_two_gather_reconstructed_Mode5',f'xeon_x21_{mode}_g{G}')
s=s.replace('Xeon_AVX512_tiled_reduce_then_Mode5',f'Xeon_AVX512_x21_{mode}_g{G}')
out=Path(f'bench_sine_53_xeon_x21_{mode}_g{G}_build.c');out.write_text(s)
print(f'X21_BUILD_PASS mode={mode} group={G} width={W} same_secant_Mode5_spine=1 same_reducer=1 same_anchor_rule=1 same_delta=1 same_Horner_FMA_order=1 coeff_source={"c0_gather_plus_sqrt_c1" if mode=="sqrt" else "c0_c1_gathers"} pair_cacheline_reuse={1 if mode=="pair" else 0}')
