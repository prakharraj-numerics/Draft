from pathlib import Path
import runpy, sys

if len(sys.argv) != 2 or sys.argv[1] not in ('tiny','pow2','partial'):
    raise SystemExit('usage: make_sine_53_xeon_intel_clues.py {tiny|pow2|partial}')
mode=sys.argv[1]

saved=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x23_hybridpi.py','14']
    runpy.run_path('make_sine_53_xeon_x23_hybridpi.py',run_name='__main__')
finally:
    sys.argv=saved

p=Path('bench_sine_53_xeon_x23_h14_build.c')
s=p.read_text()

if mode == 'partial':
    # Intel clue: let anchor/index work depend only on the high residual word;
    # the low word remains in the final local delta.  This removes rl from the
    # gather-address dependency chain and gives the scheduler a chance to issue
    # anchor work while low-word/sign work completes.
    for b in range(4):
        old=(f'        __m512d ya{b}=_mm512_add_pd(rh{b},rl{b});\n'
             f'        ji{b}=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya{b},VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);')
        new=(f'        /* X36: high residual is sufficient for anchor selection; rl stays in d. */\n'
             f'        ji{b}=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh{b},VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);')
        if old not in s:
            raise SystemExit(f'partial residual marker missing stream {b}')
        s=s.replace(old,new,1)
    s=s.replace('S53X23H14_','S53X36_')
    s=s.replace('xeon_x23_nearestpi_2f_hybrid_b14_g4_2g','xeon_x36_partial_residual_anchor_overlap_g4')
    s=s.replace('Xeon_AVX512_G4_nearestpi_2f_hybrid_b14_two_gather_Mode5','Xeon_AVX512_X36_partial_residual_anchor_overlap')
    out=Path('bench_sine_53_xeon_x36_partial_build.c')
    out.write_text(s)
    print('X36_BUILD_PASS parent=X23H14 high_residual_anchor_index=1 low_residual_final_delta=1 two_gather=1 G4=1 formula_unchanged=1 requires_Arb_regate=1')

elif mode == 'pow2':
    # Intel/Tang-Harrison clue: split the dominant derivative coefficient c1
    # into an exact power-of-two sigma plus a small correction.  sigma*d is
    # exact in binary floating point.  Protect c0 + sigma*d with TwoSum, then
    # evaluate the remaining degree-5 correction with short parallel chains.
    for b in range(4):
        old=(
          f'        c2_{b}=_mm512_mul_pd(c0_{b},MH); c3_{b}=_mm512_mul_pd(c1_{b},M6); c4_{b}=_mm512_mul_pd(c0_{b},C24); c5_{b}=_mm512_mul_pd(c1_{b},C120);\n'
          f'        p{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});\n'
          f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c3_{b});\n'
          f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c2_{b});\n'
          f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c1_{b});\n'
          f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c0_{b});')
        new=(
          f'        c2_{b}=_mm512_mul_pd(c0_{b},MH); c3_{b}=_mm512_mul_pd(c1_{b},M6); c4_{b}=_mm512_mul_pd(c0_{b},C24); c5_{b}=_mm512_mul_pd(c1_{b},C120);\n'
          f'        __m512i c1bits{b}=_mm512_castpd_si512(c1_{b});\n'
          f'        __m512d sig{b}=_mm512_castsi512_pd(_mm512_and_epi64(c1bits{b},_mm512_set1_epi64((long long)UINT64_C(0x7ff0000000000000))));\n'
          f'        __m512d sd{b}=_mm512_mul_pd(sig{b},d{b}); /* exact power-of-two product */\n'
          f'        __m512d sum{b}=_mm512_add_pd(c0_{b},sd{b});\n'
          f'        __m512d bb{b}=_mm512_sub_pd(sum{b},c0_{b});\n'
          f'        __m512d err{b}=_mm512_add_pd(_mm512_sub_pd(c0_{b},_mm512_sub_pd(sum{b},bb{b})),_mm512_sub_pd(sd{b},bb{b}));\n'
          f'        __m512d z{b}=_mm512_mul_pd(d{b},d{b});\n'
          f'        __m512d u{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});\n'
          f'        __m512d v{b}=_mm512_fmadd_pd(c3_{b},d{b},c2_{b});\n'
          f'        __m512d corr{b}=_mm512_mul_pd(z{b},_mm512_fmadd_pd(u{b},z{b},v{b}));\n'
          f'        corr{b}=_mm512_fmadd_pd(_mm512_sub_pd(c1_{b},sig{b}),d{b},corr{b});\n'
          f'        corr{b}=_mm512_add_pd(corr{b},err{b});\n'
          f'        p{b}=_mm512_add_pd(sum{b},corr{b});')
        if old not in s:
            raise SystemExit(f'pow2 evaluator marker missing stream {b}')
        s=s.replace(old,new,1)
    s=s.replace('S53X23H14_','S53X35_')
    s=s.replace('xeon_x23_nearestpi_2f_hybrid_b14_g4_2g','xeon_x35_pow2_dominant_split_g4')
    s=s.replace('Xeon_AVX512_G4_nearestpi_2f_hybrid_b14_two_gather_Mode5','Xeon_AVX512_X35_pow2_dominant_split')
    out=Path('bench_sine_53_xeon_x35_pow2_build.c')
    out.write_text(s)
    print('X35_BUILD_PASS parent=X23H14 pow2_sigma_from_c1=1 exact_sigma_times_delta=1 twosum_dominant=1 parallel_correction=1 two_gather=1 G4=1 requires_Arb_regate=1')

else:
    # Tiny active LUT: use 14 pre-existing Mode-5 anchors from the K=256 table.
    # j=0..12 correspond to 0,1/8,...,1.5; j=13 uses 1.5625 (table index 400)
    # to cover the endpoint.  c0/c1 for these anchors live in four ZMM registers,
    # selected with vpermt2pd; there are no gathers in the vector hot path.
    start=s.index('OVEC static void octant_vector_v8')
    end=s.index('\n#endif',start)
    scalar=r'''
static inline double x34_tiny_scalar(const s53w_kernel *k,double x)
{
    int64_t q; double r=reduce_scalar(x,&q);
    long j=lround(r*8.0); if(j<0)j=0; if(j>13)j=13;
    long ai=(j==13)?400:(j<<5);
    double d=fma(-(double)ai,1.0/256.0,r);
    double c0=k->tab[(size_t)ai],c1=k->tab[LUTN+(size_t)ai];
    double z=d*d,z2=z*z;
    double ta0=fma(1.0/24.0,z,-0.5),ta1=fma(1.0/40320.0,z,-1.0/720.0);
    double tb0=fma(1.0/120.0,z,-1.0/6.0),tb1=fma(1.0/362880.0,z,-1.0/5040.0);
    double A=fma(fma(ta1,z2,ta0),z,1.0);
    double B=fma(fma(tb1,z2,tb0),z,1.0);
    double y=fma(c1*d,B,c0*A);
    if(signbit(x)^(int)(q&1))y=-y;
    return y;
}

'''
    decl=[]; prep=[]; lut=[]; evals=[]; finish=[]
    for b in range(4):
        decl.append(f'        __m512d rh{b},rl{b},d{b},c0_{b},c1_{b},p{b}; __m256i jj{b}; __mmask8 s{b},g{b},a{b}; unsigned char pu{b};')
        prep += [
          f'        x12_prepare_block(x+i,{8*b},32,&rh{b},&rl{b},&s{b},&g{b},&a{b},&pu{b});',
          f'        __m512d ya{b}=_mm512_add_pd(rh{b},rl{b});',
          f'        jj{b}=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ya{b},V8),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);',
          f'        jj{b}=_mm256_min_epi32(jj{b},J13);',
          f'        __m512i j64_{b}=_mm512_cvtepi32_epi64(jj{b});',
          f'        __m256i ai{b}=_mm256_slli_epi32(jj{b},5);',
          f'        __m256i eq13_{b}=_mm256_cmpeq_epi32(jj{b},J13);',
          f'        ai{b}=_mm256_blendv_epi8(ai{b},A400,eq13_{b});',
          f'        __m512d ad{b}=_mm512_cvtepi32_pd(ai{b});',
          f'        d{b}=_mm512_sub_pd(rh{b},_mm512_mul_pd(ad{b},V256I)); d{b}=_mm512_add_pd(d{b},rl{b});'
        ]
        lut += [
          f'        c0_{b}=_mm512_permutex2var_pd(C0L,j64_{b},C0H);',
          f'        c1_{b}=_mm512_permutex2var_pd(C1L,j64_{b},C1H);'
        ]
        evals += [
          f'        __m512d z{b}=_mm512_mul_pd(d{b},d{b}),z2_{b}=_mm512_mul_pd(z{b},z{b});',
          f'        __m512d ta0_{b}=_mm512_fmadd_pd(A2,z{b},A1),ta1_{b}=_mm512_fmadd_pd(A4,z{b},A3);',
          f'        __m512d tb0_{b}=_mm512_fmadd_pd(B2,z{b},B1),tb1_{b}=_mm512_fmadd_pd(B4,z{b},B3);',
          f'        __m512d AA{b}=_mm512_fmadd_pd(_mm512_fmadd_pd(ta1_{b},z2_{b},ta0_{b}),z{b},ONE);',
          f'        __m512d BB{b}=_mm512_fmadd_pd(_mm512_fmadd_pd(tb1_{b},z2_{b},tb0_{b}),z{b},ONE);',
          f'        __m512d cd{b}=_mm512_mul_pd(c1_{b},d{b});',
          f'        p{b}=_mm512_fmadd_pd(cd{b},BB{b},_mm512_mul_pd(c0_{b},AA{b}));'
        ]
        finish += [f'        p{b}=_mm512_mask_sub_pd(p{b},s{b},Z,p{b});',f'        _mm512_storeu_pd(out+i+{8*b},p{b});']
    vec=f'''
OVEC static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0),V8=_mm512_set1_pd(8.0),V256I=_mm512_set1_pd(1.0/256.0);
    const __m512d A1=_mm512_set1_pd(-0.5),A2=_mm512_set1_pd(1.0/24.0),A3=_mm512_set1_pd(-1.0/720.0),A4=_mm512_set1_pd(1.0/40320.0);
    const __m512d B1=_mm512_set1_pd(-1.0/6.0),B2=_mm512_set1_pd(1.0/120.0),B3=_mm512_set1_pd(-1.0/5040.0),B4=_mm512_set1_pd(1.0/362880.0);
    const __m256i J13=_mm256_set1_epi32(13),A400=_mm256_set1_epi32(400);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    /* Active c0/c1 LUT is register resident: 14 anchors, padded to 16 slots. */
    const __m512d C0L=_mm512_setr_pd(tab[0],tab[32],tab[64],tab[96],tab[128],tab[160],tab[192],tab[224]);
    const __m512d C0H=_mm512_setr_pd(tab[256],tab[288],tab[320],tab[352],tab[384],tab[400],tab[400],tab[400]);
    const __m512d C1L=_mm512_setr_pd(tab[LUTN+0],tab[LUTN+32],tab[LUTN+64],tab[LUTN+96],tab[LUTN+128],tab[LUTN+160],tab[LUTN+192],tab[LUTN+224]);
    const __m512d C1H=_mm512_setr_pd(tab[LUTN+256],tab[LUTN+288],tab[LUTN+320],tab[LUTN+352],tab[LUTN+384],tab[LUTN+400],tab[LUTN+400],tab[LUTN+400]);
    size_t i=0;
    for(;i+32<=n;i+=32){{
{chr(10).join(decl)}
{chr(10).join(prep)}
        /* Four streams, register-to-register anchor lookup: zero gathers. */
{chr(10).join(lut)}
        /* Degree-9 even/odd evaluator for the coarser active anchor grid. */
{chr(10).join(evals)}
{chr(10).join(finish)}
    }}
    for(;i<n;i++) out[i]=x34_tiny_scalar(k,x[i]);
}}
'''
    s=s[:start]+scalar+vec+s[end:]
    s=s.replace('S53X23H14_','S53X34_')
    s=s.replace('xeon_x23_nearestpi_2f_hybrid_b14_g4_2g','xeon_x34_register_lut14_degree9_g4')
    s=s.replace('Xeon_AVX512_G4_nearestpi_2f_hybrid_b14_two_gather_Mode5','Xeon_AVX512_X34_register_LUT14_D9_no_gather')
    out=Path('bench_sine_53_xeon_x34_tiny_build.c')
    out.write_text(s)
    print('X34_BUILD_PASS parent=X23H14 active_anchors=14 backing_Mode5_table_unchanged=1 c0c1_register_resident=1 hot_gathers=0 degree9_evenodd=1 G4=1 requires_Arb_regate=1')
