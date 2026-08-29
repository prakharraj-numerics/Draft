from pathlib import Path
import runpy

def build_x49():
    runpy.run_path('make_sine_53_xeon_x49_production.py', run_name='__main__')
    return Path('bench_sine_53_xeon_x49_build.c').read_text()

def replace_hot(s, newhot):
    hit=s.index('octant_vector_v8(const s53w_kernel *k,')
    start=s.rfind('\n',0,hit)+1
    end=s.index('\n#endif',hit)
    return s[:start]+newhot+s[end:]

# X50: one-stream cross-iteration lookahead.
s=build_x49()
G=4
decl=[]
for b in range(G):
    decl.append(f'        __m512d rh{b},rl{b},d{b},c0_{b},c1_{b},c2_{b},c3_{b},c4_{b},c5_{b},p{b}; __m256i ji{b}; __mmask8 s{b},g{b},a{b}; unsigned char pu{b};')
prep=[]
for b in (1,2,3):
    prep += [
      f'        x12_prepare_block(x+i,{8*b},32,&rh{b},&rl{b},&s{b},&g{b},&a{b},&pu{b});',
      f'        ji{b}=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh{b},VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);'
    ]
    if b < 3:
        prep += [
          f'        c0_{b}=_mm512_i32gather_pd(ji{b},tab+0*LUTN,8);',
          f'        c1_{b}=_mm512_i32gather_pd(ji{b},tab+1*LUTN,8);'
        ]
delta=[]; recon=[]; hor=[]; finish=[]; repair=[]
for b in range(G):
    delta += [
      f'        __m512d jd{b}=_mm512_cvtepi32_pd(ji{b});',
      f'        d{b}=_mm512_fnmadd_pd(jd{b},VIK,rh{b});',
      f'        d{b}=_mm512_add_pd(d{b},rl{b});'
    ]
    recon.append(f'        c2_{b}=_mm512_mul_pd(c0_{b},MH); c3_{b}=_mm512_mul_pd(c1_{b},M6); c4_{b}=_mm512_mul_pd(c0_{b},C24); c5_{b}=_mm512_mul_pd(c1_{b},C120);')
    hor.append(f'        p{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});')
    for cp in (3,2,1,0):
        hor.append(f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c{cp}_{b});')
    finish += [f'        p{b}=_mm512_mask_sub_pd(p{b},s{b},Z,p{b});',
               f'        _mm512_storeu_pd(out+i+{8*b},p{b});']
    repair.append(f'        if(__builtin_expect(g{b}!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g{b}&(1u<<lane)) out[i+{8*b}+lane]=scalar2(k,x[i+{8*b}+lane]);')

x50=f'''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{{
    if(__builtin_expect(n<X12_TILE,0)){{octant_vector_v11_single(k,x,out,n);return;}}
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    size_t i=0;
    __m512d nrh0,nrl0,nc0_0,nc1_0; __m256i nji0; __mmask8 ns0,ng0,na0; unsigned char npu0;
    if(n>=32){{
        x12_prepare_block(x,0,32,&nrh0,&nrl0,&ns0,&ng0,&na0,&npu0);
        nji0=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(nrh0,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        nc0_0=_mm512_i32gather_pd(nji0,tab+0*LUTN,8);
        nc1_0=_mm512_i32gather_pd(nji0,tab+1*LUTN,8);
    }}
    for(;i+32<=n;i+=32){{
{chr(10).join(decl)}
        rh0=nrh0; rl0=nrl0; s0=ns0; g0=ng0; a0=na0; pu0=npu0; ji0=nji0; c0_0=nc0_0; c1_0=nc1_0;
{chr(10).join(prep)}
        c0_3=_mm512_i32gather_pd(ji3,tab+0*LUTN,8);
        c1_3=_mm512_i32gather_pd(ji3,tab+1*LUTN,8);
        if(__builtin_expect(i+64<=n,1)){{
            x12_prepare_block(x+i+32,0,32,&nrh0,&nrl0,&ns0,&ng0,&na0,&npu0);
            nji0=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(nrh0,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
            nc0_0=_mm512_i32gather_pd(nji0,tab+0*LUTN,8);
            nc1_0=_mm512_i32gather_pd(nji0,tab+1*LUTN,8);
        }}
{chr(10).join(delta)}
{chr(10).join(recon)}
{chr(10).join(hor)}
{chr(10).join(finish)}
{chr(10).join(repair)}
    }}
    if(i<n) octant_vector_x20_tail(k,x+i,out+i,n-i);
}}'''
s=replace_hot(s,x50)
s=s.replace('S53X49_','S53X50_').replace('xeon_x49_production_hw_schedule_g4','xeon_x50_cross_iteration_lookahead_g4').replace('Xeon_AVX512_X49_production_hw_schedule','Xeon_AVX512_X50_cross_iteration_lookahead')
Path('bench_sine_53_xeon_x50_build.c').write_text(s)

# X51: 256-bit AVX-512VL hot path, eight 4-lane streams.
s=build_x49()
hit=s.index('OVEC static inline void x12_prepare_block')
insert=s.rfind('\n',0,hit)+1
helper=r'''OVEC static inline void twodiff_cw256(__m256d a,__m256d b,__m256d *h,__m256d *l)
{
    __m256d x=_mm256_sub_pd(a,b);
    __m256d bv=_mm256_sub_pd(a,x);
    __m256d av=_mm256_add_pd(x,bv);
    __m256d br=_mm256_sub_pd(bv,b);
    __m256d ar=_mm256_sub_pd(a,av);
    *h=x; *l=_mm256_add_pd(ar,br);
}
OVEC static inline void x51_prepare4(const double * __restrict x,
        __m256d *rh_out,__m256d *rl_out,__mmask8 *sign_out,
        __mmask8 *guard_out,unsigned char *pure_unit_out)
{
    const __m256d Z=_mm256_setzero_pd(),ONE=_mm256_set1_pd(1.0);
    const __m256d VINVP=_mm256_set1_pd(0x1.45f306dc9c883p-2);
    const __m256d PIH=_mm256_set1_pd(0x1.921fb54442d18p+1);
    const __m256i ABSM=_mm256_set1_epi64x((long long)UINT64_C(0x7fffffffffffffff));
    __m256d vx=_mm256_loadu_pd(x);
    __m256i vxi=_mm256_castpd_si256(vx);
    __mmask8 inneg=(__mmask8)(_mm256_movepi64_mask(vxi)&0x0f);
    __m256d ax=_mm256_castsi256_pd(_mm256_and_si256(vxi,ABSM));
    __mmask8 unit=(__mmask8)(_mm256_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&0x0f);
    *pure_unit_out=(unsigned char)(unit==0x0f);
    if(unit==0x0f){*rh_out=ax;*rl_out=Z;*sign_out=inneg;*guard_out=0;return;}

    __m128i qi=_mm256_cvt_roundpd_epi32(_mm256_mul_pd(ax,VINVP),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256d qd=_mm256_cvtepi32_pd(qi);
    __m256d rh=_mm256_fnmadd_pd(qd,PIH,ax);
    __m256d rl=_mm256_mul_pd(qd,_mm256_set1_pd(-0x1.1a62633145c07p-53));
    __m256d rs_fast=_mm256_add_pd(rh,rl);
    __m256d ars=_mm256_castsi256_pd(_mm256_and_si256(_mm256_castpd_si256(rs_fast),ABSM));
    __mmask8 repair=(__mmask8)(_mm256_cmp_pd_mask(ars,_mm256_set1_pd(0x1p-14),_CMP_LT_OQ)&0x0f&~unit);
    if(__builtin_expect(repair!=0,0)){
        const __m256d PI1=_mm256_set1_pd(0x1.921fb54400000p+1);
        const __m256d PI2=_mm256_set1_pd(0x1.0b4611a600000p-33);
        const __m256d PI3=_mm256_set1_pd(0x1.3198a2e037073p-68);
        __m256d r0=_mm256_sub_pd(ax,_mm256_mul_pd(qd,PI1));
        __m256d rh3,re3;twodiff_cw256(r0,_mm256_mul_pd(qd,PI2),&rh3,&re3);
        __m256d rl3=_mm256_fnmadd_pd(qd,PI3,re3);
        rh=_mm256_mask_mov_pd(rh,repair,rh3);
        rl=_mm256_mask_mov_pd(rl,repair,rl3);
    }
    __m256d rs=_mm256_add_pd(rh,rl);
    __mmask8 rneg=(__mmask8)(_mm256_movepi64_mask(_mm256_castpd_si256(rs))&0x0f);
    rh=_mm256_mask_sub_pd(rh,rneg,Z,rh);
    rl=_mm256_mask_sub_pd(rl,rneg,Z,rl);
    rh=_mm256_mask_mov_pd(rh,unit,ax);
    rl=_mm256_mask_mov_pd(rl,unit,Z);
    __m128i parityv=_mm_slli_epi32(_mm_and_si128(qi,_mm_set1_epi32(1)),31);
    __mmask8 parity=(__mmask8)(_mm_movemask_ps(_mm_castsi128_ps(parityv))&0x0f);
    *rh_out=rh;*rl_out=rl;*sign_out=(__mmask8)((inneg^parity^rneg)&0x0f);*guard_out=0;
}

'''
s=s[:insert]+helper+s[insert:]
decl=[]; prep=[]; late=[]; recon=[]; hor=[]; finish=[]; repair=[]
for b in range(8):
    decl.append(f'        __m256d rh{b},rl{b},d{b},c0_{b},c1_{b},c2_{b},c3_{b},c4_{b},c5_{b},p{b}; __m128i ji{b}; __mmask8 s{b},g{b}; unsigned char pu{b};')
    prep += [
      f'        x51_prepare4(x+i+{4*b},&rh{b},&rl{b},&s{b},&g{b},&pu{b});',
      f'        ji{b}=_mm256_cvt_roundpd_epi32(_mm256_mul_pd(rh{b},VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);'
    ]
    if b < 6:
        prep += [f'        c0_{b}=_mm256_i32gather_pd(tab+0*LUTN,ji{b},8);',
                 f'        c1_{b}=_mm256_i32gather_pd(tab+1*LUTN,ji{b},8);']
for b in range(8):
    if b>=6:
        late += [f'        c0_{b}=_mm256_i32gather_pd(tab+0*LUTN,ji{b},8);',
                 f'        c1_{b}=_mm256_i32gather_pd(tab+1*LUTN,ji{b},8);']
    late += [f'        __m256d jd{b}=_mm256_cvtepi32_pd(ji{b});',
             f'        d{b}=_mm256_fnmadd_pd(jd{b},VIK,rh{b});',
             f'        d{b}=_mm256_add_pd(d{b},rl{b});']
    recon.append(f'        c2_{b}=_mm256_mul_pd(c0_{b},MH); c3_{b}=_mm256_mul_pd(c1_{b},M6); c4_{b}=_mm256_mul_pd(c0_{b},C24); c5_{b}=_mm256_mul_pd(c1_{b},C120);')
    hor.append(f'        p{b}=_mm256_fmadd_pd(c5_{b},d{b},c4_{b});')
    for cp in (3,2,1,0):
        hor.append(f'        p{b}=_mm256_fmadd_pd(p{b},d{b},c{cp}_{b});')
    finish += [f'        p{b}=_mm256_mask_sub_pd(p{b},s{b},Z,p{b});',
               f'        _mm256_storeu_pd(out+i+{4*b},p{b});']
    repair.append(f'        if(__builtin_expect(g{b}!=0,0)) for(unsigned lane=0;lane<4;lane++) if(g{b}&(1u<<lane)) out[i+{4*b}+lane]=scalar2(k,x[i+{4*b}+lane]);')
x51=f'''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{{
    if(__builtin_expect(n<X12_TILE,0)){{octant_vector_v11_single(k,x,out,n);return;}}
    const __m256d VK=_mm256_set1_pd(KGRID),VIK=_mm256_set1_pd(INVK),Z=_mm256_setzero_pd();
    const __m256d MH=_mm256_set1_pd(-0.5),M6=_mm256_set1_pd(-1.0/6.0),C24=_mm256_set1_pd(1.0/24.0),C120=_mm256_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    size_t i=0;
    for(;i+32<=n;i+=32){{
{chr(10).join(decl)}
{chr(10).join(prep)}
{chr(10).join(late)}
{chr(10).join(recon)}
{chr(10).join(hor)}
{chr(10).join(finish)}
{chr(10).join(repair)}
    }}
    if(i<n) octant_vector_x20_tail(k,x+i,out+i,n-i);
}}'''
s=replace_hot(s,x51)
s=s.replace('S53X49_','S53X51_').replace('xeon_x49_production_hw_schedule_g4','xeon_x51_avx512vl_256_g8').replace('Xeon_AVX512_X49_production_hw_schedule','Xeon_AVX512VL_X51_256bit_G8')
Path('bench_sine_53_xeon_x51_build.c').write_text(s)

# X53: outline B14 repair into a cold noinline helper.
s=build_x49()
marker='OVEC static inline void x12_prepare_block'
pos=s.index(marker)
insert=s.rfind('\n',0,pos)+1
cold=r'''OVEC __attribute__((noinline,cold)) static void x53_b14_repair(
        __m512d ax,__m512d qd,__mmask8 repair,__m512d *rh,__m512d *rl)
{
    const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
    const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
    const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
    __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(qd,PI1));
    __m512d rh3,re3;twodiff_cw(r0,_mm512_mul_pd(qd,PI2),&rh3,&re3);
    __m512d rl3=_mm512_fnmadd_pd(qd,PI3,re3);
    *rh=_mm512_mask_mov_pd(*rh,repair,rh3);
    *rl=_mm512_mask_mov_pd(*rl,repair,rl3);
}
'''
s=s[:insert]+cold+s[insert:]
old=r'''    if(__builtin_expect(repair!=0,0)){
        const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
        const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
        const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
        __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(qd,PI1));
        __m512d rh3,re3;twodiff_cw(r0,_mm512_mul_pd(qd,PI2),&rh3,&re3);
        __m512d rl3=_mm512_fnmadd_pd(qd,PI3,re3);
        rh=_mm512_mask_mov_pd(rh,repair,rh3);
        rl=_mm512_mask_mov_pd(rl,repair,rl3);
    }'''
new=r'''    if(__builtin_expect(repair!=0,0))
        x53_b14_repair(ax,qd,repair,&rh,&rl);'''
if old not in s: raise SystemExit('X53 B14 body marker missing')
s=s.replace(old,new,1)
s=s.replace('S53X49_','S53X53_').replace('xeon_x49_production_hw_schedule_g4','xeon_x53_cold_b14_uop_hotloop').replace('Xeon_AVX512_X49_production_hw_schedule','Xeon_AVX512_X53_cold_B14_hotloop')
Path('bench_sine_53_xeon_x53_build.c').write_text(s)
print('X50_X51_X53_BUILD_PASS X50=cross_iteration_one_stream X51=AVX512VL_256bit_G8 X53=cold_B14_frontend')
