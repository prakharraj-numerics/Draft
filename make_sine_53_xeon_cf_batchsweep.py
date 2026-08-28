from pathlib import Path
import runpy, sys

if len(sys.argv) != 2:
    raise SystemExit('usage: make_sine_53_xeon_cf_batchsweep.py G')
G = int(sys.argv[1])
if G not in (2,3,4,5,6):
    raise SystemExit('G must be one of 2,3,4,5,6')

# 1) Generate the same X20 two-gather fused architecture with the requested
# number of independent AVX-512 streams.  We patch the generator itself so the
# emitted declarations, gathers, arithmetic and loop stride are genuinely G-wide.
x20 = Path('make_sine_53_xeon_x20_fused_batch.py').read_text()
x20 = x20.replace('G=4', f'G={G}', 1)
x20 = x20.replace('for(;i+32<=n;i+=32)', f'for(;i+{8*G}<=n;i+={8*G})', 1)
x20 = x20.replace('x12_prepare_block(x+i,{8*b},32,', f'x12_prepare_block(x+i,{{8*b}},{8*G},', 1)
x20 = x20.replace('group=4', f'group={G}')
x20 = x20.replace('G4', f'G{G}')
x20 = x20.replace('_g4_', f'_g{G}_')
Path('/tmp/make_cf_x20_g.py').write_text(x20)

saved = sys.argv[:]
try:
    sys.argv = ['/tmp/make_cf_x20_g.py', '2g']
    runpy.run_path('/tmp/make_cf_x20_g.py', run_name='__main__')
finally:
    sys.argv = saved

p = Path('bench_sine_53_xeon_x20_2g_build.c')
s = p.read_text()

# 2) Replace the old octant reducer with the X21 nearest-pi 2-piece/FMA reducer.
start = s.index('OVEC static inline void x12_prepare_block')
end = s.index('OVEC static void octant_vector_x20_tail', start)
helper = r'''OVEC static inline void x12_prepare_block(const double * __restrict x,size_t base,size_t n,
        __m512d *rh_out,__m512d *rl_out,__mmask8 *sign_out,
        __mmask8 *guard_out,__mmask8 *active_out,unsigned char *pure_unit_out)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d VINVP=_mm512_set1_pd(0x1.45f306dc9c883p-2);
    const __m512d PIH=_mm512_set1_pd(0x1.921fb54442d18p+1);
    const __m512d PIL=_mm512_set1_pd(0x1.1a62633145c07p-53);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));

    unsigned rem=(unsigned)(n-base);
    __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
    __m512d vx=_mm512_maskz_loadu_pd(active,x+base);
    __m512i vxi=_mm512_castpd_si512(vx);
    __mmask8 inneg=(__mmask8)(_mm512_movepi64_mask(vxi)&active);
    __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(vxi,ABSM));
    __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);
    *pure_unit_out=(unsigned char)(unit==active);
    if(unit==active){
        *rh_out=ax;*rl_out=Z;*sign_out=inneg;*guard_out=0;*active_out=active;return;
    }

    __m256i qi=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ax,VINVP),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d qd=_mm512_cvtepi32_pd(qi);
    __m512d rh=_mm512_fnmadd_pd(qd,PIH,ax);
    __m512d rl=_mm512_mul_pd(qd,_mm512_set1_pd(-0x1.1a62633145c07p-53));

    /* X23-H14 rare repair: restore the omitted pi bits only close to a sine zero. */
    __m512d rs_fast=_mm512_add_pd(rh,rl);
    __m512i absm=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    __m512d ars=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs_fast),absm));
    __mmask8 repair=(__mmask8)(_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-14),_CMP_LT_OQ)&active&~unit);
    if(__builtin_expect(repair!=0,0)){
        const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
        const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
        const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
        __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(qd,PI1));
        __m512d rh3,re3;twodiff_cw(r0,_mm512_mul_pd(qd,PI2),&rh3,&re3);
        __m512d rl3=_mm512_fnmadd_pd(qd,PI3,re3);
        rh=_mm512_mask_mov_pd(rh,repair,rh3);
        rl=_mm512_mask_mov_pd(rl,repair,rl3);
    }

    __m512d rs=_mm512_add_pd(rh,rl);
    __mmask8 rneg=(__mmask8)(_mm512_movepi64_mask(_mm512_castpd_si512(rs))&active);
    rh=_mm512_mask_sub_pd(rh,rneg,Z,rh);
    rl=_mm512_mask_sub_pd(rl,rneg,Z,rl);
    rh=_mm512_mask_mov_pd(rh,unit,ax);
    rl=_mm512_mask_mov_pd(rl,unit,Z);

    __m256i parityv=_mm256_slli_epi32(_mm256_and_si256(qi,_mm256_set1_epi32(1)),31);
    __mmask8 parity=(__mmask8)(_mm256_movemask_ps(_mm256_castsi256_ps(parityv))&active);
    *rh_out=rh;*rl_out=rl;*sign_out=(__mmask8)((inneg^parity^rneg)&active);
    *guard_out=0;*active_out=active;
}

'''
s = s[:start] + helper + s[end:]

# 3) Carry the Mode-5 even/odd coefficient-family structure into the evaluator.
# Add ONE beside the existing vector constants.
s = s.replace(',Z=_mm512_setzero_pd();', ',Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);')

# Tail/helper evaluator occurs twice.  Convert serial Horner to CF split.
old = '''    __m512d c2=_mm512_mul_pd(c0,MH);\n    __m512d c3=_mm512_mul_pd(c1,M6);\n    __m512d c4=_mm512_mul_pd(c0,C24);\n    __m512d c5=_mm512_mul_pd(c1,C120);\n    __m512d p=_mm512_fmadd_pd(c5,d,c4);\n    p=_mm512_fmadd_pd(p,d,c3);\n    p=_mm512_fmadd_pd(p,d,c2);\n    p=_mm512_fmadd_pd(p,d,c1);\n    p=_mm512_fmadd_pd(p,d,c0);'''
new = '''    __m512d z=_mm512_mul_pd(d,d);\n    __m512d A=_mm512_fmadd_pd(z,C24,MH);\n    __m512d B=_mm512_fmadd_pd(z,C120,M6);\n    A=_mm512_fmadd_pd(A,z,ONE);\n    B=_mm512_fmadd_pd(B,z,ONE);\n    __m512d cd=_mm512_mul_pd(c1,d);\n    __m512d p=_mm512_mul_pd(c0,A);\n    p=_mm512_fmadd_pd(cd,B,p);'''
if s.count(old) != 2:
    raise SystemExit(f'expected two tail Horner blocks, got {s.count(old)}')
s = s.replace(old, new)

# G-wide hot loop.  X20 declares c2..c5; leaving unused declarations is harmless
# and lets this transform remain mechanically comparable across all G values.
for b in range(G):
    recon = (f'        c2_{b}=_mm512_mul_pd(c0_{b},MH); c3_{b}=_mm512_mul_pd(c1_{b},M6); '
             f'c4_{b}=_mm512_mul_pd(c0_{b},C24); c5_{b}=_mm512_mul_pd(c1_{b},C120);')
    repl_recon = (f'        __m512d z{b}=_mm512_mul_pd(d{b},d{b}); '
                  f'__m512d A{b}=_mm512_fmadd_pd(z{b},C24,MH); '
                  f'__m512d B{b}=_mm512_fmadd_pd(z{b},C120,M6);')
    if recon not in s:
        raise SystemExit(f'reconstruction block {b} missing')
    s = s.replace(recon, repl_recon, 1)
    hor = '\n'.join([
        f'        p{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});',
        f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c3_{b});',
        f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c2_{b});',
        f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c1_{b});',
        f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c0_{b});'])
    repl_hor = '\n'.join([
        f'        A{b}=_mm512_fmadd_pd(A{b},z{b},ONE);',
        f'        B{b}=_mm512_fmadd_pd(B{b},z{b},ONE);',
        f'        __m512d cd{b}=_mm512_mul_pd(c1_{b},d{b});',
        f'        p{b}=_mm512_mul_pd(c0_{b},A{b});',
        f'        p{b}=_mm512_fmadd_pd(cd{b},B{b},p{b});'])
    if hor not in s:
        raise SystemExit(f'Horner block {b} missing')
    s = s.replace(hor, repl_hor, 1)

# Give each emitted binary an unmistakable CF/G label.
s = s.replace('S53X20A_', f'S53CFG{G}_')
s = s.replace(f'xeon_x20_2g_g{G}_onepass', f'xeon_cf_g{G}_nearestpi_h14_2g')
s = s.replace(f'Xeon_AVX512_G{G}_two_gather_Mode5', f'Xeon_AVX512_CF_G{G}_two_gather_Mode5')

out = Path(f'bench_sine_53_xeon_cf_g{G}_build.c')
out.write_text(s)
print(f'CF_BATCH_BUILD_PASS G={G} vectors_in_flight={G} lanes_in_flight={8*G} two_gather=1 nearestpi_h14=1 CF_evenodd=1 raw_accuracy_not_assumed=1')
