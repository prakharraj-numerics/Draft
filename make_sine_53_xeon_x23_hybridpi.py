from pathlib import Path
import runpy,sys

if len(sys.argv)!=2 or sys.argv[1] not in ('20','16','12'):
    raise SystemExit('usage: make_sine_53_xeon_x23_hybridpi.py {20|16|12}')
B=int(sys.argv[1])
# Build the fast nearest-pi 2-piece/FMA reducer, then repair only lanes whose
# reduced residual is close enough to zero that the last pi bits can affect
# binary64 sine in ULPs. The repair uses the already-certified 3-piece split.
saved=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x21_nearestpi.py','2f']
    runpy.run_path('make_sine_53_xeon_x21_nearestpi.py',run_name='__main__')
finally: sys.argv=saved
p=Path('bench_sine_53_xeon_x21_2f_build.c'); s=p.read_text()
old='''    __m512d rh=_mm512_fnmadd_pd(qd,PIH,ax);
    __m512d rl=_mm512_mul_pd(qd,_mm512_set1_pd(-0x1.1a62633145c07p-53));

    __m512d rs=_mm512_add_pd(rh,rl);'''
new=f'''    __m512d rh=_mm512_fnmadd_pd(qd,PIH,ax);
    __m512d rl=_mm512_mul_pd(qd,_mm512_set1_pd(-0x1.1a62633145c07p-53));

    /* Only near a sine zero do the omitted product/cancellation bits matter in ULPs. */
    __m512d rs_fast=_mm512_add_pd(rh,rl);
    __m512i absm=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    __m512d ars=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs_fast),absm));
    __mmask8 repair=(__mmask8)(_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-{B}),_CMP_LT_OQ)&active&~unit);
    if(__builtin_expect(repair!=0,0)){{
        const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
        const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
        const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
        __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(qd,PI1));
        __m512d rh3,re3;twodiff_cw(r0,_mm512_mul_pd(qd,PI2),&rh3,&re3);
        __m512d rl3=_mm512_fnmadd_pd(qd,PI3,re3);
        rh=_mm512_mask_mov_pd(rh,repair,rh3);
        rl=_mm512_mask_mov_pd(rl,repair,rl3);
    }}

    __m512d rs=_mm512_add_pd(rh,rl);'''
if old not in s: raise SystemExit('2f residual marker missing')
s=s.replace(old,new,1)
s=s.replace('S53X21B_',f'S53X23H{B}_')
s=s.replace('xeon_x21_nearestpi_2piece_fma_g4_2g',f'xeon_x23_nearestpi_2f_hybrid_b{B}_g4_2g')
s=s.replace('Xeon_AVX512_G4_nearestpi_2pieceFMA_two_gather_Mode5',f'Xeon_AVX512_G4_nearestpi_2f_hybrid_b{B}_two_gather_Mode5')
out=Path(f'bench_sine_53_xeon_x23_h{B}_build.c'); out.write_text(s)
print(f'X23_BUILD_PASS threshold_bits={B} parent=nearestpi_2f group=4 two_gather=1 fast_2piece_fma=1 rare_3piece_zero_repair=1 formula_unchanged=1 requires_Arb_regate=1')
