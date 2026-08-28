from pathlib import Path
import runpy,sys

if len(sys.argv)!=2:
    raise SystemExit('usage: make_sine_53_xeon_x25_k1024_d4repair.py threshold_bits')
B=int(sys.argv[1])
if not (6 <= B <= 20):
    raise SystemExit('threshold_bits must be in 6..20')

# Parent: X24 = X23-H14 hybrid reducer + G4 + two-gather cache path + degree-4 local polynomial.
# Workflow fixes the dense table to K=1024 / 1609 anchors. Restore the omitted c5*d^5
# Horner path only for lanes whose nearest-pi reduced magnitude is close enough to a sine zero
# that the degree-4 truncation can exceed 1 ULP.
runpy.run_path('make_sine_53_xeon_x24_cachecompute.py',run_name='__main__')
p=Path('bench_sine_53_xeon_x24_build.c')
s=p.read_text()

for b in range(4):
    # X24's degree-4 Horner ends with p=fma(...,d,c0). Insert rare full degree-5 recomputation there.
    marker=f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c0_{b});'
    if marker not in s:
        raise SystemExit(f'final horner marker missing block {b}')
    repair=f'''        p{b}=_mm512_fmadd_pd(p{b},d{b},c0_{b});
        /* Rare cache/compute repair: near a sine zero, restore the omitted c5*d^5 term
           using the already-gathered c1. Most vectors never execute this branch. */
        __mmask8 dr{b}=(__mmask8)(_mm512_cmp_pd_mask(ya{b},_mm512_set1_pd(0x1p-{B}),_CMP_LT_OQ)&a{b});
        if(__builtin_expect(dr{b}!=0,0)){{
            c5_{b}=_mm512_mul_pd(c1_{b},C120);
            __m512d q5_{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});
            q5_{b}=_mm512_fmadd_pd(q5_{b},d{b},c3_{b});
            q5_{b}=_mm512_fmadd_pd(q5_{b},d{b},c2_{b});
            q5_{b}=_mm512_fmadd_pd(q5_{b},d{b},c1_{b});
            q5_{b}=_mm512_fmadd_pd(q5_{b},d{b},c0_{b});
            p{b}=_mm512_mask_mov_pd(p{b},dr{b},q5_{b});
        }}'''
    s=s.replace(marker,repair,1)

s=s.replace('S53X24_',f'S53X25B{B}_')
s=s.replace('xeon_x24_cachecompute_denseanchors_d4_g4_2g',f'xeon_x25_k1024_d4_rare_d5_b{B}_g4_2g')
s=s.replace('Xeon_AVX512_G4_nearestpi_h14_denseanchors_degree4_two_gather_Mode5',f'Xeon_AVX512_G4_nearestpi_h14_K1024_D4_rareD5_b{B}_two_gather_Mode5')
out=Path(f'bench_sine_53_xeon_x25_b{B}_build.c')
out.write_text(s)
print(f'X25_BUILD_PASS threshold_bits={B} parent=x24 K=1024 degree_fast=4 rare_degree5=1 two_gather=1 hybrid_reducer_b14=1 group=4 formula_spine_preserved=1 requires_Arb_regate=1')
