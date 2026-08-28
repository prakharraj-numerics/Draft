from pathlib import Path
import runpy,sys

# X24: keep X23-H14 reducer, G4 scheduling and two-gather c0/c1 cache path.
# The workflow chooses the anchor density (K=512 or 1024).  This generator
# only shortens the local polynomial from degree 5 to degree 4: at denser
# anchors the omitted c5*d^5 term is tiny, saving one coefficient multiply
# and one FMA while retaining the same c0/c1 LUT representation.

saved=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x23_hybridpi.py','14']
    runpy.run_path('make_sine_53_xeon_x23_hybridpi.py',run_name='__main__')
finally:
    sys.argv=saved

p=Path('bench_sine_53_xeon_x23_h14_build.c')
s=p.read_text()

for b in range(4):
    old=(f'        c2_{b}=_mm512_mul_pd(c0_{b},MH); c3_{b}=_mm512_mul_pd(c1_{b},M6); '
         f'c4_{b}=_mm512_mul_pd(c0_{b},C24); c5_{b}=_mm512_mul_pd(c1_{b},C120);')
    new=(f'        c2_{b}=_mm512_mul_pd(c0_{b},MH); c3_{b}=_mm512_mul_pd(c1_{b},M6); '
         f'c4_{b}=_mm512_mul_pd(c0_{b},C24);')
    if old not in s:
        raise SystemExit(f'reconstruction marker missing block {b}')
    s=s.replace(old,new,1)
    oldh=f'        p{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});'
    newh=f'        p{b}=c4_{b};'
    if oldh not in s:
        raise SystemExit(f'horner marker missing block {b}')
    s=s.replace(oldh,newh,1)

s=s.replace('S53X23H14_','S53X24_')
s=s.replace('xeon_x23_nearestpi_2f_hybrid_b14_g4_2g','xeon_x24_cachecompute_denseanchors_d4_g4_2g')
s=s.replace('Xeon_AVX512_G4_nearestpi_2f_hybrid_b14_two_gather_Mode5','Xeon_AVX512_G4_nearestpi_h14_denseanchors_degree4_two_gather_Mode5')

out=Path('bench_sine_53_xeon_x24_build.c')
out.write_text(s)
print('X24_BUILD_PASS parent=x23_h14 group=4 two_gather=1 hybrid_b14=1 dense_anchor_runtime=workflow_selected degree=4 omit_c5=1 saved_coeff_mul=1 saved_fma=1 formula_spine_preserved=1 requires_Arb_regate=1')
