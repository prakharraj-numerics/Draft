from pathlib import Path
import runpy, sys

if len(sys.argv) != 2:
    raise SystemExit('usage: make_sine_53_xeon_x28_formula_repair.py threshold_bits')
try:
    R = int(sys.argv[1])
except ValueError:
    raise SystemExit('threshold_bits must be integer')
if not (4 <= R <= 24):
    raise SystemExit('threshold_bits must be 4..24')

# Build X27 formula-carried even/odd evaluator first.
saved = sys.argv[:]
try:
    sys.argv = ['make_sine_53_xeon_x27_formula_evenodd.py']
    runpy.run_path('make_sine_53_xeon_x27_formula_evenodd.py', run_name='__main__')
finally:
    sys.argv = saved
p = Path('bench_sine_53_xeon_x27_formula_evenodd_build.c')
s = p.read_text()
th = f'0x1p-{R}'

# The regrouped formula has different rounding. Only when the reduced sine
# argument is very near a zero can that rounding become >1 ULP. In that rare
# region, recompute the same degree-5 jet using X23's serial Horner grouping.
# Common path remains the formula-carried split.
marker = '    p=_mm512_fmadd_pd(cd,B,p);\n'
if s.count(marker) != 2:
    raise SystemExit(f'expected 2 helper formula markers, found {s.count(marker)}')

repair_y = f'''    p=_mm512_fmadd_pd(cd,B,p);\n    __mmask8 fr=_mm512_cmp_pd_mask(y,_mm512_set1_pd({th}),_CMP_LT_OQ);\n    if(__builtin_expect(fr!=0,0)){{\n        __m512d c2=_mm512_mul_pd(c0,MH),c3=_mm512_mul_pd(c1,M6);\n        __m512d c4=_mm512_mul_pd(c0,C24),c5=_mm512_mul_pd(c1,C120);\n        __m512d q=_mm512_fmadd_pd(c5,d,c4);\n        q=_mm512_fmadd_pd(q,d,c3); q=_mm512_fmadd_pd(q,d,c2);\n        q=_mm512_fmadd_pd(q,d,c1); q=_mm512_fmadd_pd(q,d,c0);\n        p=_mm512_mask_mov_pd(p,fr,q);\n    }}\n'''
s = s.replace(marker, repair_y, 1)

repair_ya = f'''    p=_mm512_fmadd_pd(cd,B,p);\n    __mmask8 fr=_mm512_cmp_pd_mask(ya,_mm512_set1_pd({th}),_CMP_LT_OQ);\n    if(__builtin_expect(fr!=0,0)){{\n        __m512d c2=_mm512_mul_pd(c0,MH),c3=_mm512_mul_pd(c1,M6);\n        __m512d c4=_mm512_mul_pd(c0,C24),c5=_mm512_mul_pd(c1,C120);\n        __m512d q=_mm512_fmadd_pd(c5,d,c4);\n        q=_mm512_fmadd_pd(q,d,c3); q=_mm512_fmadd_pd(q,d,c2);\n        q=_mm512_fmadd_pd(q,d,c1); q=_mm512_fmadd_pd(q,d,c0);\n        p=_mm512_mask_mov_pd(p,fr,q);\n    }}\n'''
s = s.replace(marker, repair_ya, 1)

# G4 hot path. ya{b}=abs(reduced argument), a{b}=active lanes.
for b in range(4):
    marker_b = f'        p{b}=_mm512_fmadd_pd(cd{b},B{b},p{b});'
    if marker_b not in s:
        raise SystemExit(f'G4 formula marker {b} missing')
    repair_b = '\n'.join([
        marker_b,
        f'        __mmask8 fr{b}=(_mm512_cmp_pd_mask(ya{b},_mm512_set1_pd({th}),_CMP_LT_OQ)&a{b});',
        f'        if(__builtin_expect(fr{b}!=0,0)){{',
        f'            c2_{b}=_mm512_mul_pd(c0_{b},MH); c3_{b}=_mm512_mul_pd(c1_{b},M6); c4_{b}=_mm512_mul_pd(c0_{b},C24); c5_{b}=_mm512_mul_pd(c1_{b},C120);',
        f'            __m512d qh{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});',
        f'            qh{b}=_mm512_fmadd_pd(qh{b},d{b},c3_{b}); qh{b}=_mm512_fmadd_pd(qh{b},d{b},c2_{b});',
        f'            qh{b}=_mm512_fmadd_pd(qh{b},d{b},c1_{b}); qh{b}=_mm512_fmadd_pd(qh{b},d{b},c0_{b});',
        f'            p{b}=_mm512_mask_mov_pd(p{b},fr{b},qh{b});',
        '        }'])
    s = s.replace(marker_b, repair_b, 1)

s = s.replace('S53X27_', f'S53X28B{R}_')
s = s.replace('xeon_x27_formula_evenodd_x23h14_g4_2g',
              f'xeon_x28_formula_evenodd_repair_b{R}_g4_2g')
s = s.replace('Xeon_AVX512_G4_formula_evenodd_Mode5_two_gather',
              f'Xeon_AVX512_G4_formula_evenodd_repair_b{R}_two_gather')
out = Path(f'bench_sine_53_xeon_x28_b{R}_build.c')
out.write_text(s)
print(f'X28_BUILD_PASS threshold_bits={R} parent=x27_formula_evenodd common_formula_path=1 rare_x23_horner_repair=1 repair_near_sine_zero=1 same_reducer=1 same_anchors=1 two_gather=1 group=4 requires_Arb_regate=1')
