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

# The regrouped formula has different rounding. Only near a sine zero can that
# rounding become >1 ULP. There, recompute the same degree-5 jet with X23's
# serial-Horner grouping. Patch each helper inside its own function scope so
# the appropriate reduced-argument variable (y vs ya) is used unambiguously.
marker = '    p=_mm512_fmadd_pd(cd,B,p);\n'

def patch_helper(text, start_name, end_name, argvar, maskname):
    start = text.index(start_name)
    end = text.index(end_name, start)
    sec = text[start:end]
    if sec.count(marker) != 1:
        raise SystemExit(f'{start_name}: expected 1 formula marker, found {sec.count(marker)}')
    repair = f'''    p=_mm512_fmadd_pd(cd,B,p);\n    __mmask8 {maskname}=_mm512_cmp_pd_mask({argvar},_mm512_set1_pd({th}),_CMP_LT_OQ);\n    if(__builtin_expect({maskname}!=0,0)){{\n        __m512d c2r=_mm512_mul_pd(c0,MH),c3r=_mm512_mul_pd(c1,M6);\n        __m512d c4r=_mm512_mul_pd(c0,C24),c5r=_mm512_mul_pd(c1,C120);\n        __m512d qr=_mm512_fmadd_pd(c5r,d,c4r);\n        qr=_mm512_fmadd_pd(qr,d,c3r); qr=_mm512_fmadd_pd(qr,d,c2r);\n        qr=_mm512_fmadd_pd(qr,d,c1); qr=_mm512_fmadd_pd(qr,d,c0);\n        p=_mm512_mask_mov_pd(p,{maskname},qr);\n    }}\n'''
    sec = sec.replace(marker, repair, 1)
    return text[:start] + sec + text[end:]

s = patch_helper(s,
    'OVEC static inline __m512d mode5_poly_x11',
    'OVEC static inline __m512d mode5_poly_low_x11',
    'y', 'fr_y')
s = patch_helper(s,
    'OVEC static inline __m512d mode5_poly_low_x11',
    'OVEC static inline void twodiff_cw',
    'ya', 'fr_ya')

# G4 hot path. ya{b}=absolute reduced sine argument; a{b}=active lanes.
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
