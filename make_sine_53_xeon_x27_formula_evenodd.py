from pathlib import Path
import runpy, sys

# X27: carry the Mode-5 coefficient-family structure into the hot evaluator.
# The original Mode-5 builder factorizes the local jet as
#   even coefficients = sin(anchor) * C[k]
#   odd  coefficients = cos(anchor) * T[k]
# For degree 5 the builder-generated global factors reduce to
#   C = {1,-1/2,1/24}, T = {1,-1/6,1/120}.
# Instead of flattening these into a serial degree-5 Horner chain, evaluate
# the two independent degree-2 polynomials in z=d^2 and combine them.
# Same reducer, anchors, two gathers, G4 batch architecture and degree-5
# polynomial; only arithmetic grouping changes, so Arb re-certification is
# mandatory because binary64 rounding is different.

saved = sys.argv[:]
try:
    sys.argv = ['make_sine_53_xeon_x23_hybridpi.py', '14']
    runpy.run_path('make_sine_53_xeon_x23_hybridpi.py', run_name='__main__')
finally:
    sys.argv = saved

p = Path('bench_sine_53_xeon_x23_h14_build.c')
s = p.read_text()

# Expose one vector constant used by both independent coefficient families.
s = s.replace(',Z=_mm512_setzero_pd();',
              ',Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);')

# Small/tail helper: v17 two-gather evaluator occurs twice (normal and low-word).
old = '''    __m512d c2=_mm512_mul_pd(c0,MH);\n    __m512d c3=_mm512_mul_pd(c1,M6);\n    __m512d c4=_mm512_mul_pd(c0,C24);\n    __m512d c5=_mm512_mul_pd(c1,C120);\n    __m512d p=_mm512_fmadd_pd(c5,d,c4);\n    p=_mm512_fmadd_pd(p,d,c3);\n    p=_mm512_fmadd_pd(p,d,c2);\n    p=_mm512_fmadd_pd(p,d,c1);\n    p=_mm512_fmadd_pd(p,d,c0);'''
new = '''    /* Formula-carried Mode-5 family split: P=s*A(d^2)+c*d*B(d^2). */\n    __m512d z=_mm512_mul_pd(d,d);\n    __m512d A=_mm512_fmadd_pd(z,C24,MH);\n    __m512d B=_mm512_fmadd_pd(z,C120,M6);\n    A=_mm512_fmadd_pd(A,z,ONE);\n    B=_mm512_fmadd_pd(B,z,ONE);\n    __m512d cd=_mm512_mul_pd(c1,d);\n    __m512d p=_mm512_mul_pd(c0,A);\n    p=_mm512_fmadd_pd(cd,B,p);'''
count = s.count(old)
if count != 2:
    raise SystemExit(f'expected 2 helper Horner blocks, found {count}')
s = s.replace(old, new)

# G4 large-batch hot path: replace each stream's coefficient reconstruction +
# 5-deep Horner dependency chain with two independent 2-deep z-polynomials.
for b in range(4):
    recon = (f'        c2_{b}=_mm512_mul_pd(c0_{b},MH); c3_{b}=_mm512_mul_pd(c1_{b},M6); '
             f'c4_{b}=_mm512_mul_pd(c0_{b},C24); c5_{b}=_mm512_mul_pd(c1_{b},C120);')
    if recon not in s:
        raise SystemExit(f'G4 reconstruction block {b} missing')
    s = s.replace(recon,
        f'        __m512d z{b}=_mm512_mul_pd(d{b},d{b}); '
        f'__m512d A{b}=_mm512_fmadd_pd(z{b},C24,MH); '
        f'__m512d B{b}=_mm512_fmadd_pd(z{b},C120,M6);', 1)

    hor = '\n'.join([
        f'        p{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});',
        f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c3_{b});',
        f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c2_{b});',
        f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c1_{b});',
        f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c0_{b});'])
    repl = '\n'.join([
        f'        A{b}=_mm512_fmadd_pd(A{b},z{b},ONE);',
        f'        B{b}=_mm512_fmadd_pd(B{b},z{b},ONE);',
        f'        __m512d cd{b}=_mm512_mul_pd(c1_{b},d{b});',
        f'        p{b}=_mm512_mul_pd(c0_{b},A{b});',
        f'        p{b}=_mm512_fmadd_pd(cd{b},B{b},p{b});'])
    if hor not in s:
        raise SystemExit(f'G4 Horner block {b} missing')
    s = s.replace(hor, repl, 1)

# Label the executable unambiguously; do not claim identical floating-point bits.
s = s.replace('S53X23H14_', 'S53X27_')
s = s.replace('xeon_x23_nearestpi_2f_hybrid_b14_g4_2g',
              'xeon_x27_formula_evenodd_x23h14_g4_2g')
s = s.replace('Xeon_AVX512_G4_nearestpi_2f_hybrid_b14_two_gather_Mode5',
              'Xeon_AVX512_G4_formula_evenodd_Mode5_two_gather')

out = Path('bench_sine_53_xeon_x27_formula_evenodd_build.c')
out.write_text(s)
print('X27_BUILD_PASS parent=x23_h14 same_reducer=1 same_anchors=1 two_gather=1 group=4 degree5=1 mode5_evenodd_families=1 z=d2 independent_A_B=1 serial_horner_removed=1 runtime_multiplies=3 runtime_fmas=5 requires_Arb_regate=1')
