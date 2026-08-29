from pathlib import Path
import runpy

# Build production X50 first. Keep reducer, LUT/Mode5 anchor construction,
# angle-addition structure and degree-5 Horner exactly intact; only retune the
# odd local residual coefficients over |d| <= 1/512.
runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
p = Path('bench_sine_53_xeon_x50_build.c')
s = p.read_text()

# Degree-5 minimax/Remez coefficients for the odd local sine residual
# sin(d) ~= d + c3*d^3 + c5*d^5 on |d| <= 1/512.
# c3 ~= -0.16666666666666518016523009679787509173092381945
# c5 ~=  0.00833333220655645900311162929861565666488636358218
# binary64:
# c3 = -0x1.5555555555520p-3
# c5 =  0x1.11110ea59d51cp-7
old_const = 'const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);'
new_const = 'const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-0x1.5555555555520p-3),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(0x1.11110ea59d51cp-7);'
if old_const not in s:
    raise SystemExit('X50 constant block not found')
s = s.replace(old_const, new_const, 1)

# No Horner stage is removed. c5*d+c4 remains present in all four streams.
s = s.replace('S53X50_', 'S53X50R5_')
s = s.replace('Xeon_AVX512_X50_cross_iteration_lookahead', 'Xeon_AVX512_X50_Remez_local_d5')
Path('bench_sine_53_x50_remez_local_d5_build.c').write_text(s)
print('S53X50R5_BUILD_PASS parent=X50 formula_spine=Mode5_secant_anchor_angle_addition local_remez=1 residual_interval=abs_d_le_1_over_512 degree5=1 quintic_preserved=1 horner_stages_preserved=1 c3_hex=-0x1.5555555555520p-3 c5_hex=0x1.11110ea59d51cp-7 requires_Arb_regate=1')
