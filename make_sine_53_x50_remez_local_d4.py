from pathlib import Path
import runpy

# Build the production X50 first; only the local residual evaluator is changed.
runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
p = Path('bench_sine_53_xeon_x50_build.c')
s = p.read_text()

# Minimax/Remez coefficient for sin(d) ~= d + c3*d^3 on |d| <= 1/512.
# Solved at high precision by equioscillation of the odd-cubic error.
# c3 = -0.16666663903887434940224162825485800946631959417174
# Binary64: -0x1.555551a00cb3cp-3
old_const = 'const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);'
new_const = 'const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-0x1.555551a00cb3cp-3),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);'
if old_const not in s:
    raise SystemExit('X50 constant block not found')
s = s.replace(old_const, new_const, 1)

# Degree-5 Horner starts with c5*d+c4.  Remez candidate drops c5 and starts at c4,
# giving a degree-4 local angle-addition polynomial and one fewer hot FMA.
for b in range(4):
    old = f'p{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});'
    new = f'p{b}=c4_{b};'
    if old not in s:
        raise SystemExit(f'X50 stream {b} degree-5 Horner head not found')
    s = s.replace(old, new, 1)

s = s.replace('S53X50_', 'S53X50R4_')
s = s.replace('Xeon_AVX512_X50_cross_iteration_lookahead', 'Xeon_AVX512_X50_Remez_local_d4')
Path('bench_sine_53_x50_remez_local_d4_build.c').write_text(s)
print('S53X50R4_BUILD_PASS parent=X50 formula_spine=Mode5_secant_anchor_angle_addition local_remez=1 residual_interval=abs_d_le_1_over_512 degree4=1 dropped_d5_stage=1 hot_fma_saved_per_lane=1 c3_hex=-0x1.555551a00cb3cp-3 requires_Arb_regate=1')
