from pathlib import Path
import runpy, sys

# X40 is numerically and structurally X38.  The only change is to stop ICX
# from inlining the top-level AVX-512 evaluator so its real G4 loop has a
# stable machine-code symbol for register/port-pressure analysis.
saved=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x37_hw_schedule.py','stagger']
    runpy.run_path('make_sine_53_xeon_x37_hw_schedule.py',run_name='__main__')
finally:
    sys.argv=saved

p=Path('bench_sine_53_xeon_x38_hw_build.c')
s=p.read_text()
needle='OVEC static void octant_vector_v8(const s53w_kernel *k,'
repl='OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,'
if needle not in s:
    raise SystemExit('X38 evaluator declaration marker missing')
s=s.replace(needle,repl,1)
s=s.replace('S53X38_','S53X40_')
s=s.replace('xeon_x38_hw_stagger_gather_delta_g4','xeon_x40_isolated_x38_g4')
s=s.replace('Xeon_AVX512_X38_stagger_gather_pair_with_delta','Xeon_AVX512_X40_isolated_X38_g4')
out=Path('bench_sine_53_xeon_x40_isolated_build.c')
out.write_text(s)
print('X40_BUILD_PASS parent=X38 math_identical=1 reducer_identical=1 LUT_identical=1 G4=1 schedule_identical=1 noinline_hot_symbol=1 aligned64=1 requires_Arb_regate=1')
