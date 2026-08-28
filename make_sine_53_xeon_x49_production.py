from pathlib import Path
import runpy, sys

# X49 freezes the hardware-scheduling line selected by the cross-generation
# sweep: X41 exact branchless local delta plus X48's first-three-stream early
# coefficient-pair issue, leaving stream 3 in the original staggered order.
saved=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x46_partial_early_gather.py','3']
    runpy.run_path('make_sine_53_xeon_x46_partial_early_gather.py',run_name='__main__')
finally:
    sys.argv=saved

p=Path('bench_sine_53_xeon_x48_build.c')
s=p.read_text()
s=s.replace('S53X48_','S53X49_')
s=s.replace('xeon_x48_partial_early_pair_3_g4','xeon_x49_production_hw_schedule_g4')
s=s.replace('Xeon_AVX512_X48_partial_early_pair_3','Xeon_AVX512_X49_production_hw_schedule')
out=Path('bench_sine_53_xeon_x49_build.c')
out.write_text(s)
print('X49_BUILD_PASS parent=X48 production_candidate=1 K=256 anchors=403 two_gather=1 G4=1 B14_repair=1 exact_branchless_delta=1 early_pair_streams=3 final_stream_stagger=1 noinline_hot=1 formula_unchanged=1 requires_Arb_regate=1')
