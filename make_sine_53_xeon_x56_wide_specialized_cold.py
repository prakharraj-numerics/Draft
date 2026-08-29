from pathlib import Path
import runpy

# X56: X55 math and >1 specialization unchanged, but force the generic
# mixed-domain fallback out of the hot symbol so the compiler cannot inline
# both frontends into octant_vector_v8.
runpy.run_path('make_sine_53_xeon_x55_wide_specialized.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x55_build.c')
s=p.read_text()
needle='OVEC static inline void x55_prepare_generic('
if needle not in s:
    raise SystemExit('x55 generic helper marker not found')
s=s.replace(needle,'OVEC __attribute__((noinline,cold)) static void x55_prepare_generic(',1)
s=s.replace('S53X55_','S53X56_')
s=s.replace('xeon_x55_wide_specialized_g4','xeon_x56_wide_specialized_cold_g4')
s=s.replace('Xeon_AVX512_X55_wide_specialized','Xeon_AVX512_X56_wide_specialized_cold')
Path('bench_sine_53_xeon_x56_build.c').write_text(s)
print('S53X56_BUILD_PASS parent=X55 dedicated_gt1_full8=1 generic_mixed_fallback_noinline_cold=1 right_shifter_quotient=1 compensated_DD=1 rare_3piece_repair=1 unit_direct=1 Mode5_unchanged=1 LUT_unchanged=1 Horner_unchanged=1 G4_schedule_unchanged=1 cross_iteration_lookahead=1 requires_Arb_regate=1')
