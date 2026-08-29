from pathlib import Path
import runpy, sys

# X52: preserve X49 reducer, LUT, coefficient reconstruction and exact five-FMA
# Horner arithmetic.  Change only independent instruction order: execute each
# Horner depth across all four streams before advancing to the next depth.
# The generated assembly is inspected in CI; this is the machine-scheduling leg.
saved=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x49_production.py']
    runpy.run_path('make_sine_53_xeon_x49_production.py',run_name='__main__')
finally:
    sys.argv=saved

p=Path('bench_sine_53_xeon_x49_build.c')
s=p.read_text()
old='''        /* Preserve the five Horner FMA operations per lane. */
        p0=_mm512_fmadd_pd(c5_0,d0,c4_0);
        p0=_mm512_fmadd_pd(p0,d0,c3_0);
        p0=_mm512_fmadd_pd(p0,d0,c2_0);
        p0=_mm512_fmadd_pd(p0,d0,c1_0);
        p0=_mm512_fmadd_pd(p0,d0,c0_0);
        p1=_mm512_fmadd_pd(c5_1,d1,c4_1);
        p1=_mm512_fmadd_pd(p1,d1,c3_1);
        p1=_mm512_fmadd_pd(p1,d1,c2_1);
        p1=_mm512_fmadd_pd(p1,d1,c1_1);
        p1=_mm512_fmadd_pd(p1,d1,c0_1);
        p2=_mm512_fmadd_pd(c5_2,d2,c4_2);
        p2=_mm512_fmadd_pd(p2,d2,c3_2);
        p2=_mm512_fmadd_pd(p2,d2,c2_2);
        p2=_mm512_fmadd_pd(p2,d2,c1_2);
        p2=_mm512_fmadd_pd(p2,d2,c0_2);
        p3=_mm512_fmadd_pd(c5_3,d3,c4_3);
        p3=_mm512_fmadd_pd(p3,d3,c3_3);
        p3=_mm512_fmadd_pd(p3,d3,c2_3);
        p3=_mm512_fmadd_pd(p3,d3,c1_3);
        p3=_mm512_fmadd_pd(p3,d3,c0_3);'''
new='''        /* X52 hand schedule: round-robin four independent Horner chains. */
        p0=_mm512_fmadd_pd(c5_0,d0,c4_0);
        p1=_mm512_fmadd_pd(c5_1,d1,c4_1);
        p2=_mm512_fmadd_pd(c5_2,d2,c4_2);
        p3=_mm512_fmadd_pd(c5_3,d3,c4_3);
        p0=_mm512_fmadd_pd(p0,d0,c3_0);
        p1=_mm512_fmadd_pd(p1,d1,c3_1);
        p2=_mm512_fmadd_pd(p2,d2,c3_2);
        p3=_mm512_fmadd_pd(p3,d3,c3_3);
        p0=_mm512_fmadd_pd(p0,d0,c2_0);
        p1=_mm512_fmadd_pd(p1,d1,c2_1);
        p2=_mm512_fmadd_pd(p2,d2,c2_2);
        p3=_mm512_fmadd_pd(p3,d3,c2_3);
        p0=_mm512_fmadd_pd(p0,d0,c1_0);
        p1=_mm512_fmadd_pd(p1,d1,c1_1);
        p2=_mm512_fmadd_pd(p2,d2,c1_2);
        p3=_mm512_fmadd_pd(p3,d3,c1_3);
        p0=_mm512_fmadd_pd(p0,d0,c0_0);
        p1=_mm512_fmadd_pd(p1,d1,c0_1);
        p2=_mm512_fmadd_pd(p2,d2,c0_2);
        p3=_mm512_fmadd_pd(p3,d3,c0_3);'''
if old not in s: raise SystemExit('X49 Horner block missing')
s=s.replace(old,new,1)
s=s.replace('S53X49_','S53X52_')
s=s.replace('Xeon_AVX512_X49_production_hw_schedule','Xeon_AVX512_X52_roundrobin_horner_schedule')
s=s.replace('xeon_x49_production_hw_schedule_g4','xeon_x52_roundrobin_horner_schedule_g4')
out=Path('bench_sine_53_xeon_x52_build.c')
out.write_text(s)
print('X52_BUILD_PASS parent=X49 math_identical=1 reducer_identical=1 LUT_identical=1 coefficient_reconstruction_identical=1 five_FMA_per_lane_identical=1 horner_schedule=round_robin_four_streams requires_assembly_inspection=1 requires_Arb_regate=1')
