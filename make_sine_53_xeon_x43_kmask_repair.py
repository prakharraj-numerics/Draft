from pathlib import Path
import runpy, sys

# X43 starts from X41: X38 staggered two-gather G4 plus exact branchless
# local-delta FMA.  Only the rare B14 repair eligibility is rewritten so
# unit/near/repair stay in AVX-512 k-mask domain until the final kortest.
saved=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x41_branchless_delta.py','fma']
    runpy.run_path('make_sine_53_xeon_x41_branchless_delta.py',run_name='__main__')
finally:
    sys.argv=saved

p=Path('bench_sine_53_xeon_x41_build.c')
s=p.read_text()
old='''    __mmask8 repair=(__mmask8)(_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-14),_CMP_LT_OQ)&active&~unit);
    if(__builtin_expect(repair!=0,0)){'''
new='''    __mmask8 near0=_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-14),_CMP_LT_OQ);
    __mmask8 nonunit=_knot_mask8(unit);
    __mmask8 repair=_kand_mask8(near0,_kand_mask8(active,nonunit));
    if(__builtin_expect(!_kortestz_mask8_u8(repair,repair),0)){'''
if old not in s:
    raise SystemExit('X41 B14 repair marker missing')
s=s.replace(old,new,1)
s=s.replace('S53X41_','S53X43_')
s=s.replace('xeon_x41_branchless_exact_fma_delta_g4','xeon_x43_branchless_delta_kmask_repair_g4')
s=s.replace('Xeon_AVX512_X41_branchless_exact_fma_delta','Xeon_AVX512_X43_branchless_delta_kmask_repair')
out=Path('bench_sine_53_xeon_x43_build.c')
out.write_text(s)
print('X43_BUILD_PASS parent=X41 G4=1 two_gather=1 exact_branchless_delta=1 B14_repair_kmask_domain=1 scalar_repair_mask_logic_removed=1 requires_Arb_regate=1')
