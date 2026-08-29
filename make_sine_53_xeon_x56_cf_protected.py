from pathlib import Path
import runpy

# Build X55 first, then keep its X50 pipeline/LUT/reducer exactly while
# reorganizing only the CF evaluator so the dominant c0 + c1*d term is
# protected by FMA.  Same arithmetic-count class as X55: 3 MUL + 5 FMA per
# stream, but rounding is pushed into the O(d^2) correction.
runpy.run_path('make_sine_53_xeon_x55_cf_x50.py', run_name='__main__')
p = Path('bench_sine_53_xeon_x55_build.c')
s = p.read_text()

hit = s.index('octant_vector_v8(const s53w_kernel *k,')
start = s.rfind('\n', 0, hit) + 1
end = s.index('\n#endif', hit)
hot = s[start:end]

for b in range(4):
    old = '\n'.join([
        f'        A{b}=_mm512_fmadd_pd(A{b},z{b},ONE);',
        f'        B{b}=_mm512_fmadd_pd(B{b},z{b},ONE);',
        f'        __m512d cd{b}=_mm512_mul_pd(c1_{b},d{b});',
        f'        p{b}=_mm512_mul_pd(c0_{b},A{b});',
        f'        p{b}=_mm512_fmadd_pd(cd{b},B{b},p{b});'])
    new = '\n'.join([
        f'        __m512d base{b}=_mm512_fmadd_pd(c1_{b},d{b},c0_{b});',
        f'        __m512d cd{b}=_mm512_mul_pd(c1_{b},d{b});',
        f'        __m512d corr{b}=_mm512_mul_pd(c0_{b},A{b});',
        f'        corr{b}=_mm512_fmadd_pd(cd{b},B{b},corr{b});',
        f'        p{b}=_mm512_fmadd_pd(z{b},corr{b},base{b});'])
    if old not in hot:
        raise SystemExit(f'X56: X55 CF block {b} missing')
    hot = hot.replace(old, new, 1)

# A and B in X55 are already the correction factors
#   A=-1/2+z/24, B=-1/6+z/120
# after removing the two '+1' FMAs above.  ONE is therefore dead and ICX
# will eliminate it.
s = s[:start] + hot + s[end:]
s = s.replace('S53X55_', 'S53X56_')
s = s.replace('Xeon_AVX512_X55_CF_X50_pipeline', 'Xeon_AVX512_X56_CF_protected_X50_pipeline')
s = s.replace('xeon_x55_cf_x50_pipeline_g4', 'xeon_x56_cf_protected_x50_pipeline_g4')

out = Path('bench_sine_53_xeon_x56_build.c')
out.write_text(s)
print('X56_BUILD_PASS parent=X55 parent_hw=X50 protected_base_fma=1 correction_Od2=1 same_3mul_5fma_class=1 K=256 two_gather=1 G4=1 requires_Arb_regate=1')
