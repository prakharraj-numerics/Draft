from pathlib import Path
import runpy

# Build the current structural campaign and take X50 as the exact hardware base.
runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
p = Path('bench_sine_53_xeon_x50_build.c')
s = p.read_text()

# Restrict the transform to the named X50 AVX-512 hot function.  Reducer, LUT,
# cross-iteration lookahead, gather schedule, masks, stores and rare repair stay
# byte-for-byte source-identical outside this evaluator algebra.
hit = s.index('octant_vector_v8(const s53w_kernel *k,')
start = s.rfind('\n', 0, hit) + 1
end = s.index('\n#endif', hit)
hot = s[start:end]

old_const = '    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();'
new_const = old_const[:-1] + ',ONE=_mm512_set1_pd(1.0);'
if old_const not in hot:
    raise SystemExit('X55: X50 vector constant line not found')
hot = hot.replace(old_const, new_const, 1)

for b in range(4):
    recon = (f'        c2_{b}=_mm512_mul_pd(c0_{b},MH); c3_{b}=_mm512_mul_pd(c1_{b},M6); '
             f'c4_{b}=_mm512_mul_pd(c0_{b},C24); c5_{b}=_mm512_mul_pd(c1_{b},C120);')
    cf_recon = (f'        __m512d z{b}=_mm512_mul_pd(d{b},d{b}); '
                f'__m512d A{b}=_mm512_fmadd_pd(z{b},C24,MH); '
                f'__m512d B{b}=_mm512_fmadd_pd(z{b},C120,M6);')
    if recon not in hot:
        raise SystemExit(f'X55: reconstruction block {b} missing')
    hot = hot.replace(recon, cf_recon, 1)

    hor = '\n'.join([
        f'        p{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});',
        f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c3_{b});',
        f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c2_{b});',
        f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c1_{b});',
        f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c0_{b});'])
    cf_hor = '\n'.join([
        f'        A{b}=_mm512_fmadd_pd(A{b},z{b},ONE);',
        f'        B{b}=_mm512_fmadd_pd(B{b},z{b},ONE);',
        f'        __m512d cd{b}=_mm512_mul_pd(c1_{b},d{b});',
        f'        p{b}=_mm512_mul_pd(c0_{b},A{b});',
        f'        p{b}=_mm512_fmadd_pd(cd{b},B{b},p{b});'])
    if hor not in hot:
        raise SystemExit(f'X55: Horner block {b} missing')
    hot = hot.replace(hor, cf_hor, 1)

s = s[:start] + hot + s[end:]
s = s.replace('S53X50_', 'S53X55_')
s = s.replace('Xeon_AVX512_X50_cross_iteration_lookahead', 'Xeon_AVX512_X55_CF_X50_pipeline')
s = s.replace('xeon_x50_cross_iteration_lookahead_g4', 'xeon_x55_cf_x50_pipeline_g4')

out = Path('bench_sine_53_xeon_x55_build.c')
out.write_text(s)
print('X55_BUILD_PASS parent=X50 cross_iteration_pipeline=1 CF_evenodd_hot=1 K=256 two_gather=1 G4=1 reducer_identical=1 LUT_identical=1 requires_Arb_regate=1')
