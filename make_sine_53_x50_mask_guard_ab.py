from pathlib import Path
import re, runpy, sys

if len(sys.argv)!=2 or sys.argv[1] not in ('mask','noguard','both'):
    raise SystemExit('usage: make_sine_53_x50_mask_guard_ab.py {mask|noguard|both}')
mode=sys.argv[1]
runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py',run_name='__main__')
p=Path('bench_sine_53_xeon_x50_build.c')
s=p.read_text()

mask_count=0
guard_count=0

if mode in ('mask','both'):
    # Replace legacy int->float-signbit->movemask extraction with AVX-512VL native mask tests.
    pat=re.compile(r'_mm256_movemask_ps\(_mm256_castsi256_ps\(_mm256_slli_epi32\(([^,()]+),\s*(29|30)\)\)\)')
    def repl(m):
        global mask_count
        mask_count += 1
        bit = '4' if m.group(2)=='29' else '2'
        return f'_mm256_test_epi32_mask({m.group(1).strip()},_mm256_set1_epi32({bit}))'
    s=pat.sub(repl,s)
    if mask_count < 2:
        raise SystemExit(f'mask replacement count too small: {mask_count}')

if mode in ('noguard','both'):
    # Restrict the diagnostic edit to x12_prepare_block: preserve reduction/result math,
    # but suppress its scalar-repair guard output. Full Arb stress must re-gate safety.
    a=s.index('OVEC static inline void x12_prepare_block')
    b=s.index('\nOVEC ',a+20)
    fn=s[a:b]
    patg=re.compile(r'\*guard_out\s*=\s*([^;]+);')
    fn2,n=patg.subn('*guard_out=0;',fn)
    guard_count=n
    if guard_count < 1:
        raise SystemExit('no guard_out assignment found in x12_prepare_block')
    s=s[:a]+fn2+s[b:]

label={'mask':'MASK','noguard':'NOGUARD','both':'BOTH'}[mode]
s=s.replace('S53X50_',f'S53X50{label}_')
s=s.replace('xeon_x50_cross_iteration_lookahead_g4',f'xeon_x50_{mode}_ab_g4')
s=s.replace('Xeon_AVX512_X50_cross_iteration_lookahead',f'Xeon_AVX512_X50_{label}_AB')
out=Path(f'bench_sine_53_x50_{mode}_ab.c')
out.write_text(s)
print(f'S53X50_AB_BUILD_PASS mode={mode} mask_replacements={mask_count} guard_assignments_zeroed={guard_count} formula_unchanged=1 LUT_unchanged=1 Horner_unchanged=1 full_stress_required=1')
