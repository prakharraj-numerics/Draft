from pathlib import Path
import runpy, sys

if len(sys.argv)!=2 or sys.argv[1] not in ('fma','mulsub'):
    raise SystemExit('usage: make_sine_53_xeon_x41_branchless_delta.py {fma|mulsub}')
mode=sys.argv[1]

# Start from isolated X40, which is exactly X38 numerically/scheduling-wise
# but exposes the evaluator as a stable noinline hot symbol.
runpy.run_path('make_sine_53_xeon_x40_isolated.py',run_name='__main__')
p=Path('bench_sine_53_xeon_x40_isolated_build.c')
s=p.read_text()

for b in range(4):
    old=(f'        __m512d jd{b}=_mm512_cvtepi32_pd(ji{b});\n'
         f'        if(pu{b}) d{b}=_mm512_fnmadd_pd(jd{b},VIK,rh{b}); else {{d{b}=_mm512_sub_pd(rh{b},_mm512_mul_pd(jd{b},VIK));d{b}=_mm512_add_pd(d{b},rl{b});}}')
    if old not in s:
        raise SystemExit(f'X40 delta marker missing stream {b}')
    if mode=='fma':
        # ji is an integer anchor index and VIK=1/256 is a power of two.
        # Therefore jd*VIK is exact in binary64.  The fused subtraction is
        # bit-equivalent to mul+sub here; adding rl preserves the wide path,
        # while rl==0 on pure-unit blocks.  Arb and bit-counts re-gate this.
        new=(f'        __m512d jd{b}=_mm512_cvtepi32_pd(ji{b});\n'
             f'        d{b}=_mm512_fnmadd_pd(jd{b},VIK,rh{b});\n'
             f'        d{b}=_mm512_add_pd(d{b},rl{b});')
    else:
        # Control: remove the scalar pure-unit branch but retain the original
        # non-pure arithmetic order exactly for every block.
        new=(f'        __m512d jd{b}=_mm512_cvtepi32_pd(ji{b});\n'
             f'        d{b}=_mm512_sub_pd(rh{b},_mm512_mul_pd(jd{b},VIK));\n'
             f'        d{b}=_mm512_add_pd(d{b},rl{b});')
    s=s.replace(old,new,1)

if mode=='fma':
    tag='X41'; prefix='S53X41_'; arch='xeon_x41_branchless_exact_fma_delta_g4'; label='Xeon_AVX512_X41_branchless_exact_fma_delta'
else:
    tag='X42'; prefix='S53X42_'; arch='xeon_x42_branchless_mulsub_delta_g4'; label='Xeon_AVX512_X42_branchless_mulsub_delta'
s=s.replace('S53X40_',prefix)
s=s.replace('xeon_x40_isolated_x38_g4',arch)
s=s.replace('Xeon_AVX512_X40_isolated_X38_g4',label)
out=Path(f'bench_sine_53_xeon_{tag.lower()}_build.c')
out.write_text(s)
print(f'{tag}_BUILD_PASS parent=X40 noinline_hot_symbol=1 G4=1 pure_unit_scalar_branch_removed=1 mode={mode} exact_anchor_product=jd_times_2^-8 requires_Arb_regate=1')
