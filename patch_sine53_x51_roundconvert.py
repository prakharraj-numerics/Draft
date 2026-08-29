from pathlib import Path

p=Path('bench_sine_53_xeon_x51_build.c')
s=p.read_text()

old="""    __m128i qi=_mm256_cvt_roundpd_epi32(_mm256_mul_pd(ax,VINVP),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);"""
new="""    __m256d qnr=_mm256_round_pd(_mm256_mul_pd(ax,VINVP),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m128i qi=_mm256_cvttpd_epi32(qnr);"""
if old not in s:
    raise SystemExit('X51 reducer round-convert marker missing')
s=s.replace(old,new,1)

for b in range(8):
    old=(f'        ji{b}=_mm256_cvt_roundpd_epi32(_mm256_mul_pd(rh{b},VK),'
         '_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);')
    new=(f'        __m256d jnr{b}=_mm256_round_pd(_mm256_mul_pd(rh{b},VK),'
         '_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);\n'
         f'        ji{b}=_mm256_cvttpd_epi32(jnr{b});')
    if old not in s:
        raise SystemExit(f'X51 anchor round-convert marker missing stream {b}')
    s=s.replace(old,new,1)

p.write_text(s)
print('X51_ROUNDCONVERT_PATCH_PASS explicit_round_nearest=1 trunc_after_round=1 mxcsr_independent=1')
