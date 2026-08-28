from pathlib import Path
import runpy,sys

saved=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x23_hybridpi.py','14']
    runpy.run_path('make_sine_53_xeon_x23_hybridpi.py',run_name='__main__')
finally:
    sys.argv=saved

p=Path('bench_sine_53_xeon_x23_h14_build.c')
s=p.read_text()

for b in range(4):
    old_recon=f'        c2_{b}=_mm512_mul_pd(c0_{b},MH); c3_{b}=_mm512_mul_pd(c1_{b},M6); c4_{b}=_mm512_mul_pd(c0_{b},C24); c5_{b}=_mm512_mul_pd(c1_{b},C120);'
    new_recon=(
      old_recon+'\n'
      f'        __m512i c1bits{b}=_mm512_castpd_si512(c1_{b});\n'
      f'        __m512d sig{b}=_mm512_castsi512_pd(_mm512_and_epi64(c1bits{b},_mm512_set1_epi64((long long)UINT64_C(0x7ff0000000000000))));\n'
      f'        __m512d sd{b}=_mm512_mul_pd(sig{b},d{b});\n'
      f'        __m512d sum{b}=_mm512_add_pd(c0_{b},sd{b});\n'
      f'        __m512d bb{b}=_mm512_sub_pd(sum{b},c0_{b});\n'
      f'        __m512d err{b}=_mm512_add_pd(_mm512_sub_pd(c0_{b},_mm512_sub_pd(sum{b},bb{b})),_mm512_sub_pd(sd{b},bb{b}));\n'
      f'        __m512d z{b}=_mm512_mul_pd(d{b},d{b});\n'
      f'        __m512d u{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});\n'
      f'        __m512d v{b}=_mm512_fmadd_pd(c3_{b},d{b},c2_{b});')
    if old_recon not in s:
        raise SystemExit(f'X35 recon marker missing stream {b}')
    s=s.replace(old_recon,new_recon,1)

    old_hor=(
      f'        p{b}=_mm512_fmadd_pd(c5_{b},d{b},c4_{b});\n'
      f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c3_{b});\n'
      f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c2_{b});\n'
      f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c1_{b});\n'
      f'        p{b}=_mm512_fmadd_pd(p{b},d{b},c0_{b});')
    new_hor=(
      f'        __m512d corr{b}=_mm512_mul_pd(z{b},_mm512_fmadd_pd(u{b},z{b},v{b}));\n'
      f'        corr{b}=_mm512_fmadd_pd(_mm512_sub_pd(c1_{b},sig{b}),d{b},corr{b});\n'
      f'        corr{b}=_mm512_add_pd(corr{b},err{b});\n'
      f'        p{b}=_mm512_add_pd(sum{b},corr{b});')
    if old_hor not in s:
        raise SystemExit(f'X35 Horner marker missing stream {b}')
    s=s.replace(old_hor,new_hor,1)

s=s.replace('S53X23H14_','S53X35_')
s=s.replace('xeon_x23_nearestpi_2f_hybrid_b14_g4_2g','xeon_x35_pow2_dominant_split_g4')
s=s.replace('Xeon_AVX512_G4_nearestpi_2f_hybrid_b14_two_gather_Mode5','Xeon_AVX512_X35_pow2_dominant_split')
Path('bench_sine_53_xeon_x35_pow2_build.c').write_text(s)
print('X35_BUILD_PASS parent=X23H14 pow2_sigma_from_c1=1 exact_sigma_times_delta=1 twosum_dominant=1 parallel_correction=1 two_gather=1 G4=1 requires_Arb_regate=1')
