from pathlib import Path
import runpy

# X67: X66 topology/math frozen. Only regroup the exact same degree-5
# polynomial so the dominant c0+c1*d cancellation is performed by one FMA.
runpy.run_path('make_sine_53_x66_secant_state512_pipeline_final.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x66_build.c')
s=p.read_text()
old=r'''            __m512d z=_mm512_mul_pd(d[g],d[g]);
            __m512d ce=_mm512_fmadd_pd(z,C24,MH);
            ce=_mm512_fmadd_pd(ce,z,ONE);
            __m512d so=_mm512_fmadd_pd(z,C120,M6);
            so=_mm512_fmadd_pd(so,z,ONE);
            __m512d cd=_mm512_mul_pd(c1[g],d[g]);
            __m512d base=_mm512_mul_pd(c0[g],ce);
            pv[g]=_mm512_fmadd_pd(cd,so,base);'''
new=r'''            __m512d z=_mm512_mul_pd(d[g],d[g]);
            __m512d ec=_mm512_fmadd_pd(z,C24,MH);      /* -1/2 + z/24 */
            __m512d oc=_mm512_fmadd_pd(z,C120,M6);     /* -1/6 + z/120 */
            __m512d cd=_mm512_mul_pd(c1[g],d[g]);
            /* Same polynomial, but fuse the potentially cancelling leading
               terms first: base = c0 + c1*d.  Corrections are O(z). */
            __m512d base=_mm512_fmadd_pd(c1[g],d[g],c0[g]);
            __m512d ep=_mm512_mul_pd(c0[g],ec);
            __m512d inner=_mm512_fmadd_pd(cd,oc,ep);
            pv[g]=_mm512_fmadd_pd(z,inner,base);'''
if old not in s: raise SystemExit('X67 polynomial block not found')
s=s.replace(old,new)
s=s.replace('octant_vector_v8_rawx66','octant_vector_v8_rawx67')
s=s.replace('S53X66_','S53X67_')
s=s.replace('xeon_x66_secant_state512_pipeline_g4','xeon_x67_compensated_grouped_g4')
s=s.replace('Xeon_AVX512_X66_secant_state512_pipeline','Xeon_AVX512_X67_compensated_grouped')
Path('bench_sine_53_xeon_x67_build.c').write_text(s)
print('S53X67_BUILD_PASS parent=X66 topology_frozen=1 math_frozen=1 user_secant_spine=1 state512=1 cancellation_fused_base=1 correction_O_z=1 same_mul_fma_count_target=1 no_fallback=1 requires_Arb_regate=1')
