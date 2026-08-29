from pathlib import Path
import runpy

# X67: keep X66's successful hardware pipeline, but restore X65's certified
# degree-5 Horner evaluation order.  Mathematics/state/residual/table unchanged.
runpy.run_path('make_sine_53_x66_secant_state512_pipeline_final.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x66_build.c')
s=p.read_text()
old=r'''        /* Same degree-5 polynomial, algebraically grouped into independent
           even/odd d^2 chains: much shorter dependency chain than Horner. */
        for(int g=0;g<4;g++){
            __m512d z=_mm512_mul_pd(d[g],d[g]);
            __m512d ce=_mm512_fmadd_pd(z,C24,MH);
            ce=_mm512_fmadd_pd(ce,z,ONE);
            __m512d so=_mm512_fmadd_pd(z,C120,M6);
            so=_mm512_fmadd_pd(so,z,ONE);
            __m512d cd=_mm512_mul_pd(c1[g],d[g]);
            __m512d base=_mm512_mul_pd(c0[g],ce);
            pv[g]=_mm512_fmadd_pd(cd,so,base);
            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);
            _mm512_storeu_pd(out+i+8*g,pv[g]);
        }'''
new=r'''        /* Restore the exact X65-certified degree-5 Horner order; keep X66's
           no-prescan and cross-iteration state/gather pipeline. */
        for(int g=0;g<4;g++){
            __m512d c2=_mm512_mul_pd(c0[g],MH);
            __m512d c3=_mm512_mul_pd(c1[g],M6);
            __m512d c4=_mm512_mul_pd(c0[g],C24);
            __m512d c5=_mm512_mul_pd(c1[g],C120);
            pv[g]=_mm512_fmadd_pd(c5,d[g],c4);
            pv[g]=_mm512_fmadd_pd(pv[g],d[g],c3);
            pv[g]=_mm512_fmadd_pd(pv[g],d[g],c2);
            pv[g]=_mm512_fmadd_pd(pv[g],d[g],c1[g]);
            pv[g]=_mm512_fmadd_pd(pv[g],d[g],c0[g]);
            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);
            _mm512_storeu_pd(out+i+8*g,pv[g]);
        }'''
if old not in s: raise SystemExit('X66 grouped block not found')
s=s.replace(old,new,1)
s=s.replace('octant_vector_v8_rawx66','octant_vector_v8_rawx67')
s=s.replace('S53X66_','S53X67_')
s=s.replace('xeon_x66_secant_state512_pipeline_g4','xeon_x67_secant_state512_pipeline_horner_g4')
s=s.replace('Xeon_AVX512_X66_secant_state512_pipeline','Xeon_AVX512_X67_secant_state512_pipeline_horner')
Path('bench_sine_53_xeon_x67_build.c').write_text(s)
print('S53X67_BUILD_PASS parent=X66 frozen_math=1 user_secant_spine=1 state512=1 raw_x=1 no_domain_prescan=1 G4_AVX512=1 stream0_cross_iteration_lookahead=1 x65_certified_horner=1 split_pi512_twoFMA=1 formula_unchanged=1 requires_Arb_regate=1')
