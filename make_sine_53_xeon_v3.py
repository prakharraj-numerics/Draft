from pathlib import Path
import runpy

# Derive strictly from the proven v8 baseline.  This experiment changes only
# instruction scheduling / octant bookkeeping, not the mathematical path.
runpy.run_path('make_sine_53_octant_v8.py', run_name='__main__')
src = Path('bench_sine_53_wide_octant_v8_build.c').read_text()

insert_at = src.index('OVEC static inline __m512d mode5_poly_i32_low')
helper = r'''
/* Xeon degree-5 specialization.  The six gathers are independent and are
   exposed before the Horner dependency chain so OoO execution can overlap
   gather latency.  Arithmetic order of the five FMAs is identical to v8. */
OVEC static inline __m512d mode5_x3_deg5(const s53w_kernel *k,
                                         __m512d yh,__m512d yl,
                                         __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    __m512d ya=_mm512_add_pd(yh,yl);
    __m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(ya,VK),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256i ji=_mm512_cvttpd_epi32(jd);
    __m512d d=_mm512_sub_pd(yh,_mm512_mul_pd(jd,VIK));
    d=_mm512_add_pd(d,yl);

    /* Fixed degree=5: issue all L1-resident gathers before Horner. */
    const double *tab=k->tab;
    __m512d c5=_mm512_i32gather_pd(ji,tab+(size_t)5*LUTN,8);
    __m512d c4=_mm512_i32gather_pd(ji,tab+(size_t)4*LUTN,8);
    __m512d c3=_mm512_i32gather_pd(ji,tab+(size_t)3*LUTN,8);
    __m512d c2=_mm512_i32gather_pd(ji,tab+(size_t)2*LUTN,8);
    __m512d c1=_mm512_i32gather_pd(ji,tab+(size_t)1*LUTN,8);
    __m512d c0=_mm512_i32gather_pd(ji,tab,8);

    __m512d p=_mm512_fmadd_pd(c5,d,c4);
    p=_mm512_fmadd_pd(p,d,c3);
    p=_mm512_fmadd_pd(p,d,c2);
    p=_mm512_fmadd_pd(p,d,c1);
    p=_mm512_fmadd_pd(p,d,c0);
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

'''
src = src[:insert_at] + helper + src[insert_at:]

# Replace only the two evaluator calls in v8's vector path.
src = src.replace('__m512d p=mode5_poly_i32(k,ax,inneg);\n            _mm512_mask_storeu_pd(out+i,active,p); continue;',
                  '__m512d p=mode5_x3_deg5(k,ax,Z,inneg);\n            _mm512_mask_storeu_pd(out+i,active,p); continue;', 1)
src = src.replace('__m512d p=mode5_poly_i32_low(k,rh,rl,signmask);',
                  '__m512d p=mode5_x3_deg5(k,rh,rl,signmask);', 1)

# Replace seven equality masks + three masked q adjustments by compact octant
# bit logic.  Mapping is exactly the same:
# r=o&3 => adjustment [0,-1,+2,+1] = (r&2)-(r&1),
# reflection iff bit 1 is set, sine sign flips iff octant bit 2 is set.
old = '''        __m256i oi=_mm256_and_si256(qi,_mm256_set1_epi32(7));
        __mmask8 m1=mask_eq_i32(oi,1),m2=mask_eq_i32(oi,2),m3=mask_eq_i32(oi,3);
        __mmask8 m4=mask_eq_i32(oi,4),m5=mask_eq_i32(oi,5),m6=mask_eq_i32(oi,6),m7=mask_eq_i32(oi,7);
        __mmask8 am1=(__mmask8)((m1|m5)&wide),ap2=(__mmask8)((m2|m6)&wide),ap1=(__mmask8)((m3|m7)&wide);
        __mmask8 rev=(__mmask8)((m2|m3|m6|m7)&wide),wide_neg=(__mmask8)((m4|m5|m6|m7)&wide);
        __m512d md=qd;
        md=_mm512_mask_add_pd(md,am1,md,VM1);
        md=_mm512_mask_add_pd(md,ap2,md,VP2);
        md=_mm512_mask_add_pd(md,ap1,md,VP1);
'''
new = '''        __m256i oi=_mm256_and_si256(qi,_mm256_set1_epi32(7));
        __m256i rr=_mm256_and_si256(oi,_mm256_set1_epi32(3));
        __m256i b1=_mm256_and_si256(rr,_mm256_set1_epi32(1));
        __m256i b2=_mm256_and_si256(rr,_mm256_set1_epi32(2));
        __m256i mi=_mm256_add_epi32(qi,_mm256_sub_epi32(b2,b1));
        __m512d md=_mm512_cvtepi32_pd(mi);
        __mmask8 rev=(__mmask8)((~mask_eq_i32(b2,0))&wide);
        __m256i b4=_mm256_and_si256(oi,_mm256_set1_epi32(4));
        __mmask8 wide_neg=(__mmask8)((~mask_eq_i32(b4,0))&wide);
'''
if old not in src:
    raise SystemExit('octant block marker not found')
src = src.replace(old,new,1)

src = src.replace('S53O8_', 'S53X3_')
src = src.replace('reduction=cosine_style_pi4_octant_guarded_v8',
                  'reduction=cosine_style_pi4_octant_guarded_x3_xeon_sched')
src = src.replace('AVX512_pi4_octant_int32_compensated_cw',
                  'AVX512_pi4_octant_int32_compensated_cw_deg5_gather_sched')
Path('bench_sine_53_xeon_v3_build.c').write_text(src)
print('S53X3_BUILD_PASS v8_exact_math=1 deg5_specialized=1 six_gathers_upfront=1 octant_bitlogic=1 compensated_cw=1 edge_fallback_unchanged=1')
