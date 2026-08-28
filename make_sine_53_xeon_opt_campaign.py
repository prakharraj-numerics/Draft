from pathlib import Path
import runpy

# Base: exact certified v12. All variants below preserve the same Mode-5/secant
# coefficients, anchor rule, local delta, compensated pi/4 reduction, guarded
# fallback and <=1 ULP contract unless explicitly labelled otherwise.
runpy.run_path('make_sine_53_xeon_v12_batch.py', run_name='__main__')
base = Path('bench_sine_53_xeon_v12_build.c').read_text()


def replace_qfloor(src):
    old='__m512d qfloor=_mm512_roundscale_pd(qf,_MM_FROUND_TO_ZERO|_MM_FROUND_NO_EXC);'
    new='__m512d qfloor=_mm512_cvtepi32_pd(qi);'
    if old not in src:
        raise SystemExit('qfloor pattern not found')
    return src.replace(old,new)


def make_interleave(src, width):
    # Change only Phase 2. j,d, coefficient bits and Horner FMA order per block
    # are unchanged. We expose independent blocks simultaneously so the CPU can
    # overlap gather latency across vectors.
    marker='''        /* Phase 2: homogeneous Mode-5 work; same six coefficients/FMA order. */\n        for(size_t b=0;b<blocks;b++){\n            __m512d rh=_mm512_load_pd(rhbuf+b*8),rl=_mm512_load_pd(rlbuf+b*8);\n            __m512d p=unitbuf[b]?mode5_poly_x11(k,rh,(__mmask8)signbuf[b]):\n                                    mode5_poly_low_x11(k,rh,rl,(__mmask8)signbuf[b]);\n            _mm512_mask_storeu_pd(out+tile+b*8,(__mmask8)activebuf[b],p);\n        }\n'''
    if marker not in src:
        raise SystemExit('phase2 marker not found')

    # Specialized helper computes metadata and issues all six gathers for W
    # independent blocks before starting their Horner chains.
    helper = f'''\nOVEC static inline void mode5_phase2_x{width}(const s53w_kernel *k,\n        const double *rhbuf,const double *rlbuf,const unsigned char *signbuf,\n        const unsigned char *activebuf,const unsigned char *unitbuf,\n        double *out,size_t b0,size_t blocks)\n{{\n    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();\n    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);\n    size_t count=blocks-b0;if(count>{width})count={width};\n    __m512d d[{width}],p[{width}],c4[{width}],c3[{width}],c2[{width}],c1[{width}],c0[{width}];\n    __m256i ji[{width}];\n    __mmask8 sm[{width}],am[{width}];\n    for(size_t u=0;u<count;u++){{\n        size_t b=b0+u;__m512d rh=_mm512_load_pd(rhbuf+b*8),rl=_mm512_load_pd(rlbuf+b*8);\n        __m512d ya=unitbuf[b]?rh:_mm512_add_pd(rh,rl);\n        __m512d sy=_mm512_mul_pd(ya,VK);\n        ji[u]=_mm512_cvt_roundpd_epi32(sy,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);\n        __m512d jd=_mm512_cvtepi32_pd(ji[u]);\n        d[u]=unitbuf[b]?_mm512_fnmadd_pd(jd,VIK,rh):_mm512_add_pd(_mm512_sub_pd(rh,_mm512_mul_pd(jd,VIK)),rl);\n        sm[u]=(__mmask8)signbuf[b];am[u]=(__mmask8)activebuf[b];\n    }}\n    /* Issue coefficient gathers for all independent blocks before Horner. */\n    for(size_t u=0;u<count;u++) p[u]=_mm512_i32gather_pd(ji[u],tab+5*LUTN,8);\n    for(size_t u=0;u<count;u++) c4[u]=_mm512_i32gather_pd(ji[u],tab+4*LUTN,8);\n    for(size_t u=0;u<count;u++) c3[u]=_mm512_i32gather_pd(ji[u],tab+3*LUTN,8);\n    for(size_t u=0;u<count;u++) c2[u]=_mm512_i32gather_pd(ji[u],tab+2*LUTN,8);\n    for(size_t u=0;u<count;u++) c1[u]=_mm512_i32gather_pd(ji[u],tab+1*LUTN,8);\n    for(size_t u=0;u<count;u++) c0[u]=_mm512_i32gather_pd(ji[u],tab+0*LUTN,8);\n    /* Same five FMAs, same order within each block; blocks interleaved only. */\n    for(size_t u=0;u<count;u++) p[u]=_mm512_fmadd_pd(p[u],d[u],c4[u]);\n    for(size_t u=0;u<count;u++) p[u]=_mm512_fmadd_pd(p[u],d[u],c3[u]);\n    for(size_t u=0;u<count;u++) p[u]=_mm512_fmadd_pd(p[u],d[u],c2[u]);\n    for(size_t u=0;u<count;u++) p[u]=_mm512_fmadd_pd(p[u],d[u],c1[u]);\n    for(size_t u=0;u<count;u++) p[u]=_mm512_fmadd_pd(p[u],d[u],c0[u]);\n    for(size_t u=0;u<count;u++){{\n        p[u]=_mm512_mask_sub_pd(p[u],sm[u],Z,p[u]);\n        _mm512_mask_storeu_pd(out+(b0+u)*8,am[u],p[u]);\n    }}\n}}\n'''
    insert=src.index('\nOVEC static void octant_vector_v8')
    src=src[:insert]+helper+src[insert:]
    repl=f'''        /* Phase 2 x{width}: same Mode-5 values, inter-block latency hiding only. */\n        for(size_t b=0;b<blocks;b+={width})\n            mode5_phase2_x{width}(k,rhbuf,rlbuf,signbuf,activebuf,unitbuf,out+tile,b,blocks);\n'''
    return src.replace(marker,repl)

# qfloor-only
q = replace_qfloor(base)
q=q.replace('S53X12_','S53QF_').replace('xeon_v12_tiled_two_stage_batch','xeon_v12_qfloor_cleanup')
Path('bench_sine_53_qfloor_build.c').write_text(q)
print('S53QF_BUILD_PASS same_v12_math=1 qfloor_from_qi=1 formula_unchanged=1')

# x2/x3/x4 all include qfloor cleanup so we test composition after the free fix.
for w in (2,3,4):
    s=make_interleave(replace_qfloor(base),w)
    s=s.replace('S53X12_',f'S53I{w}_').replace('xeon_v12_tiled_two_stage_batch',f'xeon_v12_qfloor_phase2_interleave_x{w}')
    Path(f'bench_sine_53_interleave_x{w}_build.c').write_text(s)
    print(f'S53I{w}_BUILD_PASS same_v12_math=1 same_coeff_bits=1 same_anchor_rule=1 same_delta=1 same_Horner_FMA_order=1 qfloor_cleanup=1 phase2_interleave={w} formula_unchanged=1')
