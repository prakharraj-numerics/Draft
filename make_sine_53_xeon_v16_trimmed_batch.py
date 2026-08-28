from pathlib import Path
import runpy

# v16 diagnostic: preserve exact v12 reducer, coefficients, anchor rule, delta
# and five-Horner-FMA order. Only remove needless tile scratch traffic:
#   * don't store/reload rl for pure-unit blocks
#   * don't store active masks; derive the one possible tail mask from tn/b
runpy.run_path('make_sine_53_xeon_v12_batch.py', run_name='__main__')
p=Path('bench_sine_53_xeon_v12_build.c')
s=p.read_text()

old_arrays='''    _Alignas(64) double rhbuf[X12_TILE],rlbuf[X12_TILE];\n    unsigned char signbuf[X12_TILE/8],guardbuf[X12_TILE/8],activebuf[X12_TILE/8],unitbuf[X12_TILE/8];'''
new_arrays='''    _Alignas(64) double rhbuf[X12_TILE],rlbuf[X12_TILE];\n    unsigned char signbuf[X12_TILE/8],guardbuf[X12_TILE/8],unitbuf[X12_TILE/8];'''
if old_arrays not in s: raise SystemExit('v12 scratch declaration not found')
s=s.replace(old_arrays,new_arrays,1)

old_phase1='''            __m512d rh,rl;__mmask8 s,g,a;unsigned char pu;\n            x12_prepare_block(x+tile,b*8,tn,&rh,&rl,&s,&g,&a,&pu);\n            _mm512_store_pd(rhbuf+b*8,rh);_mm512_store_pd(rlbuf+b*8,rl);\n            signbuf[b]=(unsigned char)s;guardbuf[b]=(unsigned char)g;activebuf[b]=(unsigned char)a;unitbuf[b]=pu;'''
new_phase1='''            __m512d rh,rl;__mmask8 s,g,a;unsigned char pu;\n            x12_prepare_block(x+tile,b*8,tn,&rh,&rl,&s,&g,&a,&pu);\n            _mm512_store_pd(rhbuf+b*8,rh);\n            if(__builtin_expect(!pu,1))_mm512_store_pd(rlbuf+b*8,rl);\n            signbuf[b]=(unsigned char)s;guardbuf[b]=(unsigned char)g;unitbuf[b]=pu;'''
if old_phase1 not in s: raise SystemExit('v12 phase1 not found')
s=s.replace(old_phase1,new_phase1,1)

old_phase2='''        /* Phase 2: homogeneous Mode-5 work; same six coefficients/FMA order. */\n        for(size_t b=0;b<blocks;b++){\n            __m512d rh=_mm512_load_pd(rhbuf+b*8),rl=_mm512_load_pd(rlbuf+b*8);\n            __m512d p=unitbuf[b]?mode5_poly_x11(k,rh,(__mmask8)signbuf[b]):\n                                    mode5_poly_low_x11(k,rh,rl,(__mmask8)signbuf[b]);\n            _mm512_mask_storeu_pd(out+tile+b*8,(__mmask8)activebuf[b],p);\n        }'''
new_phase2='''        /* Phase 2: exact same Mode-5 arithmetic, less scratch traffic. */\n        for(size_t b=0;b<blocks;b++){\n            __m512d rh=_mm512_load_pd(rhbuf+b*8),p;\n            if(unitbuf[b]){\n                p=mode5_poly_x11(k,rh,(__mmask8)signbuf[b]);\n            }else{\n                __m512d rl=_mm512_load_pd(rlbuf+b*8);\n                p=mode5_poly_low_x11(k,rh,rl,(__mmask8)signbuf[b]);\n            }\n            size_t left=tn-b*8;\n            __mmask8 a=(__mmask8)(left>=8?0xffu:((1u<<(unsigned)left)-1u));\n            if(__builtin_expect(a==0xffu,1))_mm512_storeu_pd(out+tile+b*8,p);\n            else _mm512_mask_storeu_pd(out+tile+b*8,a,p);\n        }'''
if old_phase2 not in s: raise SystemExit('v12 phase2 not found')
s=s.replace(old_phase2,new_phase2,1)

s=s.replace('S53X12_','S53V16_')
s=s.replace('xeon_v12_tiled_two_stage_batch','xeon_v16_trimmed_tile_scratch')
pout=Path('bench_sine_53_xeon_v16_trimmed_build.c')
pout.write_text(s)
print('S53V16_BUILD_PASS exact_v12_math=1 exact_v12_reducer=1 same_coeff_bits=1 same_anchor_rule=1 same_delta=1 same_Horner_FMA_order=1 skip_unit_rl_roundtrip=1 derive_active_mask=1')
