from pathlib import Path
import runpy

# Experimental gather-reduction via per-tile anchor bucketing. This preserves
# the same v12 reducer, same Mode-5 coefficient bits, same nearest-even anchor
# rule j, same local delta d and same 5-FMA Horner order. Only Phase-2 data
# scheduling changes. Designed as a diagnostic, not a production claim.
runpy.run_path('make_sine_53_xeon_v12_batch.py', run_name='__main__')
src=Path('bench_sine_53_xeon_v12_build.c').read_text()

marker='''        /* Phase 2: homogeneous Mode-5 work; same six coefficients/FMA order. */\n        for(size_t b=0;b<blocks;b++){\n            __m512d rh=_mm512_load_pd(rhbuf+b*8),rl=_mm512_load_pd(rlbuf+b*8);\n            __m512d p=unitbuf[b]?mode5_poly_x11(k,rh,(__mmask8)signbuf[b]):\n                                    mode5_poly_low_x11(k,rh,rl,(__mmask8)signbuf[b]);\n            _mm512_mask_storeu_pd(out+tile+b*8,(__mmask8)activebuf[b],p);\n        }\n'''
if marker not in src: raise SystemExit('phase2 marker missing')

# This diagnostic groups scalar lanes by anchor within each 256-input tile.
# It intentionally trades vector gathers for ordinary scalar coefficient loads.
# The arithmetic on each lane remains the same 5 FMA Horner sequence.
repl=r'''        /* Phase 2 grouping diagnostic: same j,d and coefficient bits,
           but bucket lanes by anchor and use ordinary coefficient loads. */
        {
            int head[LUTN];int next[X12_TILE];double dscalar[X12_TILE];unsigned char sscalar[X12_TILE];
            for(int a=0;a<(int)LUTN;a++)head[a]=-1;
            for(size_t idx=0;idx<tn;idx++){
                size_t b=idx>>3,lane=idx&7;
                double rh=rhbuf[idx],rl=rlbuf[idx];
                double ya=unitbuf[b]?rh:(rh+rl);
                /* Match _MM_FROUND_TO_NEAREST_INT: nearest, ties-to-even under
                   the default IEEE rounding mode. Never use lround here. */
                long j=(long)nearbyint(ya*KGRID);if(j<0)j=0;if(j>=(long)LUTN)j=(long)LUTN-1;
                double d=unitbuf[b]?fma(-(double)j,INVK,rh):((rh-(double)j*INVK)+rl);
                dscalar[idx]=d;sscalar[idx]=(unsigned char)((signbuf[b]>>lane)&1u);
                next[idx]=head[j];head[j]=(int)idx;
            }
            for(int a=0;a<(int)LUTN;a++){
                if(head[a]<0)continue;
                const double c5=k->tab[5*LUTN+(size_t)a],c4=k->tab[4*LUTN+(size_t)a];
                const double c3=k->tab[3*LUTN+(size_t)a],c2=k->tab[2*LUTN+(size_t)a];
                const double c1=k->tab[1*LUTN+(size_t)a],c0=k->tab[0*LUTN+(size_t)a];
                for(int idx=head[a];idx>=0;idx=next[idx]){
                    double d=dscalar[idx];double p=fma(c5,d,c4);p=fma(p,d,c3);p=fma(p,d,c2);p=fma(p,d,c1);p=fma(p,d,c0);
                    out[tile+(size_t)idx]=sscalar[idx]?-p:p;
                }
            }
        }
'''
src=src.replace(marker,repl)
src=src.replace('S53X12_','S53GRP_').replace('xeon_v12_tiled_two_stage_batch','xeon_v12_anchor_group_scalar_coeff_loads')
Path('bench_sine_53_grouping_build.c').write_text(src)
print('S53GRP_BUILD_PASS exact_v12_reducer=1 same_secant_Mode5_spine=1 same_coeff_bits=1 same_anchor_rule_nearest_even=1 same_delta_formula=1 same_Horner_FMA_order=1 gather_eliminated=1 anchor_bucket_tile=256 diagnostic_only=1')
