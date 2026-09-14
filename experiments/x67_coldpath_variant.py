from pathlib import Path
import sys

if len(sys.argv) != 4:
    raise SystemExit('usage: x67_coldpath_variant.py src.c dst.c mode')
src, dst, mode = sys.argv[1:]
if mode not in ('nested_band','broad_hi','flat_band'):
    raise SystemExit('bad mode '+mode)
s = Path(src).read_text()

helper = r'''
OVEC __attribute__((noinline,cold)) static __m512d x67_rare_small_repair(
        __m512d rh,__m512d rl,__mmask8 signmask)
{
    const __m512d Z=_mm512_setzero_pd();
    const __m512d M6=_mm512_set1_pd(-1.0/6.0);
    const __m512d C120=_mm512_set1_pd(1.0/120.0);
    const __m512d N5040=_mm512_set1_pd(-1.0/5040.0);
    __m512d z=_mm512_mul_pd(rh,rh);
    __m512d p=_mm512_fmadd_pd(z,N5040,C120);
    p=_mm512_fmadd_pd(z,p,M6);
    __m512d y=_mm512_fmadd_pd(_mm512_mul_pd(rh,z),p,rh);
    y=_mm512_add_pd(y,rl);
    return _mm512_mask_sub_pd(y,signmask,Z,y);
}

'''
marker='OVEC static inline void x12_prepare_block(const double * __restrict x,size_t base,size_t n,'
if s.count(marker) != 1:
    raise SystemExit('x12 marker count')
s=s.replace(marker,helper+marker,1)

# Change only the dedicated full-width x12 path.  Its existing near-zero compare
# is already in the frozen hot path.  nested/broad reuse that compare as the
# outer gate; flat preserves it and adds a branchless narrow accuracy mask.
pos=s.index(marker)
pre, tail=s[:pos], s[pos:]
old = '''        __m512d ars=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs),ABSM));
        __mmask8 repair=_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-14),_CMP_LT_OQ);
        if(__builtin_expect(repair!=0,0)){
            const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
            const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
            const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
            __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(N,PI1));
            __m512d rh3,re3; twodiff_cw(r0,_mm512_mul_pd(N,PI2),&rh3,&re3);
            __m512d rl3=_mm512_fnmadd_pd(N,PI3,re3);
            rh=_mm512_mask_mov_pd(rh,repair,rh3);
            rl=_mm512_mask_mov_pd(rl,repair,rl3);
            rs=_mm512_add_pd(rh,rl);
        }
'''
if tail.count(old) != 1:
    raise SystemExit('full-width repair block count')
if mode == 'flat_band':
    new = '''        __m512d ars=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs),ABSM));
        __mmask8 repair=_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-14),_CMP_LT_OQ);
        __mmask8 accguard=(__mmask8)(_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1.fc00000000000p-7),_CMP_LT_OQ)&
                                    _mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1.cac083126e979p-7),_CMP_GT_OQ));
        if(__builtin_expect(repair!=0,0)){
            const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
            const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
            const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
            __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(N,PI1));
            __m512d rh3,re3; twodiff_cw(r0,_mm512_mul_pd(N,PI2),&rh3,&re3);
            __m512d rl3=_mm512_fnmadd_pd(N,PI3,re3);
            rh=_mm512_mask_mov_pd(rh,repair,rh3);
            rl=_mm512_mask_mov_pd(rl,repair,rl3);
            rs=_mm512_add_pd(rh,rl);
        }
'''
elif mode == 'nested_band':
    new = '''        __m512d ars=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs),ABSM));
        __mmask8 near=_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1.fc00000000000p-7),_CMP_LT_OQ);
        __mmask8 repair=0,accguard=0;
        if(__builtin_expect(near!=0,0)){
            repair=(__mmask8)(near&_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-14),_CMP_LT_OQ));
            accguard=(__mmask8)(near&_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1.cac083126e979p-7),_CMP_GT_OQ));
            if(repair){
                const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
                const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
                const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
                __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(N,PI1));
                __m512d rh3,re3; twodiff_cw(r0,_mm512_mul_pd(N,PI2),&rh3,&re3);
                __m512d rl3=_mm512_fnmadd_pd(N,PI3,re3);
                rh=_mm512_mask_mov_pd(rh,repair,rh3);
                rl=_mm512_mask_mov_pd(rl,repair,rl3);
                rs=_mm512_add_pd(rh,rl);
            }
        }
'''
else:
    new = '''        __m512d ars=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs),ABSM));
        __mmask8 near=_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1.fc00000000000p-7),_CMP_LT_OQ);
        __mmask8 repair=0,accguard=near;
        if(__builtin_expect(near!=0,0)){
            repair=(__mmask8)(near&_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-14),_CMP_LT_OQ));
            if(repair){
                const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
                const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
                const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
                __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(N,PI1));
                __m512d rh3,re3; twodiff_cw(r0,_mm512_mul_pd(N,PI2),&rh3,&re3);
                __m512d rl3=_mm512_fnmadd_pd(N,PI3,re3);
                rh=_mm512_mask_mov_pd(rh,repair,rh3);
                rl=_mm512_mask_mov_pd(rl,repair,rl3);
                rs=_mm512_add_pd(rh,rl);
            }
        }
'''
tail=tail.replace(old,new,1)
oldout='''        *rh_out=rh; *rl_out=rl;
        *sign_out=(__mmask8)(inneg^parity^rneg);
        *guard_out=0; *active_out=0xff; *pure_unit_out=0; return;'''
newout='''        *rh_out=rh; *rl_out=rl;
        *sign_out=(__mmask8)(inneg^parity^rneg);
        *guard_out=accguard; *active_out=0xff; *pure_unit_out=0; return;'''
if tail.count(oldout)!=1:
    raise SystemExit('guard output count')
tail=tail.replace(oldout,newout,1)
s=pre+tail

# Replace the already-existing cold hooks with the direct small-angle repair.
old_tail='''        /* Phase 3: rare exact v8 scalar repair. */
        for(size_t b=0;b<blocks;b++)if(__builtin_expect(guardbuf[b]!=0,0)){
            __mmask8 g=(__mmask8)guardbuf[b];
            for(unsigned lane=0;lane<8&&b*8+lane<tn;lane++)
                if(g&(1u<<lane))out[tile+b*8+lane]=scalar2(k,x[tile+b*8+lane]);
        }'''
new_tail='''        /* Phase 3: rare direct small-angle repair. */
        for(size_t b=0;b<blocks;b++)if(__builtin_expect(guardbuf[b]!=0,0)){
            __mmask8 g=(__mmask8)guardbuf[b];
            __m512d rh=_mm512_load_pd(rhbuf+b*8),rl=_mm512_load_pd(rlbuf+b*8);
            __m512d alt=x67_rare_small_repair(rh,rl,(__mmask8)signbuf[b]);
            _mm512_mask_storeu_pd(out+tile+b*8,g,alt);
        }'''
if s.count(old_tail)!=1:
    raise SystemExit('tail cold hook count')
s=s.replace(old_tail,new_tail,1)

old_raw='''        if(__builtin_expect(g0!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g0&(1u<<lane)) out[i+0+lane]=scalar2(k,x[i+0+lane]);
        if(__builtin_expect(g1!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g1&(1u<<lane)) out[i+8+lane]=scalar2(k,x[i+8+lane]);
        if(__builtin_expect(g2!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g2&(1u<<lane)) out[i+16+lane]=scalar2(k,x[i+16+lane]);
        if(__builtin_expect(g3!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g3&(1u<<lane)) out[i+24+lane]=scalar2(k,x[i+24+lane]);'''
new_raw='''        if(__builtin_expect(g0!=0,0)){__m512d alt=x67_rare_small_repair(rh0,rl0,s0);_mm512_mask_storeu_pd(out+i+0,g0,alt);}
        if(__builtin_expect(g1!=0,0)){__m512d alt=x67_rare_small_repair(rh1,rl1,s1);_mm512_mask_storeu_pd(out+i+8,g1,alt);}
        if(__builtin_expect(g2!=0,0)){__m512d alt=x67_rare_small_repair(rh2,rl2,s2);_mm512_mask_storeu_pd(out+i+16,g2,alt);}
        if(__builtin_expect(g3!=0,0)){__m512d alt=x67_rare_small_repair(rh3,rl3,s3);_mm512_mask_storeu_pd(out+i+24,g3,alt);}'''
if s.count(old_raw)!=1:
    raise SystemExit('raw cold hook count')
s=s.replace(old_raw,new_raw,1)

# Disable the benchmark main so the common adapter can own the executable.
q='\nint main(void)\n{'
if s.count(q)!=1:
    raise SystemExit('main count')
s=s.replace(q,'\nint sine53_production_disabled_main(void)\n{',1)
Path(dst).write_text(s)
