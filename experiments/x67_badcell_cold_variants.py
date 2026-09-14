#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 4:
    raise SystemExit('usage: x67_badcell_cold_variants.py INPUT OUTPUT MODE')
src,out,mode=sys.argv[1:]
if mode not in ('exact','abs2'):
    raise SystemExit('MODE must be exact or abs2')
s=Path(src).read_text()

fnneedle='''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawx67('''
helper=r'''OVEC __attribute__((noinline,cold)) static void x67_badcell_direct_cold(
        double *out,__m512d dv,__mmask8 mfix)
{
    _Alignas(64) double ds[8];
    _mm512_store_pd(ds,dv);
    const double DHI=0x1.921fb54442d18p-7;
    for(unsigned lane=0;lane<8;lane++) if(mfix&(1u<<lane)){
        double rr=DHI+fabs(ds[lane]);
        double z=rr*rr;
        double poly=fma(z,-1.0/5040.0,1.0/120.0);
        poly=fma(z,poly,-1.0/6.0);
        double q=fma(rr*z,poly,rr);
        if(signbit(out[lane])) q=-q;
        out[lane]=q;
    }
}

'''
if s.count(fnneedle)!=1:
    raise SystemExit(f'rawx67 function marker count={s.count(fnneedle)}')
s=s.replace(fnneedle,helper+fnneedle,1)

needle='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
if mode=='exact':
    det=r'''            __mmask8 jm2=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(
                _mm256_cmpeq_epi32(ji[g],_mm256_set1_epi32(2))));
            __mmask8 jm510=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(
                _mm256_cmpeq_epi32(ji[g],_mm256_set1_epi32(510))));
            __mmask8 mdp=_mm512_cmp_pd_mask(d[g],Z,_CMP_GT_OQ);
            __mmask8 mdn=_mm512_cmp_pd_mask(d[g],Z,_CMP_LT_OQ);
            __mmask8 mfix=(__mmask8)((jm2&mdp)|(jm510&mdn));'''
else:
    det=r'''            /* ji is modulo 512. Sign-extend its low 9 bits so
               510 becomes -2, then detect |signed_ji|==2.  The bad half-cell
               is exactly where sign(d)==sign(signed_ji). */
            __m256i sji=_mm256_srai_epi32(_mm256_slli_epi32(ji[g],23),23);
            __m256i aji=_mm256_abs_epi32(sji);
            __mmask8 mcell=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(
                _mm256_cmpeq_epi32(aji,_mm256_set1_epi32(2))));
            __mmask8 jsign=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(sji));
            __mmask8 dsign=(__mmask8)_mm512_movepi64_mask(_mm512_castpd_si512(d[g]));
            __mmask8 mfix=(__mmask8)(mcell & (unsigned char)~(jsign^dsign));'''
repl=f'''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);\n{det}\n            if(__builtin_expect(mfix!=0,0)) x67_badcell_direct_cold(out+i+8*g,d[g],mfix);'''
if s.count(needle)!=1:
    raise SystemExit(f'rawx67 polynomial/store marker count={s.count(needle)}')
Path(out).write_text(s.replace(needle,repl,1))
