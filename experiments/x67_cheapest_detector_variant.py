#!/usr/bin/env python3
from pathlib import Path
import re
import sys

if len(sys.argv) != 4:
    raise SystemExit('usage: x67_cheapest_detector_variant.py INPUT OUTPUT MODE')
src, out, mode = sys.argv[1:]
s = Path(src).read_text()

CELL = '0x1.921d1fcdec784p-7'
m = re.search(r'static const double x65_s\[512\].*?=\{(.*?)\};', s, re.S)
if not m:
    raise SystemExit('x65_s table not found')
vals = [v.strip() for v in m.group(1).split(',')]
pos = sum(v == CELL for v in vals)
neg = sum(v == '-' + CELL for v in vals)
if pos != 2 or neg != 2:
    raise SystemExit(f'expected two +CELL and two -CELL anchors in x65_s, got +{pos} -{neg}')

old = '''            __m512d z=_mm512_mul_pd(d[g],d[g]);
            __m512d ec=_mm512_fmadd_pd(z,C24,MH);      /* -1/2 + z/24 */
            __m512d oc=_mm512_fmadd_pd(z,C120,M6);     /* -1/6 + z/120 */
            __m512d cd=_mm512_mul_pd(c1[g],d[g]);
            /* Same polynomial, but fuse the potentially cancelling leading
               terms first: base = c0 + c1*d.  Corrections are O(z). */
            __m512d base=_mm512_fmadd_pd(c1[g],d[g],c0[g]);
            __m512d ep=_mm512_mul_pd(c0[g],ec);
            __m512d inner=_mm512_fmadd_pd(cd,oc,ep);
            pv[g]=_mm512_fmadd_pd(z,inner,base);
            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);
            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
if s.count(old) != 1:
    raise SystemExit(f'raw X67 polynomial block count={s.count(old)}')

if mode == 'c0abs':
    detector = '''            const __m512d CELL2=_mm512_set1_pd(0x1.921d1fcdec784p-7);
            __mmask8 mj=_mm512_cmp_pd_mask(c0[g],CELL2,_CMP_EQ_OQ);'''
elif mode == 'bitmagic':
    detector = '''            /* j in [0,511]: ((j+2)&0x1fb)==0 iff j is 2 or 510. */
            __m256i jt=_mm256_add_epi32(ji[g],_mm256_set1_epi32(2));
            jt=_mm256_and_si256(jt,_mm256_set1_epi32(0x1fb));
            __m256i jeq=_mm256_cmpeq_epi32(jt,_mm256_setzero_si256());
            __mmask8 mj=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(jeq));'''
else:
    raise SystemExit('mode must be c0abs or bitmagic')

new = '''            __m512d z=_mm512_mul_pd(d[g],d[g]);
            __m512d ec=_mm512_fmadd_pd(z,C24,MH);      /* -1/2 + z/24 */
            __m512d oc=_mm512_fmadd_pd(z,C120,M6);     /* -1/6 + z/120 */
            __m512d cd=_mm512_mul_pd(c1[g],d[g]);
            /* Same polynomial, but fuse the potentially cancelling leading
               terms first: base = c0 + c1*d.  Corrections are O(z). */
            __m512d base=_mm512_fmadd_pd(c1[g],d[g],c0[g]);
            __m512d ep=_mm512_mul_pd(c0[g],ec);
            __m512d inner=_mm512_fmadd_pd(cd,oc,ep);
            pv[g]=_mm512_fmadd_pd(z,inner,base);
''' + detector + '''
            if(__builtin_expect(mj!=0,0)){
                const __m512d BLO=_mm512_set1_pd(0x1.ff2e48e8a71dep-10); /* 0.00195 */
                const __m512d BHI=_mm512_set1_pd(0x1.8fc504816f007p-9);  /* 0.00305 */
                const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
                __m512d ad=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(d[g]),ABSM));
                __mmask8 mb=(__mmask8)(_mm512_cmp_pd_mask(ad,BLO,_CMP_GE_OQ) &
                                       _mm512_cmp_pd_mask(ad,BHI,_CMP_LE_OQ));
                /* Correct half-cell only: j=2 needs d>0, j=510 needs d<0.
                   Since c1 has opposite signs in the mirrored cells, both are cd>0. */
                __mmask8 morient=_mm512_cmp_pd_mask(cd,Z,_CMP_GT_OQ);
                __mmask8 x67_fix=(__mmask8)(mj & mb & morient);
                if(__builtin_expect(x67_fix!=0,0)){
                    const __m512d DH=_mm512_set1_pd(0x1.921fb54442d18p-7);
                    const __m512d DL=_mm512_set1_pd(0x1.1a62633145c07p-61);
                    const __m512d N5040=_mm512_set1_pd(-1.0/5040.0);
                    __m512d b=ad;
                    __m512d rh=_mm512_add_pd(DH,b);
                    __m512d re=_mm512_sub_pd(b,_mm512_sub_pd(rh,DH));
                    __m512d rl=_mm512_add_pd(re,DL);
                    __m512d zz=_mm512_mul_pd(rh,rh);
                    __m512d pp=_mm512_fmadd_pd(zz,N5040,C120);
                    pp=_mm512_fmadd_pd(zz,pp,M6);
                    __m512d alt=_mm512_fmadd_pd(_mm512_mul_pd(rh,zz),pp,rh);
                    alt=_mm512_add_pd(alt,rl);
                    pv[g]=_mm512_mask_mov_pd(pv[g],x67_fix,alt);
                }
            }
            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);
            _mm512_storeu_pd(out+i+8*g,pv[g]);'''

Path(out).write_text(s.replace(old, new, 1))
