#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 3:
    raise SystemExit('usage: x67_badcell_direct_variant.py INPUT OUTPUT')

src, out = sys.argv[1], sys.argv[2]
s = Path(src).read_text()
needle = '''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
repl = '''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n\n            /* Exact localization from the hostile scan: all frozen-X67 >1-ULP\n               cases lie in j=2,d>0 or j=510,d<0.  Re-evaluate only those\n               rare lanes as a tiny-angle sine around pi/256 rather than\n               applying a blind ULP bias. */\n            __mmask8 jm2=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(\n                _mm256_cmpeq_epi32(ji[g],_mm256_set1_epi32(2))));\n            __mmask8 jm510=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(\n                _mm256_cmpeq_epi32(ji[g],_mm256_set1_epi32(510))));\n            __mmask8 mdp=_mm512_cmp_pd_mask(d[g],Z,_CMP_GT_OQ);\n            __mmask8 mdn=_mm512_cmp_pd_mask(d[g],Z,_CMP_LT_OQ);\n            __mmask8 mfix=(__mmask8)((jm2&mdp)|(jm510&mdn));\n            if(__builtin_expect(mfix!=0,0)){\n                const __m512d DHI=_mm512_set1_pd(0x1.921fb54442d18p-7);\n                const __m512d A5040=_mm512_set1_pd(-1.0/5040.0);\n                const __m512d A120=_mm512_set1_pd(1.0/120.0);\n                __m512d ad=_mm512_abs_pd(d[g]);\n                __m512d rr=_mm512_add_pd(DHI,ad);\n                __m512d rz=_mm512_mul_pd(rr,rr);\n                __m512d rp=_mm512_fmadd_pd(rz,A5040,A120);\n                rp=_mm512_fmadd_pd(rz,rp,M6);\n                __m512d direct=_mm512_fmadd_pd(_mm512_mul_pd(rr,rz),rp,rr);\n                __mmask8 pneg=(__mmask8)_mm512_movepi64_mask(_mm512_castpd_si512(pv[g]));\n                direct=_mm512_mask_sub_pd(direct,pneg,Z,direct);\n                pv[g]=_mm512_mask_mov_pd(pv[g],mfix,direct);\n            }\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
if s.count(needle) != 1:
    raise SystemExit(f'rawx67 polynomial/store marker count={s.count(needle)}')
Path(out).write_text(s.replace(needle, repl, 1))
