#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 4:
    raise SystemExit('usage: x67_awayzero_variant.py INPUT OUTPUT THRESHOLD')

src, out, thr = map(str, sys.argv[1:])
s = Path(src).read_text()

needle = '''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''

if thr == '0':
    dpos = '_mm512_cmp_pd_mask(d[g],Z,_CMP_GT_OQ)'
    dneg = '_mm512_cmp_pd_mask(d[g],Z,_CMP_LT_OQ)'
else:
    dpos = f'_mm512_cmp_pd_mask(d[g],_mm512_set1_pd({thr}),_CMP_GT_OQ)'
    dneg = f'_mm512_cmp_pd_mask(d[g],_mm512_set1_pd(-({thr})),_CMP_LT_OQ)'

repl = f'''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            /* Accuracy experiment: the only frozen-X67 2-ULP failures are\n               j=2 with positive d or j=510 with negative d, and every one\n               is toward zero.  Move selected lanes exactly one binary64\n               representable step away from zero by incrementing magnitude. */\n            __mmask8 jm2=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(\n                _mm256_cmpeq_epi32(ji[g],_mm256_set1_epi32(2))));\n            __mmask8 jm510=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(\n                _mm256_cmpeq_epi32(ji[g],_mm256_set1_epi32(510))));\n            __mmask8 mdp={dpos};\n            __mmask8 mdn={dneg};\n            __mmask8 mfix=(__mmask8)((jm2&mdp)|(jm510&mdn));\n            __m512i pbits=_mm512_castpd_si512(pv[g]);\n            pbits=_mm512_mask_add_epi64(pbits,mfix,pbits,_mm512_set1_epi64(1));\n            pv[g]=_mm512_castsi512_pd(pbits);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''

if s.count(needle) != 1:
    raise SystemExit(f'rawx67 polynomial/store marker count={s.count(needle)}')

Path(out).write_text(s.replace(needle, repl, 1))
