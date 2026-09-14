#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv)!=3:
    raise SystemExit('usage: x67_simple_band_variant.py INPUT OUTPUT')
src,out=sys.argv[1:]
s=Path(src).read_text()

old='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
if s.count(old)!=1:
    raise SystemExit(f'raw X67 store block count={s.count(old)}')

new='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);\n\n            /* Exact nested logic: j and d already exist.  Pay the d-band\n               comparisons only when this vector actually contains j=2/510. */\n            const __m256i J2=_mm256_set1_epi32(2), J510=_mm256_set1_epi32(510);\n            __mmask8 mj2=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(ji[g],J2)));\n            __mmask8 mj510=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(ji[g],J510)));\n            __mmask8 mj=(__mmask8)(mj2|mj510);\n            if(__builtin_expect(mj!=0,0)){\n                const __m512d DLO=_mm512_set1_pd(0x1.ff2e48e8a71dep-10); /* 0.00195 */\n                const __m512d DHI=_mm512_set1_pd(0x1.8fc504816f007p-9);  /* 0.00305 */\n                __mmask8 mdpos=(__mmask8)(_mm512_cmp_pd_mask(d[g],DLO,_CMP_GE_OQ) & _mm512_cmp_pd_mask(d[g],DHI,_CMP_LE_OQ));\n                __m512d NDLO=_mm512_sub_pd(Z,DLO), NDHI=_mm512_sub_pd(Z,DHI);\n                __mmask8 mdneg=(__mmask8)(_mm512_cmp_pd_mask(d[g],NDHI,_CMP_GE_OQ) & _mm512_cmp_pd_mask(d[g],NDLO,_CMP_LE_OQ));\n                __mmask8 x67_fix=(__mmask8)((mj2 & mdpos) | (mj510 & mdneg));\n                if(__builtin_expect(x67_fix!=0,0)){\n                    for(unsigned lane=0;lane<8;lane++)\n                        if(x67_fix&(1u<<lane)) out[i+8*g+lane]=scalar2(k,x[i+8*g+lane]);\n                }\n            }'''

Path(out).write_text(s.replace(old,new,1))
