#!/usr/bin/env python3
from pathlib import Path
import sys
if len(sys.argv)!=4: raise SystemExit('usage: x67_vector_cost_variant.py INPUT OUTPUT MODE')
src,out,mode=sys.argv[1:]
s=Path(src).read_text()
old='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
if s.count(old)!=1: raise SystemExit(f'block count={s.count(old)}')
head='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            const __m256i J2=_mm256_set1_epi32(2);\n            const __m256i J510=_mm256_set1_epi32(510);\n            __mmask8 mj2=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(ji[g],J2)));\n            __mmask8 mj510=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(ji[g],J510)));\n            __mmask8 mj=(__mmask8)(mj2|mj510);'''
band='''\n            if(__builtin_expect(mj!=0,0)){\n                const __m512d BLO=_mm512_set1_pd(0x1.ff2e48e8a71dep-10);\n                const __m512d BHI=_mm512_set1_pd(0x1.8fc504816f007p-9);\n                const __m512d NBLO=_mm512_set1_pd(-0x1.ff2e48e8a71dep-10);\n                const __m512d NBHI=_mm512_set1_pd(-0x1.8fc504816f007p-9);\n                __mmask8 mdpos=(__mmask8)(_mm512_cmp_pd_mask(d[g],BLO,_CMP_GE_OQ)&_mm512_cmp_pd_mask(d[g],BHI,_CMP_LE_OQ));\n                __mmask8 mdneg=(__mmask8)(_mm512_cmp_pd_mask(d[g],NBHI,_CMP_GE_OQ)&_mm512_cmp_pd_mask(d[g],NBLO,_CMP_LE_OQ));\n                __mmask8 x67_fix=(__mmask8)((mj2&mdpos)|(mj510&mdneg));'''
full='''\n                if(__builtin_expect(x67_fix!=0,0)){\n                    const __m512d DH=_mm512_set1_pd(0x1.921fb54442d18p-7);\n                    const __m512d DL=_mm512_set1_pd(0x1.1a62633145c07p-61);\n                    const __m512d N5040=_mm512_set1_pd(-1.0/5040.0);\n                    __mmask8 m510=(__mmask8)(mj510 & x67_fix);\n                    __m512d b=_mm512_mask_sub_pd(d[g],m510,Z,d[g]);\n                    __m512d rh=_mm512_add_pd(DH,b);\n                    __m512d re=_mm512_sub_pd(b,_mm512_sub_pd(rh,DH));\n                    __m512d rl=_mm512_add_pd(re,DL);\n                    __m512d zz=_mm512_mul_pd(rh,rh);\n                    __m512d pp=_mm512_fmadd_pd(zz,N5040,C120);\n                    pp=_mm512_fmadd_pd(zz,pp,M6);\n                    __m512d alt=_mm512_fmadd_pd(_mm512_mul_pd(rh,zz),pp,rh);\n                    alt=_mm512_add_pd(alt,rl);\n                    pv[g]=_mm512_mask_mov_pd(pv[g],x67_fix,alt);\n                }\n            }'''
tail='''\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
if mode=='gate':
    body=head+'''\n            if(__builtin_expect(mj!=0,0)){ __asm__ __volatile__("" : : "r"((unsigned)mj)); }'''+tail
elif mode=='band':
    body=head+band+'''\n                if(__builtin_expect(x67_fix!=0,0)){ __asm__ __volatile__("" : : "r"((unsigned)x67_fix)); }\n            }'''+tail
elif mode=='full':
    body=head+band+full+tail
else: raise SystemExit('bad mode')
Path(out).write_text(s.replace(old,body,1))
