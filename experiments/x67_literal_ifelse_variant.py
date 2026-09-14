#!/usr/bin/env python3
from pathlib import Path
import sys

# Literal scalar if/else experiment trigger.
if len(sys.argv)!=3:
    raise SystemExit('usage: x67_literal_ifelse_variant.py INPUT OUTPUT')
src,out=sys.argv[1:]
s=Path(src).read_text()

old='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
if s.count(old)!=1:
    raise SystemExit(f'raw X67 store block count={s.count(old)}')

new='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n\n            /* Literal scalar if/else experiment. No vector mask detector.\n               j and d are the values already computed by X67 above. */\n            int32_t jlane[8] __attribute__((aligned(32)));\n            double dlane[8] __attribute__((aligned(64)));\n            double normal[8] __attribute__((aligned(64)));\n            _mm256_store_si256((__m256i*)jlane,ji[g]);\n            _mm512_store_pd(dlane,d[g]);\n            _mm512_store_pd(normal,pv[g]);\n            for(unsigned lane=0;lane<8;lane++){\n                int jj=jlane[lane];\n                double dd=dlane[lane];\n                if((jj==2 && dd>=0x1.ff2e48e8a71dep-10 && dd<=0x1.8fc504816f007p-9) ||\n                   (jj==510 && dd>=-0x1.8fc504816f007p-9 && dd<=-0x1.ff2e48e8a71dep-10)){\n                    out[i+8*g+lane]=scalar2(k,x[i+8*g+lane]);\n                }else{\n                    out[i+8*g+lane]=normal[lane];\n                }\n            }'''

Path(out).write_text(s.replace(old,new,1))
