from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit('usage: sine53_zero_guard_transform.py source.c')
p = Path(sys.argv[1])
s = p.read_text()
old = '''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);\n'''
new = '''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);\n\n            /* Experimental accuracy guard: only table anchors within two\n               slots of a sine zero are eligible for scalar/DD recompute.\n               Inputs |x|<1 remain on the original raw-X67 result. */\n            __m256i rlo=_mm256_cmpgt_epi32(_mm256_set1_epi32(3),ji[g]);\n            __m256i rhi=_mm256_cmpgt_epi32(ji[g],_mm256_set1_epi32(509));\n            __m256i rv=_mm256_or_si256(rlo,rhi);\n            __mmask8 risk=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(rv));\n            if(__builtin_expect(risk!=0,0)){\n                for(unsigned lane=0;lane<8;lane++){\n                    size_t q=i+8*(size_t)g+lane;\n                    if((risk&(1u<<lane)) && fabs(x[q])>=1.0)\n                        out[q]=scalar2(k,x[q]);\n                }\n            }\n'''
count=s.count(old)
if count != 1:
    raise SystemExit(f'expected exactly one raw-X67 polynomial store block, found {count}')
s=s.replace(old,new,1)
p.write_text(s)
