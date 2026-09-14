#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv)!=5:
    raise SystemExit('usage: x67_halfcell_coeff_variant.py INPUT OUTPUT C0_STEPS C1_MAG_STEPS')
src,out,a,b=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
s=Path(src).read_text()
needle='''            d[g]=_mm512_fnmadd_pd(N[g],HHI,vx[g]);\n            d[g]=_mm512_fnmadd_pd(N[g],HLO,d[g]);'''
if s.count(needle)!=1:
    raise SystemExit(f'reduction marker count={s.count(needle)}')
# x65_s[2] == x65_s[510].  The signed-error scan localized every frozen
# >1-ULP case to exactly one half of these two symmetric cells:
# c0==sin(pi/256) and sign(d)==sign(c1).
lines=[needle,
'''            __mmask8 x67_edge=_mm512_cmp_pd_mask(c0[g],_mm512_set1_pd(0x1.921d1fcdec784p-7),_CMP_EQ_OQ);''',
'''            __mmask8 x67_csgn=(__mmask8)_mm512_movepi64_mask(_mm512_castpd_si512(c1[g]));''',
'''            __mmask8 x67_dsgn=(__mmask8)_mm512_movepi64_mask(_mm512_castpd_si512(d[g]));''',
'''            __mmask8 x67_half=(__mmask8)(x67_edge & (unsigned char)~(x67_csgn^x67_dsgn));''']
if a:
    lines += [f'''            __m512i x67_c0b=_mm512_castpd_si512(c0[g]);''',
              f'''            x67_c0b=_mm512_mask_add_epi64(x67_c0b,x67_half,x67_c0b,_mm512_set1_epi64({a}));''',
              '''            c0[g]=_mm512_castsi512_pd(x67_c0b);''']
if b:
    # Adding to the magnitude bits moves +c1 upward and -c1 downward, preserving
    # the odd cosine symmetry between j=2 and j=510.
    lines += ['''            __m512i x67_c1b=_mm512_castpd_si512(c1[g]);''',
              f'''            x67_c1b=_mm512_mask_add_epi64(x67_c1b,x67_half,x67_c1b,_mm512_set1_epi64({b}));''',
              '''            c1[g]=_mm512_castsi512_pd(x67_c1b);''']
Path(out).write_text(s.replace(needle,'\n'.join(lines),1))
