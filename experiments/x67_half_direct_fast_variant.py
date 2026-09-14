#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv)!=4:
    raise SystemExit('usage: x67_half_direct_fast_variant.py INPUT OUTPUT MODE')
src,out,mode=sys.argv[1:]
if mode not in ('sign_direct','sign_dd','cmp_direct','cmp_dd','sign_dd_threshold'):
    raise SystemExit('bad mode')
s=Path(src).read_text()
old='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
if s.count(old)!=1:
    raise SystemExit(f'poly block count={s.count(old)}')

# c0 identifies the mirror anchor pair j=2/510.  cd=c1*d is already computed
# by the frozen polynomial.  The signed hostile replay showed the bad half is
# exactly c1*d > 0, so no extra multiply is required.
gate='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            __mmask8 x67_cell=_mm512_cmp_pd_mask(c0[g],_mm512_set1_pd(0x1.921d1fcdec784p-7),_CMP_EQ_OQ);'''
if mode.startswith('sign_'):
    gate+='''\n            __mmask8 x67_cdsign=(__mmask8)_mm512_movepi64_mask(_mm512_castpd_si512(cd));\n            __mmask8 x67_fix=(__mmask8)(x67_cell & (unsigned char)~x67_cdsign);'''
else:
    gate+='''\n            __mmask8 x67_fix=(__mmask8)(x67_cell & _mm512_cmp_pd_mask(cd,Z,_CMP_GT_OQ));'''
if mode=='sign_dd_threshold':
    gate+='''\n            x67_fix=(__mmask8)(x67_fix & _mm512_cmp_pd_mask(pv[g],_mm512_set1_pd(0x1.d14e3bcd35a86p-7),_CMP_GE_OQ));'''
gate+='''\n            if(__builtin_expect(x67_fix!=0,0)){'''

use_dd = mode in ('sign_dd','cmp_dd','sign_dd_threshold')
body=[
'const __m512d DH=_mm512_set1_pd(0x1.921fb54442d18p-7);',
'const __m512d N5040=_mm512_set1_pd(-1.0/5040.0);',
'__mmask8 m510=(__mmask8)(_mm512_movepi64_mask(_mm512_castpd_si512(c1[g]))&x67_fix);',
'__m512d b=_mm512_mask_sub_pd(d[g],m510,Z,d[g]);']
if use_dd:
    body += [
    'const __m512d DL=_mm512_set1_pd(0x1.1a62633145c07p-61);',
    '__m512d rh=_mm512_add_pd(DH,b);',
    '__m512d re=_mm512_sub_pd(b,_mm512_sub_pd(rh,DH));',
    '__m512d rl=_mm512_add_pd(re,DL);',
    '__m512d zz=_mm512_mul_pd(rh,rh);',
    '__m512d pp=_mm512_fmadd_pd(zz,N5040,C120);',
    'pp=_mm512_fmadd_pd(zz,pp,M6);',
    '__m512d alt=_mm512_fmadd_pd(_mm512_mul_pd(rh,zz),pp,rh);',
    'alt=_mm512_add_pd(alt,rl);']
else:
    body += [
    '__m512d rr=_mm512_add_pd(DH,b);',
    '__m512d zz=_mm512_mul_pd(rr,rr);',
    '__m512d pp=_mm512_fmadd_pd(zz,N5040,C120);',
    'pp=_mm512_fmadd_pd(zz,pp,M6);',
    '__m512d alt=_mm512_fmadd_pd(_mm512_mul_pd(rr,zz),pp,rr);']
body += ['pv[g]=_mm512_mask_mov_pd(pv[g],x67_fix,alt);']
new=gate+'\n'+'\n'.join('                '+x for x in body)+'''\n            }\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
Path(out).write_text(s.replace(old,new,1))
