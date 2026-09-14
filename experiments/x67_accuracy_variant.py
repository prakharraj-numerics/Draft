from pathlib import Path
import math, re, sys

if len(sys.argv) != 4:
    raise SystemExit('usage: x67_accuracy_variant.py src.c dst.c mode')
src,dst,mode = sys.argv[1:]
s = Path(src).read_text()

# Helpers to adjust one binary64 value by integer ULPs.
def step_hex(tok, steps):
    x=float.fromhex(tok)
    if steps>0:
        for _ in range(steps): x=math.nextafter(x, math.inf)
    elif steps<0:
        for _ in range(-steps): x=math.nextafter(x, -math.inf)
    return x.hex()

def edit_array(text, name, idx_steps):
    pat = rf'(static const double {name}\[512\][^=]*=\{{)(.*?)(\}};)'
    m=re.search(pat,text,re.S)
    if not m: raise SystemExit(f'array {name} not found')
    vals=[v.strip() for v in m.group(2).split(',')]
    if len(vals)!=512: raise SystemExit(f'{name} len={len(vals)}')
    for idx,st in idx_steps.items(): vals[idx]=step_hex(vals[idx],st)
    body=','.join(vals)
    return text[:m.start()] + m.group(1)+body+m.group(3) + text[m.end():]

def inject_direct_tail(text, threshold_hex, dd):
    old='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
    if dd:
        corr='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            __m256i x67_e2=_mm256_cmpeq_epi32(ji[g],_mm256_set1_epi32(2));\n            __m256i x67_e510=_mm256_cmpeq_epi32(ji[g],_mm256_set1_epi32(510));\n            __mmask8 x67_m2=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(x67_e2));\n            __mmask8 x67_m510=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(x67_e510));\n            __mmask8 x67_fix=(__mmask8)((x67_m2|x67_m510)&_mm512_cmp_pd_mask(pv[g],_mm512_set1_pd('''+threshold_hex+'''),_CMP_GE_OQ));\n            if(__builtin_expect(x67_fix!=0,0)){\n                const __m512d X67_DHI=_mm512_set1_pd(0x1.921fb54442d18p-7);\n                const __m512d X67_DLO=_mm512_set1_pd(0x1.1a62633145c07p-61);\n                const __m512d X67_N5040=_mm512_set1_pd(-1.0/5040.0);\n                __m512d x67_b=_mm512_mask_sub_pd(d[g],x67_m510,Z,d[g]);\n                __m512d x67_rh=_mm512_add_pd(X67_DHI,x67_b);\n                __m512d x67_re=_mm512_sub_pd(x67_b,_mm512_sub_pd(x67_rh,X67_DHI));\n                __m512d x67_rl=_mm512_add_pd(x67_re,X67_DLO);\n                __m512d x67_z=_mm512_mul_pd(x67_rh,x67_rh);\n                __m512d x67_poly=_mm512_fmadd_pd(x67_z,X67_N5040,C120);\n                x67_poly=_mm512_fmadd_pd(x67_z,x67_poly,M6);\n                __m512d x67_alt=_mm512_fmadd_pd(_mm512_mul_pd(x67_rh,x67_z),x67_poly,x67_rh);\n                x67_alt=_mm512_add_pd(x67_alt,x67_rl);\n                pv[g]=_mm512_mask_mov_pd(pv[g],x67_fix,x67_alt);\n            }\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
    else:
        corr='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            __m256i x67_e2=_mm256_cmpeq_epi32(ji[g],_mm256_set1_epi32(2));\n            __m256i x67_e510=_mm256_cmpeq_epi32(ji[g],_mm256_set1_epi32(510));\n            __mmask8 x67_m2=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(x67_e2));\n            __mmask8 x67_m510=(__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(x67_e510));\n            __mmask8 x67_fix=(__mmask8)((x67_m2|x67_m510)&_mm512_cmp_pd_mask(pv[g],_mm512_set1_pd('''+threshold_hex+'''),_CMP_GE_OQ));\n            if(__builtin_expect(x67_fix!=0,0)){\n                const __m512d X67_DHI=_mm512_set1_pd(0x1.921fb54442d18p-7);\n                const __m512d X67_DLO=_mm512_set1_pd(0x1.1a62633145c07p-61);\n                const __m512d X67_N5040=_mm512_set1_pd(-1.0/5040.0);\n                __m512d x67_b=_mm512_mask_sub_pd(d[g],x67_m510,Z,d[g]);\n                __m512d x67_r=_mm512_add_pd(_mm512_add_pd(X67_DHI,x67_b),X67_DLO);\n                __m512d x67_z=_mm512_mul_pd(x67_r,x67_r);\n                __m512d x67_poly=_mm512_fmadd_pd(x67_z,X67_N5040,C120);\n                x67_poly=_mm512_fmadd_pd(x67_z,x67_poly,M6);\n                __m512d x67_alt=_mm512_fmadd_pd(_mm512_mul_pd(x67_r,x67_z),x67_poly,x67_r);\n                pv[g]=_mm512_mask_mov_pd(pv[g],x67_fix,x67_alt);\n            }\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
    if text.count(old)!=1: raise SystemExit(f'direct-tail polynomial block count={text.count(old)}')
    return text.replace(old,corr,1)

def finish_source(text):
    prod_main='\nint main(void)\n{'
    if text.count(prod_main)!=1:
        raise SystemExit(f'production main count={text.count(prod_main)}')
    text=text.replace(prod_main,'\nint sine53_production_disabled_main(void)\n{',1)
    text += '\nint main(void){return 0;}\n'
    return text

extra={}
if mode=='base_add':
    old='__m512d base=_mm512_fmadd_pd(c1[g],d[g],c0[g]);'
    new='__m512d base=_mm512_add_pd(c0[g],cd);'
    if s.count(old)!=1: raise SystemExit(f'base pattern count={s.count(old)}')
    s=s.replace(old,new,1)
elif mode.startswith('hlo_'):
    st=int(mode.split('_',1)[1])
    old='0x1.1a62633145c07p-62'
    if s.count(old)!=1: raise SystemExit(f'HLO count={s.count(old)}')
    s=s.replace(old,step_hex(old,st),1)
elif mode.startswith('lut_'):
    _,a,b=mode.split('_')
    ss=int(a); cs=int(b)
    s=edit_array(s,'x65_s',{2:ss,510:ss})
    s=edit_array(s,'x65_c',{2:cs,510:-cs})
elif mode=='baseline':
    # Add focused rare-path candidates without changing the workflow matrix.
    # The threshold is below the smallest phase magnitude observed in the
    # 2-ULP family, so this only re-evaluates the vulnerable tail of j=2/510.
    extra['direct_tail142']=inject_direct_tail(s,'0x1.d14e3bcd35a86p-7',False)
    extra['direct_tail142_dd']=inject_direct_tail(s,'0x1.d14e3bcd35a86p-7',True)
else:
    raise SystemExit('unknown mode '+mode)

Path(dst).write_text(finish_source(s))
for name,text in extra.items():
    Path(dst).with_name(name+'.c').write_text(finish_source(text))
