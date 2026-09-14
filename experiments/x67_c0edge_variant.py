from pathlib import Path
import sys

if len(sys.argv) != 4:
    raise SystemExit('usage: x67_c0edge_variant.py src.c dst.c mode')
src,dst,mode=sys.argv[1:]
s=Path(src).read_text()

BLOCK='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
LOOP='''        /* Same degree-5 polynomial, algebraically grouped into independent\n           even/odd d^2 chains: much shorter dependency chain than Horner. */\n        for(int g=0;g<4;g++){'''
THR='0x1.d14e3bcd35a86p-7'
EDGE='0x1.921d1fcdec784p-7'


def finish(t):
    needle='\nint main(void)\n{'
    if t.count(needle)!=1: raise SystemExit('production main count')
    t=t.replace(needle,'\nint sine53_production_disabled_main(void)\n{',1)
    return t+'\nint main(void){return 0;}\n'


def correction_body(indent, fix, g='g', signed_store=False):
    lines=[]
    lines.append(indent+'const __m512d X67_DHI=_mm512_set1_pd(0x1.921fb54442d18p-7);')
    lines.append(indent+'const __m512d X67_DLO=_mm512_set1_pd(0x1.1a62633145c07p-61);')
    lines.append(indent+'const __m512d X67_N5040=_mm512_set1_pd(-1.0/5040.0);')
    lines.append(indent+f'__mmask8 x67_m510=(__mmask8)(_mm512_movepi64_mask(_mm512_castpd_si512(c1[{g}]))&{fix});')
    lines.append(indent+f'__m512d x67_b=_mm512_mask_sub_pd(d[{g}],x67_m510,Z,d[{g}]);')
    lines.append(indent+'__m512d x67_rh=_mm512_add_pd(X67_DHI,x67_b);')
    lines.append(indent+'__m512d x67_re=_mm512_sub_pd(x67_b,_mm512_sub_pd(x67_rh,X67_DHI));')
    lines.append(indent+'__m512d x67_rl=_mm512_add_pd(x67_re,X67_DLO);')
    lines.append(indent+'__m512d x67_z=_mm512_mul_pd(x67_rh,x67_rh);')
    lines.append(indent+'__m512d x67_poly=_mm512_fmadd_pd(x67_z,X67_N5040,C120);')
    lines.append(indent+'x67_poly=_mm512_fmadd_pd(x67_z,x67_poly,M6);')
    lines.append(indent+'__m512d x67_alt=_mm512_fmadd_pd(_mm512_mul_pd(x67_rh,x67_z),x67_poly,x67_rh);')
    lines.append(indent+'x67_alt=_mm512_add_pd(x67_alt,x67_rl);')
    if signed_store:
        lines.append(indent+f'x67_alt=_mm512_mask_sub_pd(x67_alt,sg[{g}],Z,x67_alt);')
        lines.append(indent+f'_mm512_mask_storeu_pd(out+i+8*{g},{fix},x67_alt);')
    else:
        lines.append(indent+f'pv[{g}]=_mm512_mask_mov_pd(pv[{g}],{fix},x67_alt);')
    return '\n'.join(lines)


def per8(t, threshold):
    if t.count(BLOCK)!=1: raise SystemExit('per8 block count')
    mid='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            __mmask8 x67_fix=_mm512_cmp_pd_mask(c0[g],_mm512_set1_pd('''+EDGE+'''),_CMP_EQ_OQ);\n            if(__builtin_expect(x67_fix!=0,0)){'''
    if threshold:
        mid+='''\n                x67_fix=(__mmask8)(x67_fix&_mm512_cmp_pd_mask(pv[g],_mm512_set1_pd('''+THR+'''),_CMP_GE_OQ));\n                if(x67_fix){'''
        tail='''\n                }\n            }\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
        body=correction_body('                    ','x67_fix')
    else:
        tail='''\n            }\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
        body=correction_body('                ','x67_fix')
    return t.replace(BLOCK,mid+'\n'+body+tail,1)


def grouped32(t, threshold):
    if t.count(LOOP)!=1: raise SystemExit('group loop count')
    t=t.replace(LOOP,LOOP.replace('        for(int g=0;g<4;g++){','        __mmask8 x67_edge[4];\n        for(int g=0;g<4;g++){'),1)
    if t.count(BLOCK)!=1: raise SystemExit('group body count')
    body='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            x67_edge[g]=_mm512_cmp_pd_mask(c0[g],_mm512_set1_pd('''+EDGE+'''),_CMP_EQ_OQ);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
    t=t.replace(BLOCK,body,1)
    anchor='''        }\n    }\n    if(i<n) octant_vector_x20_tail(k,x+i,out+i,n-i);'''
    if t.count(anchor)!=1: raise SystemExit('group tail anchor count')
    cold='''        }\n        __mmask8 x67_any=(__mmask8)(x67_edge[0]|x67_edge[1]|x67_edge[2]|x67_edge[3]);\n        if(__builtin_expect(x67_any!=0,0)){\n            for(int g=0;g<4;g++){\n                __mmask8 x67_fix=x67_edge[g];'''
    if threshold:
        cold+='''\n                if(x67_fix){\n                    __m512d x67_ap=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(pv[g]),_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff))));\n                    x67_fix=(__mmask8)(x67_fix&_mm512_cmp_pd_mask(x67_ap,_mm512_set1_pd('''+THR+'''),_CMP_GE_OQ));\n                }'''
    cold+='''\n                if(!x67_fix) continue;\n'''+correction_body('                ','x67_fix',signed_store=True)+'''\n            }\n        }\n    }\n    if(i<n) octant_vector_x20_tail(k,x+i,out+i,n-i);'''
    return t.replace(anchor,cold,1)

if mode=='baseline':
    out=s
elif mode=='c0edge8_dd':
    out=per8(s,False)
elif mode=='c0edge8_thr_dd':
    out=per8(s,True)
elif mode=='c0edge32_dd':
    out=grouped32(s,False)
elif mode=='c0edge32_thr_dd':
    out=grouped32(s,True)
else:
    raise SystemExit('unknown mode '+mode)

Path(dst).write_text(finish(out))