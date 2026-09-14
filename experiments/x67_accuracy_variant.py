from pathlib import Path
import math,re,sys

if len(sys.argv)!=4: raise SystemExit('usage: x67_accuracy_variant.py src.c dst.c mode')
src,dst,mode=sys.argv[1:]
s=Path(src).read_text()

def step_hex(tok,n):
    x=float.fromhex(tok)
    for _ in range(abs(n)): x=math.nextafter(x,math.inf if n>0 else -math.inf)
    return x.hex()

def edit_array(t,name,changes):
    m=re.search(rf'(static const double {name}\[512\][^=]*=\{{)(.*?)(\}};)',t,re.S)
    if not m: raise SystemExit('array '+name)
    v=[q.strip() for q in m.group(2).split(',')]
    if len(v)!=512: raise SystemExit('array length')
    for i,n in changes.items(): v[i]=step_hex(v[i],n)
    return t[:m.start()]+m.group(1)+','.join(v)+m.group(3)+t[m.end():]

def inject_c0(t,threshold):
    old='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
    if t.count(old)!=1: raise SystemExit('poly block count')
    gate='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            __mmask8 x67_fix=_mm512_cmp_pd_mask(c0[g],_mm512_set1_pd(0x1.921d1fcdec784p-7),_CMP_EQ_OQ);\n            if(__builtin_expect(x67_fix!=0,0)){'''
    if threshold:
        gate+='''\n                x67_fix=(__mmask8)(x67_fix&_mm512_cmp_pd_mask(pv[g],_mm512_set1_pd(0x1.d14e3bcd35a86p-7),_CMP_GE_OQ));\n                if(x67_fix){'''
        indent='                    '
        close='''\n                }\n            }'''
    else:
        indent='                '
        close='''\n            }'''
    body=[
      'const __m512d DH=_mm512_set1_pd(0x1.921fb54442d18p-7);',
      'const __m512d DL=_mm512_set1_pd(0x1.1a62633145c07p-61);',
      'const __m512d N5040=_mm512_set1_pd(-1.0/5040.0);',
      '__mmask8 m510=(__mmask8)(_mm512_movepi64_mask(_mm512_castpd_si512(c1[g]))&x67_fix);',
      '__m512d b=_mm512_mask_sub_pd(d[g],m510,Z,d[g]);',
      '__m512d rh=_mm512_add_pd(DH,b);',
      '__m512d re=_mm512_sub_pd(b,_mm512_sub_pd(rh,DH));',
      '__m512d rl=_mm512_add_pd(re,DL);',
      '__m512d zz=_mm512_mul_pd(rh,rh);',
      '__m512d pp=_mm512_fmadd_pd(zz,N5040,C120);',
      'pp=_mm512_fmadd_pd(zz,pp,M6);',
      '__m512d alt=_mm512_fmadd_pd(_mm512_mul_pd(rh,zz),pp,rh);',
      'alt=_mm512_add_pd(alt,rl);',
      'pv[g]=_mm512_mask_mov_pd(pv[g],x67_fix,alt);']
    new=gate+'\n'+'\n'.join(indent+x for x in body)+close+'''\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);'''
    return t.replace(old,new,1)

def inject_c0_group32(t):
    loop='''        /* Same degree-5 polynomial, algebraically grouped into independent\n           even/odd d^2 chains: much shorter dependency chain than Horner. */\n        for(int g=0;g<4;g++){'''
    if t.count(loop)!=1: raise SystemExit('group loop count')
    t=t.replace(loop,'''        /* Same degree-5 polynomial, algebraically grouped into independent\n           even/odd d^2 chains: much shorter dependency chain than Horner. */\n        unsigned x67_pack=0;\n        for(int g=0;g<4;g++){''',1)
    old='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);\n        }'''
    if t.count(old)!=1: raise SystemExit('group body count')
    new='''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            __mmask8 x67_edge=_mm512_cmp_pd_mask(c0[g],_mm512_set1_pd(0x1.921d1fcdec784p-7),_CMP_EQ_OQ);\n            x67_pack|=((unsigned)x67_edge)<<(8*g);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);\n            _mm512_storeu_pd(out+i+8*g,pv[g]);\n        }\n        if(__builtin_expect(x67_pack!=0,0)){\n            const __m512d DH=_mm512_set1_pd(0x1.921fb54442d18p-7);\n            const __m512d DL=_mm512_set1_pd(0x1.1a62633145c07p-61);\n            const __m512d N5040=_mm512_set1_pd(-1.0/5040.0);\n            const __m512i ABSM67=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));\n            for(int g=0;g<4;g++){\n                __mmask8 x67_fix=(__mmask8)(x67_pack>>(8*g));\n                if(!x67_fix) continue;\n                __m512d ap=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(pv[g]),ABSM67));\n                x67_fix=(__mmask8)(x67_fix&_mm512_cmp_pd_mask(ap,_mm512_set1_pd(0x1.d14e3bcd35a86p-7),_CMP_GE_OQ));\n                if(!x67_fix) continue;\n                __mmask8 m510=(__mmask8)(_mm512_movepi64_mask(_mm512_castpd_si512(c1[g]))&x67_fix);\n                __m512d b=_mm512_mask_sub_pd(d[g],m510,Z,d[g]);\n                __m512d rh=_mm512_add_pd(DH,b);\n                __m512d re=_mm512_sub_pd(b,_mm512_sub_pd(rh,DH));\n                __m512d rl=_mm512_add_pd(re,DL);\n                __m512d zz=_mm512_mul_pd(rh,rh);\n                __m512d pp=_mm512_fmadd_pd(zz,N5040,C120);\n                pp=_mm512_fmadd_pd(zz,pp,M6);\n                __m512d alt=_mm512_fmadd_pd(_mm512_mul_pd(rh,zz),pp,rh);\n                alt=_mm512_add_pd(alt,rl);\n                alt=_mm512_mask_sub_pd(alt,sg[g],Z,alt);\n                _mm512_mask_storeu_pd(out+i+8*g,x67_fix,alt);\n            }\n        }'''
    return t.replace(old,new,1)

def finish(t):
    q='\nint main(void)\n{'
    if t.count(q)!=1: raise SystemExit('main count')
    return t.replace(q,'\nint sine53_production_disabled_main(void)\n{',1)+'\nint main(void){return 0;}\n'

extra={}
if mode=='baseline':
    # Existing filenames are retained so the current speed/accuracy workflows
    # compare baseline, the per-8 detector, and the amortized per-32 detector.
    extra['direct_tail142']=inject_c0(s,True)
    extra['direct_tail142_dd']=inject_c0_group32(s)
elif mode=='base_add':
    a='__m512d base=_mm512_fmadd_pd(c1[g],d[g],c0[g]);'
    if s.count(a)!=1: raise SystemExit('base')
    s=s.replace(a,'__m512d base=_mm512_add_pd(c0[g],cd);',1)
elif mode.startswith('hlo_'):
    n=int(mode.split('_',1)[1]); a='0x1.1a62633145c07p-62'
    if s.count(a)!=1: raise SystemExit('hlo')
    s=s.replace(a,step_hex(a,n),1)
elif mode.startswith('lut_'):
    _,a,b=mode.split('_'); a=int(a); b=int(b)
    s=edit_array(s,'x65_s',{2:a,510:a}); s=edit_array(s,'x65_c',{2:b,510:-b})
else: raise SystemExit('unknown mode '+mode)

Path(dst).write_text(finish(s))
for name,t in extra.items(): Path(dst).with_name(name+'.c').write_text(finish(t))