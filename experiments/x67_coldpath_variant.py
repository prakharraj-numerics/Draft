from pathlib import Path
import sys

if len(sys.argv) != 4:
    raise SystemExit('usage: x67_coldpath_variant.py src.c dst.c mode')
src, dst, mode = sys.argv[1:]
if mode not in ('nested_band','broad_hi','flat_band'):
    raise SystemExit('bad mode '+mode)
s = Path(src).read_text()

# The production dispatcher sends every n>=32 call directly to rawx67.  The
# previous experiment mistakenly attached its guard to x12/general, so the
# 8192-point accuracy chunks bypassed the repair entirely.  Patch ONLY rawx67.
raw_mark = 'OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawx67('
if s.count(raw_mark) != 1:
    raise SystemExit('rawx67 marker count')

# Do not assume any particular declaration follows rawx67.  Find the matching
# closing brace of the C function itself, while ignoring braces in comments and
# quoted literals.  This makes extraction insensitive to table/function order.
def function_extent(text, marker):
    start = text.index(marker)
    brace = text.find('{', start)
    if brace < 0:
        raise SystemExit('rawx67 opening brace not found')
    depth = 0
    state = 'code'
    i = brace
    while i < len(text):
        c = text[i]
        n = text[i+1] if i+1 < len(text) else ''
        if state == 'code':
            if c == '/' and n == '/':
                state = 'line'; i += 2; continue
            if c == '/' and n == '*':
                state = 'block'; i += 2; continue
            if c == '"':
                state = 'string'; i += 1; continue
            if c == "'":
                state = 'char'; i += 1; continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return start, i + 1
        elif state == 'line':
            if c == '\n': state = 'code'
        elif state == 'block':
            if c == '*' and n == '/':
                state = 'code'; i += 2; continue
        elif state in ('string', 'char'):
            if c == '\\':
                i += 2; continue
            if (state == 'string' and c == '"') or (state == 'char' and c == "'"):
                state = 'code'
        i += 1
    raise SystemExit('rawx67 closing brace not found')

a, b = function_extent(s, raw_mark)
pre, raw, post = s[:a], s[a:b], s[b:]
if raw.count(raw_mark) != 1:
    raise SystemExit('rawx67 extraction sanity')

loop = '''        /* Same degree-5 polynomial, algebraically grouped into independent\n           even/odd d^2 chains: much shorter dependency chain than Horner. */\n        for(int g=0;g<4;g++){'''
if raw.count(loop) != 1:
    raise SystemExit('raw polynomial loop count')
raw = raw.replace(loop, '''        /* Keep the frozen X67 polynomial/store path unchanged.  Merely pack\n           suspect lane masks; a single unlikely branch after all four groups\n           enters the cold repair. */\n        unsigned x67_pack=0;\n        /* Same degree-5 polynomial, algebraically grouped into independent\n           even/odd d^2 chains: much shorter dependency chain than Horner. */\n        for(int g=0;g<4;g++){''', 1)

needle = '''            pv[g]=_mm512_fmadd_pd(z,inner,base);\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);'''
if raw.count(needle) != 1:
    raise SystemExit('raw result point count')

if mode == 'nested_band':
    # Accuracy-first reference: exact two-cell detector, no numerical band.
    det = '''            __mmask8 x67_fix=(__mmask8)(mask_eq_i32(ji[g],2)|mask_eq_i32(ji[g],510));\n            x67_pack|=((unsigned)x67_fix)<<(8*g);'''
elif mode == 'broad_hi':
    # Exact symmetric cell detector plus the conservative residual tail that
    # contains every observed 2-ULP failure.  |d| cannot exceed pi/1024 here.
    det = '''            __m256i x67_delta=_mm256_abs_epi32(_mm256_sub_epi32(ji[g],_mm256_set1_epi32(256)));\n            __mmask8 x67_fix=mask_eq_i32(x67_delta,254);\n            __m512d x67_ad=_mm512_abs_pd(d[g]);\n            x67_fix=(__mmask8)(x67_fix&_mm512_cmp_pd_mask(x67_ad,_mm512_set1_pd(0x1.f212d77318fc5p-10),_CMP_GE_OQ));\n            x67_pack|=((unsigned)x67_fix)<<(8*g);'''
else:
    # Narrow final-result band around the localized vulnerable tail.  This is
    # still evaluated before sign reconstruction, so abs() is only defensive.
    det = '''            __mmask8 x67_fix=(__mmask8)(mask_eq_i32(ji[g],2)|mask_eq_i32(ji[g],510));\n            __m512d x67_ap=_mm512_abs_pd(pv[g]);\n            x67_fix=(__mmask8)(x67_fix&_mm512_cmp_pd_mask(x67_ap,_mm512_set1_pd(0x1.cac083126e979p-7),_CMP_GE_OQ));\n            x67_fix=(__mmask8)(x67_fix&_mm512_cmp_pd_mask(x67_ap,_mm512_set1_pd(0x1.fbe76c8b43958p-7),_CMP_LE_OQ));\n            x67_pack|=((unsigned)x67_fix)<<(8*g);'''
raw = raw.replace(needle, '            pv[g]=_mm512_fmadd_pd(z,inner,base);\n'+det+'\n            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);', 1)

end = '''            _mm512_storeu_pd(out+i+8*g,pv[g]);\n        }\n    }\n    if(i<n) octant_vector_v8_x56_general(k,x+i,out+i,n-i);'''
if raw.count(end) != 1:
    raise SystemExit('raw loop end count')

cold = '''            _mm512_storeu_pd(out+i+8*g,pv[g]);\n        }\n\n        /* True cold path: only suspect lanes are recomputed.  The common\n           result above is exactly the frozen X67 output. */\n        if(__builtin_expect(x67_pack!=0,0)){\n            const __m512d DH=_mm512_set1_pd(0x1.921fb54442d18p-7);\n            const __m512d DL=_mm512_set1_pd(0x1.1a62633145c07p-61);\n            const __m512d N5040=_mm512_set1_pd(-1.0/5040.0);\n            for(int g=0;g<4;g++){\n                __mmask8 x67_fix=(__mmask8)((x67_pack>>(8*g))&0xffu);\n                if(!x67_fix) continue;\n                __mmask8 m510=(__mmask8)(mask_eq_i32(ji[g],510)&x67_fix);\n                __m512d bb=_mm512_mask_sub_pd(d[g],m510,Z,d[g]);\n                __m512d rrh=_mm512_add_pd(DH,bb);\n                __m512d rre=_mm512_sub_pd(bb,_mm512_sub_pd(rrh,DH));\n                __m512d rrl=_mm512_add_pd(rre,DL);\n                __m512d zz=_mm512_mul_pd(rrh,rrh);\n                __m512d pp=_mm512_fmadd_pd(zz,N5040,C120);\n                pp=_mm512_fmadd_pd(zz,pp,M6);\n                __m512d alt=_mm512_fmadd_pd(_mm512_mul_pd(rrh,zz),pp,rrh);\n                alt=_mm512_add_pd(alt,rrl);\n                alt=_mm512_mask_sub_pd(alt,sg[g],Z,alt);\n                _mm512_mask_storeu_pd(out+i+8*g,x67_fix,alt);\n            }\n        }\n    }\n    if(i<n) octant_vector_v8_x56_general(k,x+i,out+i,n-i);'''
raw = raw.replace(end, cold, 1)
if raw.count('unsigned x67_pack=0;') != 1 or raw.count('True cold path: only suspect lanes are recomputed.') != 1:
    raise SystemExit('rawx67 patch sanity')
s = pre + raw + post

# Adapter owns main for the MPFR scanner / speed harness.
q='\nint main(void)\n{'
if s.count(q) != 1:
    raise SystemExit('main count')
s=s.replace(q,'\nint sine53_production_disabled_main(void)\n{',1)
Path(dst).write_text(s)
