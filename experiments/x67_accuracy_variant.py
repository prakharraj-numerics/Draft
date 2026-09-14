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
    pass
else:
    raise SystemExit('unknown mode '+mode)

# The frozen X67 source is a standalone benchmark and contains its own real
# entry point.  Disable that exact multiline entry point before embedding the
# source in sine53_engine_adapter.c.  Do this here, before appending the small
# CI dummy main, so the workflow can rename the dummy unambiguously.
prod_main='\nint main(void)\n{'
if s.count(prod_main)!=1:
    raise SystemExit(f'production main count={s.count(prod_main)}')
s=s.replace(prod_main,'\nint sine53_production_disabled_main(void)\n{',1)

# The CI wrapper renames this dummy entry point before embedding the source.
# Production code is never modified.
s += '\nint main(void){return 0;}\n'
Path(dst).write_text(s)
