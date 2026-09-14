from pathlib import Path
import math,re,sys

if len(sys.argv)!=6:
    raise SystemExit('usage: x67_asym_anchor_variant.py src dst cell sstep cstep')
src,dst,cell,ss,cs=sys.argv[1:]
cell=int(cell); ss=int(ss); cs=int(cs)
if cell not in (2,510): raise SystemExit('cell must be 2 or 510')
s=Path(src).read_text()

def step_hex(tok,n):
    x=float.fromhex(tok)
    for _ in range(abs(n)):
        x=math.nextafter(x, math.inf if n>0 else -math.inf)
    return x.hex()

def edit_array(text,name,idx,step):
    pat=rf'(static const double {name}\[512\][^=]*=\{{)(.*?)(\}};)'
    m=re.search(pat,text,re.S)
    if not m: raise SystemExit(f'array {name} not found')
    vals=[v.strip() for v in m.group(2).split(',')]
    if len(vals)!=512: raise SystemExit(f'{name} len={len(vals)}')
    vals[idx]=step_hex(vals[idx],step)
    return text[:m.start()]+m.group(1)+','.join(vals)+m.group(3)+text[m.end():]

s=edit_array(s,'x65_s',cell,ss)
s=edit_array(s,'x65_c',cell,cs)
# Normalize the production entry formatting so the workflow can reliably
# rename it before linking into the scan harness. This is formatting only.
mm=list(re.finditer(r'\bint main\(void\)\s*\{',s))
if len(mm)!=1:
    raise SystemExit(f'main count={len(mm)}')
m=mm[0]
s=s[:m.start()]+'int main(void){'+s[m.end():]
Path(dst).write_text(s)
