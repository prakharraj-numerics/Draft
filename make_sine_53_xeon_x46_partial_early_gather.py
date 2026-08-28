from pathlib import Path
import runpy, sys

if len(sys.argv)!=2 or sys.argv[1] not in ('1','2','3'):
    raise SystemExit('usage: make_sine_53_xeon_x46_partial_early_gather.py {1|2|3}')
n=int(sys.argv[1])

saved=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x41_branchless_delta.py','fma']
    runpy.run_path('make_sine_53_xeon_x41_branchless_delta.py',run_name='__main__')
finally:
    sys.argv=saved

p=Path('bench_sine_53_xeon_x41_build.c')
s=p.read_text()
g0=[f'        c0_{b}=_mm512_i32gather_pd(ji{b},tab+0*LUTN,8);' for b in range(4)]
g1=[f'        c1_{b}=_mm512_i32gather_pd(ji{b},tab+1*LUTN,8);' for b in range(4)]
d=[]
for b in range(4):
    d.append(f'        __m512d jd{b}=_mm512_cvtepi32_pd(ji{b});\n'
             f'        d{b}=_mm512_fnmadd_pd(jd{b},VIK,rh{b});\n'
             f'        d{b}=_mm512_add_pd(d{b},rl{b});')

marker3='        ji3=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh3,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);'
late=[marker3,'        /* X38: per-stream gather pair, then independent delta work. */']
for b in range(4): late += [g0[b],g1[b],d[b]]
late='\n'.join(late)
if late not in s: raise SystemExit('X41 late gather sequence missing')
placeholder='        /* X46_DEFERRED */'
s=s.replace(late,marker3+'\n'+placeholder,1)

# Only first n streams issue the pair immediately after their index exists.
for b in range(n):
    m=f'        ji{b}=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh{b},VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);'
    if m not in s: raise SystemExit(f'ji marker missing {b}')
    s=s.replace(m,m+'\n        /* partial early pair */\n'+g0[b]+'\n'+g1[b],1)

# After all stream preparation, early streams only need delta; remaining
# streams retain X41's pair-then-delta order exactly.
seq=[f'        /* first {n} stream(s) gathered early; rest use X41 stagger */']
for b in range(4):
    if b<n: seq.append(d[b])
    else: seq += [g0[b],g1[b],d[b]]
s=s.replace(placeholder,'\n'.join(seq),1)

tag={1:'X46',2:'X47',3:'X48'}[n]
prefix=f'S53{tag}_'
arch=f'xeon_{tag.lower()}_partial_early_pair_{n}_g4'
label=f'Xeon_AVX512_{tag}_partial_early_pair_{n}'
s=s.replace('S53X41_',prefix)
s=s.replace('xeon_x41_branchless_exact_fma_delta_g4',arch)
s=s.replace('Xeon_AVX512_X41_branchless_exact_fma_delta',label)
out=Path(f'bench_sine_53_xeon_{tag.lower()}_build.c')
out.write_text(s)
print(f'{tag}_BUILD_PASS parent=X41 math_identical=1 reducer_identical=1 LUT_identical=1 G4=1 exact_branchless_delta=1 early_pair_streams={n} requires_Arb_regate=1')
