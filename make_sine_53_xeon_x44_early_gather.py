from pathlib import Path
import runpy, sys

if len(sys.argv)!=2 or sys.argv[1] not in ('pair','c0'):
    raise SystemExit('usage: make_sine_53_xeon_x44_early_gather.py {pair|c0}')
mode=sys.argv[1]

# Start from certified X41: X38 stagger schedule + exact branchless FMA delta.
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
delta=[]
for b in range(4):
    delta.append(
        f'        __m512d jd{b}=_mm512_cvtepi32_pd(ji{b});\n'
        f'        d{b}=_mm512_fnmadd_pd(jd{b},VIK,rh{b});\n'
        f'        d{b}=_mm512_add_pd(d{b},rl{b});')

# Remove the X38/X41 sequence which waits for all four indices before issuing
# any gather. Leave a private placeholder after ji3 for the deferred delta/c1.
marker3='        ji3=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh3,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);'
late=[marker3,'        /* X38: per-stream gather pair, then independent delta work. */']
for b in range(4):
    late += [g0[b],g1[b],delta[b]]
late_text='\n'.join(late)
if late_text not in s:
    raise SystemExit('X41 stagger sequence marker missing')
placeholder='        /* X44_DEFERRED_WORK */'
s=s.replace(late_text,marker3+'\n'+placeholder,1)

# Issue memory work immediately when each stream's anchor index becomes ready.
# This allows L1 gather latency to overlap the following streams' reducer/index
# arithmetic. Floating-point expressions and anchor indices are unchanged.
for b in range(4):
    imarker=f'        ji{b}=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh{b},VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);'
    if imarker not in s:
        raise SystemExit(f'index marker missing stream {b}')
    if mode=='pair':
        ins=imarker+'\n        /* early pair: overlap gather latency with later stream preparation */\n'+g0[b]+'\n'+g1[b]
    else:
        ins=imarker+'\n        /* early c0: overlap one gather while limiting live ZMM pressure */\n'+g0[b]
    s=s.replace(imarker,ins,1)

if mode=='pair':
    deferred='        /* both coefficient planes already issued early */\n'+'\n'.join(delta)
    tag='X44'; prefix='S53X44_'; arch='xeon_x44_early_pair_gather_g4'; label='Xeon_AVX512_X44_early_pair_gather_pipeline'
else:
    seq=['        /* c0 issued early; stagger c1 with independent delta arithmetic */']
    for b in range(4):
        seq += [g1[b],delta[b]]
    deferred='\n'.join(seq)
    tag='X45'; prefix='S53X45_'; arch='xeon_x45_early_c0_gather_g4'; label='Xeon_AVX512_X45_early_c0_gather_pipeline'
if placeholder not in s:
    raise SystemExit('deferred-work placeholder missing')
s=s.replace(placeholder,deferred,1)

s=s.replace('S53X41_',prefix)
s=s.replace('xeon_x41_branchless_exact_fma_delta_g4',arch)
s=s.replace('Xeon_AVX512_X41_branchless_exact_fma_delta',label)
out=Path(f'bench_sine_53_xeon_{tag.lower()}_build.c')
out.write_text(s)
print(f'{tag}_BUILD_PASS parent=X41 math_identical=1 reducer_identical=1 LUT_identical=1 G4=1 exact_branchless_delta=1 early_gather_mode={mode} requires_Arb_regate=1')
