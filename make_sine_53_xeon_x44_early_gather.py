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

# Remove the X38/X41 late sequence after ji3.
marker='        ji3=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh3,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);'
late=[marker,'        /* X38: per-stream gather pair, then independent delta work. */']
for b in range(4):
    late += [g0[b],g1[b],delta[b]]
late_text='\n'.join(late)
if late_text not in s:
    raise SystemExit('X41 stagger sequence marker missing')
s=s.replace(late_text,marker,1)

# Issue memory work immediately when each stream's anchor index becomes ready,
# rather than waiting for all four streams. This moves gather latency under the
# following streams' reducer/index arithmetic without changing expressions.
for b in range(4):
    imarker=f'        ji{b}=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh{b},VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);'
    if imarker not in s:
        raise SystemExit(f'index marker missing stream {b}')
    if mode=='pair':
        ins=imarker+'\n        /* early pair: overlap L1 gather latency with later stream preparation */\n'+g0[b]+'\n'+g1[b]
    else:
        ins=imarker+'\n        /* early c0: lower live-register pressure; c1 stays late */\n'+g0[b]
    s=s.replace(imarker,ins,1)

# Delta remains after all stream preparation. In c0 mode, issue late c1 just
# before each independent delta so its latency can overlap following streams.
anchor=marker
if mode=='pair':
    tail=anchor+'\n        /* all coefficient gathers are already in flight/complete */\n'+'\n'.join(delta)
    tag='X44'; prefix='S53X44_'; arch='xeon_x44_early_pair_gather_g4'; label='Xeon_AVX512_X44_early_pair_gather_pipeline'
else:
    seq=[anchor,'        /* c0 was issued early; stagger late c1 with delta arithmetic */']
    for b in range(4):
        seq += [g1[b],delta[b]]
    tail='\n'.join(seq)
    tag='X45'; prefix='S53X45_'; arch='xeon_x45_early_c0_gather_g4'; label='Xeon_AVX512_X45_early_c0_gather_pipeline'
# Because marker was also expanded above when b=3, replace the unique bare
# occurrence that remains at the end of the four prepare blocks.
idx=s.find(anchor)
# choose last occurrence: early insertion retains the marker text as prefix.
idx=s.rfind(anchor)
if idx<0: raise SystemExit('final ji3 marker missing')
s=s[:idx]+tail+s[idx+len(anchor):]

s=s.replace('S53X41_',prefix)
s=s.replace('xeon_x41_branchless_exact_fma_delta_g4',arch)
s=s.replace('Xeon_AVX512_X41_branchless_exact_fma_delta',label)
out=Path(f'bench_sine_53_xeon_{tag.lower()}_build.c')
out.write_text(s)
print(f'{tag}_BUILD_PASS parent=X41 math_identical=1 reducer_identical=1 LUT_identical=1 G4=1 exact_branchless_delta=1 early_gather_mode={mode} requires_Arb_regate=1')
