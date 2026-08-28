from pathlib import Path
import runpy, sys

# Hardware-scheduling experiments derived from certified X36.  They do not
# change any floating-point expression, coefficient, reducer constant, LUT,
# or G4 width.  Only the relative placement of independent gather and delta
# instructions in the hot loop changes.
if len(sys.argv) != 2 or sys.argv[1] not in ('allfirst','stagger','split'):
    raise SystemExit('usage: make_sine_53_xeon_x37_hw_schedule.py {allfirst|stagger|split}')
mode = sys.argv[1]

saved = sys.argv[:]
try:
    sys.argv = ['make_sine_53_xeon_intel_clues.py','partial']
    runpy.run_path('make_sine_53_xeon_intel_clues.py', run_name='__main__')
finally:
    sys.argv = saved

p = Path('bench_sine_53_xeon_x36_partial_build.c')
s = p.read_text()

# Pull the four delta-construction blocks out of the per-stream preparation.
delta = []
for b in range(4):
    block = (
        f'        __m512d jd{b}=_mm512_cvtepi32_pd(ji{b});\n'
        f'        if(pu{b}) d{b}=_mm512_fnmadd_pd(jd{b},VIK,rh{b}); else {{d{b}=_mm512_sub_pd(rh{b},_mm512_mul_pd(jd{b},VIK));d{b}=_mm512_add_pd(d{b},rl{b});}}\n'
    )
    if block not in s:
        raise SystemExit(f'delta marker missing stream {b}')
    s = s.replace(block, '', 1)
    delta.append(block.rstrip())

g0 = [f'        c0_{b}=_mm512_i32gather_pd(ji{b},tab+0*LUTN,8);' for b in range(4)]
g1 = [f'        c1_{b}=_mm512_i32gather_pd(ji{b},tab+1*LUTN,8);' for b in range(4)]

if mode == 'allfirst':
    # Existing grouped gathers remain in place.  Insert all jd/d work only
    # after the final c1 gather, so the gather memory operations can overlap
    # with conversion/local-delta arithmetic.
    marker = g1[3]
    if marker not in s:
        raise SystemExit('last c1 gather marker missing')
    late = marker + '\n        /* X37: all eight gathers issued before independent local-delta work. */\n' + '\n'.join(delta)
    s = s.replace(marker, late, 1)
    tag, prefix, arch, label = (
        'X37', 'S53X37_', 'xeon_x37_hw_all_gathers_first_g4',
        'Xeon_AVX512_X37_all_gathers_first_then_delta')
elif mode == 'stagger':
    # Remove the grouped gather planes.  After all four indices exist, issue
    # each stream's c0/c1 pair and immediately fill its gather latency with
    # the stream's independent jd/d arithmetic.
    for line in g0 + g1:
        if line not in s:
            raise SystemExit('grouped gather marker missing: ' + line)
        s = s.replace(line + '\n', '', 1)
    marker = '        ji3=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh3,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);'
    if marker not in s:
        raise SystemExit('last index marker missing')
    seq = [marker, '        /* X38: per-stream gather pair, then independent delta work. */']
    for b in range(4):
        seq += [g0[b], g1[b], delta[b]]
    s = s.replace(marker, '\n'.join(seq), 1)
    tag, prefix, arch, label = (
        'X38', 'S53X38_', 'xeon_x38_hw_stagger_gather_delta_g4',
        'Xeon_AVX512_X38_stagger_gather_pair_with_delta')
else:
    # Keep c0 plane early, place all jd/d work between c0 and c1 planes.  This
    # deliberately spreads the gather pressure while preserving four-way ILP.
    marker = g0[3]
    if marker not in s:
        raise SystemExit('last c0 gather marker missing')
    mid = marker + '\n        /* X39: fill the gap between gather planes with local-delta work. */\n' + '\n'.join(delta)
    s = s.replace(marker, mid, 1)
    tag, prefix, arch, label = (
        'X39', 'S53X39_', 'xeon_x39_hw_split_gather_planes_g4',
        'Xeon_AVX512_X39_c0_then_delta_then_c1')

s = s.replace('S53X36_', prefix)
s = s.replace('xeon_x36_partial_residual_anchor_overlap_g4', arch)
s = s.replace('Xeon_AVX512_X36_partial_residual_anchor_overlap', label)
out = Path(f'bench_sine_53_xeon_{tag.lower()}_hw_build.c')
out.write_text(s)
print(f'{tag}_BUILD_PASS parent=X36 math_identical=1 reducer_identical=1 LUT_identical=1 G4=1 schedule_only=1 mode={mode} requires_Arb_regate=1')
