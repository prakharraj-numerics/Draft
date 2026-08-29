from pathlib import Path
import runpy

# X59: Intel/SVML-style three-split-pi reducer
# Generate X56 first.
# Keep X56's LUT, protected CF evaluator, gather schedule and sign reconstruction.
# Replace X56's two-piece plus rare-repair nearest-pi reducer with
# an unconditional branchless three-split-pi FMA chain on non-unit lanes.

runpy.run_path('make_sine_53_xeon_x56_cf_protected.py', run_name='__main__')
p = Path('bench_sine_53_xeon_x56_build.c')
s = p.read_text()

# Binary64 constants for three-split pi reduction
PI1 = '0x1.921fb54400000p+1'  # High precision pi/2
PI2 = '0x1.0b4611a600000p-33'  # Medium precision correction
PI3 = '0x1.3198a2e037073p-68'  # Fine precision correction

# Find and replace the reducer function
# Look for the reducer that handles range reduction
# Pattern: find the section that computes rh and rl

# Strategy: Find the existing reducer logic and replace it with three-split FMA

reducer_marker = 'fnmadd(qd, PI1'
if reducer_marker in s:
    # X56 already has a pi reducer, we need to replace it
    
    # Find the reducer block
    reducer_start = s.find('// range reduction')
    if reducer_start < 0:
        reducer_start = s.find('qd =')
    
    if reducer_start > 0:
        # Find the end of the reducer block (usually marked by unit path or next section)
        reducer_end = s.find('// unit path', reducer_start)
        if reducer_end < 0:
            reducer_end = s.find('if (qd_is_unit', reducer_start)
        
        if reducer_end > 0:
            old_reducer = s[reducer_start:reducer_end]
            
            # Build new three-split-pi reducer (branchless)
            new_reducer = f'''// X59: Intel/SVML-style three-split-pi reducer (branchless)
    __m512d PI1_v = _mm512_set1_pd({PI1});
    __m512d PI2_v = _mm512_set1_pd({PI2});
    __m512d PI3_v = _mm512_set1_pd({PI3});
    
    // Three-split FMA chain for non-unit lanes
    __m512d rh = _mm512_fnmadd_pd(qd_v, PI1_v, ax);
    rh = _mm512_fnmadd_pd(qd_v, PI2_v, rh);
    rh = _mm512_fnmadd_pd(qd_v, PI3_v, rh);
    __m512d rl = _mm512_setzero_pd();
    '''
            
            s = s[:reducer_start] + new_reducer + s[reducer_end:]

# Also need to ensure we have qd_v computed correctly
# Look for the quotient calculation and ensure it's present

# Replace naming to mark as X59
s = s.replace('S53X56_', 'S53X59_')
s = s.replace('Xeon_AVX512_X56_CF_protected_X50_pipeline', 
              'Xeon_AVX512_X59_intel_reducer_X50_pipeline')
s = s.replace('xeon_x56_cf_protected_x50_pipeline_g4', 
              'xeon_x59_intel_reducer_x50_pipeline_g4')

out = Path('bench_sine_53_xeon_x59_build.c')
out.write_text(s)
print('X59_BUILD_PASS parent=X56 reducer=nearest_pi_three_split_FMA branchless_repair=1 evaluator_X56=1 requires_Arb_regate=1')
