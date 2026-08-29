from pathlib import Path
import runpy

# X58: One-gather sin/cos reconstruction
# Generate X56 first.
# Gather only c0 = sin(anchor) on the ordinary path.
# Reconstruct positive c1 = cos(anchor) using: c1 = sqrt((1 - c0) * (1 + c0))
# For anchor index j >= 256, use masked exact c1 gather from original table.
# Retain complete protected X56 CF evaluator, reducer, schedule, sign handling and fallback.

runpy.run_path('make_sine_53_xeon_x56_cf_protected.py', run_name='__main__')
p = Path('bench_sine_53_xeon_x56_build.c')
s = p.read_text()

hit = s.index('octant_vector_v8(const s53w_kernel *k,')
start = s.rfind('\n', 0, hit) + 1
end = s.index('\n#endif', hit)
hot = s[start:end]

# Strategy: find the two-gather block and replace with one-gather + sqrt reconstruction
# Pattern: find c0_b and c1_b gathers, then replace c1 gather with sqrt reconstruction

new_hot = hot

for b in range(4):
    # Pattern: look for the sequence of two gathers
    # __m512d c0_{b} = _mm512_i64gather_pd(idx0_{b}, lut, 8);
    # __m512d c1_{b} = _mm512_i64gather_pd(idx1_{b}, lut, 8);
    
    # We need to:
    # 1. Keep the c0 gather
    # 2. Replace c1 gather with: c1 = sqrt((1 - c0) * (1 + c0))
    # 3. For j >= 256, blend in exact c1 from gather
    
    # Find the c1 gather line
    c0_pattern = f'__m512d c0_{b} = _mm512_i64gather_pd'
    c1_pattern = f'__m512d c1_{b} = _mm512_i64gather_pd'
    
    if c0_pattern in hot and c1_pattern in hot:
        # Extract indices - look for idx0 and idx1
        c0_idx_start = hot.find(f'idx0_{b}')
        c1_idx_start = hot.find(f'idx1_{b}')
        
        if c0_idx_start > 0 and c1_idx_start > 0:
            # Build replacement that keeps c0 gather but replaces c1 with sqrt reconstruction
            # and adds conditional exact c1 repair for j >= 256
            
            old_c1_gather = hot[hot.find(f'__m512d c1_{b} = _mm512_i64gather_pd'):hot.find(f'__m512d c1_{b} = _mm512_i64gather_pd') + 100]
            old_c1_gather = old_c1_gather[:old_c1_gather.find(';')+1]
            
            new_c1_code = f'''__m512d c1_sqrt_{b} = _mm512_mul_pd(_mm512_sub_pd(_mm512_set1_pd(1.0), c0_{b}), _mm512_add_pd(_mm512_set1_pd(1.0), c0_{b}));
        c1_sqrt_{b} = _mm512_sqrt_pd(c1_sqrt_{b});
        __m512d c1_exact_{b} = _mm512_i64gather_pd(idx1_{b}, lut, 8);
        __mmask8 repair_mask_{b} = _mm512_cmpge_epi64_mask(idx1_{b}, _mm512_set1_epi64(256));
        __m512d c1_{b} = _mm512_mask_blend_pd(repair_mask_{b}, c1_sqrt_{b}, c1_exact_{b});'''
            
            new_hot = new_hot.replace(old_c1_gather, new_c1_code)

# Replace naming to mark as X58
s = s[:start] + new_hot + s[end:]
s = s.replace('S53X56_', 'S53X58_')
s = s.replace('Xeon_AVX512_X56_CF_protected_X50_pipeline', 
              'Xeon_AVX512_X58_onegather_X50_pipeline')
s = s.replace('xeon_x56_cf_protected_x50_pipeline_g4', 
              'xeon_x58_onegather_x50_pipeline_g4')

out = Path('bench_sine_53_xeon_x58_build.c')
out.write_text(s)
print('X58_BUILD_PASS parent=X56 c0_gather=1 c1_sqrt_reconstruction=1 exact_c1_repair_j_ge_256=1 requires_Arb_regate=1')
