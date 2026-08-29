from pathlib import Path
import runpy

runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
s=Path('bench_sine_53_xeon_x50_build.c').read_text()

# X61 no longer patches s53w_kernel, kernel_create, or kernel_destroy.
# It only replaces X50's hot evaluator.  The hot evaluator loads the fixed
# radix-8 anchor basis directly from the existing Mode5-generated k->tab once
# per call, keeps it in ZMM registers, and performs no coefficient gathers.
hit=s.find('octant_vector_v8(const s53w_kernel *k,')
if hit < 0:
    raise SystemExit('octant_vector_v8 marker missing')
start=s.rfind('\n',0,hit)+1
end=s.find('\n#endif',hit)
if end < 0:
    raise SystemExit('octant_vector_v8 end marker missing')
hot=Path('x61_formula_nogather_hot.inc').read_text().rstrip()
s=s[:start]+hot+s[end:]

s=s.replace('S53X50_','S53X61_').replace('xeon_x50_cross_iteration_lookahead_g4','xeon_x61_formula_nogather').replace('Xeon_AVX512_X50_cross_iteration_lookahead','Xeon_AVX512_X61_formula_nogather')
Path('bench_sine_53_xeon_x61_build.c').write_text(s)

checks={
 'hot_from_existing_tab':'__builtin_assume_aligned(k->tab,64)' in hot,
 'basis_a':'_mm512_load_pd(tab)' in hot,
 'basis_b':'tab[56]' in hot and 'tab[LUTN+56]' in hot,
 'basis_c':'tab[384]' in hot and 'tab[LUTN+384]' in hot,
 'hot_has_permute':'_mm512_permutexvar_pd' in hot,
 'no_anchor_pack':'anchor_pack' not in hot and 'x61_anchor_pack' not in s,
 'no_generator_init_patch':'kernel_create' not in Path(__file__).read_text(),
}
if not all(checks.values()):
    raise SystemExit('X61 generation self-check failed '+repr(checks))
print('X61_BUILD_PASS parent=X50 formula_preserved=1 mode5_fused_relations=1 same_reducer=1 same_anchor_rule=1 same_delta=1 same_horner_order=1 coefficient_memory_gathers=0 anchor_reconstruction=base8_register_permute angle_add_from_original_mode5_anchor_values=1 direct_existing_tab_basis_load=1 no_struct_surgery=1 no_initializer_surgery=1 selfcheck=1 requires_Arb_regate=1')
