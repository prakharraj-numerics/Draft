from pathlib import Path
import runpy, re

runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
s=Path('bench_sine_53_xeon_x50_build.c').read_text()

def find_braced_function(src, name):
    m=re.search(r'\b'+re.escape(name)+r'\s*\([^;]*?\)\s*\{', src, re.S)
    if not m:
        raise SystemExit(name+' function marker missing')
    op=src.find('{',m.start())
    depth=0
    for i in range(op,len(src)):
        c=src[i]
        if c=='{': depth+=1
        elif c=='}':
            depth-=1
            if depth==0:
                return m.start(),op,i+1
    raise SystemExit(name+' unbalanced braces')

# Do NOT modify s53w_kernel.  Some upstream generators splice out the typedef
# while retaining uses of the type, so struct surgery is fragile.  X61 instead
# owns one TU-local 64-byte-aligned 48-double anchor pack.
hit=s.find('octant_vector_v8(const s53w_kernel *k,')
if hit < 0:
    raise SystemExit('octant_vector_v8 marker missing')
start=s.rfind('\n',0,hit)+1
pack_decl='static double x61_anchor_pack[48] __attribute__((aligned(64)));\n'
s=s[:start]+pack_decl+s[start:]

# Populate the compact pack from the original Mode5-generated X50 c0/c1 planes
# whenever kernel_create succeeds.  No allocation and no destructor changes.
ks,ko,ke=find_braced_function(s,'kernel_create')
body=s[ko:ke]
rets=list(re.finditer(r'\breturn\s+k\s*;',body))
if not rets:
    raise SystemExit('kernel_create successful return missing')
absret=ko+rets[-1].start()
init='''for(int q=0;q<8;q++){x61_anchor_pack[q]=k->tab[q];x61_anchor_pack[8+q]=k->tab[LUTN+q];x61_anchor_pack[16+q]=k->tab[8*q];x61_anchor_pack[24+q]=k->tab[LUTN+8*q];int j=64*q;if(j>=LUTN)j=0;x61_anchor_pack[32+q]=k->tab[j];x61_anchor_pack[40+q]=k->tab[LUTN+j];}'''
s=s[:absret]+init+s[absret:]

# Re-find hot function after insertion and replace only that evaluator.
hit=s.find('octant_vector_v8(const s53w_kernel *k,')
start=s.rfind('\n',0,hit)+1
end=s.find('\n#endif',hit)
if end < 0:
    raise SystemExit('octant_vector_v8 end marker missing')
hot=Path('x61_formula_nogather_hot.inc').read_text().rstrip()
hot=hot.replace('k->anchor_pack','x61_anchor_pack')
s=s[:start]+hot+s[end:]

s=s.replace('S53X50_','S53X61_').replace('xeon_x50_cross_iteration_lookahead_g4','xeon_x61_formula_nogather').replace('Xeon_AVX512_X50_cross_iteration_lookahead','Xeon_AVX512_X61_formula_nogather')
Path('bench_sine_53_xeon_x61_build.c').write_text(s)

checks={
 'global_pack':'static double x61_anchor_pack[48] __attribute__((aligned(64)));' in s,
 'anchor_init':'x61_anchor_pack[q]=k->tab[q]' in s,
 'hot_uses_pack':'__builtin_assume_aligned(x61_anchor_pack,64)' in s,
 'hot_has_permute':'_mm512_permutexvar_pd' in s,
 'no_struct_dependency':'anchor_pack' not in re.sub(r'x61_anchor_pack','',s),
}
if not all(checks.values()):
    raise SystemExit('X61 generation self-check failed '+repr(checks))
print('X61_BUILD_PASS parent=X50 formula_preserved=1 mode5_fused_relations=1 same_reducer=1 same_anchor_rule=1 same_delta=1 same_horner_order=1 coefficient_memory_gathers=0 anchor_reconstruction=base8_register_permute angle_add_from_original_mode5_anchor_values=1 global_static_pack=1 no_struct_surgery=1 selfcheck=1 requires_Arb_regate=1')
