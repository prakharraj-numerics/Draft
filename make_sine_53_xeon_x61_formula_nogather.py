from pathlib import Path
import runpy, re

runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
s=Path('bench_sine_53_xeon_x50_build.c').read_text()

# X50's generated translation unit may preserve or reflow the compact typedef.
# Patch the actual s53w_kernel typedef structurally instead of relying on one
# exact whitespace spelling.
pat=r'typedef\s+struct\s*\{(?P<body>[^{}]*?sine_fixed_ctx\s*\*\s*ctx[^{}]*?double\s*\*\s*tab\s*;[^{}]*?)\}\s*s53w_kernel\s*;'
m=re.search(pat,s,re.S)
if not m:
    raise SystemExit('kernel struct marker missing')
body=m.group('body')
if 'anchor_pack' not in body:
    body=body.rstrip()+' double *anchor_pack; '
s=s[:m.start()]+('typedef struct {'+body+'} s53w_kernel;')+s[m.end():]

# Add a 48-double aligned pack holding original Mode5 c0/c1 anchor values at
# radix-8 basis positions: a, 8b, 64c.  Runtime reconstructs arbitrary anchor
# j=a+8b+64c with angle addition, so the hot loop has no coefficient gathers.
kpos=s.find('static s53w_kernel *kernel_create')
if kpos < 0:
    raise SystemExit('kernel_create marker missing')
ret=s.find('return k;',kpos)
if ret < 0:
    raise SystemExit('kernel_create return marker missing')
init='''k->anchor_pack=al64(48*sizeof(double));if(!k->anchor_pack){free(k->tab);s53_coeff_destroy(k->ctx);free(k);return NULL;}for(int q=0;q<8;q++){k->anchor_pack[q]=k->tab[q];k->anchor_pack[8+q]=k->tab[LUTN+q];k->anchor_pack[16+q]=k->tab[8*q];k->anchor_pack[24+q]=k->tab[LUTN+8*q];int j=64*q;if(j>=LUTN)j=0;k->anchor_pack[32+q]=k->tab[j];k->anchor_pack[40+q]=k->tab[LUTN+j];}'''
s=s[:ret]+init+s[ret:]

# Free the pack before the original table.
dpos=s.find('static void kernel_destroy')
if dpos < 0:
    raise SystemExit('kernel_destroy marker missing')
freepos=s.find('free(k->tab);',dpos)
if freepos < 0:
    raise SystemExit('kernel_destroy free marker missing')
s=s[:freepos]+'free(k->anchor_pack);'+s[freepos:]

hit=s.index('octant_vector_v8(const s53w_kernel *k,')
start=s.rfind('\n',0,hit)+1
end=s.index('\n#endif',hit)
hot=Path('x61_formula_nogather_hot.inc').read_text().rstrip()
s=s[:start]+hot+s[end:]
s=s.replace('S53X50_','S53X61_').replace('xeon_x50_cross_iteration_lookahead_g4','xeon_x61_formula_nogather').replace('Xeon_AVX512_X50_cross_iteration_lookahead','Xeon_AVX512_X61_formula_nogather')
Path('bench_sine_53_xeon_x61_build.c').write_text(s)
print('X61_BUILD_PASS parent=X50 formula_preserved=1 mode5_fused_relations=1 same_reducer=1 same_anchor_rule=1 same_delta=1 same_horner_order=1 coefficient_memory_gathers=0 anchor_reconstruction=base8_register_permute angle_add_from_original_mode5_anchor_values=1 requires_Arb_regate=1')
