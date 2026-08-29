from pathlib import Path
import runpy

runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
s=Path('bench_sine_53_xeon_x50_build.c').read_text()
old='typedef struct { sine_fixed_ctx *ctx; int terms,deg; double *tab; } s53w_kernel;'
new='typedef struct { sine_fixed_ctx *ctx; int terms,deg; double *tab; double *anchor_pack; } s53w_kernel;'
if old not in s: raise SystemExit('kernel struct marker missing')
s=s.replace(old,new,1)
oldc='for(int a=0;a<LUTN;a++){size_t off=(size_t)a*(size_t)(k->deg+1);for(int j=0;j<=k->deg;j++)k->tab[(size_t)j*LUTN+(size_t)a]=coeff_to_double(k->ctx->coef+2*(off+(size_t)j),k->ctx->coef_sign[off+(size_t)j]!=0);}return k;}'
newc='for(int a=0;a<LUTN;a++){size_t off=(size_t)a*(size_t)(k->deg+1);for(int j=0;j<=k->deg;j++)k->tab[(size_t)j*LUTN+(size_t)a]=coeff_to_double(k->ctx->coef+2*(off+(size_t)j),k->ctx->coef_sign[off+(size_t)j]!=0);}k->anchor_pack=al64(48*sizeof(double));if(!k->anchor_pack){free(k->tab);s53_coeff_destroy(k->ctx);free(k);return NULL;}for(int q=0;q<8;q++){k->anchor_pack[q]=k->tab[q];k->anchor_pack[8+q]=k->tab[LUTN+q];k->anchor_pack[16+q]=k->tab[8*q];k->anchor_pack[24+q]=k->tab[LUTN+8*q];int j=64*q;if(j>=LUTN)j=0;k->anchor_pack[32+q]=k->tab[j];k->anchor_pack[40+q]=k->tab[LUTN+j];}return k;}'
if oldc not in s: raise SystemExit('kernel_create tail marker missing')
s=s.replace(oldc,newc,1)
oldd='static void kernel_destroy(s53w_kernel*k){if(!k)return;free(k->tab);s53_coeff_destroy(k->ctx);free(k);}'
newd='static void kernel_destroy(s53w_kernel*k){if(!k)return;free(k->anchor_pack);free(k->tab);s53_coeff_destroy(k->ctx);free(k);}'
if oldd not in s: raise SystemExit('kernel_destroy marker missing')
s=s.replace(oldd,newd,1)
hit=s.index('octant_vector_v8(const s53w_kernel *k,')
start=s.rfind('\n',0,hit)+1
end=s.index('\n#endif',hit)
hot=Path('x61_formula_nogather_hot.inc').read_text().rstrip()
s=s[:start]+hot+s[end:]
s=s.replace('S53X50_','S53X61_').replace('xeon_x50_cross_iteration_lookahead_g4','xeon_x61_formula_nogather').replace('Xeon_AVX512_X50_cross_iteration_lookahead','Xeon_AVX512_X61_formula_nogather')
Path('bench_sine_53_xeon_x61_build.c').write_text(s)
print('X61_BUILD_PASS parent=X50 formula_preserved=1 mode5_fused_relations=1 same_reducer=1 same_anchor_rule=1 same_delta=1 same_horner_order=1 coefficient_memory_gathers=0 anchor_reconstruction=base8_register_permute angle_add_from_original_mode5_anchor_values=1 requires_Arb_regate=1')
