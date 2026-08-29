from pathlib import Path
import runpy, re

runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
s=Path('bench_sine_53_xeon_x50_build.c').read_text()

# Patch the generated X50 object by C structure/function identity, not by an
# exact one-line spelling.  This deliberately makes the X61 generator immune
# to whitespace/reflow and to harmless changes in the fields preceding tab.
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

# s53w_kernel is a flat typedef in this translation unit.  Match only by the
# typedef's public type name, then insert the side-pack pointer immediately
# before its closing brace.
tm=re.search(r'typedef\s+struct\s*\{.*?\}\s*s53w_kernel\s*;',s,re.S)
if not tm:
    # Emit useful diagnostics into Actions if upstream generation ever changes.
    near=[ln for ln in s.splitlines() if 's53w_kernel' in ln][:8]
    raise SystemExit('s53w_kernel typedef missing; sightings='+repr(near))
txt=tm.group(0)
if 'anchor_pack' not in txt:
    close=txt.rfind('}')
    txt=txt[:close]+' double *anchor_pack; '+txt[close:]
    s=s[:tm.start()]+txt+s[tm.end():]

# Locate kernel_create structurally and insert initialization immediately
# before its final successful `return k;`, never before an error return.
ks,ko,ke=find_braced_function(s,'kernel_create')
body=s[ko:ke]
rets=list(re.finditer(r'\breturn\s+k\s*;',body))
if not rets:
    raise SystemExit('kernel_create successful return missing')
r=rets[-1]
absret=ko+r.start()
init='''k->anchor_pack=al64(48*sizeof(double));if(!k->anchor_pack){free(k->tab);s53_coeff_destroy(k->ctx);free(k);return NULL;}for(int q=0;q<8;q++){k->anchor_pack[q]=k->tab[q];k->anchor_pack[8+q]=k->tab[LUTN+q];k->anchor_pack[16+q]=k->tab[8*q];k->anchor_pack[24+q]=k->tab[LUTN+8*q];int j=64*q;if(j>=LUTN)j=0;k->anchor_pack[32+q]=k->tab[j];k->anchor_pack[40+q]=k->tab[LUTN+j];}'''
s=s[:absret]+init+s[absret:]

# Locate kernel_destroy structurally and free the side pack immediately before
# the existing table free.  Again, no full-line marker is assumed.
ds,do,de=find_braced_function(s,'kernel_destroy')
dbody=s[do:de]
fm=re.search(r'free\s*\(\s*k\s*->\s*tab\s*\)\s*;',dbody)
if not fm:
    raise SystemExit('kernel_destroy table free missing')
absfree=do+fm.start()
s=s[:absfree]+'free(k->anchor_pack);'+s[absfree:]

# Replace only X50's named hot evaluator with X61's formula-preserving no-gather
# evaluator.  The surrounding reducer, scalar repair path and harness remain X50.
hit=s.find('octant_vector_v8(const s53w_kernel *k,')
if hit<0:
    raise SystemExit('octant_vector_v8 marker missing')
start=s.rfind('\n',0,hit)+1
end=s.find('\n#endif',hit)
if end<0:
    raise SystemExit('octant_vector_v8 end marker missing')
hot=Path('x61_formula_nogather_hot.inc').read_text().rstrip()
s=s[:start]+hot+s[end:]

s=s.replace('S53X50_','S53X61_').replace('xeon_x50_cross_iteration_lookahead_g4','xeon_x61_formula_nogather').replace('Xeon_AVX512_X50_cross_iteration_lookahead','Xeon_AVX512_X61_formula_nogather')
Path('bench_sine_53_xeon_x61_build.c').write_text(s)

# Self-check the generated translation unit before allowing Actions to compile.
checks={
 'anchor_field':'anchor_pack' in txt,
 'anchor_init':'k->anchor_pack=al64(48*sizeof(double))' in s,
 'anchor_free':'free(k->anchor_pack);' in s,
 'hot_uses_pack':'__builtin_assume_aligned(k->anchor_pack,64)' in s,
 'hot_has_permute':'_mm512_permutexvar_pd' in s,
}
if not all(checks.values()):
    raise SystemExit('X61 generation self-check failed '+repr(checks))
print('X61_BUILD_PASS parent=X50 formula_preserved=1 mode5_fused_relations=1 same_reducer=1 same_anchor_rule=1 same_delta=1 same_horner_order=1 coefficient_memory_gathers=0 anchor_reconstruction=base8_register_permute angle_add_from_original_mode5_anchor_values=1 structural_generator=1 selfcheck=1 requires_Arb_regate=1')
