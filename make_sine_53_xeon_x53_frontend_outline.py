from pathlib import Path
import runpy, sys

# X53: preserve X49 arithmetic/schedule.  Move the four rare guarded scalar
# repair loops completely out of the AVX-512 hot function, and align the loop
# body boundary.  Goal: reduce hot code/uop footprint and DSB pressure.
saved=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x49_production.py']
    runpy.run_path('make_sine_53_xeon_x49_production.py',run_name='__main__')
finally:
    sys.argv=saved
p=Path('bench_sine_53_xeon_x49_build.c')
s=p.read_text()
needle='OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,'
helper=r'''__attribute__((noinline,cold)) static void x53_repair8(const s53w_kernel *k,const double *x,double *out,__mmask8 g)
{
    for(unsigned lane=0;lane<8;lane++) if(g&(1u<<lane)) out[lane]=scalar2(k,x[lane]);
}

'''
if needle not in s: raise SystemExit('X49 hot declaration missing')
s=s.replace(needle,helper+needle,1)
s=s.replace('    for(;i+32<=n;i+=32){','    __asm__ __volatile__(".p2align 6");\n    for(;i+32<=n;i+=32){',1)
for off in (0,8,16,24):
    b=off//8
    old=f'        if(__builtin_expect(g{b}!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g{b}&(1u<<lane)) out[i+{off}+lane]=scalar2(k,x[i+{off}+lane]);'
    new=f'        if(__builtin_expect(g{b}!=0,0)) x53_repair8(k,x+i+{off},out+i+{off},g{b});'
    if old not in s: raise SystemExit(f'repair marker {b} missing')
    s=s.replace(old,new,1)
s=s.replace('S53X49_','S53X53_')
s=s.replace('Xeon_AVX512_X49_production_hw_schedule','Xeon_AVX512_X53_frontend_outlined_repairs')
s=s.replace('xeon_x49_production_hw_schedule_g4','xeon_x53_frontend_outlined_repairs_g4')
out=Path('bench_sine_53_xeon_x53_build.c')
out.write_text(s)
print('X53_BUILD_PASS parent=X49 math_identical=1 schedule_identical=1 rare_repairs_out_of_line=1 hot_loop_align64_directive=1 hot_code_shrink_target=1 requires_Arb_regate=1')
