from pathlib import Path
import runpy, sys, re

# X54: architecture selector.  Same numerical algorithm on both arms.
# Ice Lake Xeon (CPUID family 6 model 0x6a/0x6c) uses X50 rolling lookahead,
# which the hardware campaign showed materially faster on 8370C.  Other Xeons
# use X49.  The CPUID decision is cached; no per-lane or per-block dispatch.
saved=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x49_production.py']
    runpy.run_path('make_sine_53_xeon_x49_production.py',run_name='__main__')
    x49=Path('bench_sine_53_xeon_x49_build.c').read_text()
    sys.argv=['make_sine_53_xeon_x50_lookahead.py']
    runpy.run_path('make_sine_53_xeon_x50_lookahead.py',run_name='__main__')
    x50=Path('bench_sine_53_xeon_x50_build.c').read_text()
finally:
    sys.argv=saved

def extract(src):
    sig='OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8'
    a=src.index(sig)
    b=src.index('\n#endif\n\nstatic inline double unit_scalar_v8',a)
    return src[a:b]

f49=extract(x49).replace('static void octant_vector_v8','static void x54_x49_impl',1)
f50=extract(x50).replace('static void octant_vector_v8','static void x54_x50_impl',1)

# Use X50 as the container because it has every helper required by both bodies.
s=x50
start=s.index('OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8')
end=s.index('\n#endif\n\nstatic inline double unit_scalar_v8',start)
wrap=r'''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    static int cached_pick=-1;
    int pick=cached_pick;
    if(__builtin_expect(pick<0,0)){
        unsigned eax,ebx,ecx,edx;
        eax=1; ecx=0;
        __asm__ __volatile__("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));
        unsigned fam=(eax>>8)&0xfu;
        unsigned model=(eax>>4)&0xfu;
        if(fam==6u) model|=((eax>>16)&0xfu)<<4;
        /* Ice Lake-SP / Ice Lake-D model IDs. */
        pick=(fam==6u && (model==0x6au || model==0x6cu)) ? 50 : 49;
        cached_pick=pick;
    }
    if(__builtin_expect(pick==50,0)) x54_x50_impl(k,x,out,n);
    else x54_x49_impl(k,x,out,n);
}'''
s=s[:start]+f49+'\n\n'+f50+'\n\n'+wrap+s[end:]
# cpuid inline asm needs no additional header.
s=s.replace('S53X50_','S53X54_')
s=s.replace('Xeon_AVX512_X50_cross_iteration_lookahead','Xeon_AVX512_X54_cpu_dispatch_X49_X50')
s=s.replace('xeon_x50_cross_iteration_lookahead_g4','xeon_x54_cpu_dispatch')
out=Path('bench_sine_53_xeon_x54_build.c')
out.write_text(s)
print('X54_BUILD_PASS parent=X49+X50 numerical_paths_identical_contract=1 cpuid_cached=1 icelake_models=0x6a,0x6c pick_icelake=X50 pick_other=X49 dispatch_per_array_call=1 requires_Arb_regate=1')
