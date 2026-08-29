from pathlib import Path
import runpy

# Build frozen X67 wide implementation.
runpy.run_path('make_sine_53_x67_compensated_grouped_final.py', run_name='__main__')
x67p=Path('bench_sine_53_xeon_x67_build.c')
s=x67p.read_text()

# Separately build the proven X50 unit-domain implementation and transplant only
# its hot octant_vector_v8 body under a new name. Dependencies already exist in
# the common lineage used by X67.
runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
t=Path('bench_sine_53_xeon_x50_build.c').read_text()
hit=t.index('octant_vector_v8(const s53w_kernel *k,')
start=t.rfind('\n',0,hit)+1
end=t.index('\n#endif',hit)
x50=t[start:end]
x50=x50.replace('octant_vector_v8(const s53w_kernel *k,','octant_vector_v8_x50unit(const s53w_kernel *k,',1)

# Insert the X50 unit entrypoint immediately before X67 raw kernel.
ins=s.index('OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawx67')
s=s[:ins]+x50+'\n\n'+s[ins:]

# Dedicated no-dispatch entrypoints.  Wide path remains byte-for-byte X67 hot
# implementation; unit path remains X50 hot implementation.
mainpos=s.index('\nint main(void)')
entry=r'''
OVEC __attribute__((noinline,hot,aligned(64))) static void sine53_x68_unit(
        const s53w_kernel *k,const double * __restrict x,double * __restrict out,size_t n)
{
    octant_vector_v8_x50unit(k,x,out,n);
}
OVEC __attribute__((noinline,hot,aligned(64))) static void sine53_x68_wide(
        const s53w_kernel *k,const double * __restrict x,double * __restrict out,size_t n)
{
    octant_vector_v8_rawx67(k,x,out,n);
}
'''
s=s[:mainpos]+entry+s[mainpos:]
Path('bench_sine_53_xeon_x68_build.c').write_text(s)
print('S53X68_BUILD_PASS unit_entry=X50_frozen wide_entry=X67_frozen no_shared_dispatch_tax=1 formula_unchanged=1 requires_Arb_regate=1')
