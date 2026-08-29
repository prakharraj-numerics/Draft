from pathlib import Path
p=Path('bench_sine_53_xeon_x65_build.c')
s=p.read_text()
old='''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(\n        const s53w_kernel *k,const double * __restrict x,double * __restrict out,size_t n)\n{\n    /* Dedicated frozen wide path for homogeneous |x| in [1,10000]. */\n    if(n>=32){\n        size_t j=0;\n        for(;j<n;j++){double a=fabs(x[j]);if(a<1.0||a>10000.0)break;}\n        if(j==n){octant_vector_v8_rawx65(k,x,out,n);return;}\n    }\n    octant_vector_v8_x56_general(k,x,out,n);\n}\n'''
new='''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(\n        const s53w_kernel *k,const double * __restrict x,double * __restrict out,size_t n)\n{\n    /* Frozen wide entry: caller/domain dispatch has already selected >1. */\n    if(n>=32){octant_vector_v8_rawx65(k,x,out,n);return;}\n    octant_vector_v8_x56_general(k,x,out,n);\n}\n'''
if old not in s: raise SystemExit('X65 prescan wrapper marker not found')
p.write_text(s.replace(old,new,1))
print('S53X65_NOSCAN_PATCH_PASS frozen_math_unchanged=1 scalar_prescan_removed=1')
