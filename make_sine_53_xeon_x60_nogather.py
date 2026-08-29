from pathlib import Path
import runpy

# X60: remove coefficient-table gathers from the hot vector path.
# Keep X50/X49 range reduction and sign reconstruction, but evaluate sin(r)
# directly with a fixed degree-21 odd polynomial after nearest-pi reduction.
runpy.run_path('make_sine_53_xeon_x50_x53_hw_campaign.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x50_build.c')
s=p.read_text()

def replace_hot(src,newhot):
    hit=src.index('octant_vector_v8(const s53w_kernel *k,')
    start=src.rfind('\n',0,hit)+1
    end=src.index('\n#endif',hit)
    return src[:start]+newhot+src[end:]

x60=r'''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    size_t i=0;
    const __m512d Z=_mm512_setzero_pd();
    const __m512d C3 =_mm512_set1_pd(-1.0/6.0);
    const __m512d C5 =_mm512_set1_pd( 1.0/120.0);
    const __m512d C7 =_mm512_set1_pd(-1.0/5040.0);
    const __m512d C9 =_mm512_set1_pd( 1.0/362880.0);
    const __m512d C11=_mm512_set1_pd(-1.0/39916800.0);
    const __m512d C13=_mm512_set1_pd( 1.0/6227020800.0);
    const __m512d C15=_mm512_set1_pd(-1.0/1307674368000.0);
    const __m512d C17=_mm512_set1_pd( 1.0/355687428096000.0);
    const __m512d C19=_mm512_set1_pd(-1.0/121645100408832000.0);
    const __m512d C21=_mm512_set1_pd( 1.0/51090942171709440000.0);
#define X60_EVAL(B,OFF,NBLK) do { \
        __m512d rh##B,rl##B; __mmask8 sg##B,gd##B,aa##B; unsigned char pu##B; \
        x12_prepare_block(x+i,(OFF),(NBLK),&rh##B,&rl##B,&sg##B,&gd##B,&aa##B,&pu##B); \
        __m512d r##B=_mm512_add_pd(rh##B,rl##B); \
        __m512d z##B=_mm512_mul_pd(r##B,r##B); \
        __m512d q##B=C21; \
        q##B=_mm512_fmadd_pd(q##B,z##B,C19); q##B=_mm512_fmadd_pd(q##B,z##B,C17); \
        q##B=_mm512_fmadd_pd(q##B,z##B,C15); q##B=_mm512_fmadd_pd(q##B,z##B,C13); \
        q##B=_mm512_fmadd_pd(q##B,z##B,C11); q##B=_mm512_fmadd_pd(q##B,z##B,C9); \
        q##B=_mm512_fmadd_pd(q##B,z##B,C7);  q##B=_mm512_fmadd_pd(q##B,z##B,C5); \
        q##B=_mm512_fmadd_pd(q##B,z##B,C3); \
        __m512d rz##B=_mm512_mul_pd(r##B,z##B); \
        __m512d y##B=_mm512_fmadd_pd(rz##B,q##B,r##B); \
        y##B=_mm512_mask_sub_pd(y##B,sg##B,Z,y##B); \
        _mm512_storeu_pd(out+i+(OFF),y##B); \
        if(__builtin_expect(gd##B!=0,0)) for(unsigned lane=0;lane<8;lane++) \
            if(gd##B&(1u<<lane)) out[i+(OFF)+lane]=scalar2(k,x[i+(OFF)+lane]); \
    } while(0)
    for(;i+32<=n;i+=32){
        X60_EVAL(0,0,32); X60_EVAL(1,8,32); X60_EVAL(2,16,32); X60_EVAL(3,24,32);
    }
    for(;i+8<=n;i+=8){
        X60_EVAL(4,0,8);
    }
#undef X60_EVAL
    if(i<n) octant_vector_x20_tail(k,x+i,out+i,n-i);
}'''

s=replace_hot(s,x60)
s=s.replace('S53X50_','S53X60_')
s=s.replace('xeon_x50_cross_iteration_lookahead_g4','xeon_x60_nogather_fixedpoly_g4')
s=s.replace('Xeon_AVX512_X50_cross_iteration_lookahead','Xeon_AVX512_X60_nogather_fixedpoly')
Path('bench_sine_53_xeon_x60_build.c').write_text(s)
print('X60_BUILD_PASS parent=X50 coefficient_gathers=0 fixed_odd_degree=21 same_reducer=1 same_sign_reconstruction=1 hot_G4=1 tail8_nogather=1 residual_tail_lt8_old=1 requires_Arb_regate=1')
