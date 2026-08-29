from pathlib import Path
import runpy

# X57: physically split homogeneous |x|>=1 batches into their own hot symbol.
# The general/mixed path remains X56 for correctness.  A cheap batch classifier
# dispatches all-wide batches to a separate reducer/evaluator body so icx cannot
# merge the <1/mixed frontend into the >1 hot instruction stream.
runpy.run_path('make_sine_53_xeon_x56_wide_specialized_cold.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x56_build.c')
s=p.read_text()

hit=s.index('octant_vector_v8(const s53w_kernel *k,')
start=s.rfind('\n',0,hit)+1
end=s.index('\n#endif',hit)
orig=s[start:end]

general=orig.replace('octant_vector_v8(', 'octant_vector_v8_x56_general(', 1)
gt1=orig.replace('octant_vector_v8(', 'octant_vector_v8_gt1(', 1)
gt1=gt1.replace('x12_prepare_block(', 'x57_prepare_gt1(', )

helper=r'''OVEC static inline void x57_prepare_gt1(const double * __restrict x,size_t base,size_t n,
        __m512d *rh_out,__m512d *rl_out,__mmask8 *sign_out,
        __mmask8 *guard_out,__mmask8 *active_out,unsigned char *pure_unit_out)
{
    (void)n;
    const __m512d Z=_mm512_setzero_pd();
    const __m512d VINVP=_mm512_set1_pd(0x1.45f306dc9c883p-2);
    const __m512d RS=_mm512_set1_pd(0x1.8p52);
    const __m512d PIH=_mm512_set1_pd(0x1.921fb54442d18p+1);
    const __m512d PIL=_mm512_set1_pd(-0x1.1a62633145c07p-53);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    __m512d vx=_mm512_loadu_pd(x+base);
    __m512i vxi=_mm512_castpd_si512(vx);
    __mmask8 inneg=_mm512_movepi64_mask(vxi);
    __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(vxi,ABSM));

    __m512d Y=_mm512_fmadd_pd(ax,VINVP,RS);
    __m512d N=_mm512_sub_pd(Y,RS);
    __m512i Ybits=_mm512_castpd_si512(Y);
    __m512i paritybits=_mm512_slli_epi64(Ybits,63);
    __mmask8 parity=_mm512_movepi64_mask(paritybits);

    __m512d rh=_mm512_fnmadd_pd(N,PIH,ax);
    __m512d rl=_mm512_mul_pd(N,PIL);
    __m512d rs=_mm512_add_pd(rh,rl);
    __m512d ars=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs),ABSM));
    __mmask8 repair=_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-14),_CMP_LT_OQ);
    if(__builtin_expect(repair!=0,0)){
        const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
        const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
        const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
        __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(N,PI1));
        __m512d rh3,re3; twodiff_cw(r0,_mm512_mul_pd(N,PI2),&rh3,&re3);
        __m512d rl3=_mm512_fnmadd_pd(N,PI3,re3);
        rh=_mm512_mask_mov_pd(rh,repair,rh3);
        rl=_mm512_mask_mov_pd(rl,repair,rl3);
        rs=_mm512_add_pd(rh,rl);
    }
    __mmask8 rneg=_mm512_movepi64_mask(_mm512_castpd_si512(rs));
    rh=_mm512_mask_sub_pd(rh,rneg,Z,rh);
    rl=_mm512_mask_sub_pd(rl,rneg,Z,rl);
    *rh_out=rh; *rl_out=rl; *sign_out=(__mmask8)(inneg^parity^rneg);
    *guard_out=0; *active_out=0xff; *pure_unit_out=0;
}
'''

dispatch=r'''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}
    const __m512d ONE=_mm512_set1_pd(1.0);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    size_t j=0;
    for(;j+8<=n;j+=8){
        __m512i xi=_mm512_castpd_si512(_mm512_loadu_pd(x+j));
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(xi,ABSM));
        if(__builtin_expect(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)!=0,0)){
            octant_vector_v8_x56_general(k,x,out,n); return;
        }
    }
    for(;j<n;j++){
        uint64_t u; memcpy(&u,x+j,sizeof(u)); u&=UINT64_C(0x7fffffffffffffff);
        double a; memcpy(&a,&u,sizeof(a));
        if(a<1.0){octant_vector_v8_x56_general(k,x,out,n);return;}
    }
    octant_vector_v8_gt1(k,x,out,n);
}
'''

s=s[:start]+helper+'\n'+general+'\n'+gt1+'\n'+dispatch+s[end:]
s=s.replace('S53X56_','S53X57_')
s=s.replace('xeon_x56_wide_specialized_cold_g4','xeon_x57_batch_gt1_split_g4')
s=s.replace('Xeon_AVX512_X56_wide_specialized_cold','Xeon_AVX512_X57_batch_gt1_split')
Path('bench_sine_53_xeon_x57_build.c').write_text(s)
print('S53X57_BUILD_PASS parent=X56 batch_gt1_classifier=1 separate_gt1_hot_symbol=1 gt1_no_unit_masks=1 right_shifter_DD=1 rare_3piece_repair=1 mixed_fallback=X56 Mode5_unchanged=1 LUT_unchanged=1 Horner_unchanged=1 G4_schedule_unchanged=1 requires_Arb_regate=1')
