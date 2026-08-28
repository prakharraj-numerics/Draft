from pathlib import Path
import runpy, sys

if len(sys.argv)!=2 or sys.argv[1] not in ('3p','2f'):
    raise SystemExit('usage: make_sine_53_xeon_x21_nearestpi.py {3p|2f}')
mode=sys.argv[1]

# Start from the current throughput champion: x20A = G4 one-pass + two gathers.
saved_argv=sys.argv[:]
try:
    sys.argv=['make_sine_53_xeon_x20_fused_batch.py','2g']
    runpy.run_path('make_sine_53_xeon_x20_fused_batch.py',run_name='__main__')
finally:
    sys.argv=saved_argv
p=Path('bench_sine_53_xeon_x20_2g_build.c')
s=p.read_text()

start=s.index('OVEC static inline void x12_prepare_block')
end=s.index('OVEC static void octant_vector_x20_tail',start)

if mode=='3p':
    helper=r'''OVEC static inline void x12_prepare_block(const double * __restrict x,size_t base,size_t n,
        __m512d *rh_out,__m512d *rl_out,__mmask8 *sign_out,
        __mmask8 *guard_out,__mmask8 *active_out,unsigned char *pure_unit_out)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d VINVP=_mm512_set1_pd(0x1.45f306dc9c883p-2);
    /* pi = PI1 + PI2 + PI3, obtained by scaling the already-certified pi/4 split. */
    const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
    const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
    const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));

    unsigned rem=(unsigned)(n-base);
    __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
    __m512d vx=_mm512_maskz_loadu_pd(active,x+base);
    __m512i vxi=_mm512_castpd_si512(vx);
    __mmask8 inneg=(__mmask8)(_mm512_movepi64_mask(vxi)&active);
    __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(vxi,ABSM));
    __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);
    *pure_unit_out=(unsigned char)(unit==active);
    if(unit==active){
        *rh_out=ax;*rl_out=Z;*sign_out=inneg;*guard_out=0;*active_out=active;return;
    }

    /* Sine-specific reduction: q=nearest integer to |x|/pi, so r is in [-pi/2,pi/2].
       Unlike octant reduction, either nearest choice at a half-period boundary is valid. */
    __m256i qi=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ax,VINVP),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d qd=_mm512_cvtepi32_pd(qi);
    __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(qd,PI1));
    __m512d rh,re;twodiff_cw(r0,_mm512_mul_pd(qd,PI2),&rh,&re);
    __m512d rl=_mm512_fnmadd_pd(qd,PI3,re);

    __m512d rs=_mm512_add_pd(rh,rl);
    __mmask8 rneg=(__mmask8)(_mm512_movepi64_mask(_mm512_castpd_si512(rs))&active);
    rh=_mm512_mask_sub_pd(rh,rneg,Z,rh);
    rl=_mm512_mask_sub_pd(rl,rneg,Z,rl);
    rh=_mm512_mask_mov_pd(rh,unit,ax);
    rl=_mm512_mask_mov_pd(rl,unit,Z);

    __m256i parityv=_mm256_slli_epi32(_mm256_and_si256(qi,_mm256_set1_epi32(1)),31);
    __mmask8 parity=(__mmask8)(_mm256_movemask_ps(_mm256_castsi256_ps(parityv))&active);
    *rh_out=rh;*rl_out=rl;*sign_out=(__mmask8)((inneg^parity^rneg)&active);
    *guard_out=0;*active_out=active;
}

'''
    prefix='S53X21A_'; arch='xeon_x21_nearestpi_3piece_g4_2g'; label='Xeon_AVX512_G4_nearestpi_3piece_two_gather_Mode5'
else:
    helper=r'''OVEC static inline void x12_prepare_block(const double * __restrict x,size_t base,size_t n,
        __m512d *rh_out,__m512d *rl_out,__mmask8 *sign_out,
        __mmask8 *guard_out,__mmask8 *active_out,unsigned char *pure_unit_out)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d VINVP=_mm512_set1_pd(0x1.45f306dc9c883p-2);
    const __m512d PIH=_mm512_set1_pd(0x1.921fb54442d18p+1);
    const __m512d PIL=_mm512_set1_pd(0x1.1a62633145c07p-53);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));

    unsigned rem=(unsigned)(n-base);
    __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
    __m512d vx=_mm512_maskz_loadu_pd(active,x+base);
    __m512i vxi=_mm512_castpd_si512(vx);
    __mmask8 inneg=(__mmask8)(_mm512_movepi64_mask(vxi)&active);
    __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(vxi,ABSM));
    __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);
    *pure_unit_out=(unsigned char)(unit==active);
    if(unit==active){
        *rh_out=ax;*rl_out=Z;*sign_out=inneg;*guard_out=0;*active_out=active;return;
    }

    __m256i qi=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ax,VINVP),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d qd=_mm512_cvtepi32_pd(qi);
    /* One rounded pi product is avoided: FMA gives the high residual directly;
       keep the true-pi tail as a separate low word for Mode-5 delta formation. */
    __m512d rh=_mm512_fnmadd_pd(qd,PIH,ax);
    __m512d rl=_mm512_mul_pd(qd,_mm512_set1_pd(-0x1.1a62633145c07p-53));

    __m512d rs=_mm512_add_pd(rh,rl);
    __mmask8 rneg=(__mmask8)(_mm512_movepi64_mask(_mm512_castpd_si512(rs))&active);
    rh=_mm512_mask_sub_pd(rh,rneg,Z,rh);
    rl=_mm512_mask_sub_pd(rl,rneg,Z,rl);
    rh=_mm512_mask_mov_pd(rh,unit,ax);
    rl=_mm512_mask_mov_pd(rl,unit,Z);

    __m256i parityv=_mm256_slli_epi32(_mm256_and_si256(qi,_mm256_set1_epi32(1)),31);
    __mmask8 parity=(__mmask8)(_mm256_movemask_ps(_mm256_castsi256_ps(parityv))&active);
    *rh_out=rh;*rl_out=rl;*sign_out=(__mmask8)((inneg^parity^rneg)&active);
    *guard_out=0;*active_out=active;
}

'''
    prefix='S53X21B_'; arch='xeon_x21_nearestpi_2piece_fma_g4_2g'; label='Xeon_AVX512_G4_nearestpi_2pieceFMA_two_gather_Mode5'

s=s[:start]+helper+s[end:]
s=s.replace('S53X20A_',prefix)
s=s.replace('xeon_x20_2g_g4_onepass',arch)
s=s.replace('Xeon_AVX512_G4_two_gather_Mode5',label)
out=Path(f'bench_sine_53_xeon_x21_{mode}_build.c')
out.write_text(s)
print(f'X21_BUILD_PASS mode={mode} parent=x20A group=4 two_gather=1 nearest_pi_reduction=1 octant_map_removed=1 boundary_guard_removed=1 residual_low_preserved=1 formula_unchanged=1 requires_Arb_regate=1')
