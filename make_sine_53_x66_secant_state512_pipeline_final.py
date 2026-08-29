from pathlib import Path
import runpy

# X66: frozen X65 mathematics, hardware-only optimization.
# Changes: remove O(n) domain pre-scan for the documented |x|<=10000 kernel,
# one-stream cross-iteration state+gather lookahead, grouped even/odd degree-5
# evaluation to shorten dependency depth. Formula/state/table/residual unchanged.
runpy.run_path('make_sine_53_x65_secant_state512_final.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x65_build.c')
s=p.read_text()
hit=s.index('octant_vector_v8_rawx65(')
start=s.rfind('\nOVEC',0,hit)+1
end=s.index('\n#endif',hit)

raw=r'''OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawx66(
        const s53w_kernel *k,const double * __restrict x,double * __restrict out,size_t n)
{
    (void)k;
    const __m512d VINV=_mm512_set1_pd(0x1.45f306dc9c883p+7);
    const __m512d RS=_mm512_set1_pd(0x1.8p52);
    const __m512d HHI=_mm512_set1_pd(0x1.921fb54442d18p-8);
    const __m512d HLO=_mm512_set1_pd(0x1.1a62633145c07p-62);
    const __m512d ONE=_mm512_set1_pd(1.0),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const __m256i I511=_mm256_set1_epi32(511);
    size_t i=0;

    __m512d nvx0,nN0,nd0,nc0_0,nc1_0; __m256i nji0; __mmask8 nsg0;
    if(n>=32){
        nvx0=_mm512_loadu_pd(x);
        __m512d Y=_mm512_fmadd_pd(nvx0,VINV,RS);
        nN0=_mm512_sub_pd(Y,RS);
        __m512i yb=_mm512_castpd_si512(Y);
        nji0=_mm256_and_si256(_mm512_cvtepi64_epi32(yb),I511);
        nsg0=_mm512_movepi64_mask(_mm512_slli_epi64(yb,54));
        nc0_0=_mm512_i32gather_pd(nji0,x65_s,8);
        nc1_0=_mm512_i32gather_pd(nji0,x65_c,8);
        nd0=_mm512_fnmadd_pd(nN0,HHI,nvx0);
        nd0=_mm512_fnmadd_pd(nN0,HLO,nd0);
    }

    for(;i+32<=n;i+=32){
        __m512d vx[4],N[4],d[4],c0[4],c1[4],pv[4];
        __m256i ji[4]; __mmask8 sg[4];
        vx[0]=nvx0; N[0]=nN0; d[0]=nd0; c0[0]=nc0_0; c1[0]=nc1_0; ji[0]=nji0; sg[0]=nsg0;

        for(int g=1;g<4;g++){
            vx[g]=_mm512_loadu_pd(x+i+8*g);
            __m512d Y=_mm512_fmadd_pd(vx[g],VINV,RS);
            N[g]=_mm512_sub_pd(Y,RS);
            __m512i yb=_mm512_castpd_si512(Y);
            ji[g]=_mm256_and_si256(_mm512_cvtepi64_epi32(yb),I511);
            sg[g]=_mm512_movepi64_mask(_mm512_slli_epi64(yb,54));
            c0[g]=_mm512_i32gather_pd(ji[g],x65_s,8);
            c1[g]=_mm512_i32gather_pd(ji[g],x65_c,8);
            d[g]=_mm512_fnmadd_pd(N[g],HHI,vx[g]);
            d[g]=_mm512_fnmadd_pd(N[g],HLO,d[g]);
        }

        /* Launch next iteration's first gather before current polynomial chains. */
        if(__builtin_expect(i+64<=n,1)){
            nvx0=_mm512_loadu_pd(x+i+32);
            __m512d Y=_mm512_fmadd_pd(nvx0,VINV,RS);
            nN0=_mm512_sub_pd(Y,RS);
            __m512i yb=_mm512_castpd_si512(Y);
            nji0=_mm256_and_si256(_mm512_cvtepi64_epi32(yb),I511);
            nsg0=_mm512_movepi64_mask(_mm512_slli_epi64(yb,54));
            nc0_0=_mm512_i32gather_pd(nji0,x65_s,8);
            nc1_0=_mm512_i32gather_pd(nji0,x65_c,8);
            nd0=_mm512_fnmadd_pd(nN0,HHI,nvx0);
            nd0=_mm512_fnmadd_pd(nN0,HLO,nd0);
        }

        /* Same degree-5 polynomial, algebraically grouped into independent
           even/odd d^2 chains: much shorter dependency chain than Horner. */
        for(int g=0;g<4;g++){
            __m512d z=_mm512_mul_pd(d[g],d[g]);
            __m512d ce=_mm512_fmadd_pd(z,C24,MH);
            ce=_mm512_fmadd_pd(ce,z,ONE);
            __m512d so=_mm512_fmadd_pd(z,C120,M6);
            so=_mm512_fmadd_pd(so,z,ONE);
            __m512d cd=_mm512_mul_pd(c1[g],d[g]);
            __m512d base=_mm512_mul_pd(c0[g],ce);
            pv[g]=_mm512_fmadd_pd(cd,so,base);
            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);
            _mm512_storeu_pd(out+i+8*g,pv[g]);
        }
    }
    if(i<n) octant_vector_v8_x56_general(k,x+i,out+i,n-i);
}

OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(
        const s53w_kernel *k,const double * __restrict x,double * __restrict out,size_t n)
{
    /* X66 is the dedicated documented |x|<=10000 kernel.  No O(n) pre-scan. */
    if(__builtin_expect(n>=32,1)){octant_vector_v8_rawx66(k,x,out,n);return;}
    octant_vector_v8_x56_general(k,x,out,n);
}
'''

s=s[:start]+raw+s[end:]
s=s.replace('S53X65_','S53X66_')
s=s.replace('xeon_x65_secant_state512_final_g4','xeon_x66_secant_state512_pipeline_g4')
s=s.replace('Xeon_AVX512_X65_secant_state512_final','Xeon_AVX512_X66_secant_state512_pipeline')
Path('bench_sine_53_xeon_x66_build.c').write_text(s)
print('S53X66_BUILD_PASS parent=X65 frozen_math=1 user_secant_spine=1 state512=1 raw_x=1 no_domain_prescan=1 domain_abs_le_10000=1 G4_AVX512=1 stream0_cross_iteration_lookahead=1 grouped_even_odd_degree5=1 split_pi512_twoFMA=1 gathers=2 formula_unchanged=1 requires_Arb_regate=1')
