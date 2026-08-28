from pathlib import Path
import runpy

# Preserve v8 as the proven correctness baseline and derive a separate Xeon path.
runpy.run_path('make_sine_53_octant_v8.py', run_name='__main__')
src = Path('bench_sine_53_wide_octant_v8_build.c').read_text()

insert_at = src.index('OVEC static inline __m512d mode5_poly_i32_low')
anchor_code = r'''
/* Xeon experiment: replace six per-degree Mode-5 coefficient gathers by two
   anchor gathers (sin(a), cos(a)) and evaluate the same secant-spine C/T local
   identity entirely in registers.  In production these 403 pairs are shipped
   constants; MPFR generation here is setup-only and excluded from timing. */
static double *xeon_asin, *xeon_acos;
static int xeon_anchor_init(void)
{
    xeon_asin=al64((size_t)LUTN*sizeof(double));
    xeon_acos=al64((size_t)LUTN*sizeof(double));
    if(!xeon_asin||!xeon_acos)return 0;
    mpfr_t a,s,c; mpfr_init2(a,256); mpfr_init2(s,256); mpfr_init2(c,256);
    for(size_t i=0;i<LUTN;i++){
        mpfr_set_ui(a,(unsigned long)i,MPFR_RNDN);
        mpfr_div_2ui(a,a,SF_K,MPFR_RNDN);
        mpfr_sin_cos(s,c,a,MPFR_RNDN);
        xeon_asin[i]=mpfr_get_d(s,MPFR_RNDN);
        xeon_acos[i]=mpfr_get_d(c,MPFR_RNDN);
    }
    mpfr_clear(c); mpfr_clear(s); mpfr_clear(a); return 1;
}

OVEC static inline __m512d xeon_ct_i32(__m512d yh,__m512d yl,__mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d ONE=_mm512_set1_pd(1.0),MH=_mm512_set1_pd(-0.5),C24=_mm512_set1_pd(1.0/24.0);
    const __m512d MSIX=_mm512_set1_pd(-1.0/6.0),C120=_mm512_set1_pd(1.0/120.0);
    __m512d ya=_mm512_add_pd(yh,yl);
    __m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(ya,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256i ji=_mm512_cvttpd_epi32(jd);
    __m512d d=_mm512_fnmadd_pd(jd,VIK,yh);
    d=_mm512_add_pd(d,yl);
    __m512d z=_mm512_mul_pd(d,d);
    __m512d C=_mm512_fmadd_pd(z,_mm512_fmadd_pd(z,C24,MH),ONE);
    __m512d T=_mm512_fmadd_pd(z,_mm512_fmadd_pd(z,C120,MSIX),ONE);
    __m512d sa=_mm512_i32gather_pd(ji,xeon_asin,8);
    __m512d ca=_mm512_i32gather_pd(ji,xeon_acos,8);
    __m512d dt=_mm512_mul_pd(d,T);
    __m512d p=_mm512_fmadd_pd(ca,dt,_mm512_mul_pd(sa,C));
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

'''
src = src[:insert_at] + anchor_code + src[insert_at:]

start = src.index('OVEC static void octant_vector_v8')
end = src.index('\n#endif', start)
new_vec = r'''OVEC static void octant_vector_v8(const s53w_kernel *k,const double *x,
                                  double *out,size_t n)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d V4OPI=_mm512_set1_pd(FOUR_OVER_PI);
    const __m512d VC1=_mm512_set1_pd(PIO4_CW1),VC2=_mm512_set1_pd(PIO4_CW2),VC3=_mm512_set1_pd(PIO4_CW3);
    const __m512d VFT=_mm512_set1_pd(BOUND_TAU*FOUR_OVER_PI),V1MFT=_mm512_set1_pd(1.0-BOUND_TAU*FOUR_OVER_PI);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    const __m256i I1=_mm256_set1_epi32(1),I2=_mm256_set1_epi32(2),I3=_mm256_set1_epi32(3),I4=_mm256_set1_epi32(4),I7=_mm256_set1_epi32(7);

    for(size_t i=0;i<n;i+=8){
        unsigned rem=(unsigned)(n-i);
        __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
        __m512d vx=_mm512_maskz_loadu_pd(active,x+i);
        __mmask8 inneg=(__mmask8)(_mm512_cmp_pd_mask(vx,Z,_CMP_LT_OQ)&active);
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vx),ABSM));
        __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);

        if(unit==active){
            __m512d p=xeon_ct_i32(ax,Z,inneg);
            _mm512_mask_storeu_pd(out+i,active,p);
            continue;
        }
        __mmask8 wide=(__mmask8)(active&~unit);

        __m512d qf=_mm512_mul_pd(ax,V4OPI);
        __m256i qi=_mm512_cvttpd_epi32(qf);
        __m512d qd=_mm512_cvtepi32_pd(qi);
        __m512d frac=_mm512_sub_pd(qf,qd);
        __mmask8 guarded=(__mmask8)((_mm512_cmp_pd_mask(frac,VFT,_CMP_LT_OQ)|
                                     _mm512_cmp_pd_mask(frac,V1MFT,_CMP_GT_OQ))&wide);

        /* Octant logic from bits instead of seven equality masks.
           r=o&3; adjustment=[0,-1,+2,+1]=(r&2)-(r&1). */
        __m256i oi=_mm256_and_si256(qi,I7);
        __m256i r=_mm256_and_si256(oi,I3);
        __m256i b1=_mm256_and_si256(r,I1), b2=_mm256_and_si256(r,I2);
        __m256i adj=_mm256_sub_epi32(b2,b1);
        __m256i mi=_mm256_add_epi32(qi,adj);
        __m512d md=_mm512_cvtepi32_pd(mi);
        __mmask8 rev=(__mmask8)((~mask_eq_i32(b2,0))&wide);
        __m256i b4=_mm256_and_si256(oi,I4);
        __mmask8 wide_neg=(__mmask8)((~mask_eq_i32(b4,0))&wide);

        /* Keep v8's proven compensated Cody-Waite residual. */
        __m512d t1=_mm512_mul_pd(md,VC1);
        __m512d r0=_mm512_sub_pd(ax,t1);
        __m512d t2=_mm512_mul_pd(md,VC2);
        __m512d rh,re; twodiff_cw(r0,t2,&rh,&re);
        __m512d rl=_mm512_fnmadd_pd(md,VC3,re);
        rh=_mm512_mask_sub_pd(rh,rev,Z,rh);
        rl=_mm512_mask_sub_pd(rl,rev,Z,rl);

        rh=_mm512_mask_mov_pd(rh,unit,ax);
        rl=_mm512_mask_mov_pd(rl,unit,Z);
        rh=_mm512_mask_mov_pd(rh,guarded,Z);
        rl=_mm512_mask_mov_pd(rl,guarded,Z);
        __mmask8 signmask=(__mmask8)(((wide_neg^inneg)&wide)|(inneg&unit));
        __m512d p=xeon_ct_i32(rh,rl,signmask);
        _mm512_mask_storeu_pd(out+i,active,p);

        if(__builtin_expect(guarded!=0,0)){
            for(unsigned lane=0;lane<8&&i+lane<n;lane++)
                if(guarded&(1u<<lane)) out[i+lane]=scalar2(k,x[i+lane]);
        }
    }
}'''
src = src[:start] + new_vec + src[end:]

# Setup-only anchor creation before any verification/timing.
needle='s53w_kernel *k=kernel_create(2);'
if needle not in src:
    raise SystemExit('kernel-create marker not found')
src=src.replace(needle,'if(!xeon_anchor_init())return 2;'+needle,1)

# Distinguish output and function names from the v8 baseline.
src=src.replace('S53O8_', 'S53X1_')
src=src.replace('octant_v8', 'octant_x1')
src=src.replace('_v8', '_x1')
src=src.replace('cosine_style_pi4_octant_guarded_x1_compensated_cw','cosine_style_pi4_octant_guarded_xeon2g_compensated_cw')
src=src.replace('AVX512_pi4_octant_int32_compensated_cw','AVX512_pi4_octant_int32_compensated_cw_2anchor_gather')
Path('bench_sine_53_xeon_v1_build.c').write_text(src)
print('S53X1_BUILD_PASS two_anchor_gathers=1 octant_bitlogic=1 v8_compensated_cw=1 rare_table_DD_boundary=1 formula=unchanged_secant_spine_CT')
