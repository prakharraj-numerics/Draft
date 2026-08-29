from pathlib import Path
import runpy

# X59: X58 architecture, still with NO pi/modulo/quadrant reduction for the
# homogeneous >1 hot path, but carry a double-double sin/cos state through the
# local Mode5-derived evaluation and repeated double-angle reconstruction.
# Low anchor parts come from the existing 103-bit coefficient source at setup.
runpy.run_path('make_sine_53_xeon_x58_raw_pow2_double.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x58_build.c')
s=p.read_text()

# Add a side table for low parts of the two anchor planes used at runtime.
old='typedef struct { sine_fixed_ctx *ctx; int terms,deg; double *tab; } s53w_kernel;'
new='typedef struct { sine_fixed_ctx *ctx; int terms,deg; double *tab; double *lotab; } s53w_kernel;'
if old not in s: raise SystemExit('kernel struct marker not found')
s=s.replace(old,new,1)

# Replace kernel construction/destruction.  MPFR is setup-only and extracts a
# faithful low double from the already-existing 103-bit fixed coefficient.
k0=s.index('static s53w_kernel *kernel_create(')
k1=s.index('\nstatic inline double reduce_scalar',k0)
replacement=r'''static void coeff_to_hi_lo103(const mp_limb_t q[2],int neg,double *hi,double *lo)
{
    mpfr_t v,t; mpfr_init2(v,192); mpfr_init2(t,192);
    mpfr_set_ui(v,(unsigned long)q[1],MPFR_RNDN); mpfr_mul_2si(v,v,-39,MPFR_RNDN);
    mpfr_set_ui(t,(unsigned long)q[0],MPFR_RNDN); mpfr_mul_2si(t,t,-103,MPFR_RNDN);
    mpfr_add(v,v,t,MPFR_RNDN); if(neg) mpfr_neg(v,v,MPFR_RNDN);
    *hi=mpfr_get_d(v,MPFR_RNDN); mpfr_sub_d(v,v,*hi,MPFR_RNDN); *lo=mpfr_get_d(v,MPFR_RNDN);
    mpfr_clear(t); mpfr_clear(v);
}
static s53w_kernel *kernel_create(int terms)
{
    if(terms<1||terms>3)return NULL;
    s53w_kernel*k=calloc(1,sizeof(*k)); if(!k)return NULL;
    k->ctx=s53_coeff_create_terms(terms); if(!k->ctx){free(k);return NULL;}
    k->terms=terms;k->deg=k->ctx->poly_deg;
    k->tab=al64((size_t)(k->deg+1)*LUTN*sizeof(double));
    k->lotab=al64((size_t)2*LUTN*sizeof(double));
    if(!k->tab||!k->lotab){free(k->lotab);free(k->tab);s53_coeff_destroy(k->ctx);free(k);return NULL;}
    for(int a=0;a<LUTN;a++){
        size_t off=(size_t)a*(size_t)(k->deg+1);
        for(int j=0;j<=k->deg;j++){
            const mp_limb_t *q=k->ctx->coef+2*(off+(size_t)j);
            int neg=k->ctx->coef_sign[off+(size_t)j]!=0;
            k->tab[(size_t)j*LUTN+(size_t)a]=coeff_to_double(q,neg);
            if(j<2){double hi,lo;coeff_to_hi_lo103(q,neg,&hi,&lo);k->tab[(size_t)j*LUTN+(size_t)a]=hi;k->lotab[(size_t)j*LUTN+(size_t)a]=lo;}
        }
    }
    return k;
}
static void kernel_destroy(s53w_kernel*k){if(!k)return;free(k->lotab);free(k->tab);s53_coeff_destroy(k->ctx);free(k);}
'''
s=s[:k0]+replacement+s[k1:]

hit=s.index('octant_vector_v8_rawpow2(const s53w_kernel *k,')
start=s.rfind('\n',0,hit)+1
end=s.index('\nOVEC ',hit+20)

helpers=r'''OVEC static inline void x59_norm(__m512d a,__m512d e,__m512d *h,__m512d *l)
{
    __m512d z=_mm512_add_pd(a,e); *h=z; *l=_mm512_add_pd(_mm512_sub_pd(a,z),e);
}
OVEC static inline void x59_scale(__m512d ah,__m512d al,__m512d k,__m512d *h,__m512d *l)
{
    __m512d p=_mm512_mul_pd(ah,k);
    __m512d e=_mm512_fmadd_pd(ah,k,_mm512_sub_pd(_mm512_setzero_pd(),p));
    e=_mm512_fmadd_pd(al,k,e); x59_norm(p,e,h,l);
}
OVEC static inline void x59_horner(__m512d ph,__m512d pl,__m512d d,__m512d ch,__m512d cl,__m512d *h,__m512d *l)
{
    __m512d p=_mm512_mul_pd(ph,d);
    __m512d pe=_mm512_fmadd_pd(ph,d,_mm512_sub_pd(_mm512_setzero_pd(),p));
    pe=_mm512_fmadd_pd(pl,d,pe);
    __m512d z=_mm512_add_pd(p,ch),bb=_mm512_sub_pd(z,p);
    __m512d se=_mm512_add_pd(_mm512_sub_pd(p,_mm512_sub_pd(z,bb)),_mm512_sub_pd(ch,bb));
    __m512d e=_mm512_add_pd(_mm512_add_pd(pe,cl),se); x59_norm(z,e,h,l);
}
OVEC static inline void x59_mul(__m512d ah,__m512d al,__m512d bh,__m512d bl,__m512d *h,__m512d *l)
{
    __m512d p=_mm512_mul_pd(ah,bh);
    __m512d e=_mm512_fmadd_pd(ah,bh,_mm512_sub_pd(_mm512_setzero_pd(),p));
    e=_mm512_fmadd_pd(ah,bl,e); e=_mm512_fmadd_pd(al,bh,e); x59_norm(p,e,h,l);
}
OVEC static inline void x59_double_state(__m512d sh,__m512d sl,__m512d ch,__m512d cl,
                                         __m512d *nsh,__m512d *nsl,__m512d *nch,__m512d *ncl)
{
    __m512d ph,pl; x59_mul(sh,sl,ch,cl,&ph,&pl);
    *nsh=_mm512_add_pd(ph,ph); *nsl=_mm512_add_pd(pl,pl);
    x59_mul(ch,cl,ch,cl,&ph,&pl); ph=_mm512_add_pd(ph,ph); pl=_mm512_add_pd(pl,pl);
    __m512d one=_mm512_set1_pd(1.0),z=_mm512_sub_pd(ph,one),bb=_mm512_sub_pd(z,ph);
    __m512d se=_mm512_add_pd(_mm512_sub_pd(ph,_mm512_sub_pd(z,bb)),_mm512_sub_pd(_mm512_sub_pd(_mm512_setzero_pd(),one),bb));
    x59_norm(z,_mm512_add_pd(pl,se),nch,ncl);
}
'''

decls=[]; loads=[]; coeff=[]; hor=[]; dbl=[]; finish=[]
for b in range(4):
    o=8*b
    decls.append(f'        __m512d u{b},d{b},ah{b},al{b},bh{b},bl{b},sh{b},sl{b},ch{b},cl{b}; __m256i ji{b}; __m512i nv{b}; __mmask8 neg{b};')
    loads += [
        f'        __m512i xb{b}=_mm512_castpd_si512(_mm512_loadu_pd(x+i+{o})); neg{b}=_mm512_movepi64_mask(xb{b});',
        f'        __m512i ab{b}=_mm512_and_epi64(xb{b},ABSM); __m512i eb{b}=_mm512_and_epi64(_mm512_srli_epi64(ab{b},52),EMASK); nv{b}=_mm512_sub_epi64(eb{b},E1022);',
        f'        __m512i ub{b}=_mm512_or_epi64(_mm512_and_epi64(ab{b},MANT),UEXP); u{b}=_mm512_castsi512_pd(ub{b});',
        f'        ji{b}=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(u{b},VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);',
        f'        ah{b}=_mm512_i32gather_pd(ji{b},tab+0*LUTN,8); bh{b}=_mm512_i32gather_pd(ji{b},tab+1*LUTN,8);',
        f'        al{b}=_mm512_i32gather_pd(ji{b},lotab+0*LUTN,8); bl{b}=_mm512_i32gather_pd(ji{b},lotab+1*LUTN,8);',
        f'        __m512d jd{b}=_mm512_cvtepi32_pd(ji{b}); d{b}=_mm512_fnmadd_pd(jd{b},VIK,u{b});'
    ]
    # Build DD coefficients and Horner from degree 5 down.
    coeff += [
        f'        __m512d s5h{b},s5l{b},s4h{b},s4l{b},s3h{b},s3l{b},s2h{b},s2l{b};',
        f'        x59_scale(bh{b},bl{b},C120,&s5h{b},&s5l{b}); x59_scale(ah{b},al{b},C24,&s4h{b},&s4l{b});',
        f'        x59_scale(bh{b},bl{b},M6,&s3h{b},&s3l{b}); s2h{b}=_mm512_mul_pd(ah{b},MH); s2l{b}=_mm512_mul_pd(al{b},MH);',
        f'        __m512d c5h{b},c5l{b},c4h{b},c4l{b},c3h{b},c3l{b},c2h{b},c2l{b},c1h{b},c1l{b};',
        f'        x59_scale(ah{b},al{b},NC120,&c5h{b},&c5l{b}); x59_scale(bh{b},bl{b},C24,&c4h{b},&c4l{b});',
        f'        x59_scale(ah{b},al{b},P6,&c3h{b},&c3l{b}); c2h{b}=_mm512_mul_pd(bh{b},MH); c2l{b}=_mm512_mul_pd(bl{b},MH);',
        f'        c1h{b}=_mm512_sub_pd(Z,ah{b}); c1l{b}=_mm512_sub_pd(Z,al{b});'
    ]
    hor += [
        f'        sh{b}=s5h{b}; sl{b}=s5l{b}; x59_horner(sh{b},sl{b},d{b},s4h{b},s4l{b},&sh{b},&sl{b});',
        f'        x59_horner(sh{b},sl{b},d{b},s3h{b},s3l{b},&sh{b},&sl{b}); x59_horner(sh{b},sl{b},d{b},s2h{b},s2l{b},&sh{b},&sl{b});',
        f'        x59_horner(sh{b},sl{b},d{b},bh{b},bl{b},&sh{b},&sl{b}); x59_horner(sh{b},sl{b},d{b},ah{b},al{b},&sh{b},&sl{b});',
        f'        ch{b}=c5h{b}; cl{b}=c5l{b}; x59_horner(ch{b},cl{b},d{b},c4h{b},c4l{b},&ch{b},&cl{b});',
        f'        x59_horner(ch{b},cl{b},d{b},c3h{b},c3l{b},&ch{b},&cl{b}); x59_horner(ch{b},cl{b},d{b},c2h{b},c2l{b},&ch{b},&cl{b});',
        f'        x59_horner(ch{b},cl{b},d{b},c1h{b},c1l{b},&ch{b},&cl{b}); x59_horner(ch{b},cl{b},d{b},bh{b},bl{b},&ch{b},&cl{b});'
    ]
for step in range(14):
    for b in range(4):
        dbl += [
            f'        __mmask8 m{step}_{b}=_mm512_cmp_epi64_mask(nv{b},_mm512_set1_epi64({step}),_MM_CMPINT_GT);',
            f'        __m512d nsh{step}_{b},nsl{step}_{b},nch{step}_{b},ncl{step}_{b}; x59_double_state(sh{b},sl{b},ch{b},cl{b},&nsh{step}_{b},&nsl{step}_{b},&nch{step}_{b},&ncl{step}_{b});',
            f'        sh{b}=_mm512_mask_mov_pd(sh{b},m{step}_{b},nsh{step}_{b}); sl{b}=_mm512_mask_mov_pd(sl{b},m{step}_{b},nsl{step}_{b});',
            f'        ch{b}=_mm512_mask_mov_pd(ch{b},m{step}_{b},nch{step}_{b}); cl{b}=_mm512_mask_mov_pd(cl{b},m{step}_{b},ncl{step}_{b});'
        ]
for b in range(4):
    o=8*b
    finish += [f'        __m512d out{b}=_mm512_add_pd(sh{b},sl{b}); out{b}=_mm512_mask_sub_pd(out{b},neg{b},Z,out{b}); _mm512_storeu_pd(out+i+{o},out{b});']

raw=f'''{helpers}
OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawpow2(const s53w_kernel *k,
                                  const double * __restrict x,double * __restrict out,size_t n)
{{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),P6=_mm512_set1_pd(1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0),NC120=_mm512_set1_pd(-1.0/120.0);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff)),MANT=_mm512_set1_epi64((long long)UINT64_C(0x000fffffffffffff)),EMASK=_mm512_set1_epi64(0x7ff),E1022=_mm512_set1_epi64(1022),UEXP=_mm512_set1_epi64((long long)UINT64_C(0x3fe0000000000000));
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64); const double *lotab=(const double *)__builtin_assume_aligned(k->lotab,64);
    size_t i=0; for(;i+32<=n;i+=32){{
{chr(10).join(decls)}
{chr(10).join(loads)}
{chr(10).join(coeff)}
{chr(10).join(hor)}
{chr(10).join(dbl)}
{chr(10).join(finish)}
    }}
    if(i<n) octant_vector_v8_x56_general(k,x+i,out+i,n-i);
}}'''
s=s[:start]+raw+s[end:]
s=s.replace('S53X58_','S53X59_').replace('xeon_x58_raw_pow2_double_g4','xeon_x59_raw_pow2_dd_g4').replace('Xeon_AVX512_X58_raw_pow2_double','Xeon_AVX512_X59_raw_pow2_dd')
Path('bench_sine_53_xeon_x59_build.c').write_text(s)
print('S53X59_BUILD_PASS parent=X58 raw_gt1=1 pi_reduction=0 modulo=0 quadrant=0 parity=0 exact_power2_scale=1 anchor_low_from_103bit_source=1 DD_local_sincos=1 DD_double_angle=1 requires_Arb_regate=1')
