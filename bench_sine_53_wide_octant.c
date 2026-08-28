#define _GNU_SOURCE
#define main s53f2_disabled_main
#include "bench_sine_53_wide_fast2.c"
#undef main
#include <mpfr.h>

/*
 * Cosine-inspired wide binary64 sine dispatcher.
 *
 * Common path:
 *   |x| < 1 : exact same direct Mode-5 unit-domain path, no wide tax.
 *   |x| >=1 : AVX-512 pi/4 octant reduction using int32 quotient arithmetic,
 *             split pi/4 constants, branchless octant folding, then the same
 *             K=8 degree-5 Mode-5 evaluator.
 *
 * Rare guard path:
 *   Only lanes whose cheap remainder is within 2^-32 of an n*pi/4 boundary
 *   are recomputed by the already-proven table-DD scalar reducer from fast2.
 *   This preserves the signed residual near n*pi and handles n*pi/2 and
 *   n*pi/4 boundaries without taxing ordinary lanes.
 */

#define OTRIALS 11
#define OROUNDS 220000
#define OSTRESS 32768
#define PIO4_HI 0x1.921fb54442d18p-1
#define PIO4_LO 0x1.1a62633145c07p-55
#define PIO2_HI 0x1.921fb54442d18p+0
#define PIO2_LO 0x1.1a62633145c07p-54
#define FOUR_OVER_PI 0x1.45f306dc9c883p+0
#define BOUND_TAU 0x1p-32

#if defined(__x86_64__) || defined(__i386__)
#define OVEC __attribute__((target("avx512f,avx512dq,avx2,fma")))

OVEC static inline __mmask8 mask_eq_i32(__m256i a, int v)
{
    __m256i c = _mm256_cmpeq_epi32(a, _mm256_set1_epi32(v));
    return (__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(c));
}

OVEC static inline __m512d poly_i32(const s53w_kernel *k, __m512d y,
                                    __mmask8 signmask)
{
    const __m512d VK = _mm512_set1_pd(KGRID);
    const __m512d VIK = _mm512_set1_pd(INVK);
    const __m512d Z = _mm512_setzero_pd();
    __m512d jd = _mm512_roundscale_pd(_mm512_mul_pd(y, VK),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i ji = _mm512_cvttpd_epi32(jd);
    __m512d d = _mm512_fnmadd_pd(jd, VIK, y);
    __m512d p = _mm512_i32gather_pd(ji,
                    k->tab + (size_t)k->deg * LUTN, 8);
    for (int j = k->deg - 1; j >= 0; j--) {
        __m512d c = _mm512_i32gather_pd(ji,
                        k->tab + (size_t)j * LUTN, 8);
        p = _mm512_fmadd_pd(p, d, c);
    }
    return _mm512_mask_sub_pd(p, signmask, Z, p);
}

OVEC static void octant_vector(const s53w_kernel *k, const double *x,
                               double *out, size_t n)
{
    const __m512d Z = _mm512_setzero_pd();
    const __m512d ONE = _mm512_set1_pd(1.0);
    const __m512d V4OPI = _mm512_set1_pd(FOUR_OVER_PI);
    const __m512d VP4H = _mm512_set1_pd(PIO4_HI);
    const __m512d VP4L = _mm512_set1_pd(PIO4_LO);
    const __m512d VP2H = _mm512_set1_pd(PIO2_HI);
    const __m512d VP2L = _mm512_set1_pd(PIO2_LO);
    const __m512d VTAU = _mm512_set1_pd(BOUND_TAU);
    const __m512i ABSM = _mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));

    for (size_t i = 0; i < n; i += 8) {
        unsigned rem = (unsigned)(n - i);
        __mmask8 active = (__mmask8)(rem >= 8 ? 0xffu : ((1u << rem) - 1u));
        __m512d vx = _mm512_maskz_loadu_pd(active, x + i);
        __mmask8 inneg = (__mmask8)(_mm512_cmp_pd_mask(vx, Z, _CMP_LT_OQ) & active);
        __m512d ax = _mm512_castsi512_pd(
                        _mm512_and_epi64(_mm512_castpd_si512(vx), ABSM));
        __mmask8 unit = (__mmask8)(_mm512_cmp_pd_mask(ax, ONE, _CMP_LT_OQ) & active);

        /* Entirely-unit blocks take literally the winning unit path. */
        if (unit == active) {
            __m512d p = poly_i32(k, ax, inneg);
            _mm512_mask_storeu_pd(out + i, active, p);
            continue;
        }

        __mmask8 wide = (__mmask8)(active & ~unit);

        /* k=floor(|x|*4/pi), represented as 8 int32 lanes.  Our contract
           |x|<=10000 gives k<=12732, so int32 is ample and cheaper than the
           old int64 conversion path. */
        __m512d kf0 = _mm512_mul_pd(ax, V4OPI);
        __m256i ki = _mm512_cvttpd_epi32(kf0);
        __m512d kd = _mm512_cvtepi32_pd(ki);

        /* Cheap split-constant remainder. FMA makes cancellation benign on
           ordinary lanes; only ultra-near boundaries are sent to the DD guard. */
        __m512d w = _mm512_fnmadd_pd(kd, VP4H, ax);
        w = _mm512_fnmadd_pd(kd, VP4L, w);

        __m512d dp4 = _mm512_sub_pd(VP4H, w);
        __mmask8 near0 = _mm512_cmp_pd_mask(w, VTAU, _CMP_LT_OQ);
        __mmask8 near4 = _mm512_cmp_pd_mask(dp4, VTAU, _CMP_LT_OQ);
        __mmask8 guarded = (__mmask8)((near0 | near4) & wide);

        __m256i oi = _mm256_and_si256(ki, _mm256_set1_epi32(7));
        __mmask8 m1 = mask_eq_i32(oi, 1), m2 = mask_eq_i32(oi, 2);
        __mmask8 m3 = mask_eq_i32(oi, 3), m4 = mask_eq_i32(oi, 4);
        __mmask8 m5 = mask_eq_i32(oi, 5), m6 = mask_eq_i32(oi, 6);
        __mmask8 m7 = mask_eq_i32(oi, 7);
        __mmask8 add4 = (__mmask8)((m1 | m5) & wide);
        __mmask8 sub2 = (__mmask8)((m2 | m6) & wide);
        __mmask8 sub4 = (__mmask8)((m3 | m7) & wide);
        __mmask8 wide_neg = (__mmask8)((m4 | m5 | m6 | m7) & wide);

        /* Fold octants to y in [0,pi/2].  The low split is included in every
           cancellation-sensitive reconstruction, exactly the lesson from the
           guarded cosine code. */
        __m512d y = w;
        __m512d ya = _mm512_add_pd(_mm512_add_pd(VP4H, w), VP4L);
        __m512d y2 = _mm512_add_pd(_mm512_sub_pd(VP2H, w), VP2L);
        __m512d y4 = _mm512_add_pd(_mm512_sub_pd(VP4H, w), VP4L);
        y = _mm512_mask_mov_pd(y, add4, ya);
        y = _mm512_mask_mov_pd(y, sub2, y2);
        y = _mm512_mask_mov_pd(y, sub4, y4);

        __mmask8 wide_sign = (__mmask8)(wide_neg ^ inneg);
        __m512d pwide = poly_i32(k, y, wide_sign);

        if (unit) {
            __m512d punit = poly_i32(k, ax, inneg);
            pwide = _mm512_mask_mov_pd(pwide, unit, punit);
        }
        _mm512_mask_storeu_pd(out + i, active, pwide);

        /* Rare cosine-style guarded boundary recovery.  Keep the old table-DD
           reducer only here; it retains the signed tiny residual near n*pi and
           is already validated against Arb on the same wide domain. */
        if (__builtin_expect(guarded != 0, 0)) {
            for (unsigned lane = 0; lane < 8 && i + lane < n; lane++)
                if (guarded & (1u << lane))
                    out[i + lane] = scalar2(k, x[i + lane]);
        }
    }
}
#endif

static void octant_eval(const s53w_kernel *k, const double *x, double *y, size_t n)
{
#if defined(__x86_64__) || defined(__i386__)
    if (hav2()) { octant_vector(k, x, y, n); return; }
#endif
    for (size_t i = 0; i < n; i++) {
        if (fabs(x[i]) < 1.0) {
            double ax = fabs(x[i]);
            long a = lround(ax * KGRID);
            if (a < 0) a = 0; if (a >= LUTN) a = LUTN - 1;
            double d = fma(-(double)a, INVK, ax);
            double p = k->tab[(size_t)k->deg * LUTN + (size_t)a];
            for (int j = k->deg - 1; j >= 0; j--)
                p = fma(p, d, k->tab[(size_t)j * LUTN + (size_t)a]);
            y[i] = signbit(x[i]) ? -p : p;
        } else y[i] = scalar2(k, x[i]);
    }
}

static int guard_count(const double *x, int n)
{
    int c = 0;
    for (int i = 0; i < n; i++) {
        double ax = fabs(x[i]); if (ax < 1.0) continue;
        int k = (int)(ax * FOUR_OVER_PI);
        double w = fma(-(double)k, PIO4_HI, ax);
        w = fma(-(double)k, PIO4_LO, w);
        if (w < BOUND_TAU || (PIO4_HI - w) < BOUND_TAU) c++;
    }
    return c;
}

static int verify_oct(const char *tag, const s53w_kernel *k,
                      const double *x, int n)
{
    double *o = al64((size_t)n * sizeof(double));
    double *in = al64((size_t)n * sizeof(double));
    if (!o || !in) { free(in); free(o); return 0; }
    octant_eval(k, x, o, (size_t)n); vmdSin(n, x, in, VML_HA);
    arb_t ax, ay; arf_t lo, hi; arb_init(ax); arb_init(ay); arf_init(lo); arf_init(hi);
    int uq=0, oe=0, o1=0, ie=0, i1=0; uint64_t om=0, im=0;
    for (int i=0;i<n;i++) {
        arb_set_d(ax,x[i]); arb_sin(ay,ax,256);
        arb_get_lbound_arf(lo,ay,256); arb_get_ubound_arf(hi,ay,256);
        double a=arf_get_d(lo,ARF_RND_NEAR), b=arf_get_d(hi,ARF_RND_NEAR);
        if (dbits(a)!=dbits(b)) continue;
        uq++; uint64_t uo=ulpd(o[i],a), ui=ulpd(in[i],a);
        if (!uo) oe++; if (uo<=1) o1++; if (uo>om) om=uo;
        if (!ui) ie++; if (ui<=1) i1++; if (ui>im) im=ui;
    }
    printf("S53O_VERIFY tag=%s cases=%d unique_ref=%d ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu guarded_lanes=%d reference=Arb256\n",
           tag,n,uq,oe,o1,(unsigned long)om,ie,i1,(unsigned long)im,guard_count(x,n));
    arf_clear(hi); arf_clear(lo); arb_clear(ay); arb_clear(ax); free(in); free(o);
    return uq==n && om<=1;
}

static void make_octant_stress(double *x)
{
    mpfr_t pi,p4,t; mpfr_init2(pi,256); mpfr_init2(p4,256); mpfr_init2(t,256);
    mpfr_const_pi(pi,MPFR_RNDN); mpfr_div_ui(p4,pi,4,MPFR_RNDN);
    for (int i=0;i<OSTRESS;i++) {
        uint64_t h=mix64(UINT64_C(2026082897)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));
        unsigned q=(unsigned)((h>>16)%12733U);
        mpfr_mul_ui(t,p4,q,MPFR_RNDN);
        double b=mpfr_get_d(t,MPFR_RNDN), v=b;
        switch (i&7) {
            case 0: break;
            case 1: v=nextafter(b,INFINITY); break;
            case 2: v=nextafter(b,-INFINITY); break;
            case 3: v=nextafter(nextafter(b,INFINITY),INFINITY); break;
            case 4: v=nextafter(nextafter(b,-INFINITY),-INFINITY); break;
            case 5: v=nextafter(nextafter(nextafter(b,INFINITY),INFINITY),INFINITY); break;
            case 6: v=nextafter(nextafter(nextafter(b,-INFINITY),-INFINITY),-INFINITY); break;
            default: v=b; break;
        }
        if (v<0.0) v=0.0; if (v>10000.0) v=10000.0;
        x[i]=(h&1)?-v:v;
    }
    mpfr_clear(t); mpfr_clear(p4); mpfr_clear(pi);
}

static uint64_t run_oct(const s53w_kernel *k, const double *x, int rounds,
                        volatile double *sink)
{
    double y[CASES]; uint64_t t=now_ns();
    for (int r=0;r<rounds;r++) octant_eval(k,x,y,CASES);
    t=now_ns()-t; *sink+=y[CASES-1]; return t;
}

static int bench_oct(const s53w_kernel *k, const double *x)
{
    if (!hav2()) { printf("S53O_SKIP_TIMING reason=no_avx512\n"); return 0; }
    volatile double sink=0; run_oct(k,x,5000,&sink); run_intel(x,5000,&sink);
    double ot[OTRIALS],it[OTRIALS],calls=(double)OROUNDS*CASES;
    for (int t=0;t<OTRIALS;t++) {
        uint64_t a,b;
        if (t&1) { b=run_intel(x,OROUNDS,&sink); a=run_oct(k,x,OROUNDS,&sink); }
        else { a=run_oct(k,x,OROUNDS,&sink); b=run_intel(x,OROUNDS,&sink); }
        ot[t]=(double)a/calls; it[t]=(double)b/calls;
        printf("S53O_TRIAL trial=%d ours_ns=%.6f intel_ns=%.6f intel_over_ours=%.6fx\n",
               t+1,ot[t],it[t],it[t]/ot[t]);
    }
    qsort(ot,OTRIALS,sizeof(double),cmpd); qsort(it,OTRIALS,sizeof(double),cmpd);
    double om=ot[OTRIALS/2],im=it[OTRIALS/2];
    printf("S53O_RESULT cases=150 terms=2 degree=5 K=%d ours_e2e_ns_per_input=%.6f intel_ha_ns_per_input=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx throughput_advantage_pct=%.3f requested_guarded_lanes=%d reduction=AVX512_pi4_octant_int32_split common_boundary_guard=2^-32 rare_fallback=table_DD_qpi unit_dispatch=direct formula=unchanged_Mode5_secant_spine accuracy_contract=le1ulp all_raw_input_work_included=1 sink=%.17g\n",
           SF_K,om,im,om/im,im/om,(im/om-1.0)*100.0,guard_count(x,CASES),(double)sink);
    return 0;
}

int main(void)
{
    int cpu=pin(); mkl_set_num_threads_local(1);
    printf("S53O_DOMAIN cpu_pin=%d target=binary64_53bit cases=150 bands=0_to_1,1_to_500,1000_to_10000 signs=25pos_25neg_each intel=oneMKL_vmdSin_VML_HA reduction=cosine_style_pi4_octant_guarded formula=unchanged_Mode5_secant_spine\n",cpu);
    if (!redtab2_init()) return 2;
    s53w_kernel *k=kernel_create(2); if (!k) return 3;
    double x[CASES]; make_bench(x);
    if (!verify_oct("requested150",k,x,CASES)) { kernel_destroy(k); redtab2_clear(); return 4; }
    bands2(k,x); /* legacy fast2 band print retained only as a reference trace */

    double *st=al64(STRESS*sizeof(double));
    if (!st) { kernel_destroy(k); redtab2_clear(); return 5; }
    make_stress(st);
    if (!verify_oct("legacy_npi_npi2_stress",k,st,STRESS)) {
        free(st); kernel_destroy(k); redtab2_clear(); return 6;
    }
    free(st);

    double *os=al64(OSTRESS*sizeof(double));
    if (!os) { kernel_destroy(k); redtab2_clear(); return 7; }
    make_octant_stress(os);
    if (!verify_oct("all_npi4_boundary_stress",k,os,OSTRESS)) {
        free(os); kernel_destroy(k); redtab2_clear(); return 8;
    }
    free(os);

    int rc=bench_oct(k,x);
    kernel_destroy(k); redtab2_clear(); flint_cleanup_master(); return rc;
}
