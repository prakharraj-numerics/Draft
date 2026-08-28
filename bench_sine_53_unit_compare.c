#define _GNU_SOURCE
#define main s53f2_disabled_main
#include "bench_sine_53_wide_fast2.c"
#undef main

/* Pure [0,1) comparison on the exact deterministic 150-input corpus used by
   the original 53-bit experiment.  Compile this translation unit with K=8
   (SF_K=8, KGRID=256, INVK=1/256).  The same Mode-5 degree-5 coefficient
   machinery is used in both our paths:
     unit_direct: no pi reduction, because the domain contract says x in [0,1)
     wide_unified: the current universal table-DD wide reducer + Mode-5
   Intel receives the same raw x[] through vmdSin(...,VML_HA). */

#define UTRIALS 11
#define UROUNDS 300000

static void make_unit150(double *x)
{
    for (int i = 0; i < CASES; i++) {
        uint64_t h = mix64(UINT64_C(2026082879) +
                           (uint64_t)i * UINT64_C(0x9e3779b97f4a7c15));
        uint64_t m = h & ((UINT64_C(1) << 53) - 1U);
        if (!m) m = 1;
        x[i] = ldexp((double)m, -53);
        if (x[i] >= 1.0) x[i] = nextafter(1.0, 0.0);
    }
}

static inline double unit_scalar_one(const s53w_kernel *k, double x)
{
    long a = lround(x * KGRID);
    if (a < 0) a = 0;
    if (a >= LUTN) a = LUTN - 1;
    double d = fma(-(double)a, INVK, x);
    double p = k->tab[(size_t)k->deg * LUTN + (size_t)a];
    for (int j = k->deg - 1; j >= 0; j--)
        p = fma(p, d, k->tab[(size_t)j * LUTN + (size_t)a]);
    return p;
}

#if defined(__x86_64__) || defined(__i386__)
#define UVEC __attribute__((target("avx512f,avx512dq,fma")))
UVEC static void unit_vector(const s53w_kernel *k, const double *x, double *y, size_t n)
{
    const __m512d VK = _mm512_set1_pd(KGRID), VIK = _mm512_set1_pd(INVK);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d vx = _mm512_loadu_pd(x + i);
        __m512d jd = _mm512_roundscale_pd(_mm512_mul_pd(vx, VK),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512i ji = _mm512_cvttpd_epi64(jd);
        __m512d d = _mm512_fnmadd_pd(jd, VIK, vx);
        __m512d p = _mm512_i64gather_pd(ji,
                        k->tab + (size_t)k->deg * LUTN, 8);
        for (int j = k->deg - 1; j >= 0; j--) {
            __m512d c = _mm512_i64gather_pd(ji,
                            k->tab + (size_t)j * LUTN, 8);
            p = _mm512_fmadd_pd(p, d, c);
        }
        _mm512_storeu_pd(y + i, p);
    }
    for (; i < n; i++) y[i] = unit_scalar_one(k, x[i]);
}
#endif

static void unit_eval(const s53w_kernel *k, const double *x, double *y, size_t n)
{
#if defined(__x86_64__) || defined(__i386__)
    if (hav2()) { unit_vector(k, x, y, n); return; }
#endif
    for (size_t i = 0; i < n; i++) y[i] = unit_scalar_one(k, x[i]);
}

static int verify_outputs(const char *tag, const double *x, const double *o, int n)
{
    arb_t ax, ay; arf_t lo, hi;
    arb_init(ax); arb_init(ay); arf_init(lo); arf_init(hi);
    int uq = 0, exact = 0, le1 = 0; uint64_t mx = 0;
    for (int i = 0; i < n; i++) {
        arb_set_d(ax, x[i]); arb_sin(ay, ax, 256);
        arb_get_lbound_arf(lo, ay, 256); arb_get_ubound_arf(hi, ay, 256);
        double a = arf_get_d(lo, ARF_RND_NEAR), b = arf_get_d(hi, ARF_RND_NEAR);
        if (dbits(a) != dbits(b)) continue;
        uq++; uint64_t u = ulpd(o[i], a);
        if (!u) exact++; if (u <= 1) le1++; if (u > mx) mx = u;
    }
    printf("S53U_VERIFY tag=%s cases=%d unique_ref=%d exact=%d le1ulp=%d max_ulp=%lu reference=Arb256\n",
           tag, n, uq, exact, le1, (unsigned long)mx);
    arf_clear(hi); arf_clear(lo); arb_clear(ay); arb_clear(ax);
    return uq == n && mx <= 1;
}

static uint64_t run_unit(const s53w_kernel *k, const double *x, int rounds, volatile double *sink)
{
    double y[CASES]; uint64_t t = now_ns();
    for (int r = 0; r < rounds; r++) unit_eval(k, x, y, CASES);
    t = now_ns() - t; *sink += y[CASES-1]; return t;
}
static uint64_t run_wide_u(const s53w_kernel *k, const double *x, int rounds, volatile double *sink)
{
    double y[CASES]; uint64_t t = now_ns();
    for (int r = 0; r < rounds; r++) eval2(k, x, y, CASES);
    t = now_ns() - t; *sink += y[CASES-1]; return t;
}

int main(void)
{
    int cpu = pin(); mkl_set_num_threads_local(1);
    printf("S53U_DOMAIN domain=[0,1) cases=150 target=binary64_53bit K=%d KGRID=%.0f cpu_pin=%d intel=oneMKL_vmdSin_VML_HA formula=unchanged_Mode5_secant_spine\n",
           SF_K, KGRID, cpu);
    if (!redtab2_init()) return 2;
    s53w_kernel *k = kernel_create(2); if (!k) return 3;
    double x[CASES], u[CASES], w[CASES], in[CASES]; make_unit150(x);
    unit_eval(k, x, u, CASES); eval2(k, x, w, CASES); vmdSin(CASES, x, in, VML_HA);
    int uw_same = 0;
    for (int i = 0; i < CASES; i++) if (dbits(u[i]) == dbits(w[i])) uw_same++;
    printf("S53U_PATH_CHECK unit_vs_wide_bit_identical=%d/150\n", uw_same);
    if (!verify_outputs("unit_direct", x, u, CASES) ||
        !verify_outputs("wide_unified", x, w, CASES) ||
        !verify_outputs("intel_ha", x, in, CASES)) {
        kernel_destroy(k); redtab2_clear(); return 4;
    }
    if (!hav2()) {
        printf("S53U_SKIP_TIMING reason=no_avx512\n");
        kernel_destroy(k); redtab2_clear(); flint_cleanup_master(); return 0;
    }
    volatile double sink = 0;
    run_unit(k, x, 5000, &sink); run_wide_u(k, x, 5000, &sink); run_intel(x, 5000, &sink);
    double ut[UTRIALS], wt[UTRIALS], it[UTRIALS];
    double calls = (double)UROUNDS * CASES;
    for (int t = 0; t < UTRIALS; t++) {
        uint64_t a, b, c;
        if (t & 1) {
            c = run_intel(x, UROUNDS, &sink);
            b = run_wide_u(k, x, UROUNDS, &sink);
            a = run_unit(k, x, UROUNDS, &sink);
        } else {
            a = run_unit(k, x, UROUNDS, &sink);
            b = run_wide_u(k, x, UROUNDS, &sink);
            c = run_intel(x, UROUNDS, &sink);
        }
        ut[t] = (double)a / calls; wt[t] = (double)b / calls; it[t] = (double)c / calls;
        printf("S53U_TRIAL trial=%d unit_ns=%.6f wide_ns=%.6f intel_ns=%.6f intel_over_unit=%.6fx intel_over_wide=%.6fx\n",
               t+1, ut[t], wt[t], it[t], it[t]/ut[t], it[t]/wt[t]);
    }
    qsort(ut, UTRIALS, sizeof(double), cmpd); qsort(wt, UTRIALS, sizeof(double), cmpd); qsort(it, UTRIALS, sizeof(double), cmpd);
    double um = ut[UTRIALS/2], wm = wt[UTRIALS/2], im = it[UTRIALS/2];
    printf("S53U_RESULT cases=150 K=%d terms=2 degree=5 unit_direct_ns=%.6f wide_unified_ns=%.6f intel_ha_ns=%.6f intel_over_unit=%.6fx unit_throughput_adv_pct=%.3f intel_over_wide=%.6fx wide_vs_unit_overhead=%.6fx formula=unchanged_Mode5_secant_spine accuracy_contract=le1ulp all_raw_input_work_included=1 sink=%.17g\n",
           SF_K, um, wm, im, im/um, (im/um-1.0)*100.0, im/wm, wm/um, (double)sink);
    kernel_destroy(k); redtab2_clear(); flint_cleanup_master(); return 0;
}
