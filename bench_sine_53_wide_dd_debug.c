#define main s53w_dd_main_base
#include "bench_sine_53_wide_dd_driver.c"
#undef main

int main(void)
{
    int cpu = pin();
    mkl_set_num_threads_local(1);
    printf("S53W_DEBUG backend=%s cpu_pin=%d\n", backend_dd(), cpu);
    s53w_kernel *k = kernel_create(2);
    if (!k) return 2;
    double x[CASES], vec[CASES];
    make_bench(x);
    eval_e2e_dd(k, x, vec, CASES);
    int same = 0, mism = 0;
    for (int i = 0; i < CASES; i++)
    {
        double sc = eval_scalar_dd_one(k, x[i]);
        if (dbits(sc) == dbits(vec[i])) same++;
        else if (++mism <= 12)
        {
            int64_t q; double rh, rl; int rn;
            reduce_dd_scalar(x[i], &q, &rh, &rl, &rn);
            printf("S53W_MISMATCH i=%d x=%.17g scalar=%.17g vector=%.17g ulp=%lu q=%ld rneg=%d rh=%.17g rl=%.17g\n",
                   i, x[i], sc, vec[i], (unsigned long) ulpd(sc, vec[i]),
                   (long) q, rn, rh, rl);
        }
    }
    printf("S53W_VECTOR_SCALAR bit_identical=%d/%d mismatches=%d\n", same, CASES, mism);
    kernel_destroy(k);
    flint_cleanup_master();
    return mism ? 1 : 0;
}
