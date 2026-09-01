#define _GNU_SOURCE
#include <stddef.h>

#ifndef SINE53_GENERATED_SOURCE
#error "compile with -DSINE53_GENERATED_SOURCE=\"generated source\""
#endif

#include SINE53_GENERATED_SOURCE

static s53w_kernel *sine53_adapter_kernel = NULL;

int sine53_engine_init(void)
{
    if (!redtab2_init()) return 0;
    sine53_adapter_kernel = kernel_create(2);
    if (!sine53_adapter_kernel) {
        redtab2_clear();
        return 0;
    }
    return 1;
}

void sine53_engine_eval(double *out, const double *in, size_t n)
{
    octant_eval_v8(sine53_adapter_kernel, in, out, n);
}

void sine53_engine_cleanup(void)
{
    if (sine53_adapter_kernel) {
        kernel_destroy(sine53_adapter_kernel);
        sine53_adapter_kernel = NULL;
    }
    redtab2_clear();
    flint_cleanup_master();
}
