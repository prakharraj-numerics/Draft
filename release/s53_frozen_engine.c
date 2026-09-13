#include <stddef.h>

#ifndef SINE53_GENERATED_SOURCE
#error missing_generated_source
#endif
#ifndef S53_FROZEN_TABLES_HEADER
#error missing_frozen_tables
#endif
#ifndef ENGINE_PREFIX
#error missing_engine_prefix
#endif

#define S53_CAT2(a,b) a##b
#define S53_CAT(a,b) S53_CAT2(a,b)
#define S53_FN(n) S53_CAT(ENGINE_PREFIX,n)

#include SINE53_GENERATED_SOURCE
#include S53_FROZEN_TABLES_HEADER

static s53w_kernel S53_FN(_kobj);
static s53w_kernel *S53_FN(_kptr);

__attribute__((visibility("hidden"))) int S53_FN(_init)(void)
{
    pih2 = (double *)(void *)s53_r0;
    pil2 = (double *)(void *)s53_r1;
    S53_FN(_kobj).ctx = NULL;
    S53_FN(_kobj).terms = 2;
    S53_FN(_kobj).deg = S53_FROZEN_DEG;
    S53_FN(_kobj).tab = (double *)(void *)s53_r2;
    S53_FN(_kptr) = &S53_FN(_kobj);
    return 1;
}

__attribute__((visibility("hidden"))) void S53_FN(_eval)(double *out, const double *in, size_t n)
{
    octant_eval_v8(S53_FN(_kptr), in, out, n);
}

__attribute__((visibility("hidden"))) void S53_FN(_close)(void)
{
    S53_FN(_kptr) = NULL;
    pih2 = NULL;
    pil2 = NULL;
}
