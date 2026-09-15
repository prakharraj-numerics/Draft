#include <stddef.h>

#ifndef C53_GENERATED_SOURCE
#error missing_generated_source
#endif
#ifndef C53_FROZEN_TABLES_HEADER
#error missing_frozen_tables
#endif
#ifndef ENGINE_PREFIX
#error missing_engine_prefix
#endif

#define C53_CAT2(a,b) a##b
#define C53_CAT(a,b) C53_CAT2(a,b)
#define C53_FN(n) C53_CAT(ENGINE_PREFIX,n)

/* The benchmark translation units carry dormant non-static main replacements.
   Give each embedded engine private names so the two production objects can be
   linked together without exposing or retaining those harness entry points. */
#define c53_support_disabled_main C53_FN(_s0)
#define c53_wide_support_disabled_main C53_FN(_s1)
#define c53_unit_disabled_main C53_FN(_s2)
#define c53_wide_disabled_main C53_FN(_s3)
#include C53_GENERATED_SOURCE
#undef c53_wide_disabled_main
#undef c53_unit_disabled_main
#undef c53_wide_support_disabled_main
#undef c53_support_disabled_main

#include C53_FROZEN_TABLES_HEADER

static s53w_kernel C53_FN(_kobj);
static s53w_kernel *C53_FN(_kptr);

__attribute__((visibility("hidden"))) int C53_FN(_init)(void)
{
    pih2 = (double *)(void *)c53_r0;
    pil2 = (double *)(void *)c53_r1;
    C53_FN(_kobj).ctx = NULL;
    C53_FN(_kobj).terms = 2;
    C53_FN(_kobj).deg = C53_FROZEN_DEG;
    C53_FN(_kobj).tab = (double *)(void *)c53_r2;
    C53_FN(_kptr) = &C53_FN(_kobj);
    return 1;
}

__attribute__((visibility("hidden"))) void C53_FN(_eval)(double *out, const double *in, size_t n)
{
    octant_eval_v8(C53_FN(_kptr), in, out, n);
}

__attribute__((visibility("hidden"))) void C53_FN(_close)(void)
{
    C53_FN(_kptr) = NULL;
    pih2 = NULL;
    pil2 = NULL;
}
