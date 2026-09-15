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

#include C53_GENERATED_SOURCE
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
