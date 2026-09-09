#include <math.h>
#include <stddef.h>

__attribute__((noinline))
void svml_sin_high(const double *restrict x, double *restrict y, size_t n)
{
#pragma clang loop vectorize(enable) interleave(enable)
    for (size_t i = 0; i < n; ++i)
        y[i] = sin(x[i]);
}
