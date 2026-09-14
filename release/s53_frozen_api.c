#include <stddef.h>
#include <string.h>

int u_init(void);
void u_eval(double *, const double *, size_t);
void u_close(void);
int w_init(void);
void w_eval(double *, const double *, size_t);
void w_close(void);

__attribute__((visibility("hidden"), noinline))
void *_intel_fast_memset(void *dst, int c, size_t n)
{
    static void *(*volatile fn)(void *, int, size_t) = memset;
    return fn(dst, c, n);
}

static int ready;

__attribute__((visibility("default"))) int s53_init(void)
{
    if (ready) return 1;
    if (!u_init()) return 0;
    if (!w_init()) { u_close(); return 0; }
    ready = 1;
    return 1;
}

__attribute__((visibility("default"))) void s53_eval(double *out, const double *in, size_t n, unsigned profile)
{
    if (profile == 0) u_eval(out, in, n);
    else w_eval(out, in, n);
}

__attribute__((visibility("default"))) void s53_close(void)
{
    if (!ready) return;
    w_close();
    u_close();
    ready = 0;
}
