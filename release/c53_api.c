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

static int c_ready;

__attribute__((visibility("default"))) int c53_init(void)
{
    if (c_ready) return 0;
    if (!u_init()) return -1;
    if (!w_init()) { u_close(); return -1; }
    c_ready = 1;
    return 0;
}

__attribute__((visibility("default"))) int c53_eval(double *out, const double *in, size_t n, unsigned profile)
{
    if (!c_ready) return -2;
    if ((n && (!out || !in)) || profile > 1U) return -1;
    if (!n) return 0;

    if (profile == 0U) u_eval(out, in, n);
    else w_eval(out, in, n);
    return 0;
}

__attribute__((visibility("default"))) void c53_close(void)
{
    if (!c_ready) return;
    w_close();
    u_close();
    c_ready = 0;
}
