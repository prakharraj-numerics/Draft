// EARTH-SINE — SME streaming diagnostic kernel derived from frozen COS53 EARTH.
// Source analogue: Draft-cosine benchmark_support/apple_cos53_earth_frozen.cpp
// Definition: precomputed delta/c0/c1/signbits + SME streaming Horner/sign only.
//
// SINE53 preparation contract:
//   c0[i]       = sin(anchor_i)
//   c1[i]       = cos(anchor_i)
//   delta[i]    = reduced residual from anchor_i
//   signbits[i] bit 0 = final sine sign parity after range reduction,
//                       including the original input sign.
//
// The streaming kernel intentionally does no range reduction, LUT lookup,
// coefficient generation, scheduling, or x-dependent work. As with COS53
// EARTH, those stages are expected to be prepared outside this kernel.
//
// For f=sin, the local derivative recurrence is identical in form to cosine:
//   f''  = -f,  f''' = -f'
// hence c2=-c0/2 and c3=-c1/6 and the same cubic Horner structure applies.

#include <arm_sve.h>
#include <cstddef>
#include <cstdint>

static constexpr double EARTH_SINE_MH = -0x1.ffffff92c5f94p-2;
static constexpr double EARTH_SINE_M6 = -0x1.5555551eb851fp-3;

extern "C" void sine53_earth_stream(
    const double* x,
    double* y,
    size_t n,
    const double* delta,
    const double* c0,
    const double* c1,
    const uint64_t* signbits) __arm_streaming;

extern "C" void sine53_earth_stream(
    const double* x,
    double* y,
    size_t n,
    const double* delta,
    const double* c0,
    const double* c1,
    const uint64_t* signbits) __arm_streaming {
    (void)x;
    const svuint64_t one = svdup_u64(1);

    for (size_t i = 0; i < n; i += 8) {
        svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)n);

        svfloat64_t de = svld1_f64(pg, delta + i);
        svfloat64_t a0 = svld1_f64(pg, c0 + i);
        svfloat64_t a1 = svld1_f64(pg, c1 + i);

        svfloat64_t c2 = svmul_n_f64_x(pg, a0, EARTH_SINE_MH);
        svfloat64_t c3 = svmul_n_f64_x(pg, a1, EARTH_SINE_M6);

        svfloat64_t p = svmla_f64_x(pg, c2, c3, de);
        p = svmla_f64_x(pg, a1, p, de);
        p = svmla_f64_x(pg, a0, p, de);

        svuint64_t sb = svld1_u64(pg, signbits + i);
        svuint64_t parity = svand_u64_x(pg, sb, one);
        svuint64_t outsign = svlsl_n_u64_x(pg, parity, 63);
        p = svreinterpret_f64_u64(
            sveor_u64_x(pg, svreinterpret_u64_f64(p), outsign));

        svst1_f64(pg, y + i, p);
    }
}
