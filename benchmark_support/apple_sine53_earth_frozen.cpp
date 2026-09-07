// EARTH-SINE — SME streaming diagnostic kernel derived from frozen COS53 EARTH.
// Source analogue: Draft-cosine benchmark_support/apple_cos53_earth_frozen.cpp
// Definition: precomputed delta/c0/c1/signbits + SME streaming Horner/sign only.
//
// Unlike COS53 EARTH (terms=1 / degree=3), frozen SINE53 production is
// terms=2 / degree=5.  This kernel therefore preserves the SINE53 polynomial
// order exactly instead of copying cosine's cubic approximation.
//
// SINE53 preparation contract:
//   c0[i], c1[i] = frozen SINE53 anchor coefficient planes 0 and 1
//   delta[i]      = reduced residual from the selected 1/256 anchor
//   signbits[i] bit 0 = final sine sign parity after range reduction,
//                       including the original input sign.
//
// Frozen SINE53 recurrence/order, matching mode5_poly_x11:
//   c2 = -c0/2
//   c3 = -c1/6
//   c4 =  c0/24
//   c5 =  c1/120
//   p  = (((((c5*d+c4)*d+c3)*d+c2)*d+c1)*d+c0)

#include <arm_sve.h>
#include <cstddef>
#include <cstdint>

static constexpr double EARTH_SINE_MH   = -0.5;
static constexpr double EARTH_SINE_M6   = -1.0 / 6.0;
static constexpr double EARTH_SINE_C24  =  1.0 / 24.0;
static constexpr double EARTH_SINE_C120 =  1.0 / 120.0;

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
        svfloat64_t c4 = svmul_n_f64_x(pg, a0, EARTH_SINE_C24);
        svfloat64_t c5 = svmul_n_f64_x(pg, a1, EARTH_SINE_C120);

        svfloat64_t p = svmla_f64_x(pg, c4, c5, de);
        p = svmla_f64_x(pg, c3, p, de);
        p = svmla_f64_x(pg, c2, p, de);
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
