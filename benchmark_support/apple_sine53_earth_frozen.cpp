// EARTH — direct sine transfer of frozen COS53 EARTH diagnostic kernel from run 34060621422
// Source commit provenance: Draft-cosine 1ba471c9e612d7dd4a5149711e65c4e207f21a6f
//
// SME contract intentionally preserved from EARTH:
//   - streaming mode
//   - M4-class 64-byte SVL => 8 FP64 lanes
//   - terms=1 / degree=3 cubic
//   - identical tuned MH/M6 constants
//   - identical Horner/FMA ordering
//
// Minimal cosine->sine semantic change:
// COS53 EARTH's incoming anchor pair is (cos(a), -sin(a)).
// For sine we need (sin(a), cos(a)), therefore per lane:
//   s0 = -c1
//   s1 =  c0
// The final sign stream is supplied precomputed for sine; the SME sign-XOR
// mechanism itself is unchanged.

#include <arm_sve.h>
#include <cstddef>
#include <cstdint>

static constexpr double EARTH_MH = -0x1.ffffff92c5f94p-2;
static constexpr double EARTH_M6 = -0x1.5555551eb851fp-3;

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

        // Exact COS53 EARTH anchor streams, reinterpreted minimally for sine:
        //   incoming c0 = cos(a), incoming c1 = -sin(a)
        //   sine s0 = sin(a) = -c1, sine s1 = cos(a) = c0
        svfloat64_t cos_a = svld1_f64(pg, c0 + i);
        svfloat64_t neg_sin_a = svld1_f64(pg, c1 + i);
        svfloat64_t a0 = svneg_f64_x(pg, neg_sin_a);
        svfloat64_t a1 = cos_a;

        svfloat64_t c2 = svmul_n_f64_x(pg, a0, EARTH_MH);
        svfloat64_t c3 = svmul_n_f64_x(pg, a1, EARTH_M6);

        svfloat64_t p = svmla_f64_x(pg, c2, c3, de);
        p = svmla_f64_x(pg, a1, p, de);
        p = svmla_f64_x(pg, a0, p, de);

        svuint64_t qb = svld1_u64(pg, signbits + i);
        svuint64_t parity = svand_u64_x(pg, qb, one);
        svuint64_t outsign = svlsl_n_u64_x(pg, parity, 63);
        p = svreinterpret_f64_u64(
            sveor_u64_x(pg, svreinterpret_u64_f64(p), outsign));

        svst1_f64(pg, y + i, p);
    }
}
