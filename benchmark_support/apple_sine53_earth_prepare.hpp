#pragma once

#if !defined(__APPLE__)
#error "apple_sine53_earth_prepare.hpp is Apple-only"
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

/* Literal sine transfer of the frozen COS53 PLUTO/EARTH front-end.
 *
 * Preserved from cosine:
 *   INVPI, KGRID=1280, split reciprocal, PI_P1/PI_P2 reduction,
 *   2^52 magic rounding for q and j, 2009-entry AoS LUT contract.
 *
 * The cosine AoS table stores (cos(a), -sin(a)).  Therefore the sine
 * cubic coefficients at the same anchor are obtained without regeneration:
 *   sine c0 =  sin(a) = -cosine_c1
 *   sine c1 =  cos(a) =  cosine_c0
 *
 * Sine's final sign is sign(x) XOR q-parity XOR sign(reduced residual).
 * Rare j>=2009 lanes use the same scalar-repair philosophy as frozen cosine.
 */

namespace apple_sine53_earth {

static constexpr double INVPI = 0x1.45f306dc9c883p-2;
static constexpr double KGRID = 1280.0;
static constexpr double NINVK_HI = -0x1.999999999999ap-11;
static constexpr double NINVK_LO =  0x1.999999999999ap-65;
static constexpr double PI_P1 = 0x1.921fb54442000p+1;
static constexpr double PI_P2 = 0x1.a308d313198a3p-40;
static constexpr double MAGIC = 0x1p52;
static constexpr std::size_t LUTN = 2009;
static constexpr std::uint64_t JMASK = (UINT64_C(1) << 52) - 1;

struct PrepareTables {
    const double* cosine_aos; // [cos(a0),-sin(a0), cos(a1),-sin(a1), ...]
    std::size_t lutn;
};

static inline std::uint64_t bits(double x) noexcept {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

static inline std::size_t prepare(const double* x,
                                  std::size_t n,
                                  const PrepareTables& tab,
                                  double* delta,
                                  double* c0,
                                  double* c1,
                                  std::uint64_t* signbits) noexcept {
    std::size_t repairs = 0;
    constexpr std::uint64_t SIGN = UINT64_C(0x8000000000000000);

    for (std::size_t i = 0; i < n; ++i) {
        const double xi = x[i];
        const std::uint64_t xsign = bits(xi) & SIGN;
        const double ax = std::fabs(xi);

        if (!std::isfinite(ax)) {
            ++repairs;
            delta[i] = 0.0;
            c0[i] = std::sin(xi);
            c1[i] = 0.0;
            signbits[i] = 0;
            continue;
        }

        const double qscaled = ax * INVPI;
        const double qmagic = qscaled + MAGIC;
        const double qd = qmagic - MAGIC;
        const std::uint64_t qbits = bits(qmagic);

        const double qp1 = qd * PI_P1;
        const double t = ax - qp1;
        const double rh = std::fma(qd, -PI_P2, t);
        const double d = t - rh;
        const double rl = std::fma(qd, -PI_P2, d);

        const std::uint64_t rsign = bits(rh) & SIGN;
        const double ah = std::fabs(rh);
        const double al = rsign ? -rl : rl;

        const double jscaled = ah * KGRID;
        const double jmagic = jscaled + MAGIC;
        const double jd = jmagic - MAGIC;
        const std::uint64_t j = bits(jmagic) & JMASK;

        if (j >= tab.lutn) {
            // Same rare scalar repair as the frozen cosine PLUTO path.
            ++repairs;
            delta[i] = 0.0;
            c0[i] = std::sin(xi);
            c1[i] = 0.0;
            signbits[i] = 0;
            continue;
        }

        double de = std::fma(jd, NINVK_HI, ah);
        de = std::fma(jd, NINVK_LO, de);
        de += al;
        delta[i] = de;

        const double cos_c0 = tab.cosine_aos[2*j + 0];
        const double cos_c1 = tab.cosine_aos[2*j + 1];
        c0[i] = -cos_c1; // sin(anchor)
        c1[i] =  cos_c0; // cos(anchor)

        const std::uint64_t parity = qbits & UINT64_C(1);
        signbits[i] = ((xsign != 0) ^ (parity != 0) ^ (rsign != 0)) ? 1u : 0u;
    }

    return repairs;
}

} // namespace apple_sine53_earth
