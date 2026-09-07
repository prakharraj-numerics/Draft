#pragma once

#if !defined(__APPLE__)
#error "apple_sine53_earth_prepare.hpp is Apple-only"
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

/* Genuine preparation front-end for the Apple SINE53 EARTH SME kernel.
 *
 * Streams produced for apple_sine53_earth_frozen.cpp:
 *   delta[i]    local displacement from the 1/256 anchor
 *   c0[i]       sin(anchor)
 *   c1[i]       cos(anchor)
 *   signbits[i] bit 0 = final sign of sin(x)
 *
 * Reduction follows the proven Apple COS53 nearest-pi scheme:
 *   q  = nearest(|x|/pi)
 *   r  = |x| - q*pi using a hi/lo q*pi table
 *   rr = |r|
 *   j  = nearest(rr*256)
 *   d  = rr - j/256
 *
 * Because sin(x)=(-1)^q sin(r) and sin is odd, final sign is
 *   sign(x) XOR (q&1) XOR (r<0).
 */

namespace apple_sine53_earth {

static constexpr double kInvPi = 0x1.45f306dc9c883p-2;
static constexpr double kGrid = 256.0;
static constexpr double kInvGrid = 1.0 / 256.0;

struct PrepareTables {
    const double* sin_tab;
    const double* cos_tab;
    std::size_t lutn;
    const double* pih;
    const double* pil;
    std::size_t redn;
};

static inline void exact_two_sum(double a, double b,
                                 double& hi, double& lo) noexcept {
    hi = a + b;
    const double bv = hi - a;
    const double av = hi - bv;
    const double br = b - bv;
    const double ar = a - av;
    lo = ar + br;
}

static inline std::size_t prepare(const double* x,
                                  std::size_t n,
                                  const PrepareTables& t,
                                  double* delta,
                                  double* c0,
                                  double* c1,
                                  std::uint64_t* signbits) noexcept {
    std::size_t bad = 0;

    for (std::size_t i = 0; i < n; ++i) {
        const double xi = x[i];
        const bool input_neg = std::signbit(xi);
        const double ax = std::fabs(xi);

        if (!std::isfinite(ax)) {
            ++bad;
            delta[i] = c0[i] = c1[i] = std::numeric_limits<double>::quiet_NaN();
            signbits[i] = 0;
            continue;
        }

        const long long qll = std::llrint(ax * kInvPi);
        if (qll < 0 || static_cast<std::size_t>(qll) >= t.redn) {
            ++bad;
            delta[i] = c0[i] = c1[i] = std::numeric_limits<double>::quiet_NaN();
            signbits[i] = 0;
            continue;
        }
        const std::size_t q = static_cast<std::size_t>(qll);

        const double s = ax - t.pih[q];
        const double b = -t.pil[q];
        double rh, rl;
        exact_two_sum(s, b, rh, rl);

        const double rs = rh + rl;
        const bool residual_neg = std::signbit(rs) && rs != 0.0;
        if (residual_neg) {
            rh = -rh;
            rl = -rl;
        }
        const double rr = rh + rl;

        long long jll = std::llrint(rr * kGrid);
        if (jll < 0) jll = 0;
        if (static_cast<std::size_t>(jll) >= t.lutn)
            jll = static_cast<long long>(t.lutn - 1);
        const std::size_t j = static_cast<std::size_t>(jll);

        const double jd = static_cast<double>(jll);
        delta[i] = (rh - jd * kInvGrid) + rl;
        c0[i] = t.sin_tab[j];
        c1[i] = t.cos_tab[j];
        signbits[i] = static_cast<std::uint64_t>(input_neg ^
                                                 ((q & 1u) != 0) ^
                                                 residual_neg);
    }

    return bad;
}

} // namespace apple_sine53_earth
