#pragma once

/* SINE53 X50/X67 copy path with custom2 removed.
 *
 * The frozen Intel/Xeon X50 and X67 source files are NOT modified:
 *   - sine53_x50_unit_production.c
 *   - sine53_x67_wide_production.c
 *
 * This is the copied outer batch-routing layer only.  The supplied evaluator
 * continues to provide the existing X50/X67 math/routing.  Unlike
 * sine53_batch_production.hpp, this copy never constructs or calls custom2.
 */

#include <cstddef>

class Sine53BatchX50X67NoCustom2 {
public:
    using fn_t = void (*)(double *, const double *, size_t);

    explicit Sine53BatchX50X67NoCustom2(fn_t current_eval)
        : current_eval_(current_eval) {}

    Sine53BatchX50X67NoCustom2(const Sine53BatchX50X67NoCustom2&) = delete;
    Sine53BatchX50X67NoCustom2& operator=(const Sine53BatchX50X67NoCustom2&) = delete;

    void run(double *out, const double *in, size_t n) const {
        current_eval_(out, in, n);
    }

private:
    fn_t current_eval_;
};
