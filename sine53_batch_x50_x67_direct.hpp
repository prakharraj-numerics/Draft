#pragma once

/* SINE53 X50/X67 direct copied batch path.
 *
 * The frozen Intel/Xeon X50 and X67 source files are NOT modified:
 *   - sine53_x50_unit_production.c
 *   - sine53_x67_wide_production.c
 *
 * This copied outer batch layer always calls the supplied current evaluator
 * directly. The supplied evaluator continues to provide the existing X50/X67
 * mathematics and routing for every batch size.
 */

#include <cstddef>

class Sine53BatchX50X67Direct {
public:
    using fn_t = void (*)(double *, const double *, size_t);

    explicit Sine53BatchX50X67Direct(fn_t current_eval)
        : current_eval_(current_eval) {}

    Sine53BatchX50X67Direct(const Sine53BatchX50X67Direct&) = delete;
    Sine53BatchX50X67Direct& operator=(const Sine53BatchX50X67Direct&) = delete;

    void run(double *out, const double *in, size_t n) const {
        current_eval_(out, in, n);
    }

private:
    fn_t current_eval_;
};
