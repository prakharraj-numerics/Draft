#pragma once

/* EXPERIMENTAL ONLY: SINE53 lazy-4500 routing candidate.

   Frozen production remains untouched at its 1600 activation boundary.

   Temporary experimental routing:
       n < 4500   -> current SINE53 evaluator; custom2 helper does not exist
       n >= 4500  -> construct/use exact frozen permanent 2-core scheduler

   Mathematics and the frozen custom2 implementation are unchanged.
*/

#include <cstddef>
#include <memory>
#include "sine53_custom_2core_1600_frozen.hpp"

class Sine53BatchLazy4500Candidate {
public:
    using fn_t = void (*)(double *, const double *, size_t);
    static constexpr size_t kCustom2MinN = 4500;

    explicit Sine53BatchLazy4500Candidate(fn_t current_eval)
        : current_eval_(current_eval) {}

    Sine53BatchLazy4500Candidate(const Sine53BatchLazy4500Candidate&) = delete;
    Sine53BatchLazy4500Candidate& operator=(const Sine53BatchLazy4500Candidate&) = delete;

    void run(double *out, const double *in, size_t n) {
        if (n < kCustom2MinN) {
            current_eval_(out, in, n);
            return;
        }
        if (!custom2_) {
            custom2_ = std::make_unique<Sine53CustomPermanent2Core1600Frozen>(current_eval_);
        }
        custom2_->run(out, in, n);
    }

    bool helper_started() const noexcept { return static_cast<bool>(custom2_); }

private:
    fn_t current_eval_;
    std::unique_ptr<Sine53CustomPermanent2Core1600Frozen> custom2_;
};
