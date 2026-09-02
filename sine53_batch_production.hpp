#pragma once

/* SINE53 FROZEN production batch routing.

   Resource-elastic routing frozen from exact Intel Xeon 6973P-C evidence:

       n < 4500   -> current SINE53 evaluator; custom2 helper does not exist
       n >= 4500  -> construct/use the exact frozen permanent 2-core scheduler

   Once constructed, the helper remains alive for later large batches.

   Evidence for this lifecycle/routing promotion:
     Full native + SDE benchmark: GitHub Actions run 33693912241.
     Exact Xeon 6973P-C artifacts: shards 17, 44, 67, and 73.
     Tested sizes: 100, 700, 3500, 4500, 5000, 8000, 15000, 20000,
                   25000, 30000, 50000, 1000000, 2000000.

   The promotion changes scheduling/lifecycle only. It does not change SINE53
   mathematics, the supplied evaluator, X50/X67 routing, or the frozen custom2
   implementation.

   Rationale for the 4500 boundary:
     - below 4500, deferring helper construction substantially improves process
       CPU-time efficiency and resource proportionality versus the old eager
       production lifecycle;
     - this intentionally trades some wall speed in part of the 1600-4499 band
       for better CPU-resource efficiency;
     - at 4500 and above, the same frozen custom2 scheduler is used and the
       large wall-speed advantage returns.

   Accuracy in the promoted benchmark remained within an observed maximum
   comparator difference of 2 ULP versus Intel oneMKL vmdSin(..., VML_HA) over
   the tested map. This is a comparator difference, not an absolute-error bound.

   Historical speed-only evidence established the earlier 1600 custom2 crossover
   (runs 33568489785, 33567930913, and 33567193984). That evidence remains valid,
   but the production objective is now explicitly balanced wall performance plus
   CPU-resource proportionality rather than minimum wall time at every small N.

   FROZEN: do not change the <4500 lazy region, the 4500 activation boundary,
   or the frozen scheduler without a new benchmark and explicit promotion.
*/

#include <cstddef>
#include <memory>
#include "sine53_custom_2core_1600_frozen.hpp"

class Sine53BatchProductionFrozen {
public:
    using fn_t = void (*)(double *, const double *, size_t);
    static constexpr size_t kCustom2MinN = 4500;

    explicit Sine53BatchProductionFrozen(fn_t current_eval)
        : current_eval_(current_eval) {}

    Sine53BatchProductionFrozen(const Sine53BatchProductionFrozen&) = delete;
    Sine53BatchProductionFrozen& operator=(const Sine53BatchProductionFrozen&) = delete;

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
