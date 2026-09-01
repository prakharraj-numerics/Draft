#pragma once

/* SINE53 FROZEN production batch routing.

   Routing rule frozen from exact Intel Xeon 6973P-C benchmark evidence:

       n < 5000   -> current SINE53 evaluator
       n >= 5000  -> frozen custom permanent 2-core scheduler

   Evidence:
     GitHub Actions run 33567193984, exact Xeon shards 37 and 25.

     custom2 was bit-identical to current SINE53 over all 72 requested cells on
     both exact-Xeon artifacts. At every tested point from n=5000 through
     n=4000000, custom2 beat current SINE53 in all six sign/range cells on both
     shards. At n=1200, current SINE53 was still faster than custom2.

   This dispatcher changes scheduling only. The supplied evaluator remains the
   same current SINE53 implementation (X50/X67 routing is external to this
   scheduler and remains unchanged).

   FROZEN: do not change the 5000 threshold or scheduler in this file without a
   new benchmark and an explicit production promotion.
*/

#include <cstddef>
#include "sine53_custom_2core_5000_frozen.hpp"

class Sine53BatchProductionFrozen {
public:
    using fn_t = void (*)(double *, const double *, size_t);
    static constexpr size_t kCustom2MinN = 5000;

    explicit Sine53BatchProductionFrozen(fn_t current_eval)
        : current_eval_(current_eval), custom2_(current_eval) {}

    Sine53BatchProductionFrozen(const Sine53BatchProductionFrozen&) = delete;
    Sine53BatchProductionFrozen& operator=(const Sine53BatchProductionFrozen&) = delete;

    void run(double *out, const double *in, size_t n) {
        if (n >= kCustom2MinN) {
            custom2_.run(out, in, n);
        } else {
            current_eval_(out, in, n);
        }
    }

private:
    fn_t current_eval_;
    Sine53CustomPermanent2Core5000Frozen custom2_;
};
