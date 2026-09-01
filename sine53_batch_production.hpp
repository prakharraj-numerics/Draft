#pragma once

/* SINE53 FROZEN production batch routing.

   Routing rule frozen from exact Intel Xeon 6973P-C benchmark evidence:

       n < 1600   -> current SINE53 evaluator
       n >= 1600  -> frozen custom permanent 2-core scheduler

   Evidence:
     Focused boundary run 33568489785, exact Xeon shard 61:
       - n=1600 through 1900: custom2 beat current in all six requested
         sign/range cells at every tested size
       - custom2 remained bit-identical to current on the tested grid

     Focused boundary run 33567930913, exact Xeon shard 50:
       - n=1500: current SINE53 remained faster on the six-case average
       - n=2000 through 4500: custom2 beat current in all six requested
         sign/range cells at every tested size
       - custom2 remained bit-identical to current on the boundary grid

     Broad run 33567193984, exact Xeon shards 37 and 25:
       - n=5000 through 4000000: custom2 beat current in all six requested
         sign/range cells at every tested size on both shards
       - custom2 remained bit-identical to current over the tested cells

   This dispatcher changes scheduling only. The supplied evaluator remains the
   same current SINE53 implementation (X50/X67 routing is external to this
   scheduler and remains unchanged).

   FROZEN: do not change the 1600 threshold or scheduler in this file without a
   new benchmark and an explicit production promotion.
*/

#include <cstddef>
#include "sine53_custom_2core_1600_frozen.hpp"

class Sine53BatchProductionFrozen {
public:
    using fn_t = void (*)(double *, const double *, size_t);
    static constexpr size_t kCustom2MinN = 1600;

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
    Sine53CustomPermanent2Core1600Frozen custom2_;
};
