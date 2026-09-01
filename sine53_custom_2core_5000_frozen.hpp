#pragma once

/* SINE53 FROZEN permanent 2-core scheduler.

   Production threshold frozen from exact Intel Xeon 6973P-C evidence:
     - three-way benchmark run 33567193984
     - exact Xeon shards 37 and 25

   Validated behavior relevant to the frozen production rule:
     - custom2 output was bit-identical to current SINE53 over all 72 tested cells
       on both exact-Xeon artifacts
     - at every tested point from n=5000 through n=4000000, custom2 beat current
       SINE53 in all six requested sign/range cells on both shards
     - at n=1200, current SINE53 remained faster than custom2
     - n=250 is a strong custom2 loss and remains on the current evaluator

   Scheduling only. This file does not alter sine mathematics, constants, range
   reduction, polynomial evaluation, or output semantics.

   Execution model on the validated 2-physical-core Xeon runner:
     - caller thread pinned to CPU0
     - one permanent helper pinned to CPU2
     - work split on a 32-double boundary
     - release/acquire generation handoff and completion counter
     - no queue, work stealing, task allocation, or per-call thread creation

   The <64-value fallback is retained: fewer than two complete 32-value blocks
   execute on the caller only.

   FROZEN: do not modify. Future experiments must use a new candidate file.
*/

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

class Sine53CustomPermanent2Core5000Frozen {
public:
    using fn_t = void (*)(double *, const double *, size_t);

    explicit Sine53CustomPermanent2Core5000Frozen(fn_t fn)
        : generation_(0), completed_(0), stop_(false),
          out_(nullptr), in_(nullptr), n2_(0), fn_(fn) {
        pin_current_thread(0);
        helper_ = std::thread([this] { helper_loop(); });
        while (!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }

    ~Sine53CustomPermanent2Core5000Frozen() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }

    Sine53CustomPermanent2Core5000Frozen(const Sine53CustomPermanent2Core5000Frozen&) = delete;
    Sine53CustomPermanent2Core5000Frozen& operator=(const Sine53CustomPermanent2Core5000Frozen&) = delete;

    void run(double *out, const double *in, size_t n) {
        if (!n) return;
        const size_t full32 = n / 32;
        if (full32 < 2) {
            fn_(out, in, n);
            return;
        }

        const size_t split_blocks = full32 / 2;
        const size_t split = split_blocks * 32;
        if (split == 0 || split >= n) {
            fn_(out, in, n);
            return;
        }

        out_ = out + split;
        in_ = in + split;
        n2_ = n - split;

        const uint64_t g = generation_.fetch_add(1, std::memory_order_release) + 1;
        fn_(out, in, split);
        while (completed_.load(std::memory_order_acquire) != g) _mm_pause();
    }

private:
    static void pin_current_thread(int cpu) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }

    void helper_loop() {
        pin_current_thread(2);
        helper_ready_.store(true, std::memory_order_release);
        uint64_t seen = generation_.load(std::memory_order_relaxed);
        for (;;) {
            uint64_t g;
            while ((g = generation_.load(std::memory_order_acquire)) == seen) _mm_pause();
            seen = g;
            if (stop_.load(std::memory_order_relaxed)) return;

            double *out = out_;
            const double *in = in_;
            const size_t n = n2_;
            fn_(out, in, n);
            completed_.store(g, std::memory_order_release);
        }
    }

    std::thread helper_;
    alignas(64) std::atomic<uint64_t> generation_;
    alignas(64) std::atomic<uint64_t> completed_;
    alignas(64) std::atomic<bool> helper_ready_{false};
    std::atomic<bool> stop_;

    double *out_;
    const double *in_;
    size_t n2_;
    fn_t fn_;
};
