#ifndef SINE_LOWPREC_SPINE_FUSED_H
#define SINE_LOWPREC_SPINE_FUSED_H

#include <flint/flint.h>
#include <flint/fmpq.h>
#include <flint/fmpz.h>
#include <flint/arf.h>
#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

void *sine_fixed_create(unsigned digits);
void sine_fixed_destroy(void *ctx);
slong sine_fixed_nlimbs(const void *ctx);
slong sine_fixed_B(const void *ctx);
slong sine_fixed_target_bits(const void *ctx);
int sine_fixed_degree(const void *ctx);
int sine_fixed_prepare(void *ctx, slong *anchor, int *delta_neg,
                       mp_limb_t *delta, const fmpq_t x);
int sine_fixed_eval_prepared(void *ctx, mp_limb_t *out,
                             slong anchor, int delta_neg,
                             const mp_limb_t *delta);
int sine_fixed_eval_fmpq(void *ctx, mp_limb_t *out, const fmpq_t x);
int sine_fixed_get_arf(void *ctx, arf_t out, const mp_limb_t *in);
int sine_fixed_get_fixed(void *ctx, fmpz_t out, const mp_limb_t *in);

#ifdef __cplusplus
}
#endif

#endif
