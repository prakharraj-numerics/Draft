#ifndef SINE_17_VECTOR_Q103_H
#define SINE_17_VECTOR_Q103_H

#include <stddef.h>
#include <gmp.h>
#include <flint/flint.h>
#include <flint/fmpq.h>
#include <flint/fmpz.h>

typedef struct sine17q_vec_plan sine17q_vec_plan;

void *sine17q_create_terms(int terms);
void sine17q_destroy(void *ctx);
slong sine17q_target_bits(const void *ctx);
int sine17q_degree(const void *ctx);
int sine17q_prepare(void *ctx, slong *anchor, int *delta_neg,
                    mp_limb_t delta[2], const fmpq_t x);
int sine17q_eval_prepared(void *ctx, mp_limb_t out[2], slong anchor,
                          int delta_neg, const mp_limb_t delta[2]);
int sine17q_eval_fmpq(void *ctx, mp_limb_t out[2], const fmpq_t x);
int sine17q_get_fixed(void *ctx, fmpz_t out, const mp_limb_t in[2]);

sine17q_vec_plan *sine17q_vec_plan_create(void *ctx, const slong *anchor,
                                          const unsigned char *delta_neg,
                                          const mp_limb_t *delta_aos,
                                          size_t count);
void sine17q_vec_plan_destroy(sine17q_vec_plan *p);
size_t sine17q_vec_plan_count(const sine17q_vec_plan *p);
size_t sine17q_vec_plan_padded_count(const sine17q_vec_plan *p);
int sine17q_vec_eval(const sine17q_vec_plan *p, mp_limb_t *out_lo,
                     mp_limb_t *out_hi, unsigned char *sign_masks);
const char *sine17q_vec_backend(void);

#endif
