# Sine53 Xeon campaign variants

All variants preserve the secant/Mode-5 construction; no minimax/refit/SVML is substituted.

- baseline: certified v12
- qfloor: replace redundant roundscale(qf,to_zero) with cvtepi32_pd(qi)
- interleave_x2/x3/x4: qfloor cleanup plus Phase-2 only inter-block latency hiding. Same coefficient bits, anchor rule, local delta, five Horner FMAs and their per-block order.
- grouping / denser-grid variants are intentionally separate because they alter data organization or coefficient generation and require fresh certification.
