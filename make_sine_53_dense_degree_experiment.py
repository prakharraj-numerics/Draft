from pathlib import Path

# Feasibility driver only. We do NOT substitute a minimax/Taylor fit. This file
# records the intended experiment boundary for a denser-grid lower-degree
# Mode-5 build: regenerate coefficients from the existing secant/Mode-5 builder,
# then certify. If the existing builder cannot emit the requested degree/grid
# combination without changing the formula family, the experiment must stop.

print('S53DENSE_PLAN same_secant_Mode5_generator_required=1 no_minimax=1 no_Taylor_substitute=1 certification_required=1')
