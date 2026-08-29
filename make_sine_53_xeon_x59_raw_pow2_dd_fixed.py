from pathlib import Path

# Mechanical wrapper: make the X59 generator robust to whitespace/style changes
# in the generated s53w_kernel typedef, then execute the otherwise unchanged X59.
src = Path('make_sine_53_xeon_x59_raw_pow2_dd.py').read_text()
old = """old='typedef struct { sine_fixed_ctx *ctx; int terms,deg; double *tab; } s53w_kernel;'\nnew='typedef struct { sine_fixed_ctx *ctx; int terms,deg; double *tab; double *lotab; } s53w_kernel;'\nif old not in s: raise SystemExit('kernel struct marker not found')\ns=s.replace(old,new,1)"""
new = """import re\npat=r'typedef\\s+struct\\s*\\{\\s*sine_fixed_ctx\\s*\\*ctx;\\s*int\\s+terms,deg;\\s*double\\s*\\*tab;\\s*\\}\\s*s53w_kernel;'\nrep='typedef struct { sine_fixed_ctx *ctx; int terms,deg; double *tab; double *lotab; } s53w_kernel;'\ns,nsub=re.subn(pat,rep,s,count=1)\nif nsub != 1: raise SystemExit('kernel struct regex marker not found')"""
if old not in src:
    raise SystemExit('X59 wrapper patch target not found')
src = src.replace(old,new,1)
exec(compile(src,'make_sine_53_xeon_x59_raw_pow2_dd.py','exec'))
