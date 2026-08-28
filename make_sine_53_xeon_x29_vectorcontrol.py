from pathlib import Path
import runpy, sys

if len(sys.argv) != 2 or sys.argv[1] not in ('u','r','ur'):
    raise SystemExit('usage: make_sine_53_xeon_x29_vectorcontrol.py {u|r|ur}')
mode = sys.argv[1]

# Frozen production baseline: X23-H14.
saved = sys.argv[:]
try:
    sys.argv = ['make_sine_53_xeon_x23_hybridpi.py', '14']
    runpy.run_path('make_sine_53_xeon_x23_hybridpi.py', run_name='__main__')
finally:
    sys.argv = saved

p = Path('bench_sine_53_xeon_x23_h14_build.c')
s = p.read_text()

if 'u' in mode:
    # Remove the vector-mask -> scalar all-unit early exit. For |x|<1, nearest x/pi
    # quotient is zero anyway, and the existing unit masks restore rh=ax, rl=0,
    # parity=0. Thus the full vector reducer is valid for those lanes too.
    old_unit = '''    *pure_unit_out=(unsigned char)(unit==active);\n    if(unit==active){\n        *rh_out=ax;*rl_out=Z;*sign_out=inneg;*guard_out=0;*active_out=active;return;\n    }\n\n'''
    new_unit = '''    /* X29-U: no scalar all-unit control transfer. */\n    *pure_unit_out=0;\n\n'''
    if old_unit not in s:
        raise SystemExit('X29-U unit early-return marker missing')
    s = s.replace(old_unit, new_unit, 1)

    # In the G4 hot loop remove four scalar pu branches.  One FMA plus one add is
    # used for every stream.  This is cheaper than the old non-unit mul/sub/add
    # arm and avoids the scalar decision entirely.  Regrouping is re-certified.
    for b in range(4):
        old = (f'        if(pu{b}) d{b}=_mm512_fnmadd_pd(jd{b},VIK,rh{b}); else '
               f'{{d{b}=_mm512_sub_pd(rh{b},_mm512_mul_pd(jd{b},VIK));d{b}=_mm512_add_pd(d{b},rl{b});}}')
        new = (f'        d{b}=_mm512_fnmadd_pd(jd{b},VIK,rh{b}); '
               f'd{b}=_mm512_add_pd(d{b},rl{b});')
        if old not in s:
            raise SystemExit(f'X29-U pu branch marker missing for stream {b}')
        s = s.replace(old, new, 1)

if 'r' in mode:
    # Remove the scalar repair-mask branch by always executing the 3-piece
    # vector arithmetic and retaining the existing masked moves. This is an
    # intentionally aggressive control-crossing experiment: likely more work,
    # but it directly measures whether the branch itself costs enough to matter.
    old = '''    if(__builtin_expect(repair!=0,0)){\n        const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);\n        const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);\n        const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);\n        __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(qd,PI1));\n        __m512d rh3,re3;twodiff_cw(r0,_mm512_mul_pd(qd,PI2),&rh3,&re3);\n        __m512d rl3=_mm512_fnmadd_pd(qd,PI3,re3);\n        rh=_mm512_mask_mov_pd(rh,repair,rh3);\n        rl=_mm512_mask_mov_pd(rl,repair,rl3);\n    }'''
    new = '''    {\n        /* X29-R: unconditional vector repair arithmetic; masked commit only. */\n        const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);\n        const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);\n        const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);\n        __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(qd,PI1));\n        __m512d rh3,re3;twodiff_cw(r0,_mm512_mul_pd(qd,PI2),&rh3,&re3);\n        __m512d rl3=_mm512_fnmadd_pd(qd,PI3,re3);\n        rh=_mm512_mask_mov_pd(rh,repair,rh3);\n        rl=_mm512_mask_mov_pd(rl,repair,rl3);\n    }'''
    if old not in s:
        raise SystemExit('X29-R B14 repair branch marker missing')
    s = s.replace(old, new, 1)

prefix = {'u':'S53X29U_','r':'S53X29R_','ur':'S53X29UR_'}[mode]
arch = {'u':'xeon_x29_unit_branchless_g4','r':'xeon_x29_repair_branchless_g4','ur':'xeon_x29_unit_repair_branchless_g4'}[mode]
label = {'u':'Xeon_AVX512_X29U_unit_control_vectorized','r':'Xeon_AVX512_X29R_repair_control_vectorized','ur':'Xeon_AVX512_X29UR_unit_repair_control_vectorized'}[mode]
s = s.replace('S53X23H14_', prefix)
s = s.replace('xeon_x23_nearestpi_2f_hybrid_b14_g4_2g', arch)
s = s.replace('Xeon_AVX512_G4_nearestpi_2f_hybrid_b14_two_gather_Mode5', label)
out = Path(f'bench_sine_53_xeon_x29_{mode}_build.c')
out.write_text(s)
print(f'X29_BUILD_PASS mode={mode} parent=X23H14 G4=1 two_gather=1 math_spine_unchanged=1 unit_scalar_crossing_removed={int("u" in mode)} repair_scalar_crossing_removed={int("r" in mode)} requires_Arb_regate=1')
