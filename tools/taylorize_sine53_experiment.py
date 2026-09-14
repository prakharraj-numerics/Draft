#!/usr/bin/env python3
"""Experimental SINE53 source transformer.

Base: freeze/sine53-cleanup-20260912 @ 9f4b58c18eecb99c9220099eb0cd312702400e8f

This deliberately changes only the normal AVX-512 degree-5 local evaluator in
X50/X67.  Reduction, 1/256 anchor selection, tables, dispatch, sign handling,
and rare scalar safety fallbacks remain unchanged.

The current hot evaluator builds Taylor coefficients from the two anchor planes
and evaluates them as a serial degree-5 Horner chain.  This experiment exposes
the same local sine Taylor polynomial directly as two independent quadratics in
z=d^2:

    A(z) = 1 - z/2 + z^2/24
    B(z) = 1 - z/6 + z^2/120
    sin(a+d) ~= sin(a)*A(z) + d*cos(a)*B(z)

Run from repository root.  The transform is idempotent.
"""

from pathlib import Path

FILES = (
    Path("sine53_x50_unit_production.c"),
    Path("sine53_x67_wide_production.c"),
)

OLD_NOTE = " * Mathematical evaluator is unchanged Mode-5/secant-spine degree 5."
NEW_NOTE = " * EXPERIMENT: same anchors/reduction, direct local sine-Taylor degree 5."

TAYLOR_X11 = r'''OVEC static inline __m512d taylor_poly_x11(const s53w_kernel *k,__m512d y,
                                           __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d ONE=_mm512_set1_pd(1.0);
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0);
    const __m512d C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    __m512d sy=_mm512_mul_pd(y,VK);
    __m256i ji=_mm512_cvt_roundpd_epi32(sy,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd=_mm512_cvtepi32_pd(ji);
    __m512d d=_mm512_fnmadd_pd(jd,VIK,y);
    __m512d s=_mm512_i32gather_pd(ji,tab+0*LUTN,8);
    __m512d c=_mm512_i32gather_pd(ji,tab+1*LUTN,8);

    /* Direct local Taylor architecture.  A and B are independent chains. */
    __m512d z=_mm512_mul_pd(d,d);
    __m512d A=_mm512_fmadd_pd(z,C24,MH);
    __m512d B=_mm512_fmadd_pd(z,C120,M6);
    A=_mm512_fmadd_pd(z,A,ONE);
    B=_mm512_fmadd_pd(z,B,ONE);

    __m512d dc=_mm512_mul_pd(d,c);
    __m512d sA=_mm512_mul_pd(s,A);
    __m512d p=_mm512_fmadd_pd(dc,B,sA);
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}'''

TAYLOR_LOW_X11 = r'''OVEC static inline __m512d taylor_poly_low_x11(const s53w_kernel *k,
                                               __m512d yh,__m512d yl,
                                               __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d ONE=_mm512_set1_pd(1.0);
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0);
    const __m512d C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    __m512d ya=_mm512_add_pd(yh,yl);
    __m512d sy=_mm512_mul_pd(ya,VK);
    __m256i ji=_mm512_cvt_roundpd_epi32(sy,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd=_mm512_cvtepi32_pd(ji);
    __m512d d=_mm512_sub_pd(yh,_mm512_mul_pd(jd,VIK));
    d=_mm512_add_pd(d,yl);
    __m512d s=_mm512_i32gather_pd(ji,tab+0*LUTN,8);
    __m512d c=_mm512_i32gather_pd(ji,tab+1*LUTN,8);

    /* Same polynomial as taylor_poly_x11; retain the low reduction word. */
    __m512d z=_mm512_mul_pd(d,d);
    __m512d A=_mm512_fmadd_pd(z,C24,MH);
    __m512d B=_mm512_fmadd_pd(z,C120,M6);
    A=_mm512_fmadd_pd(z,A,ONE);
    B=_mm512_fmadd_pd(z,B,ONE);

    __m512d dc=_mm512_mul_pd(d,c);
    __m512d sA=_mm512_mul_pd(s,A);
    __m512d p=_mm512_fmadd_pd(dc,B,sA);
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}'''


def replace_function(src: str, name: str, replacement: str) -> str:
    marker = f"OVEC static inline __m512d {name}("
    start = src.find(marker)
    if start < 0:
        raise RuntimeError(f"could not find {name}")
    brace = src.find("{", start)
    if brace < 0:
        raise RuntimeError(f"could not find opening brace for {name}")
    depth = 0
    end = None
    for i in range(brace, len(src)):
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise RuntimeError(f"could not find closing brace for {name}")
    return src[:start] + replacement + src[end:]


def transform(path: Path) -> None:
    src = path.read_text()
    if "taylor_poly_x11(" in src and "taylor_poly_low_x11(" in src:
        print(f"{path}: already Taylorized")
        return

    # Rename only the optimized hot evaluators and every call to them.
    if src.count("mode5_poly_x11(") < 2 or src.count("mode5_poly_low_x11(") < 2:
        raise RuntimeError(f"{path}: unexpected X11 call/definition layout")
    src = src.replace("mode5_poly_low_x11(", "taylor_poly_low_x11(")
    src = src.replace("mode5_poly_x11(", "taylor_poly_x11(")

    src = replace_function(src, "taylor_poly_x11", TAYLOR_X11)
    src = replace_function(src, "taylor_poly_low_x11", TAYLOR_LOW_X11)
    if OLD_NOTE in src:
        src = src.replace(OLD_NOTE, NEW_NOTE, 1)

    path.write_text(src)
    print(f"{path}: Taylor hot path installed")


def main() -> None:
    for path in FILES:
        if not path.exists():
            raise SystemExit(f"missing expected frozen source: {path}")
        transform(path)


if __name__ == "__main__":
    main()
