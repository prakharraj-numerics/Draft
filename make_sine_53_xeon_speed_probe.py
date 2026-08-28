from pathlib import Path
import runpy

# Build the exact four-gather DD candidate first.
runpy.run_path('make_sine_53_xeon_v1.py', run_name='__main__')
p = Path('bench_sine_53_xeon_v2_build.c')
src = p.read_text()

old1 = 'double *st=al64(STRESS*sizeof(double));if(!st)return 5;make_stress(st);if(!verify_x2("legacy_npi_npi2_stress",k,st,STRESS)){free(st);kernel_destroy(k);redtab2_clear();return 6;}free(st);'
new1 = 'double *st=al64(STRESS*sizeof(double));if(!st)return 5;make_stress(st);int sp1=verify_x2("legacy_npi_npi2_stress",k,st,STRESS);printf("S53X2_SPEED_PROBE stress_legacy_pass=%d contractual=0\\n",sp1);free(st);'
old2 = 'double *os=al64(OSTRESS*sizeof(double));if(!os)return 7;make_pi4_stress_x2(os);if(!verify_x2("all_npi4_boundary_stress",k,os,OSTRESS)){free(os);kernel_destroy(k);redtab2_clear();return 8;}free(os);'
new2 = 'double *os=al64(OSTRESS*sizeof(double));if(!os)return 7;make_pi4_stress_x2(os);int sp2=verify_x2("all_npi4_boundary_stress",k,os,OSTRESS);printf("S53X2_SPEED_PROBE stress_pi4_pass=%d contractual=0\\n",sp2);free(os);'

if old1 not in src:
    raise SystemExit('legacy stress gate marker not found')
if old2 not in src:
    raise SystemExit('pi4 stress gate marker not found')
src = src.replace(old1,new1,1).replace(old2,new2,1)
src = src.replace('S53X2_DOMAIN ', 'S53X2_SPEED_PROBE_DOMAIN ', 1)
p.write_text(src)
print('S53X2_SPEED_PROBE_BUILD requested150_strict=1 stress_nonfatal=1 candidate=four_gather_DD')
