from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit('usage: patch_sine53_x27_speed_ceiling.py generated.c')
p = Path(sys.argv[1])
s = p.read_text()

# Keep the unique9600 Arb check visible, but do not let it suppress timing in
# this deliberately diagnostic speed-ceiling executable.
old_gate = 'if(!verify_v8("unique9600",k,x,n)){printf("S53UNIQ_INVALID accuracy_gate=fail\\n");free(yi);free(yo);free(x);return;}'
new_gate = 'int uq_ok=verify_v8("unique9600",k,x,n); printf("S53UNIQ_DIAGNOSTIC accuracy_ok=%d timing_even_if_invalid=1\\n",uq_ok);'
if old_gate not in s:
    raise SystemExit('unique accuracy gate marker missing')
s = s.replace(old_gate, new_gate, 1)

# Replace the final certification-first main completely. We already know X27
# fails 41 legacy cases by 2 ULP; this executable exists only to quantify its
# raw throughput ceiling versus X23 on the same CPU. Kernel setup and timing
# functions are otherwise unchanged.
mainpos = s.rfind('\nint main(void)')
if mainpos < 0:
    raise SystemExit('final main not found')
new_main = r'''
int main(void)
{
    int cpu=pin();
    mkl_set_num_threads_local(1);
    printf("S53_SPEED_CEILING_DOMAIN cpu_pin=%d diagnostic_only=1 known_x27_legacy_invalid=1 intel=oneMKL_vmdSin_VML_HA\n",cpu);
    if(!redtab2_init()) return 2;
    s53w_kernel *k=kernel_create(2);
    if(!k){redtab2_clear();return 3;}
    double x[CASES];
    make_bench(x);
    bench_batches_x11(k,x);
    bench_unique9600(k);
    kernel_destroy(k);
    redtab2_clear();
    flint_cleanup_master();
    return 0;
}
'''
s = s[:mainpos] + new_main
p.write_text(s)
print('X27_SPEED_CEILING_PATCH_PASS certification_main_bypassed=1 repeated_timing=1 unique_timing_even_if_invalid=1')
