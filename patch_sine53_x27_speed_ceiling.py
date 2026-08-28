from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit('usage: patch_sine53_x27_speed_ceiling.py generated.c')
p = Path(sys.argv[1])
s = p.read_text()
old_gate = 'if(!verify_v8("unique9600",k,x,n)){printf("S53UNIQ_INVALID accuracy_gate=fail\\n");free(yi);free(yo);free(x);return;}'
new_gate = 'int uq_ok=verify_v8("unique9600",k,x,n); printf("S53UNIQ_DIAGNOSTIC accuracy_ok=%d timing_even_if_invalid=1\\n",uq_ok);'
if old_gate not in s:
    raise SystemExit('unique accuracy gate marker missing')
s = s.replace(old_gate, new_gate, 1)
old_tail = 'int rc=bench_v8(k,x);bench_batches_x11(k,x);bench_unique9600(k);kernel_destroy(k);redtab2_clear();flint_cleanup_master();return rc;'
new_tail = 'bench_batches_x11(k,x);bench_unique9600(k);int rc=bench_v8(k,x);kernel_destroy(k);redtab2_clear();flint_cleanup_master();return rc;'
if old_tail not in s:
    raise SystemExit('diagnostic main tail marker missing')
s = s.replace(old_tail, new_tail, 1)
p.write_text(s)
print('X27_SPEED_CEILING_PATCH_PASS timing_before_accuracy_abort=1 unique_timing_even_if_invalid=1')
