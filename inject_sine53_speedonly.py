from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit('usage: inject_sine53_speedonly.py generated.c')

p = Path(sys.argv[1])
s = p.read_text()
mainpos = s.index('\nint main(void)')

helper = r'''
static void speedonly_make_unique9600(double *x)
{
    const int per=3200;
    for(int i=0;i<9600;i++){
        int band=i%3,j=i/3;
        unsigned q=(unsigned)(((unsigned long long)j*1031ULL+17ULL)%3200ULL);
        double u=((double)q+0.5)/(double)per;
        double a=band==0 ? u : (band==1 ? 1.0+499.0*u : 1000.0+9000.0*u);
        x[i]=(j&1)?-a:a;
    }
}

static void speedonly_unique9600(const s53w_kernel *k)
{
    const int n=9600,rr=3125;
    double *x=al64((size_t)n*sizeof(double));
    double *yo=al64((size_t)n*sizeof(double));
    double *yi=al64((size_t)n*sizeof(double));
    if(!x||!yo||!yi){free(yi);free(yo);free(x);printf("S53SPEEDONLY_ALLOC_FAIL\n");return;}
    speedonly_make_unique9600(x);
    volatile double sink=0;
    for(int r=0;r<100;r++){octant_eval_v8(k,x,yo,(size_t)n);vmdSin(n,x,yi,VML_HA);}sink+=yo[n-1]+yi[n-1];
    double ot[9],it[9],calls=(double)n*(double)rr;
    for(int t=0;t<9;t++){
        uint64_t a,z,t0;
        if(t&1){
            t0=now_ns();for(int r=0;r<rr;r++)vmdSin(n,x,yi,VML_HA);z=now_ns()-t0;
            t0=now_ns();for(int r=0;r<rr;r++)octant_eval_v8(k,x,yo,(size_t)n);a=now_ns()-t0;
        }else{
            t0=now_ns();for(int r=0;r<rr;r++)octant_eval_v8(k,x,yo,(size_t)n);a=now_ns()-t0;
            t0=now_ns();for(int r=0;r<rr;r++)vmdSin(n,x,yi,VML_HA);z=now_ns()-t0;
        }
        ot[t]=(double)a/calls;it[t]=(double)z/calls;sink+=yo[n-1]+yi[n-1];
    }
    qsort(ot,9,sizeof(double),cmpd);qsort(it,9,sizeof(double),cmpd);
    printf("S53SPEEDONLY_UNIQUE cases=9600 ours_ns=%.6f intel_ns=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx speed_advantage_vs_intel_pct=%.3f accuracy_gate=disabled diagnostic_only=1 sink=%.17g\n",
           ot[4],it[4],ot[4]/it[4],it[4]/ot[4],(it[4]/ot[4]-1.0)*100.0,(double)sink);
    free(yi);free(yo);free(x);
}
'''

main = r'''
int main(void)
{
    int cpu=pin();
    mkl_set_num_threads_local(1);
    printf("S53SPEEDONLY_DOMAIN cpu_pin=%d intel=oneMKL_vmdSin_VML_HA one_thread=1 accuracy_gate=disabled diagnostic_only=1\n",cpu);
    if(!redtab2_init()) return 2;
    s53w_kernel *k=kernel_create(2);
    if(!k){redtab2_clear();return 3;}
    double x[CASES];
    make_bench(x);
    int rc=bench_v8(k,x);
    bench_batches_x11(k,x);
    speedonly_unique9600(k);
    kernel_destroy(k);
    redtab2_clear();
    flint_cleanup_master();
    return rc;
}
'''

s = s[:mainpos] + helper + main
p.write_text(s)
print('S53SPEEDONLY_INJECT_PASS accuracy_gate=disabled repeated_and_unique_timing=1 oneMKL_HA=1')
