from pathlib import Path
import sys

if len(sys.argv)!=2:
    raise SystemExit('usage: inject_sine53_unique9600.py generated.c')
p=Path(sys.argv[1]);s=p.read_text()
mainpos=s.index('\nint main(void)')
helper=r'''
static void make_unique9600(double *x)
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

static void bench_unique9600(const s53w_kernel *k)
{
    const int n=9600,rr=3125;double *x=al64((size_t)n*sizeof(double));
    double *yo=al64((size_t)n*sizeof(double)),*yi=al64((size_t)n*sizeof(double));
    if(!x||!yo||!yi){free(yi);free(yo);free(x);printf("S53UNIQ_ALLOC_FAIL\n");return;}
    make_unique9600(x);
    if(!verify_v8("unique9600",k,x,n)){printf("S53UNIQ_INVALID accuracy_gate=fail\n");free(yi);free(yo);free(x);return;}
    volatile double sink=0;for(int r=0;r<100;r++){octant_eval_v8(k,x,yo,(size_t)n);vmdSin(n,x,yi,VML_HA);}sink+=yo[n-1]+yi[n-1];
    double ot[7],it[7],calls=(double)n*(double)rr;
    for(int t=0;t<7;t++){
        uint64_t a,z,t0;if(t&1){t0=now_ns();for(int r=0;r<rr;r++)vmdSin(n,x,yi,VML_HA);z=now_ns()-t0;t0=now_ns();for(int r=0;r<rr;r++)octant_eval_v8(k,x,yo,(size_t)n);a=now_ns()-t0;}
        else{t0=now_ns();for(int r=0;r<rr;r++)octant_eval_v8(k,x,yo,(size_t)n);a=now_ns()-t0;t0=now_ns();for(int r=0;r<rr;r++)vmdSin(n,x,yi,VML_HA);z=now_ns()-t0;}
        ot[t]=(double)a/calls;it[t]=(double)z/calls;sink+=yo[n-1]+yi[n-1];
    }
    qsort(ot,7,sizeof(double),cmpd);qsort(it,7,sizeof(double),cmpd);
    printf("S53UNIQ_RESULT cases=9600 ours_ns=%.6f intel_ns=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx unique_inputs=1 Arb256_all=1 sink=%.17g\n",ot[3],it[3],ot[3]/it[3],it[3]/ot[3],(double)sink);
    free(yi);free(yo);free(x);
}
'''
s=s[:mainpos]+helper+s[mainpos:]
needle='int rc=bench_v8(k,x);bench_batches_x11(k,x);kernel_destroy(k);redtab2_clear();flint_cleanup_master();return rc;'
if needle not in s:
    raise SystemExit('v11/v12 main tail not found for unique9600 injection')
s=s.replace(needle,'int rc=bench_v8(k,x);bench_batches_x11(k,x);bench_unique9600(k);kernel_destroy(k);redtab2_clear();flint_cleanup_master();return rc;',1)
p.write_text(s)
print('S53UNIQ_INJECT_PASS cases=9600 deterministic_unique=1 Arb256_gate=1')
