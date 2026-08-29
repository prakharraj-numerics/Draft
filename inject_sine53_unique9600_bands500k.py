from pathlib import Path
import sys

if len(sys.argv)!=2:
    raise SystemExit('usage: inject_sine53_unique9600_bands500k.py generated.c')
p=Path(sys.argv[1]);s=p.read_text()
mainpos=s.index('\nint main(void)')
helper=r'''
static void make_unique_band3200_500k(double *x,int band)
{
    const int n=3200;
    for(int j=0;j<n;j++){
        unsigned q=(unsigned)(((unsigned long long)j*1031ULL+17ULL)%3200ULL);
        double u=((double)q+0.5)/(double)n;
        double a=band==0 ? u : (band==1 ? 1.0+499.0*u : 1000.0+499000.0*u);
        x[j]=(j&1)?-a:a;
    }
}

static void bench_unique_band3200_500k(const s53w_kernel *k,int band,const char *label)
{
    const int n=3200,rr=9375; /* 30M input evaluations per trial */
    double *x=al64((size_t)n*sizeof(double));
    double *yo=al64((size_t)n*sizeof(double)),*yi=al64((size_t)n*sizeof(double));
    if(!x||!yo||!yi){free(yi);free(yo);free(x);printf("S53U500_ALLOC_FAIL band=%s\n",label);return;}
    make_unique_band3200_500k(x,band);
    if(!verify_v8(label,k,x,n)){printf("S53U500_INVALID band=%s accuracy_gate=fail\n",label);free(yi);free(yo);free(x);return;}
    volatile double sink=0;
    for(int r=0;r<100;r++){octant_eval_v8(k,x,yo,(size_t)n);vmdSin(n,x,yi,VML_HA);} sink+=yo[n-1]+yi[n-1];
    double ot[7],it[7],calls=(double)n*(double)rr;
    for(int t=0;t<7;t++){
        uint64_t a,z,t0;
        if(t&1){
            t0=now_ns();for(int r=0;r<rr;r++)vmdSin(n,x,yi,VML_HA);z=now_ns()-t0;
            t0=now_ns();for(int r=0;r<rr;r++)octant_eval_v8(k,x,yo,(size_t)n);a=now_ns()-t0;
        } else {
            t0=now_ns();for(int r=0;r<rr;r++)octant_eval_v8(k,x,yo,(size_t)n);a=now_ns()-t0;
            t0=now_ns();for(int r=0;r<rr;r++)vmdSin(n,x,yi,VML_HA);z=now_ns()-t0;
        }
        ot[t]=(double)a/calls;it[t]=(double)z/calls;sink+=yo[n-1]+yi[n-1];
    }
    qsort(ot,7,sizeof(double),cmpd);qsort(it,7,sizeof(double),cmpd);
    printf("S53U500_RESULT band=%s cases=3200 ours_min_ns=%.6f ours_median_ns=%.6f ours_max_ns=%.6f intel_min_ns=%.6f intel_median_ns=%.6f intel_max_ns=%.6f median_ours_over_intel=%.6fx unique_inputs=1 Arb256_all=1 sink=%.17g\n",
           label,ot[0],ot[3],ot[6],it[0],it[3],it[6],ot[3]/it[3],(double)sink);
    free(yi);free(yo);free(x);
}

static void bench_unique9600_bands500k(const s53w_kernel *k)
{
    bench_unique_band3200_500k(k,0,"abs_lt_1");
    bench_unique_band3200_500k(k,1,"abs_1_to_500");
    bench_unique_band3200_500k(k,2,"abs_1000_to_500000");
}
'''
s=s[:mainpos]+helper+s[mainpos:]
needle='int rc=bench_v8(k,x);bench_batches_x11(k,x);kernel_destroy(k);redtab2_clear();flint_cleanup_master();return rc;'
if needle not in s:
    raise SystemExit('v11/v12 main tail not found for banded unique9600 injection')
s=s.replace(needle,'int rc=bench_v8(k,x);bench_unique9600_bands500k(k);kernel_destroy(k);redtab2_clear();flint_cleanup_master();return rc;',1)
p.write_text(s)
print('S53U500_INJECT_PASS bands=3 cases_per_band=3200 total_unique=9600 trials=7 ranges=0_1,1_500,1000_500000 Arb256_gate=1')
