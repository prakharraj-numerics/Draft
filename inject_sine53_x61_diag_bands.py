from pathlib import Path
import sys
if len(sys.argv)!=2: raise SystemExit('usage: inject_sine53_x61_diag_bands.py generated.c')
p=Path(sys.argv[1]); s=p.read_text(); mainpos=s.index('\nint main(void)')
helper=r'''
static void x61_make_band(double *x,int band)
{
    const int n=3200;
    for(int j=0;j<n;j++){
        unsigned q=(unsigned)(((unsigned long long)j*1031ULL+17ULL)%3200ULL);
        double u=((double)q+0.5)/(double)n;
        double a=band==0 ? (1.0+499.0*u) : (1000.0+9000.0*u);
        x[j]=(j&1)?-a:a;
    }
}
static void x61_diag_band(const s53w_kernel *k,int band,const char *label)
{
    const int n=3200,rr=8;
    double *x=al64((size_t)n*sizeof(double)),*yo=al64((size_t)n*sizeof(double)),*yi=al64((size_t)n*sizeof(double));
    if(!x||!yo||!yi){printf("X61_ALLOC_FAIL band=%s\n",label);return;}
    x61_make_band(x,band);
    int acc=verify_v8(label,k,x,n);
    printf("X61_ACCURACY_GATE band=%s pass=%d\n",label,acc);
    volatile double sink=0;
    for(int r=0;r<3;r++){octant_eval_v8(k,x,yo,(size_t)n);vmdSin(n,x,yi,VML_HA);} sink+=yo[n-1]+yi[n-1];
    double ot[5],it[5],calls=(double)n*(double)rr;
    for(int t=0;t<5;t++){
        uint64_t a,z,t0;
        if(t&1){t0=now_ns();for(int r=0;r<rr;r++)vmdSin(n,x,yi,VML_HA);z=now_ns()-t0;t0=now_ns();for(int r=0;r<rr;r++)octant_eval_v8(k,x,yo,(size_t)n);a=now_ns()-t0;}
        else{t0=now_ns();for(int r=0;r<rr;r++)octant_eval_v8(k,x,yo,(size_t)n);a=now_ns()-t0;t0=now_ns();for(int r=0;r<rr;r++)vmdSin(n,x,yi,VML_HA);z=now_ns()-t0;}
        ot[t]=(double)a/calls;it[t]=(double)z/calls;sink+=yo[n-1]+yi[n-1];
    }
    qsort(ot,5,sizeof(double),cmpd);qsort(it,5,sizeof(double),cmpd);
    printf("X61_DIAG_RESULT band=%s cases=3200 accuracy_pass=%d ours_ns=%.6f intel_ns=%.6f ours_over_intel=%.3fx sink=%.17g\n",label,acc,ot[2],it[2],ot[2]/it[2],(double)sink);
    free(yi);free(yo);free(x);
}
'''
s=s[:mainpos]+helper+s[mainpos:]
mainpos=s.index('\nint main(void)')
rawmain=r'''
int main(void)
{
    int cpu=pin();mkl_set_num_threads_local(1);
    printf("X61_DIAGNOSTIC_MAIN cpu_pin=%d raw_input=1 no_range_reduction=1 no_anchor=1 no_lut=1 factors=4096\n",cpu);
    s53w_kernel *k=kernel_create(2); if(!k) return 5;
    x61_diag_band(k,0,"abs_1_to_500");
    x61_diag_band(k,1,"abs_1000_to_10000");
    kernel_destroy(k);redtab2_clear();flint_cleanup_master();return 0;
}
'''
s=s[:mainpos]+rawmain
p.write_text(s)
print('X61_DIAG_INJECT_PASS bands=2 cases=3200 timing_even_if_accuracy_fails=1')
