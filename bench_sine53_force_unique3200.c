#ifndef SOURCE_FILE
#error SOURCE_FILE must name generated X50 C source
#endif

#define main s53_force_disabled_original_main
#include SOURCE_FILE
#undef main

static void force_make_band3200(double *x,int band)
{
    const int n=3200;
    for(int j=0;j<n;j++){
        unsigned q=(unsigned)(((unsigned long long)j*1031ULL+17ULL)%3200ULL);
        double u=((double)q+0.5)/(double)n;
        double a=band==0 ? u : (band==1 ? 1.0+499.0*u : 1000.0+9000.0*u);
        x[j]=(j&1)?-a:a;
    }
}

static void force_bench_band(const s53w_kernel *k,int band,const char *label)
{
    const int n=3200,rr=9375;
    double *x=al64((size_t)n*sizeof(double));
    double *yo=al64((size_t)n*sizeof(double));
    double *yi=al64((size_t)n*sizeof(double));
    if(!x||!yo||!yi){printf("S53FORCE_ALLOC_FAIL band=%s\n",label);free(yi);free(yo);free(x);return;}
    force_make_band3200(x,band);

    /* Accuracy is reported but NEVER blocks timing. This is a performance
       diagnostic requested to measure raw-input cost independent of reduction. */
    int valid=verify_v8(label,k,x,n);

    volatile double sink=0;
    for(int r=0;r<100;r++){
        octant_eval_v8(k,x,yo,(size_t)n);
        vmdSin(n,x,yi,VML_HA);
    }
    sink+=yo[n-1]+yi[n-1];

    double ot[7],it[7],calls=(double)n*(double)rr;
    for(int t=0;t<7;t++){
        uint64_t a,z,t0;
        if(t&1){
            t0=now_ns();for(int r=0;r<rr;r++)vmdSin(n,x,yi,VML_HA);z=now_ns()-t0;
            t0=now_ns();for(int r=0;r<rr;r++)octant_eval_v8(k,x,yo,(size_t)n);a=now_ns()-t0;
        }else{
            t0=now_ns();for(int r=0;r<rr;r++)octant_eval_v8(k,x,yo,(size_t)n);a=now_ns()-t0;
            t0=now_ns();for(int r=0;r<rr;r++)vmdSin(n,x,yi,VML_HA);z=now_ns()-t0;
        }
        ot[t]=(double)a/calls;
        it[t]=(double)z/calls;
        sink+=yo[n-1]+yi[n-1];
    }
    qsort(ot,7,sizeof(double),cmpd);
    qsort(it,7,sizeof(double),cmpd);
    printf("S53FORCE_RESULT band=%s cases=3200 valid_le1ulp=%d ours_min_ns=%.6f ours_median_ns=%.6f ours_max_ns=%.6f intel_min_ns=%.6f intel_median_ns=%.6f intel_max_ns=%.6f ours_over_intel=%.6fx unique_inputs=1 sink=%.17g\n",
           label,valid,ot[0],ot[3],ot[6],it[0],it[3],it[6],ot[3]/it[3],(double)sink);
    free(yi);free(yo);free(x);
}

int main(void)
{
    int cpu=pin();
    mkl_set_num_threads_local(1);
    printf("S53FORCE_DOMAIN cpu_pin=%d backend=%s LUTN=%lu terms=2 bands=abs_lt_1,abs_1_to_500,abs_1000_to_10000 trials=7 evals=30000000_per_band accuracy_nonblocking=1\n",
           cpu,backend(),(unsigned long)LUTN);
    s53w_kernel *k=kernel_create(2);
    if(!k){fprintf(stderr,"kernel_create failed\n");return 2;}
    force_bench_band(k,0,"abs_lt_1");
    force_bench_band(k,1,"abs_1_to_500");
    force_bench_band(k,2,"abs_1000_to_10000");
    kernel_destroy(k);
    redtab2_clear();
    flint_cleanup_master();
    return 0;
}
