#define main s53x12_original_main
#include "bench_sine_53_xeon_v12_build.c"
#undef main

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROFILE_N 9600

int main(int argc,char **argv)
{
    if(argc < 2 || (strcmp(argv[1],"ours") && strcmp(argv[1],"intel"))) {
        fprintf(stderr,"usage: %s ours|intel [rounds]\n", argv[0]);
        return 64;
    }
    int rounds = argc > 2 ? atoi(argv[2]) : 120000;
    if(rounds < 1) rounds = 1;

    int cpu=pin();
    mkl_set_num_threads_local(1);
    if(!redtab2_init()) return 2;
    s53w_kernel *k=kernel_create(2);
    if(!k){redtab2_clear();return 3;}

    double base[CASES];
    make_bench(base);
    double *x=al64((size_t)PROFILE_N*sizeof(double));
    double *y=al64((size_t)PROFILE_N*sizeof(double));
    if(!x||!y){free(y);free(x);kernel_destroy(k);redtab2_clear();return 4;}
    for(int i=0;i<PROFILE_N;i++) x[i]=base[i%CASES];

    /* Warm the exact path that will be profiled. */
    if(!strcmp(argv[1],"ours")) {
        for(int r=0;r<200;r++) octant_eval_v8(k,x,y,PROFILE_N);
    } else {
        for(int r=0;r<200;r++) vmdSin(PROFILE_N,x,y,VML_HA);
    }

    volatile double sink=0.0;
    uint64_t t0=now_ns();
    if(!strcmp(argv[1],"ours")) {
        for(int r=0;r<rounds;r++) octant_eval_v8(k,x,y,PROFILE_N);
    } else {
        for(int r=0;r<rounds;r++) vmdSin(PROFILE_N,x,y,VML_HA);
    }
    uint64_t dt=now_ns()-t0;
    sink += y[PROFILE_N-1];
    printf("S53PERF mode=%s cpu_pin=%d n=%d rounds=%d ns_per_input=%.9f sink=%.17g\n",
           argv[1],cpu,PROFILE_N,rounds,(double)dt/((double)PROFILE_N*(double)rounds),(double)sink);

    free(y);free(x);kernel_destroy(k);redtab2_clear();flint_cleanup_master();
    return 0;
}
