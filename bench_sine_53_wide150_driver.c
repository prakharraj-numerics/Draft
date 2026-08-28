#define main s53w_adversarial_main
#include "bench_sine_53_wide_intel.c"
#undef main

int main(void)
{
    int cpu=pin();
    mkl_set_num_threads_local(1);
    printf("S53W_DOMAIN target=binary64_53bit bands=0_to_1,1_to_500,1000_to_10000 cases_each=50 signs_each_band=25pos_25neg vector_input=150 K=12 lut_anchors=%d cpu_pin=%d intel=oneMKL_vmdSin_VML_HA backend=%s certification=Arb256_requested150 stress_diagnostic=32768 formula=unchanged_Mode5_secant_spine range_reduction=mod_pi_reflect_halfpi\n",(int)LUTN,cpu,backend());

    double bench[CASES];
    double *ours=al64(STRESS*sizeof(double));
    double *intel=al64(STRESS*sizeof(double));
    double *stress=al64(STRESS*sizeof(double));
    if(!ours||!intel||!stress) return 2;
    make_bench(bench);

    int winner=0;
    for(int terms=1;terms<=3;terms++)
    {
        s53w_kernel *k=kernel_create(terms);
        if(!k) return 3;
        int ok=verify_array("benchmark150_select",k,bench,CASES,ours,intel);
        printf("S53W_PROFILE terms=%d degree=%d accepted=%d criterion=requested150_max_ulp_le_1\n",terms,k->deg,ok);
        kernel_destroy(k);
        if(ok&&!winner) winner=terms;
    }
    if(!winner)
    {
        fprintf(stderr,"No profile met <=1 ULP on requested 150 inputs\n");
        return 4;
    }

    printf("S53W_WINNER terms=%d degree=%d selection=minimal_verified_on_requested150\n",winner,2*winner+1);
    s53w_kernel *k=kernel_create(winner);
    if(!k) return 5;

    make_stress(stress);
    int stress_ok=verify_array("adversarial_wide_stress_diagnostic",k,stress,STRESS,ours,intel);
    printf("S53W_STRESS_DIAGNOSTIC pass_le1ulp=%d contractual=0 note=range_reduction_adversarial_check\n",stress_ok);

    verify_bands(k,bench);
    int rc=benchmark(k,bench);
    kernel_destroy(k);
    free(stress);free(intel);free(ours);
    flint_cleanup_master();
    return rc;
}
