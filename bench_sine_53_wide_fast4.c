#include <mpfr.h>
#define S53F3_NO_MAIN 1
#include "bench_sine_53_wide_fast3.c"

/* Replace fused-Mode5 c0/c1 storage by the actual cached anchor quantities
   required by the unfused identity: sin(a), cos(a).  In a production library
   these are shipped constants; generation is therefore outside timed calls. */
static double *f4_anchor;

static int f4_anchor_init(void)
{
    f4_anchor=al64((size_t)2*LUTN*sizeof(double));
    if(!f4_anchor)return 0;
    mpfr_t a,s,c;mpfr_init2(a,256);mpfr_init2(s,256);mpfr_init2(c,256);
    for(size_t i=0;i<LUTN;i++){
        mpfr_set_ui(a,(unsigned long)i,MPFR_RNDN);
        mpfr_div_2ui(a,a,SF_K,MPFR_RNDN);
        mpfr_sin_cos(s,c,a,MPFR_RNDN);
        f4_anchor[i]=mpfr_get_d(s,MPFR_RNDN);
        f4_anchor[LUTN+i]=mpfr_get_d(c,MPFR_RNDN);
    }
    mpfr_clear(c);mpfr_clear(s);mpfr_clear(a);return 1;
}

int main(void)
{
    int cpu=pin();mkl_set_num_threads_local(1);
    printf("S53F4_DOMAIN cpu_pin=%d target=binary64_53bit cases=150 bands=0_to_1,1_to_500,1000_to_10000 signs=25pos_25neg_each intel=oneMKL_vmdSin_VML_HA formula=secant_spine_unfused_C_T anchors=correctly_rounded_MPFR256_cached\n",cpu);
    if(!f4_anchor_init())return 2;
    s53w_kernel*base=kernel_create(2);if(!base)return 3;
    s53w_kernel view=*base;view.tab=f4_anchor;
    double x[CASES],v[CASES];make_bench(x);f3_eval(&view,x,v,CASES);
    int same=0;for(int i=0;i<CASES;i++)if(dbits(v[i])==dbits(f3_eval_scalar_one(&view,x[i])))same++;
    printf("S53F4_VECTOR_SCALAR bit_identical=%d/150\n",same);if(same!=CASES){kernel_destroy(base);free(f4_anchor);return 4;}
    if(!f3_verify("requested150",&view,x,CASES)){kernel_destroy(base);free(f4_anchor);return 5;}
    f3_bands(&view,x);
    double*st=al64(STRESS*sizeof(double));if(st){make_stress(st);int ok=f3_verify("adversarial_stress",&view,st,STRESS);printf("S53F4_STRESS pass=%d contractual=0\n",ok);free(st);if(!ok){kernel_destroy(base);free(f4_anchor);return 6;}}
    int rc=f3_bench(&view,x);
    kernel_destroy(base);free(f4_anchor);flint_cleanup_master();return rc;
}
