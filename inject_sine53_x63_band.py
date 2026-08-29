from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text();mainpos=s.index('\nint main(void)')
helper=r'''
static void x63_make(double*x){const int n=3200;for(int j=0;j<n;j++){unsigned q=(unsigned)(((unsigned long long)j*1031ULL+17ULL)%3200ULL);double u=((double)q+0.5)/(double)n;double a=1.0+499.0*u;x[j]=(j&1)?-a:a;}}
static void x63_run(const s53w_kernel*k){const int n=3200,rr=16;double*x=al64(n*sizeof(double)),*yo=al64(n*sizeof(double)),*yi=al64(n*sizeof(double));x63_make(x);int acc=verify_v8("abs_1_to_500",k,x,n);printf("X63_ACCURACY pass=%d\n",acc);volatile double sink=0;for(int r=0;r<3;r++){octant_eval_v8(k,x,yo,n);vmdSin(n,x,yi,VML_HA);}double ot[5],it[5],calls=(double)n*rr;for(int t=0;t<5;t++){uint64_t a,b,t0;if(t&1){t0=now_ns();for(int r=0;r<rr;r++)vmdSin(n,x,yi,VML_HA);b=now_ns()-t0;t0=now_ns();for(int r=0;r<rr;r++)octant_eval_v8(k,x,yo,n);a=now_ns()-t0;}else{t0=now_ns();for(int r=0;r<rr;r++)octant_eval_v8(k,x,yo,n);a=now_ns()-t0;t0=now_ns();for(int r=0;r<rr;r++)vmdSin(n,x,yi,VML_HA);b=now_ns()-t0;}ot[t]=(double)a/calls;it[t]=(double)b/calls;sink+=yo[n-1]+yi[n-1];}qsort(ot,5,sizeof(double),cmpd);qsort(it,5,sizeof(double),cmpd);printf("X63_RESULT ours_ns=%.6f intel_ns=%.6f ours_over_intel=%.3fx acc=%d sink=%.17g\n",ot[2],it[2],ot[2]/it[2],acc,(double)sink);free(yi);free(yo);free(x);}
'''
s=s[:mainpos]+helper+s[mainpos:];mainpos=s.index('\nint main(void)')
s=s[:mainpos]+r'''
int main(void){int cpu=pin();mkl_set_num_threads_local(1);printf("X63_MAIN cpu=%d raw_x=1 band=1_to_500\n",cpu);s53w_kernel*k=kernel_create(2);if(!k)return 5;x63_run(k);kernel_destroy(k);redtab2_clear();flint_cleanup_master();return 0;}
'''
p.write_text(s)
