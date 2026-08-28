from pathlib import Path
import sys

if len(sys.argv)!=3:
    raise SystemExit('usage: make_sine_53_xeon_x26_parallel.py input.c output.c')

src=Path(sys.argv[1])
out=Path(sys.argv[2])
s=src.read_text()
needle='int main(void)'
pos=s.rfind(needle)
if pos<0:
    raise SystemExit('final main marker not found')
s=s[:pos]+'int x23_original_main(void)'+s[pos+len(needle):]

s += r'''

#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <limits.h>

#define X26_MAX_THREADS 64

typedef struct x26_pool x26_pool;
typedef struct { x26_pool *p; int id; int cpu; } x26_worker;
struct x26_pool {
    int t, stop, mode;
    size_t n;
    const double *x;
    double *out;
    const s53w_kernel *k;
    pthread_barrier_t start, done;
    pthread_t th[X26_MAX_THREADS];
    x26_worker w[X26_MAX_THREADS];
};

static inline uint64_t x26_now_ns(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW,&ts);
    return (uint64_t)ts.tv_sec*UINT64_C(1000000000)+(uint64_t)ts.tv_nsec;
}
static int x26_cmpd(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y);}
static long long x26_gcd(long long a,long long b){while(b){long long t=a%b;a=b;b=t;}return a<0?-a:a;}

static int x26_read_int(const char *path,int fallback){
    FILE *f=fopen(path,"r"); int v=fallback; if(f){ if(fscanf(f,"%d",&v)!=1) v=fallback; fclose(f);} return v;
}
static int x26_cpu_order(int *order,int *nphys_out){
    cpu_set_t set; CPU_ZERO(&set); if(sched_getaffinity(0,sizeof(set),&set)!=0) return 0;
    int allowed[X26_MAX_THREADS],na=0;
    for(int c=0;c<CPU_SETSIZE && na<X26_MAX_THREADS;c++) if(CPU_ISSET(c,&set)) allowed[na++]=c;
    int np=0;
    int key_pkg[X26_MAX_THREADS],key_core[X26_MAX_THREADS];
    for(int i=0;i<na;i++){
        char p1[160],p2[160];
        snprintf(p1,sizeof(p1),"/sys/devices/system/cpu/cpu%d/topology/physical_package_id",allowed[i]);
        snprintf(p2,sizeof(p2),"/sys/devices/system/cpu/cpu%d/topology/core_id",allowed[i]);
        int pkg=x26_read_int(p1,0), core=x26_read_int(p2,allowed[i]);
        int seen=0; for(int j=0;j<np;j++) if(key_pkg[j]==pkg && key_core[j]==core){seen=1;break;}
        if(!seen){ key_pkg[np]=pkg; key_core[np]=core; order[np++]=allowed[i]; }
    }
    int nt=np;
    for(int i=0;i<na;i++){
        int used=0; for(int j=0;j<np;j++) if(order[j]==allowed[i]){used=1;break;}
        if(!used && nt<X26_MAX_THREADS) order[nt++]=allowed[i];
    }
    *nphys_out=np; return nt;
}
static void x26_pin_cpu(int cpu){
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu,&s); pthread_setaffinity_np(pthread_self(),sizeof(s),&s);
}

static void *x26_worker_main(void *vp){
    x26_worker *w=(x26_worker*)vp; x26_pool *p=w->p; x26_pin_cpu(w->cpu); mkl_set_num_threads_local(1);
    for(;;){
        pthread_barrier_wait(&p->start);
        if(p->stop){ pthread_barrier_wait(&p->done); break; }
        size_t q=((p->n/(size_t)p->t)/32u)*32u;
        size_t lo=(size_t)w->id*q;
        size_t hi=(w->id==p->t-1)?p->n:lo+q;
        if(hi>lo){
            if(p->mode==0) octant_vector_v8(p->k,p->x+lo,p->out+lo,hi-lo);
            else vmdSin((MKL_INT)(hi-lo),p->x+lo,p->out+lo,VML_HA);
        }
        pthread_barrier_wait(&p->done);
    }
    return 0;
}
static int x26_pool_init(x26_pool *p,int t,const int *cpus,const s53w_kernel *k){
    memset(p,0,sizeof(*p)); p->t=t; p->k=k;
    if(pthread_barrier_init(&p->start,0,(unsigned)t+1u)) return -1;
    if(pthread_barrier_init(&p->done,0,(unsigned)t+1u)) return -1;
    for(int i=0;i<t;i++){
        p->w[i].p=p; p->w[i].id=i; p->w[i].cpu=cpus[i];
        if(pthread_create(&p->th[i],0,x26_worker_main,&p->w[i])) return -1;
    }
    return 0;
}
static void x26_pool_destroy(x26_pool *p){
    p->stop=1; pthread_barrier_wait(&p->start); pthread_barrier_wait(&p->done);
    for(int i=0;i<p->t;i++) pthread_join(p->th[i],0);
    pthread_barrier_destroy(&p->start); pthread_barrier_destroy(&p->done);
}
static void x26_pool_run(x26_pool *p,int mode,const double*x,double*out,size_t n){
    p->mode=mode; p->x=x; p->out=out; p->n=n;
    pthread_barrier_wait(&p->start); pthread_barrier_wait(&p->done);
}
static void x26_direct_run(int mode,const s53w_kernel*k,const double*x,double*out,size_t n){
    if(mode==0) octant_vector_v8(k,x,out,n); else vmdSin((MKL_INT)n,x,out,VML_HA);
}

static void x26_make_input(double*x,size_t n){
    size_t m=(n+2u)/3u;
    long long a=(long long)(104729u % (m?m:1u)); if(a<3) a=3; if((a&1)==0) a++;
    while(x26_gcd(a,(long long)m)!=1) a+=2;
    for(size_t i=0;i<n;i++){
        size_t k=i/3u; size_t j=(size_t)(((unsigned long long)k*(unsigned long long)a+12345ull)%(unsigned long long)m);
        double u=((double)j+0.5)/(double)m;
        double v;
        switch(i%3u){case 0:v=0x1p-20+(1.0-0x1p-19)*u;break;case 1:v=1.0+499.0*u;break;default:v=1000.0+9000.0*u;break;}
        x[i]=(i&1u)?-v:v;
    }
}

static void x26_measure(size_t n,int t,const int*cpus,const s53w_kernel*k,double*x,double*out,
                        double *ours_ns,double *intel_ns){
    const int trials=(n<=9600?31:(n<=65536?21:11));
    const int warm=(n<=9600?20:5);
    double ot[31],it[31];
    x26_pool pool; int usepool=(t>1);
    if(usepool && x26_pool_init(&pool,t,cpus,k)!=0){fprintf(stderr,"X26 pool init failed\n");exit(7);}    
    if(t==1) x26_pin_cpu(cpus[0]);
    for(int w=0;w<warm;w++){
        if(usepool){x26_pool_run(&pool,0,x,out,n);x26_pool_run(&pool,1,x,out,n);} else {x26_direct_run(0,k,x,out,n);x26_direct_run(1,k,x,out,n);}    
    }
    for(int r=0;r<trials;r++){
        uint64_t a,b;
        if((r&1)==0){
            a=x26_now_ns(); if(usepool)x26_pool_run(&pool,0,x,out,n);else x26_direct_run(0,k,x,out,n); b=x26_now_ns(); ot[r]=(double)(b-a)/(double)n;
            a=x26_now_ns(); if(usepool)x26_pool_run(&pool,1,x,out,n);else x26_direct_run(1,k,x,out,n); b=x26_now_ns(); it[r]=(double)(b-a)/(double)n;
        }else{
            a=x26_now_ns(); if(usepool)x26_pool_run(&pool,1,x,out,n);else x26_direct_run(1,k,x,out,n); b=x26_now_ns(); it[r]=(double)(b-a)/(double)n;
            a=x26_now_ns(); if(usepool)x26_pool_run(&pool,0,x,out,n);else x26_direct_run(0,k,x,out,n); b=x26_now_ns(); ot[r]=(double)(b-a)/(double)n;
        }
    }
    qsort(ot,trials,sizeof(double),x26_cmpd); qsort(it,trials,sizeof(double),x26_cmpd);
    *ours_ns=ot[trials/2]; *intel_ns=it[trials/2];
    if(usepool)x26_pool_destroy(&pool);
}

int main(void){
    mkl_set_dynamic(0); mkl_set_num_threads(1); mkl_set_num_threads_local(1);
    int cpus[X26_MAX_THREADS],nphys=0; int nlog=x26_cpu_order(cpus,&nphys);
    if(nlog<1){fprintf(stderr,"X26 no allowed CPUs\n");return 2;}
    printf("X26_TOPOLOGY logical=%d physical=%d cpu_order=",nlog,nphys); for(int i=0;i<nlog;i++)printf("%s%d",i?",":"",cpus[i]); printf("\n");
    s53w_kernel *k=kernel_create(2); if(!k)return 3;
    const size_t sizes[]={1200u,9600u,65536u,1048576u};
    const int nsizes=4;
    size_t maxn=sizes[nsizes-1];
    double *x=(double*)aligned_alloc(64,((maxn*sizeof(double)+63u)/64u)*64u);
    double *out=(double*)aligned_alloc(64,((maxn*sizeof(double)+63u)/64u)*64u);
    if(!x||!out)return 4;
    int ts[3],nt=0; ts[nt++]=1; if(nphys>1)ts[nt++]=nphys; if(nlog>1 && (nt==0||ts[nt-1]!=nlog))ts[nt++]=nlog;
    if(nt>3)nt=3;
    for(int si=0;si<nsizes;si++){
        size_t n=sizes[si]; x26_make_input(x,n);
        double base_o=0,base_i=0;
        for(int ti=0;ti<nt;ti++){
            int t=ts[ti]; if(t>nlog)t=nlog;
            double on,in; x26_measure(n,t,cpus,k,x,out,&on,&in);
            if(t==1){base_o=on;base_i=in;}
            double os=(base_o>0?base_o/on:1.0), is=(base_i>0?base_i/in:1.0);
            double oe=os/(double)t, ie=is/(double)t;
            const char *kind=(t<=nphys?"physical":"logical_smt");
            printf("X26_RESULT n=%zu workers=%d kind=%s ours_ns=%.6f intel_ns=%.6f ours_over_intel=%.6fx ours_speedup=%.4fx intel_speedup=%.4fx ours_eff=%.4f intel_eff=%.4f\n",
                   n,t,kind,on,in,on/in,os,is,oe,ie);
        }
    }
    free(out);free(x);kernel_destroy(k);flint_cleanup_master();return 0;
}
'''

out.write_text(s)
print('X26_BUILD_PASS outer_parallel=persistent_pthreads static_contiguous_chunks=1 physical_core_first=1 smt_after_physical=1 same_outer_pool_for_ours_and_intel=1 mkl_internal_threads=1 x23_h14_math_unchanged=1')
