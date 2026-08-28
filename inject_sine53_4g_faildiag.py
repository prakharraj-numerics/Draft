from pathlib import Path
import sys

path=Path(sys.argv[1] if len(sys.argv)>1 else 'bench_sine_53_xeon_v2_build.c')
s=path.read_text()
mainpos=s.rfind('\nint main(void)')
if mainpos < 0: raise SystemExit('main not found')
code=r'''
static void s53x2_fail_geometry(void)
{
    const int N=32768;
    double *x=al64((size_t)N*sizeof(double)),*o=al64((size_t)N*sizeof(double));
    if(!x||!o){free(o);free(x);return;}
    make_stress(x);
    s53w_kernel *k=kernel_create(2);if(!k){free(o);free(x);return;}
    octant_eval_x2(k,x,o,N);
    arb_t ax,ay; arf_t lo,hi; arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);
    int bad=0,anchor_bad[LUTN]; memset(anchor_bad,0,sizeof(anchor_bad));
    for(int i=0;i<N;i++){
        arb_set_d(ax,x[i]);arb_sin(ay,ax,256);arb_get_lbound_arf(lo,ay,256);arb_get_ubound_arf(hi,ay,256);
        double a=arf_get_d(lo,ARF_RND_NEAR),b=arf_get_d(hi,ARF_RND_NEAR);if(dbits(a)!=dbits(b))continue;
        uint64_t u=ulpd(o[i],a); if(u<=1)continue;
        double xx=fabs(x[i]),rh,rl; int q=0,oi=0,m=0,rev=0;
        if(xx<1.0){rh=xx;rl=0.0;}
        else{
            double qf=xx*FOUR_OVER_PI; q=(int)qf; oi=q&7;
            static const int adj[8]={0,-1,2,1,0,-1,2,1}; m=q+adj[oi]; rev=((oi&2)!=0);
            double md=(double)m,t1=md*PIO4_CW1,r0=xx-t1,t2=md*PIO4_CW2;
            double h=r0-t2,bv=r0-h,av=h+bv,br=bv-t2,ar=r0-av;
            rh=h;rl=fma(-md,PIO4_CW3,ar+br); if(rev){rh=-rh;rl=-rl;}
        }
        double ya=rh+rl; long j=(long)nearbyint(ya*KGRID);if(j<0)j=0;if(j>=(long)LUTN)j=(long)LUTN-1;
        double d=(rh-(double)j*INVK)+rl;
        anchor_bad[j]++; bad++;
        if(bad<=256)printf("S53X2_FAIL_GEOM i=%d x=%.17g ulp=%lu q=%d oct=%d anchor=%ld d=%+.17e absd=%.17e\n",i,x[i],(unsigned long)u,q,oi,j,d,fabs(d));
    }
    printf("S53X2_FAIL_SUMMARY bad=%d",bad);for(int j=0;j<(int)LUTN;j++)if(anchor_bad[j])printf(" j%d=%d",j,anchor_bad[j]);printf("\n");
    arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);kernel_destroy(k);free(o);free(x);
}
'''
s=s[:mainpos]+code+s[mainpos:]
needle='int main(void){'
pos=s.rfind(needle)
if pos<0: raise SystemExit('main marker not found after insert')
brace=pos+len(needle)
s=s[:brace]+'s53x2_fail_geometry();'+s[brace:]
path.write_text(s)
print('S53X2_FAILDIAG_INJECT_PASS')
