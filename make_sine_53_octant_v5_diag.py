from pathlib import Path
import runpy

# Build the v4 source first, then add diagnostics only.  Numerical path is unchanged.
runpy.run_path('make_sine_53_octant_v4.py', run_name='__main__')
src = Path('bench_sine_53_wide_octant_v4_build.c').read_text()

needle = 'uq++;uint64_t uo=ulpd(o[i],a),ui=ulpd(in[i],a);if(!uo)oe++;if(uo<=1)o1++;if(uo>om)om=uo;if(!ui)ie++;if(ui<=1)i1++;if(ui>im)im=ui;'
repl = r'''uq++;uint64_t uo=ulpd(o[i],a),ui=ulpd(in[i],a);
        if(uo>1){
            double xx=x[i], axx=fabs(xx), qf=axx*FOUR_OVER_PI, qd=trunc(qf), frac=qf-qd;
            const double ft=BOUND_TAU*FOUR_OVER_PI;
            int gd=(axx>=1.0)&&(frac<ft||frac>1.0-ft);
            double sf=scalar2(k,xx); uint64_t su=ulpd(sf,a);
            int qi=(int)qd, oct=qi&7, m=qi, rev=0;
            if(oct==1||oct==5)m=qi-1;
            else if(oct==2||oct==6){m=qi+2;rev=1;}
            else if(oct==3||oct==7){m=qi+1;rev=1;}
            double yy=rev?fma((double)m,PIO4_HI,-axx):fma(-(double)m,PIO4_HI,axx);
            yy=rev?fma((double)m,PIO4_LO,yy):fma(-(double)m,PIO4_LO,yy);
            yy=rev?fma((double)m,PIO4_TINY,yy):fma(-(double)m,PIO4_TINY,yy);
            printf("S53O5_MISS tag=%s i=%d x=%.17g ours=%.17g ref=%.17g ulp=%lu scalar2=%.17g scalar2_ulp=%lu q=%d oct=%d m=%d rev=%d y3=%.17g frac=%.17g guard=%d ft=%.17g\n",
                   tag,i,xx,o[i],a,(unsigned long)uo,sf,(unsigned long)su,qi,oct,m,rev,yy,frac,gd,ft);
        }
        if(!uo)oe++;if(uo<=1)o1++;if(uo>om)om=uo;if(!ui)ie++;if(ui<=1)i1++;if(ui>im)im=ui;'''
if needle not in src:
    raise SystemExit('verify needle missing')
src = src.replace(needle,repl,1)
src = src.replace('S53O4_', 'S53O5_')
Path('bench_sine_53_wide_octant_v5_diag_build.c').write_text(src)
print('S53O5_BUILD_PASS path=v4_unchanged diagnostics=miss_guard_scalar2')
