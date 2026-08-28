from pathlib import Path

src = Path('bench_sine_53_wide_octant_v2.c').read_text()

start = src.index('OVEC static void octant_vector_v2')
end = src.index('\n#endif', start)
new_vec = r'''OVEC static inline __m512d mode5_poly_dd_i32(const s53w_kernel *k,
                                             __m512d yh,__m512d yl,
                                             __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    __m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(yh,VK),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256i ji=_mm512_cvttpd_epi32(jd);
    __m512d d=_mm512_fnmadd_pd(jd,VIK,yh);
    __m512d p=_mm512_i32gather_pd(ji,k->tab+(size_t)k->deg*LUTN,8),dp=Z;
    for(int j=k->deg-1;j>=0;j--){
        dp=_mm512_fmadd_pd(dp,d,p);
        __m512d c=_mm512_i32gather_pd(ji,k->tab+(size_t)j*LUTN,8);
        p=_mm512_fmadd_pd(p,d,c);
    }
    /* P(d+yl)=P(d)+yl*P'(d)+O(yl^2); yl is at the subtraction-roundoff scale. */
    p=_mm512_fmadd_pd(yl,dp,p);
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

OVEC static void octant_vector_v2(const s53w_kernel *k,const double *x,
                                  double *out,size_t n)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d V4OPI=_mm512_set1_pd(FOUR_OVER_PI);
    const __m512d VP4H=_mm512_set1_pd(PIO4_HI),VP4L=_mm512_set1_pd(PIO4_LO);
    const __m512d VFT=_mm512_set1_pd(BOUND_TAU*FOUR_OVER_PI);
    const __m512d V1MFT=_mm512_set1_pd(1.0-BOUND_TAU*FOUR_OVER_PI);
    const __m512d VM1=_mm512_set1_pd(-1.0),VP1=_mm512_set1_pd(1.0),VP2=_mm512_set1_pd(2.0);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));

    for(size_t i=0;i<n;i+=8){
        unsigned rem=(unsigned)(n-i);
        __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
        __m512d vx=_mm512_maskz_loadu_pd(active,x+i);
        __mmask8 inneg=(__mmask8)(_mm512_cmp_pd_mask(vx,Z,_CMP_LT_OQ)&active);
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vx),ABSM));
        __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);

        /* Keep the established unit-domain champion literally untouched. */
        if(unit==active){
            __m512d p=mode5_poly_i32(k,ax,inneg);
            _mm512_mask_storeu_pd(out+i,active,p);
            continue;
        }

        __mmask8 wide=(__mmask8)(active&~unit);
        __m512d qf=_mm512_mul_pd(ax,V4OPI);
        __m256i ki=_mm512_cvttpd_epi32(qf);
        __m512d kd=_mm512_cvtepi32_pd(ki);

        __m512d frac=_mm512_sub_pd(qf,kd);
        __mmask8 near0=_mm512_cmp_pd_mask(frac,VFT,_CMP_LT_OQ);
        __mmask8 near1=_mm512_cmp_pd_mask(frac,V1MFT,_CMP_GT_OQ);
        __mmask8 guarded=(__mmask8)((near0|near1)&wide);

        __m256i oi=_mm256_and_si256(ki,_mm256_set1_epi32(7));
        __mmask8 m1=mask_eq_i32(oi,1),m2=mask_eq_i32(oi,2),m3=mask_eq_i32(oi,3);
        __mmask8 m4=mask_eq_i32(oi,4),m5=mask_eq_i32(oi,5),m6=mask_eq_i32(oi,6),m7=mask_eq_i32(oi,7);
        __mmask8 am1=(__mmask8)((m1|m5)&wide);
        __mmask8 ap2=(__mmask8)((m2|m6)&wide);
        __mmask8 ap1=(__mmask8)((m3|m7)&wide);
        __mmask8 rev=(__mmask8)((m2|m3|m6|m7)&wide);
        __mmask8 wide_neg=(__mmask8)((m4|m5|m6|m7)&wide);

        __m512d md=kd;
        md=_mm512_mask_add_pd(md,am1,md,VM1);
        md=_mm512_mask_add_pd(md,ap2,md,VP2);
        md=_mm512_mask_add_pd(md,ap1,md,VP1);

        /* Direct folded-angle DD residual, no q*pi table gathers.
           ph+pe is the exact product md*PIO4_HI (TwoProduct via FMA).
           pl supplies the split-low piece. TwoSum/TwoDiff preserve the final
           subtraction roundoff as yl, which is then consumed by P'(d). */
        __m512d ph=_mm512_mul_pd(md,VP4H);
        __m512d pe=_mm512_fmadd_pd(md,VP4H,_mm512_sub_pd(Z,ph));
        __m512d pl=_mm512_mul_pd(md,VP4L);

        __m512d bhp,blp,bhn,bln;
        twos2v(ax,_mm512_sub_pd(Z,ph),&bhp,&blp);       /* ax-ph */
        twos2v(ph,_mm512_sub_pd(Z,ax),&bhn,&bln);       /* ph-ax */
        __m512d ep=_mm512_sub_pd(Z,_mm512_add_pd(pe,pl));
        __m512d en=_mm512_add_pd(pe,pl);
        __m512d yhp,e2p,yhn,e2n;
        twos2v(bhp,ep,&yhp,&e2p);
        twos2v(bhn,en,&yhn,&e2n);
        __m512d ylp=_mm512_add_pd(blp,e2p);
        __m512d yln=_mm512_add_pd(bln,e2n);

        __m512d yh=_mm512_mask_mov_pd(yhp,rev,yhn);
        __m512d yl=_mm512_mask_mov_pd(ylp,rev,yln);
        yh=_mm512_mask_mov_pd(yh,unit,ax);
        yl=_mm512_mask_mov_pd(yl,unit,Z);

        /* Boundary lanes are recomputed with the older full table-DD reducer. */
        yh=_mm512_mask_mov_pd(yh,guarded,Z);
        yl=_mm512_mask_mov_pd(yl,guarded,Z);

        __mmask8 signmask=(__mmask8)(((wide_neg^inneg)&wide)|(inneg&unit));
        __m512d p=mode5_poly_dd_i32(k,yh,yl,signmask);
        _mm512_mask_storeu_pd(out+i,active,p);

        if(__builtin_expect(guarded!=0,0)){
            for(unsigned lane=0;lane<8&&i+lane<n;lane++)
                if(guarded&(1u<<lane)) out[i+lane]=scalar2(k,x[i+lane]);
        }
    }
}'''
src = src[:start] + new_vec + src[end:]

# Make diagnostics count exactly the same rare path used by the vector code.
gs = src.index('static int guard_count_v2')
ge = src.index('\n}\n\nstatic int verify_v2', gs) + 2
new_guard = r'''static int guard_count_v2(const double *x,int n)
{
    int c=0; const double ft=BOUND_TAU*FOUR_OVER_PI;
    for(int i=0;i<n;i++){
        double ax=fabs(x[i]);if(ax<1.0)continue;
        double qf=ax*FOUR_OVER_PI;double q=trunc(qf);double frac=qf-q;
        if(frac<ft||frac>1.0-ft)c++;
    }
    return c;
}'''
src = src[:gs] + new_guard + src[ge:]

# Keep failure diagnostics until all three accuracy gates pass on AVX-512.
needle = 'uq++;uint64_t uo=ulpd(o[i],a),ui=ulpd(in[i],a);if(!uo)oe++;if(uo<=1)o1++;if(uo>om)om=uo;if(!ui)ie++;if(ui<=1)i1++;if(ui>im)im=ui;'
repl = r'''uq++;uint64_t uo=ulpd(o[i],a),ui=ulpd(in[i],a);
        if(uo>1){
            double xx=x[i],axx=fabs(xx),qf=axx*FOUR_OVER_PI,qd=trunc(qf),frac=qf-qd;
            int qi=(int)qd,oct=qi&7,m=qi,rev=0;
            if(oct==1||oct==5)m=qi-1;
            else if(oct==2||oct==6){m=qi+2;rev=1;}
            else if(oct==3||oct==7){m=qi+1;rev=1;}
            double yy=rev?fma((double)m,PIO4_HI,-axx):fma(-(double)m,PIO4_HI,axx);
            yy=rev?fma((double)m,PIO4_LO,yy):fma(-(double)m,PIO4_LO,yy);
            double sf=scalar2(k,xx);uint64_t su=ulpd(sf,a);double ft=BOUND_TAU*FOUR_OVER_PI;
            int gd=(axx>=1.0)&&(frac<ft||frac>1.0-ft);
            printf("S53O3_MISS tag=%s i=%d x=%.17g ours=%.17g ref=%.17g ulp=%lu scalar_dd=%.17g scalar_dd_ulp=%lu q=%d oct=%d m=%d folded_y=%.17g frac=%.17g guarded=%d\n",
                   tag,i,xx,o[i],a,(unsigned long)uo,sf,(unsigned long)su,qi,oct,m,yy,frac,gd);
        }
        if(!uo)oe++;if(uo<=1)o1++;if(uo>om)om=uo;if(!ui)ie++;if(ui<=1)i1++;if(ui>im)im=ui;'''
if needle not in src:
    raise SystemExit('verify needle missing')
src = src.replace(needle,repl,1)

src = src.replace('S53O2_', 'S53O3_')
src = src.replace('octant_v2', 'octant_v3')
src = src.replace('_v2', '_v3')
src = src.replace('guarded_v2', 'guarded_v3')
src = src.replace('cosine_style_pi4_octant_guarded_v2', 'cosine_style_pi4_octant_guarded_v3_direct_multiple_ddlow')
src = src.replace('AVX512_pi4_octant_int32_split', 'AVX512_pi4_octant_direct_multiple_DDlow_int32')
Path('bench_sine_53_wide_octant_v3_build.c').write_text(src)
print('S53O3_BUILD_PASS direct_multiple_fold=1 dd_low_residual=1 derivative_correction=1 rare_dd_boundary=1 unit_direct=1 miss_diag=1')
