OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const __m512i M7=_mm512_set1_epi64(7);
    const double *ap=(const double *)__builtin_assume_aligned(k->anchor_pack,64);
    const __m512d SA=_mm512_load_pd(ap+0*8),CA=_mm512_load_pd(ap+1*8);
    const __m512d SB=_mm512_load_pd(ap+2*8),CB=_mm512_load_pd(ap+3*8);
    const __m512d SC=_mm512_load_pd(ap+4*8),CC=_mm512_load_pd(ap+5*8);
    size_t i=0;
    for(;i+32<=n;i+=32){
        for(int b=0;b<4;b++){
            __m512d rh,rl; __mmask8 sg,gd,aa; unsigned char pu;
            x12_prepare_block(x+i,8*b,32,&rh,&rl,&sg,&gd,&aa,&pu);
            __m256i j32=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
            __m512i j=_mm512_cvtepi32_epi64(j32);
            __m512i ia=_mm512_and_epi64(j,M7);
            __m512i ib=_mm512_and_epi64(_mm512_srli_epi64(j,3),M7);
            __m512i ic=_mm512_srli_epi64(j,6);
            __m512d sa=_mm512_permutexvar_pd(ia,SA),ca=_mm512_permutexvar_pd(ia,CA);
            __m512d sb=_mm512_permutexvar_pd(ib,SB),cb=_mm512_permutexvar_pd(ib,CB);
            __m512d sc=_mm512_permutexvar_pd(ic,SC),cc=_mm512_permutexvar_pd(ic,CC);
            __m512d s01=_mm512_fmadd_pd(sa,cb,_mm512_mul_pd(ca,sb));
            __m512d c01=_mm512_fnmadd_pd(sa,sb,_mm512_mul_pd(ca,cb));
            __m512d c0=_mm512_fmadd_pd(s01,cc,_mm512_mul_pd(c01,sc));
            __m512d c1=_mm512_fnmadd_pd(s01,sc,_mm512_mul_pd(c01,cc));
            __m512d jd=_mm512_cvtepi32_pd(j32);
            __m512d d=_mm512_add_pd(_mm512_fnmadd_pd(jd,VIK,rh),rl);
            __m512d c2=_mm512_mul_pd(c0,MH),c3=_mm512_mul_pd(c1,M6),c4=_mm512_mul_pd(c0,C24),c5=_mm512_mul_pd(c1,C120);
            __m512d y=_mm512_fmadd_pd(c5,d,c4);
            y=_mm512_fmadd_pd(y,d,c3); y=_mm512_fmadd_pd(y,d,c2);
            y=_mm512_fmadd_pd(y,d,c1); y=_mm512_fmadd_pd(y,d,c0);
            y=_mm512_mask_sub_pd(y,sg,Z,y);
            _mm512_storeu_pd(out+i+8*b,y);
            if(__builtin_expect(gd!=0,0)) for(unsigned lane=0;lane<8;lane++) if(gd&(1u<<lane)) out[i+8*b+lane]=scalar2(k,x[i+8*b+lane]);
        }
    }
    if(i<n) octant_vector_x20_tail(k,x+i,out+i,n-i);
}
