from pathlib import Path
import runpy, math
import mpmath as mp

# X65 FINAL frozen math:
# user spine b=sec*tan, D=sec-1, sin=b/(1+D)^2.
# Algebraic addition inside that spine gives
# sin(a+d)=sin(a)cos(d)+cos(a)sin(d).
# State a=n*pi/512; tiny d=x-a. Anchor sin/cos are generated OFFLINE from
# the same Mode-5 secant builder: secant-derived cos(pi/512),
# sin=sqrt((1-c)(1+c)), then high-precision angle-addition recurrence.
# Runtime: raw x -> right-shifter n -> j=n mod 512 -> split pi/512 residual
# -> two gathers -> same degree-5 local polynomial as <1.
# No runtime external sin/cos/fmod/remainder.
runpy.run_path('make_sine_53_xeon_x56_wide_specialized_cold.py', run_name='__main__')
p=Path('bench_sine_53_xeon_x56_build.c')
s=p.read_text()
hit=s.index('octant_vector_v8(const s53w_kernel *k,')
start=s.rfind('\n',0,hit)+1
end=s.index('\n#endif',hit)
orig=s[start:end]
general=orig.replace('octant_vector_v8(', 'octant_vector_v8_x56_general(', 1)

# ---- reproduce the Mode-5 secant seed in high precision (no sin/cos oracle) ----
mp.mp.dps=180
pi=mp.pi
A0=mp.mpf(4)/pi
A1=-A0/3
Q0=mp.mpf(4)/(pi*pi)
Q1=Q0/9
q0=mp.mpf(1); q1=-(Q0+Q1); q2=Q0*Q1
p0=A0*Q0+A1*Q1
p1=-q2*(A0+A1)

# Euler numbers E_0,E_2,... from the exact recurrence.
E={0:1}
for n in range(2,50,2):
    E[n] = -sum(math.comb(n,k)*E[k] for k in range(0,n,2))
res=[]
qp0=Q0; qp1=Q1
for k in range(24):
    m=k+1
    G=mp.mpf(abs(E[2*m]))/mp.factorial(2*m)
    res.append(G-(A0*qp0+A1*qp1))
    qp0*=Q0; qp1*=Q1

delta=pi/mp.mpf(512)
z=delta*delta
R=res[-1]
for v in reversed(res[:-1]): R=R*z+v
gf=1+z*R
Q=(q2*z+q1)*z+q0
P=p1*z+p0
den=Q*gf+z*P
seedc=Q/den
seeds=mp.sqrt((1-seedc)*(1+seedc))

ct=[mp.mpf(1)]; st=[mp.mpf(0)]
for j in range(1,512):
    c=ct[-1]*seedc-st[-1]*seeds
    q=st[-1]*seedc+ct[-1]*seeds
    ct.append(c); st.append(q)
# exact cardinal anchors, important for near-zero/near-extremum lanes
ct[0]=mp.mpf(1); st[0]=mp.mpf(0)
ct[256]=mp.mpf(0); st[256]=mp.mpf(1)

def hx(v): return float(v).hex()
CT=','.join(hx(v) for v in ct)
ST=','.join(hx(v) for v in st)
h=pi/mp.mpf(512)
hhi=mp.mpf(float(h))
hlo=h-hhi
vinv=mp.mpf(512)/pi

raw=f'''static const double x65_s[512] __attribute__((aligned(64)))={{ {ST} }};
static const double x65_c[512] __attribute__((aligned(64)))={{ {CT} }};

OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawx65(
        const s53w_kernel *k,const double * __restrict x,double * __restrict out,size_t n)
{{
    (void)k;
    const __m512d VINV=_mm512_set1_pd({hx(vinv)});
    const __m512d RS=_mm512_set1_pd(0x1.8p52);
    const __m512d HHI=_mm512_set1_pd({hx(hhi)});
    const __m512d HLO=_mm512_set1_pd({hx(hlo)});
    const __m512d Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const __m256i I511=_mm256_set1_epi32(511);
    size_t i=0;
    for(;i+32<=n;i+=32){{
        __m512d vx[4],N[4],d[4],c0[4],c1[4],p[4];
        __m256i ji[4]; __mmask8 sg[4];
        /* coefficient-major / G4: form all states first */
        for(int g=0;g<4;g++){{
            vx[g]=_mm512_loadu_pd(x+i+8*g);
            __m512d Y=_mm512_fmadd_pd(vx[g],VINV,RS);
            N[g]=_mm512_sub_pd(Y,RS);
            __m512i yb=_mm512_castpd_si512(Y);
            /* low 9 bits are n mod 512; bit 9 is the pi-cell sign. */
            ji[g]=_mm256_and_si256(_mm512_cvtepi64_epi32(yb),I511);
            sg[g]=_mm512_movepi64_mask(_mm512_slli_epi64(yb,54));
        }}
        /* Gather early so memory latency overlaps residual work. */
        for(int g=0;g<4;g++){{
            c0[g]=_mm512_i32gather_pd(ji[g],x65_s,8);
            c1[g]=_mm512_i32gather_pd(ji[g],x65_c,8);
        }}
        for(int g=0;g<4;g++){{
            /* Two-FMA split pi/512 reduction. FMA cancellation makes this DD-like
               without an explicit error accumulator. */
            d[g]=_mm512_fnmadd_pd(N[g],HHI,vx[g]);
            d[g]=_mm512_fnmadd_pd(N[g],HLO,d[g]);
        }}
        /* Same degree-5 local evaluator as the successful <1 core. */
        for(int g=0;g<4;g++){{
            __m512d c2=_mm512_mul_pd(c0[g],MH);
            __m512d c3=_mm512_mul_pd(c1[g],M6);
            __m512d c4=_mm512_mul_pd(c0[g],C24);
            __m512d c5=_mm512_mul_pd(c1[g],C120);
            p[g]=_mm512_fmadd_pd(c5,d[g],c4);
            p[g]=_mm512_fmadd_pd(p[g],d[g],c3);
            p[g]=_mm512_fmadd_pd(p[g],d[g],c2);
            p[g]=_mm512_fmadd_pd(p[g],d[g],c1[g]);
            p[g]=_mm512_fmadd_pd(p[g],d[g],c0[g]);
            p[g]=_mm512_mask_sub_pd(p[g],sg[g],Z,p[g]);
            _mm512_storeu_pd(out+i+8*g,p[g]);
        }}
    }}
    if(i<n) octant_vector_v8_x56_general(k,x+i,out+i,n-i);
}}

OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(
        const s53w_kernel *k,const double * __restrict x,double * __restrict out,size_t n)
{{
    /* Dedicated frozen wide path for homogeneous |x| in [1,10000]. */
    if(n>=32){{
        size_t j=0;
        for(;j<n;j++){{double a=fabs(x[j]);if(a<1.0||a>10000.0)break;}}
        if(j==n){{octant_vector_v8_rawx65(k,x,out,n);return;}}
    }}
    octant_vector_v8_x56_general(k,x,out,n);
}}
'''

s=s[:start]+general+'\n'+raw+s[end:]
s=s.replace('S53X56_','S53X65_')
s=s.replace('xeon_x56_wide_specialized_cold_g4','xeon_x65_secant_state512_final_g4')
s=s.replace('Xeon_AVX512_X56_wide_specialized_cold','Xeon_AVX512_X65_secant_state512_final')
Path('bench_sine_53_xeon_x65_build.c').write_text(s)
print('S53X65_BUILD_PASS frozen_math=1 user_secant_spine=1 state512=1 anchors_from_Mode5_secant=1 raw_x=1 right_shifter_state=1 split_pi512_twoFMA=1 G4_AVX512=1 coefficient_major=1 gathers=2 degree5_same_as_lt1=1 parity_final_mask=1 runtime_sin_cos_fmod_remainder=0 requires_Arb_regate=1')
