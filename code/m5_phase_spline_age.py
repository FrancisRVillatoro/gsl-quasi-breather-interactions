
from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from ssf4_gsl import grid, ssf4_step

HERE=Path(__file__).resolve().parents[1] / "data"
H=(1,3,5,7)

def initial_qb(x,w):
    q=1/np.sqrt(1+w*w)
    return np.zeros_like(x),4*q/np.cosh(q*x)

def G(u,b):
    y=1-np.cos(u)
    return 2*y/(1+np.sqrt(1+b*b*y))

def positive_zero_crossings(t,y):
    ids=np.where((y[:-1]<=0)&(y[1:]>0))[0]
    tc=[]
    for i in ids:
        a=-y[i]/(y[i+1]-y[i])
        tc.append(t[i]+a*(t[i+1]-t[i]))
    return np.asarray(tc)

def build_phase_spline(t,y):
    tc=positive_zero_crossings(t,y)
    # arbitrary phase origin; only phase differences matter
    ph=2*np.pi*np.arange(len(tc),dtype=float)
    spl=CubicSpline(tc,ph)
    return spl,tc

def phase_design(tau,theta):
    cols=[np.ones_like(tau),tau]
    for n in H:
        c=np.cos(n*theta); s=np.sin(n*theta)
        cols += [c,s,tau*c,tau*s]
    return np.column_stack(cols)

def fixed_coeff(t,U,V,tref,theta):
    tau=t-tref
    A=phase_design(tau,theta)
    CU,*_=np.linalg.lstsq(A,U,rcond=None)
    CV,*_=np.linalg.lstsq(A,V,rcond=None)
    rowsU=[CU[0]]; rowsV=[CV[0]]; j=2
    for n in H:
        rowsU += [CU[j],CU[j+1]]
        rowsV += [CV[j],CV[j+1]]
        j+=4
    return np.asarray(rowsU),np.asarray(rowsV)

def fit_tail(x,amp,kpred):
    peak=float(amp.max()); cand=[]
    for up,lo in [(0.45,0.08),(0.4,0.05),(0.3,0.03),(0.2,0.02)]:
        m=(x>0)&(amp<up*peak)&(amp>lo*peak)
        if m.sum()<12: continue
        rr=x[m]; yy=np.log(amp[m])
        sl,it=np.polyfit(rr,yy,1); kap=-float(sl)
        if kap<=0: continue
        yf=sl*rr+it
        r2=1-((yy-yf)**2).sum()/((yy-yy.mean())**2).sum()
        score=r2-0.02*abs(kap-kpred)/max(kpred,1e-12)
        cand.append((score,r2,kap,float(np.exp(it)),float(rr.min()),float(rr.max())))
    if not cand:
        return (np.nan,)*5
    cand.sort(key=lambda z:z[0],reverse=True)
    _,r2,kap,C,rmin,rmax=cand[0]
    return r2,kap,C,rmin,rmax

def fit_phase(delta,F,nmax=4):
    cols=[np.ones_like(delta)]
    for n in range(1,nmax+1):
        cols += [np.cos(n*delta),np.sin(n*delta)]
    A=np.column_stack(cols)
    c,*_=np.linalg.lstsq(A,F,rcond=None)
    z={}
    for n in range(1,nmax+1):
        z[f"cos{n}"]=float(c[1+2*(n-1)])
        z[f"sin{n}"]=float(c[2+2*(n-1)])
    return z

def analyze(x,k,b,w,t,U,V,tref,T0,phase_spline,halfwindow):
    tau=t-tref
    m=np.abs(tau)<=halfwindow*T0+1e-12
    t=t[m]; U=U[m]; V=V[m]; tau=t-tref
    theta=phase_spline(t)
    theta0=float(phase_spline(tref))

    CU,CV=fixed_coeff(t,U,V,tref,theta)
    CUx=np.fft.irfft(1j*k[None,:]*np.fft.rfft(CU,axis=1),n=len(x),axis=1)

    omega=float(phase_spline(tref,1))
    alpha=float(phase_spline(tref,2))
    kappa=np.sqrt(max(0,1-omega*omega))

    # reconstruction RMS at center
    ic=len(x)//2
    yrec=CU[0,ic]*np.ones_like(t)
    tau=t-tref
    for n in H:
        j=1+2*H.index(n)
        yrec += (CU[j,ic]+tau*0)*np.cos(n*theta) + (CU[j+1,ic]+tau*0)*np.sin(n*theta)
    rms=float(np.sqrt(np.mean((U[:,ic]-yrec)**2)))

    amp1=np.sqrt(CU[1]**2+CU[2]**2)
    tr2,ktail,Ctail,rmin,rmax=fit_tail(x,amp1,kappa)

    def point(phi,idx):
        # phi is relative phase offset added to absolute theta0.
        th=theta0+phi
        uu=float(CU[0,idx]); vv=float(CV[0,idx]); ux=float(CUx[0,idx])
        for n in H:
            j=1+2*H.index(n); c=math.cos(n*th); s=math.sin(n*th)
            uu += CU[j,idx]*c+CU[j+1,idx]*s
            vv += CV[j,idx]*c+CV[j+1,idx]*s
            ux += CUx[j,idx]*c+CUx[j+1,idx]*s
        return uu,vv,ux

    # Better: relative-phase average uses a common phase psi, not absolute theta0.
    def point_psi(psi,idx):
        uu=float(CU[0,idx]); vv=float(CV[0,idx]); ux=float(CUx[0,idx])
        for n in H:
            j=1+2*H.index(n); c=math.cos(n*psi); s=math.sin(n*psi)
            uu += CU[j,idx]*c+CU[j+1,idx]*s
            vv += CV[j,idx]*c+CV[j+1,idx]*s
            ux += CUx[j,idx]*c+CUx[j+1,idx]*s
        return uu,vv,ux

    def fund_psi(psi,idx):
        c=math.cos(psi); s=math.sin(psi)
        return (float(CU[1,idx]*c+CU[2,idx]*s),
                float(CV[1,idx]*c+CV[2,idx]*s),
                float(CUx[1,idx]*c+CUx[2,idx]*s))

    def cross(Lp,Rp):
        uL,vL,xL=Lp; uR,vR,xR=Rp
        return float(G(uL+uR,b)-G(uL,b)-G(uR,b)-vL*vR-xL*xR)

    def qcross(Lp,Rp):
        uL,vL,xL=Lp; uR,vR,xR=Rp
        return float(uL*uR-vL*vR-xL*xR)

    D=2+3*b*b
    mu=(41*b**4+28*b*b+4)/(D*D)
    nd=48; nt=192
    deltas=2*np.pi*np.arange(nd)/nd
    psis=2*np.pi*np.arange(nt)/nt
    Z=CU[1]-1j*CU[2]
    Zx=CUx[1]-1j*CUx[2]
    dx=x[1]-x[0]

    fr=[]
    for target in (6.,7.,8.):
        mm=max(1,int(round((target/kappa)/(2*dx))))
        R=2*mm*dx; sact=kappa*R
        ip=ic+mm; im=ic-mm

        F=np.empty(nd); Fq=np.empty(nd)
        for jd,dlt in enumerate(deltas):
            vals=[]; valsq=[]
            for psi in psis:
                vals.append(cross(point_psi(psi-dlt/2,ip),point_psi(psi+dlt/2,im)))
                valsq.append(qcross(fund_psi(psi-dlt/2,ip),fund_psi(psi+dlt/2,im)))
            F[jd]=np.mean(vals); Fq[jd]=np.mean(valsq)

        fp=fit_phase(deltas,F); fq=fit_phase(deltas,Fq)
        A1=fp["cos1"]; A2=fp["cos2"]; A1loc=fq["cos1"]
        amid=np.sqrt(abs(Z[ip])*abs(Z[im]))
        A2loc=-(D/64)*amid**4

        Lam=-Zx[ip]/Z[ip]
        R1=D*np.exp(sact)*A1/(128*kappa**4)
        R2=-D*np.exp(2*sact)*A2/(256*kappa**4)
        fr.append({
            "b":b,"w":w,"age_T0":tref/T0,"halfwindow_T0":halfwindow,
            "omega":omega,"alpha":alpha,"kappa_core":kappa,
            "kappa_tail_global":ktail,"tail_R2":tr2,
            "target_kappaR":target,"actual_kappaR":sact,"R":R,
            "A1":A1,"A2":A2,"A1_local":A1loc,"A2_local":A2loc,
            "A1_over_local":A1/A1loc,"A2_over_local":A2/A2loc,
            "R1_corrected":R1/(1+mu*kappa*kappa),"R2_leading":R2,
            "lambda_real_local":float(np.real(Lam)),
            "lambda_imag_local":float(np.imag(Lam)),
            "lambda_real_over_kappa":float(np.real(Lam)/kappa),
            "lambda_imag_over_kappa":float(np.imag(Lam)/kappa),
            "lambda_mismatch_over_kappa":float(abs(Lam-kappa)/kappa),
        })

    summary={
        "b":b,"w":w,"age_T0":tref/T0,"halfwindow_T0":halfwindow,
        "omega":omega,"alpha":alpha,"kappa_core":kappa,
        "kappa_tail_global":ktail,
        "kappa_tail_over_core":ktail/kappa if np.isfinite(ktail) else np.nan,
        "tail_R2":tr2,"C1_tail":Ctail,"tail_rmin":rmin,"tail_rmax":rmax,
        "center_reconstruction_rms":rms,
    }
    return summary,fr

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--w",type=float,required=True)
    ap.add_argument("--b",type=float,default=1.0)
    ap.add_argument("--spp",type=int,default=160)
    ap.add_argument("--L",type=float,default=1200)
    ap.add_argument("--N",type=int,default=4096)
    args=ap.parse_args()

    b=args.b; w=args.w
    ages=(20.,40.,80.)
    maxhw=5.0

    x,dx,k,omega_k=grid(args.N,args.L)
    ic=args.N//2
    omega0=w/np.sqrt(1+w*w)
    T0=2*np.pi/omega0
    dt=T0/args.spp
    t_end=(max(ages)+maxhw)*T0
    nsteps=int(round(t_end/dt))
    u,v=initial_qb(x,w)

    # Entire center trace for phase spline; fields only in union of max windows.
    tc=[]; yc=[]
    stores={age:{"t":[],"U":[],"V":[]} for age in ages}
    every=2

    for n in range(nsteps):
        u,v=ssf4_step(u,v,dt,b,omega_k)
        if (n+1)%every: continue
        t=(n+1)*dt; tp=t/T0
        tc.append(t); yc.append(u[ic])
        for age in ages:
            if age-maxhw <= tp <= age+maxhw:
                stores[age]["t"].append(t)
                stores[age]["U"].append(u.copy())
                stores[age]["V"].append(v.copy())

    tc=np.asarray(tc); yc=np.asarray(yc)
    phase_spline,zc=build_phase_spline(tc,yc)

    summaries=[]; forces=[]
    for age in ages:
        S=stores[age]
        tt=np.asarray(S["t"]); U=np.asarray(S["U"]); V=np.asarray(S["V"])
        tref=age*T0
        for hw in (3.0,5.0):
            sm,fr=analyze(x,k,b,w,tt,U,V,tref,T0,phase_spline,hw)
            summaries.append(sm); forces.extend(fr)

    tag=f"w{w:g}".replace(".","p")
    sp=HERE/f"m5_phase_age_summary_{tag}.csv"
    fp=HERE/f"m5_phase_age_forces_{tag}.csv"
    with open(sp,"w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(summaries[0].keys()))
        wr.writeheader(); wr.writerows(summaries)
    with open(fp,"w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(forces[0].keys()))
        wr.writeheader(); wr.writerows(forces)

    print(f"M5 phase-spline b={b}, w={w}; zero crossings={len(zc)}")
    for sm in summaries:
        if sm["halfwindow_T0"]!=5.0: continue
        rr=min([r for r in forces if r["age_T0"]==sm["age_T0"] and r["halfwindow_T0"]==5.0],
               key=lambda r:abs(r["target_kappaR"]-7))
        # window sensitivity at same age
        sm3=[s for s in summaries if s["age_T0"]==sm["age_T0"] and s["halfwindow_T0"]==3.0][0]
        print(
            f"age={sm['age_T0']:.0f}T: omega={sm['omega']:.9f} "
            f"(hw3 diff={sm3['omega']-sm['omega']:+.2e}), "
            f"k={sm['kappa_core']:.6f}, ktail/k={sm['kappa_tail_over_core']:.3f}, "
            f"tailR2={sm['tail_R2']:.3f}, A1/local={rr['A1_over_local']:.4f}, "
            f"A2/local={rr['A2_over_local']:.4f}, "
            f"ReLam/k={rr['lambda_real_over_kappa']:.3f}, "
            f"ImLam/k={rr['lambda_imag_over_kappa']:+.3f}, "
            f"mismatch={rr['lambda_mismatch_over_kappa']:.3f}"
        )

if __name__=="__main__":
    main()
