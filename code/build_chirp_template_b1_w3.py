
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from ssf4_gsl import grid, ssf4_step

HERE=Path(__file__).resolve().parent
HARMONICS=(1,3,5,7)

def initial_qb(x,w):
    q=1/np.sqrt(1+w*w)
    return np.zeros_like(x),4*q/np.cosh(q*x)

def phase_design(tau, omega, alpha, with_trends=True):
    theta=omega*tau+0.5*alpha*tau*tau
    cols=[np.ones_like(tau)]
    if with_trends:
        cols.append(tau)
    for n in HARMONICS:
        c=np.cos(n*theta); s=np.sin(n*theta)
        cols += [c,s]
        if with_trends:
            cols += [tau*c,tau*s]
    return np.column_stack(cols)

def fit_chirped_phase(t,y,omega_guess):
    tref=float(np.mean(t)); tau=t-tref
    scale_t=max(abs(tau).max(),1.0)
    # Scale alpha parameter so optimizer sees comparable magnitudes.
    def objective(z):
        omega=float(z[0])
        alpha=float(z[1])/(scale_t*scale_t)
        A=phase_design(tau,omega,alpha,True)
        c,*_=np.linalg.lstsq(A,y,rcond=None)
        r=y-A@c
        return float(np.dot(r,r))
    z0=np.array([omega_guess,0.0])
    res=minimize(objective,z0,method="Nelder-Mead",
                 options={"xatol":1e-12,"fatol":1e-14,"maxiter":1000})
    omega=float(res.x[0])
    alpha=float(res.x[1])/(scale_t*scale_t)
    return tref,omega,alpha,float(np.sqrt(objective(res.x)/len(y)))

def fixed_age_coefficients(t,U,V,tref,omega,alpha):
    tau=t-tref
    A=phase_design(tau,omega,alpha,True)
    CU,*_=np.linalg.lstsq(A,U,rcond=None)
    CV,*_=np.linalg.lstsq(A,V,rcond=None)
    # Keep coefficients at tau=0 only:
    # design order = mean, mean*tau,
    # for each n: cos,sin,tau*cos,tau*sin
    rowsU=[CU[0]]
    rowsV=[CV[0]]
    j=2
    for n in HARMONICS:
        rowsU += [CU[j],CU[j+1]]
        rowsV += [CV[j],CV[j+1]]
        j += 4
    return np.asarray(rowsU),np.asarray(rowsV)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--spp",type=int,default=160)
    ap.add_argument("--tag",default="spp160")
    args=ap.parse_args()

    b=1.0; w=3.0; L=600.0; N=4096
    age_periods=40.0
    window_periods=10.0
    sample_every=2

    x,dx,k,omega_k=grid(N,L)
    ic=N//2
    omega0=w/np.sqrt(1+w*w)
    T0=2*np.pi/omega0
    dt=T0/args.spp
    nsteps=int(round(age_periods*args.spp))
    t0=(age_periods-window_periods)*T0

    u,v=initial_qb(x,w)
    ts=[]; Us=[]; Vs=[]; yc=[]
    for n in range(nsteps):
        u,v=ssf4_step(u,v,dt,b,omega_k)
        t=(n+1)*dt
        if (n+1)%sample_every==0 and t>=t0:
            ts.append(t); Us.append(u.copy()); Vs.append(v.copy()); yc.append(u[ic])

    ts=np.asarray(ts); Us=np.asarray(Us); Vs=np.asarray(Vs); yc=np.asarray(yc)

    # Initial guess from positive-slope zero crossings.
    ids=np.where((yc[:-1]<=0)&(yc[1:]>0))[0]
    tc=[]
    for i in ids:
        a=-yc[i]/(yc[i+1]-yc[i])
        tc.append(ts[i]+a*(ts[i+1]-ts[i]))
    omega_guess=2*np.pi/np.mean(np.diff(tc))

    tref,omega,alpha,rms=fit_chirped_phase(ts,yc,omega_guess)
    CU,CV=fixed_age_coefficients(ts,Us,Vs,tref,omega,alpha)

    kappa=np.sqrt(1-omega*omega)

    # spatial derivatives of U coefficients
    CUx=np.fft.irfft(1j*k[None,:]*np.fft.rfft(CU,axis=1),n=N,axis=1)

    np.savez_compressed(
        HERE/f"chirp_template_{args.tag}.npz",
        x=x,CU=CU,CV=CV,CUx=CUx,omega=omega,alpha=alpha,
        kappa=kappa,tref=tref,fit_rms=rms,dt=dt,L=L,N=N,
        window_periods=window_periods
    )

    # diagnostics: amplitudes and outgoing/Helmholtz residuals.
    rows=[]
    for n in HARMONICS:
        j=1+2*HARMONICS.index(n)
        Z=CU[j]-1j*CU[j+1]
        Zx=CUx[j]-1j*CUx[j+1]
        amp=np.abs(Z)
        if n==1:
            kval=kappa
            # evanescent right-tail residual Zx + kappa Z
            for xp in (5,7,9,12,15,20):
                ii=int(np.argmin(abs(x-xp)))
                res=abs(Zx[ii]+kval*Z[ii])/(kval*abs(Z[ii])+1e-300)
                rows.append({"n":n,"x":x[ii],"amp":amp[ii],
                             "k_theory":kval,"outgoing_or_tail_residual":res})
        else:
            kval=np.sqrt((n*omega)**2-1)
            # outgoing right-wave residual Zx + i k Z
            for xp in (5,7,9,12,15,20,30,40):
                ii=int(np.argmin(abs(x-xp)))
                res=abs(Zx[ii]+1j*kval*Z[ii])/(kval*abs(Z[ii])+1e-300)
                rows.append({"n":n,"x":x[ii],"amp":amp[ii],
                             "k_theory":kval,"outgoing_or_tail_residual":res})

    with open(HERE/f"chirp_template_diagnostics_{args.tag}.csv","w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    # plot temporal fit reconstruction at center
    tau=ts-tref
    theta=omega*tau+0.5*alpha*tau*tau
    yrec=CU[0,ic]*np.ones_like(ts)
    for n in HARMONICS:
        j=1+2*HARMONICS.index(n)
        yrec += CU[j,ic]*np.cos(n*theta)+CU[j+1,ic]*np.sin(n*theta)

    plt.figure(figsize=(7.6,4.8))
    plt.plot(tau,yc,label="numerical center signal")
    plt.plot(tau,yrec,"--",label="chirp-aware frozen-age reconstruction")
    plt.xlabel(r"$t-t_*$")
    plt.ylabel(r"$u(0,t)$")
    plt.title("Chirp-aware quasi-breather template")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(HERE/f"chirp_template_center_fit_{args.tag}.png",dpi=180)
    plt.close()

    # harmonic amplitudes versus x
    plt.figure(figsize=(7.6,5.0))
    pos=(x>=0)&(x<=50)
    for n in HARMONICS:
        j=1+2*HARMONICS.index(n)
        amp=np.sqrt(CU[j]**2+CU[j+1]**2)
        plt.semilogy(x[pos],amp[pos],label=f"n={n}")
    plt.xlabel("x")
    plt.ylabel("temporal-harmonic amplitude")
    plt.title("Chirp-aware harmonic tails")
    plt.grid(True,which="both"); plt.legend(); plt.tight_layout()
    plt.savefig(HERE/f"chirp_template_harmonic_amplitudes_{args.tag}.png",dpi=180)
    plt.close()

    print("CHIRP-AWARE TEMPLATE")
    print(f"spp={args.spp}")
    print(f"omega_guess={omega_guess:.12e}")
    print(f"omega_mid={omega:.12e}")
    print(f"alpha=domega/dt={alpha:.12e}")
    print(f"kappa={kappa:.12e}")
    print(f"center_fit_rms={rms:.12e}")
    print(f"tref={tref:.12e}")
    for n in HARMONICS:
        j=1+2*HARMONICS.index(n)
        amp=np.sqrt(CU[j]**2+CU[j+1]**2)
        for xp in (7,9,15,30):
            ii=int(np.argmin(abs(x-xp)))
            print(f"n={n} amp(x={x[ii]:.3f})={amp[ii]:.9e}")

if __name__=="__main__":
    main()
