"""Reproduce the isolated b=1, w=3 quasi-breather diagnostics for Figure 2.

By default, use the published 160 steps per seed period. Write the summary,
the fitted profile and diagnostic plots in the current working directory.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from ssf4_gsl import grid, ssf4_step, energy, momentum

def initial_qb(x, w):
    q = 1.0 / np.sqrt(1.0 + w*w)
    return np.zeros_like(x), 4.0*q/np.cosh(q*x)

def zero_crossing_frequency(t, y):
    idx = np.where((y[:-1] <= 0) & (y[1:] > 0))[0]
    tc = []
    for i in idx:
        a = -y[i] / (y[i+1]-y[i])
        tc.append(t[i] + a*(t[i+1]-t[i]))
    tc = np.asarray(tc)
    if len(tc) < 3:
        return np.nan
    return 2*np.pi/np.mean(np.diff(tc))

def harmonic_fit(t, y, omega_guess, harmonics=(1,3,5)):
    tt = t - t[0]
    def mat(om):
        cols = [np.ones_like(tt)]
        for n in harmonics:
            cols += [np.cos(n*om*tt), np.sin(n*om*tt)]
        return np.column_stack(cols)
    def obj(om):
        A = mat(om)
        c, *_ = np.linalg.lstsq(A,y,rcond=None)
        r = y-A@c
        return float(r@r)
    lo=max(0.5,0.96*omega_guess); hi=min(0.999999,1.04*omega_guess)
    res=minimize_scalar(obj,bounds=(lo,hi),method="bounded",
                        options={"xatol":1e-12})
    om=float(res.x)
    A=mat(om)
    c,*_=np.linalg.lstsq(A,y,rcond=None)
    amps={}
    j=1
    for n in harmonics:
        amps[n]=float(np.hypot(c[j],c[j+1])); j+=2
    rms=float(np.sqrt(obj(om)/len(y)))
    return om,amps,rms

def choose_tail_fit(x, amp, kpred):
    """
    Fit the pre-radiative fundamental tail.

    For the isolated QB the demodulated fundamental decreases exponentially
    from the core and eventually reaches a radiation/leakage floor.  We fit
    the intermediate amplitude band

        0.05 max|U1| < |U1| < 0.40 max|U1|,

    separately on x>0.  This avoids both the nonlinear core and the far-field
    radiation floor.  If too few points are available, progressively broader
    bands are tried.
    """
    r=np.abs(x); peak=float(amp.max())
    bands=[(0.40,0.05),(0.50,0.05),(0.40,0.03),(0.30,0.03)]
    best=None
    for up,low in bands:
        m=(x>0)&(amp<up*peak)&(amp>low*peak)
        if m.sum()<12:
            continue
        rr=r[m]; yy=np.log(amp[m])
        slope,inter=np.polyfit(rr,yy,1)
        kap=-float(slope)
        if kap<=0:
            continue
        yf=slope*rr+inter
        ssr=((yy-yf)**2).sum(); sst=((yy-yy.mean())**2).sum()
        r2=1-ssr/sst
        item=(r2,kap,float(np.exp(inter)),rr.min(),rr.max(),m,up,low)
        # Prefer the first (narrowest prescribed) band if it is already
        # an excellent exponential. Otherwise retain the best R^2.
        if best is None or item[0]>best[0]:
            best=item
        if r2>0.9995 and (up,low)==(0.40,0.05):
            break
    if best is None:
        raise RuntimeError("No usable pre-radiative tail interval")
    return best

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spp", type=int, default=160, help="steps per seed period (default: 160)")
    parser.add_argument("--tag", help="output filename suffix (default: spp followed by the number of steps per period)")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(),
                        help="directory for the CSV, NPZ and diagnostic plots (default: current directory)")
    args = parser.parse_args()
    if args.spp <= 0:
        parser.error("--spp must be positive")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    b=1.0; w=3.0
    L=160.0; N=1024
    spp=args.spp
    tag=args.tag if args.tag is not None else f"spp{spp}"
    total_periods=80.0
    ana_start=50.0
    ana_periods=16.0
    sample_every=2

    x,dx,k,omega_k=grid(N,L)
    q=1/np.sqrt(1+w*w)
    omega0=w/np.sqrt(1+w*w)
    T0=2*np.pi/omega0
    dt=T0/spp
    nsteps=int(round(total_periods*spp))
    ta=ana_start*T0; tb=(ana_start+ana_periods)*T0

    u,v=initial_qb(x,w)
    H0=energy(u,v,dx,k,b)
    P0=momentum(u,v,dx,k)
    ic=N//2

    ts=[]; uc=[]; Hs=[]; Hcores=[]; Ps=[]
    core_radius=12.0
    core_mask=np.abs(x)<=core_radius
    st=[]; su=[]
    for n in range(nsteps):
        u,v=ssf4_step(u,v,dt,b,omega_k)
        t=(n+1)*dt
        if (n+1)%sample_every==0:
            ts.append(t); uc.append(u[ic])
            Hs.append(energy(u,v,dx,k,b))
            # Core energy using the same spectral derivative as the Hamiltonian.
            ux=np.fft.irfft(1j*k*np.fft.rfft(u),n=N)
            ypot=1.0-np.cos(u)
            G=2.0*ypot/(1.0+np.sqrt(1.0+b*b*ypot))
            edens=0.5*v*v+0.5*ux*ux+G
            Hcores.append(dx*np.sum(edens[core_mask]))
            Ps.append(momentum(u,v,dx,k))
            if ta<=t<=tb:
                st.append(t); su.append(u.copy())

    ts=np.asarray(ts); uc=np.asarray(uc)
    Hs=np.asarray(Hs); Hcores=np.asarray(Hcores); Ps=np.asarray(Ps)
    st=np.asarray(st); su=np.asarray(su)

    m=(ts>=ta)&(ts<=tb)
    om_zc=zero_crossing_frequency(ts[m],uc[m])
    om,amps,rms=harmonic_fit(ts[m],uc[m],om_zc)
    kpred=np.sqrt(1-om*om)

    tmid=0.5*(st[0]+st[-1])
    ph=np.exp(-1j*om*(st-tmid))
    U1=2*np.mean(su*ph[:,None],axis=0)
    a1=np.abs(U1)

    r2,kfit,C1,rmin,rmax,tailmask,tail_up,tail_low=choose_tail_fit(x,a1,kpred)

    chi=(41*b**4+28*b*b+4)/(2*(3*b*b+2)**2)
    A=4*np.sqrt(2)/np.sqrt(2+3*b*b)
    C1as=2*A*kpred*(1+chi*kpred*kpred)

    shcore,_=np.polyfit(ts[m],Hcores[m],1)
    relCoreSlope=shcore/Hcores[m].mean()
    relCoreChange=(Hcores[m][-1]-Hcores[m][0])/Hcores[m][0]

    summary={
      "b":b,"w":w,"q_initial":q,"omega_initial_SG":omega0,
      "omega_zero_crossing":om_zc,"omega_fit":om,
      "kappa_pred":kpred,"kappa_fit":kfit,
      "relative_kappa_error":(kfit-kpred)/kpred,
      "C1_fit":C1,"C1_asym_O_kappa3":C1as,
      "relative_C1_error":(C1-C1as)/C1as,
      "tail_R2":r2,"tail_rmin":rmin,"tail_rmax":rmax,
      "A1_center":amps[1],"A3_center":amps[3],"A5_center":amps[5],
      "A3_over_A1":amps[3]/amps[1],"harmonic_fit_rms":rms,
      "H0":H0,"core_radius":core_radius,
      "relative_core_energy_slope_per_time":relCoreSlope,
      "relative_core_energy_change_analysis_window":relCoreChange,
      "max_relative_total_energy_error":float(np.max(np.abs((Hs-H0)/H0))),
      "tail_band_upper_fraction":tail_up,
      "tail_band_lower_fraction":tail_low,
      "max_abs_momentum_change":float(np.max(np.abs(Ps-P0))),
      "dt":dt,"N":N,"L":L,"steps_per_period":spp
    }

    with open(output_dir / f"isolated_qb_summary_{tag}.csv","w",newline="") as f:
        wr=csv.writer(f, lineterminator="\n"); wr.writerow(["quantity","value"])
        for kv in summary.items(): wr.writerow(kv)

    np.savez_compressed(output_dir / f"isolated_qb_data_{tag}.npz",
                       x=x,ts=ts,center_u=uc,H=Hs,Hcore=Hcores,P=Ps,U1=U1)

    plt.figure(figsize=(7,4.5))
    plt.plot((ts[m]-ta)/T0,uc[m])
    plt.xlabel("time from analysis start / initial period")
    plt.ylabel("u(0,t)")
    plt.title("Isolated GSL quasi-breather: b=1, w=3")
    plt.grid(True); plt.tight_layout()
    plt.savefig(output_dir / f"isolated_qb_center_{tag}.png",dpi=180); plt.close()

    pos=x>0
    plt.figure(figsize=(7,5))
    plt.semilogy(x[pos],a1[pos],label="|U1(x)|")
    rr=np.linspace(rmin,rmax,300)
    plt.semilogy(rr,C1*np.exp(-kfit*rr),"--",
                 label=f"fit kappa={kfit:.6f}")
    plt.semilogy(rr,C1*np.exp(-kpred*rr),":",
                 label=f"sqrt(1-Omega^2)={kpred:.6f}")
    plt.xlim(0,65); plt.ylim(max(a1.max()*1e-9,1e-14),1.2*a1.max())
    plt.xlabel("x"); plt.ylabel("|U1|")
    plt.title("Fundamental tail")
    plt.grid(True,which="both"); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / f"isolated_qb_tail_{tag}.png",dpi=180); plt.close()

    plt.figure(figsize=(7,4.5))
    plt.plot(ts/T0,(Hcores-Hcores[0])/Hcores[0])
    plt.axvspan(ana_start,ana_start+ana_periods,alpha=0.15)
    plt.xlabel("time / initial period"); plt.ylabel("(Hcore-Hcore(0))/Hcore(0)")
    plt.title("Core-energy diagnostic")
    plt.grid(True); plt.tight_layout()
    plt.savefig(output_dir / f"isolated_qb_energy_{tag}.png",dpi=180); plt.close()

    print("ISOLATED GSL QB: b=1, w=3")
    for k0,v0 in summary.items():
        if isinstance(v0,(float,np.floating)):
            print(f"{k0:42s} = {v0:.12e}")
        else:
            print(f"{k0:42s} = {v0}")

if __name__=="__main__":
    main()
