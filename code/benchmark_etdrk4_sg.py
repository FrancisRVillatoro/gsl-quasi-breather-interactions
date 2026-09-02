
"""
benchmark_etdrk4_sg.py
======================

Temporal convergence benchmark of ETDRK4 against the exact sine-Gordon
breather, directly comparable with benchmark_sg.py used for SSF4.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from ssf4_gsl import (
    grid, energy, momentum,
    sine_gordon_breather, sine_gordon_breather_energy
)
from etdrk4_gsl import ETDRK4Coefficients, etdrk4_step

HERE=Path(__file__).resolve().parent

def l2(z,dx):
    return np.sqrt(dx*np.sum(z*z))

def run_case(spp,w=2.0,L=120.0,N=2048,nperiods=10.25):
    x,dx,k,omega_k=grid(N,L)
    omega=w/np.sqrt(1+w*w)
    T=2*np.pi/omega
    dt=T/spp
    nsteps=int(round(nperiods*spp))
    tf=nsteps*dt

    coeff=ETDRK4Coefficients(omega_k,dt)
    u,v=sine_gordon_breather(x,0.0,w)
    H0=energy(u,v,dx,k,0.0)
    max_dH=0.0
    max_P=0.0

    for n in range(nsteps):
        u,v=etdrk4_step(u,v,0.0,coeff)
        if (n+1)%max(1,spp//10)==0 or n+1==nsteps:
            H=energy(u,v,dx,k,0.0)
            P=momentum(u,v,dx,k)
            max_dH=max(max_dH,abs(H-H0)/abs(H0))
            max_P=max(max_P,abs(P))

    ue,ve=sine_gordon_breather(x,tf,w)
    eru=l2(u-ue,dx)/l2(ue,dx)
    erphase=np.sqrt(l2(u-ue,dx)**2+l2(v-ve,dx)**2)/np.sqrt(
        l2(ue,dx)**2+l2(ve,dx)**2
    )
    return {
        "steps_per_period":spp,"dt":dt,
        "rel_u_l2":float(eru),
        "rel_phase_l2":float(erphase),
        "max_rel_energy_error":float(max_dH),
        "max_abs_momentum":float(max_P),
        "H0_numeric":float(H0),
        "H_exact":float(sine_gordon_breather_energy(w)),
    }

def add_order(rows,key):
    rows[0]["p_"+key]=""
    for j in range(1,len(rows)):
        rows[j]["p_"+key]=float(np.log(rows[j-1][key]/rows[j][key])/np.log(2))

def main():
    rows=[run_case(spp) for spp in (20,40,80,160)]
    for key in ("rel_u_l2","rel_phase_l2","max_rel_energy_error"):
        add_order(rows,key)

    path=HERE/"benchmark_etdrk4_sg_results.csv"
    with open(path,"w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    dt=np.array([r["dt"] for r in rows])
    err=np.array([r["rel_u_l2"] for r in rows])
    ref=err[-1]*(dt/dt[-1])**4
    plt.figure(figsize=(7,5))
    plt.loglog(dt,err,"o-",label="ETDRK4 relative L2 error")
    plt.loglog(dt,ref,"--",label="O(dt^4)")
    plt.xlabel("dt"); plt.ylabel("error")
    plt.title("ETDRK4 temporal convergence: sine-Gordon breather")
    plt.grid(True,which="both"); plt.legend(); plt.tight_layout()
    plt.savefig(HERE/"benchmark_etdrk4_convergence.png",dpi=180)
    plt.close()

    print("ETDRK4 SG BENCHMARK")
    print("spp        dt           relL2(u)       p      phaseL2        p      max dH/H       p")
    for r in rows:
        p1="-" if r["p_rel_u_l2"]=="" else f"{r['p_rel_u_l2']:.4f}"
        p2="-" if r["p_rel_phase_l2"]=="" else f"{r['p_rel_phase_l2']:.4f}"
        p3="-" if r["p_max_rel_energy_error"]=="" else f"{r['p_max_rel_energy_error']:.4f}"
        print(f"{r['steps_per_period']:3d} {r['dt']:12.5e} {r['rel_u_l2']:12.5e} {p1:>7s} "
              f"{r['rel_phase_l2']:12.5e} {p2:>7s} {r['max_rel_energy_error']:12.5e} {p3:>7s}")
    print("H relative initialization error =",
          abs(rows[0]["H0_numeric"]-rows[0]["H_exact"])/rows[0]["H_exact"])

if __name__=="__main__":
    main()
