#!/usr/bin/env python3
"""Three-period sine-Gordon deformation check at omega=0.9, kappa R=7.

Shows that the two-body deformation coefficients are short-time response
coefficients, not stationary effective-potential constants.
"""
from pathlib import Path
import csv, numpy as np
from ssf4_gsl import grid, ssf4_step, spectral_derivative
from v8_phase_resolved_deformation import fit_fourier


def main():
    omega = 0.9; kappa = np.sqrt(1 - omega**2); T = 2*np.pi/omega; s = 7.0; R = s/kappa
    N = 1024; L = 180.0; spp = 120; nphase = 12; nperiods = 3
    x, dx, k, omega_k = grid(N, L); ic = N//2; dt = T/spp
    delta = 2*np.pi*np.arange(nphase)/nphase
    raw = [{"delta": [], "sup": [], "full": []} for _ in range(nperiods)]

    def uv(center, theta):
        X = kappa*(x-center); S = 1/np.cosh(X); z = kappa/omega*np.sin(theta)*S
        return 4*np.arctan(z), 4*kappa*np.cos(theta)*S/(1+z*z)
    def sigma(u,v,ux): return float(1-np.cos(u[ic])-.5*v[ic]**2-.5*ux[ic]**2)

    for d in delta:
        uL,vL=uv(-R/2,-d/2); uR,vR=uv(R/2,d/2); u=uL+uR; v=vL+vR
        times=[]; sup=[]; full=[]
        for n in range(nperiods*spp+1):
            if n%4==0 or n==nperiods*spp:
                uxL=spectral_derivative(uL,k); uxR=spectral_derivative(uR,k); ux=spectral_derivative(u,k)
                sup.append(sigma(uL+uR,vL+vR,uxL+uxR)-sigma(uL,vL,uxL)-sigma(uR,vR,uxR))
                full.append(sigma(u,v,ux)-sigma(uL,vL,uxL)-sigma(uR,vR,uxR)); times.append(n*dt)
            if n<nperiods*spp:
                uL,vL=ssf4_step(uL,vL,dt,0.,omega_k); uR,vR=ssf4_step(uR,vR,dt,0.,omega_k); u,v=ssf4_step(u,v,dt,0.,omega_k)
        times=np.asarray(times); sup=np.asarray(sup); full=np.asarray(full)
        for p in range(nperiods):
            m=(times>=p*T-1e-10)&(times<=(p+1)*T+1e-10); tt=times[m]
            av=lambda q: float(np.trapezoid(q[m],tt)/(tt[-1]-tt[0]))
            raw[p]["delta"].append(d); raw[p]["sup"].append(av(sup)); raw[p]["full"].append(av(full))

    rows=[]
    for p,r in enumerate(raw,1):
        d=np.asarray(r["delta"]); q=fit_fourier(d,np.asarray(r["full"])-np.asarray(r["sup"]),3)
        rows.append({"period":p,"def_c0":q["c0"],"def_A1":q["A1"],"def_A2":q["A2"],"def_A3":q["A3"]})
    out=Path(__file__).resolve().parents[1]/"data"
    with open(out/"v8_sg_deformation_multiperiod_omega09.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for r in rows: print(r)

if __name__ == "__main__": main()
