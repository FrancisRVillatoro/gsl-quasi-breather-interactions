#!/usr/bin/env python3
"""Decompose one-period frozen-template drift into U, Lambda, Gamma and A1^(2)."""
from pathlib import Path
import csv, numpy as np
from ssf4_gsl import grid, ssf4_step, spectral_derivative


def main():
    root=Path(__file__).resolve().parents[1]; data=root/"data"
    d=np.load(data/"v3_allharmonic_template_spp320.npz")
    x=d["x"].copy(); CU=d["CU"].copy(); CV=d["CV"].copy(); CUx=d["CUx"].copy(); omega=float(d["omega"]); kappa=float(d["kappa"])
    H=(1,3,5,7); N=len(x); dx=x[1]-x[0]; L=N*dx; _,_,k,omega_k=grid(N,L); T=2*np.pi/omega; spp=320; dt=T/spp
    def rec(theta):
        u=CU[0].copy(); v=CV[0].copy()
        for n in H:
            j=1+2*H.index(n); u+=CU[j]*np.cos(n*theta)+CU[j+1]*np.sin(n*theta); v+=CV[j]*np.cos(n*theta)+CV[j+1]*np.sin(n*theta)
        return u,v
    targets=(6.,7.,8.); idx=[np.argmin(abs(x-(s/kappa)/2)) for s in targets]
    u,v=rec(0.); times=[]; U=[[] for _ in idx]; V=[[] for _ in idx]; D=[[] for _ in idx]
    for n in range(spp+1):
        if n%2==0 or n==spp:
            ux=spectral_derivative(u,k); times.append(n*dt)
            for j,i in enumerate(idx): U[j].append(u[i]); V[j].append(v[i]); D[j].append(ux[i])
        if n<spp: u,v=ssf4_step(u,v,dt,1.0,omega_k)
    times=np.asarray(times); m=times<T-1e-12; tt=times[m]
    cols=[np.ones_like(tt)]
    for n in H: cols += [np.cos(n*omega*tt),np.sin(n*omega*tt)]
    A=np.column_stack(cols); rows=[]
    for target,i,uu,vv,dd in zip(targets,idx,U,V,D):
        ce=np.linalg.lstsq(A,np.asarray(uu)[m],rcond=None)[0]; cv=np.linalg.lstsq(A,np.asarray(vv)[m],rcond=None)[0]; cd=np.linalg.lstsq(A,np.asarray(dd)[m],rcond=None)[0]
        Ue=ce[1]-1j*ce[2]; Ve=cv[1]-1j*cv[2]; De=cd[1]-1j*cd[2]; Uf=CU[1,i]-1j*CU[2,i]; Vf=CV[1,i]-1j*CV[2,i]; Df=CUx[1,i]-1j*CUx[2,i]
        def diag(U,V,D):
            A1=.5*(abs(U)**2-abs(V)**2+abs(D)**2); Lam=-D/U; Gam=V/(1j*omega*U); return A1,Lam,Gam
        A1f,Lf,Gf=diag(Uf,Vf,Df); A1e,Le,Ge=diag(Ue,Ve,De)
        rows.append({"target_kappaR":target,"x_eval":x[i],"actual_kappaR":2*kappa*x[i],"Uabs_frozen":abs(Uf),"Uabs_evolved":abs(Ue),"Uabs_ratio":abs(Ue)/abs(Uf),"U2_ratio":abs(Ue)**2/abs(Uf)**2,"Lambda_frozen_real":Lf.real,"Lambda_frozen_imag":Lf.imag,"Lambda_evolved_real":Le.real,"Lambda_evolved_imag":Le.imag,"Lambda_abs_ratio":abs(Le)/abs(Lf),"Gamma_frozen_real":Gf.real,"Gamma_frozen_imag":Gf.imag,"Gamma_evolved_real":Ge.real,"Gamma_evolved_imag":Ge.imag,"A1quad_frozen":A1f,"A1quad_evolved":A1e,"A1quad_ratio":A1e/A1f})
    with open(data/"v8_template_tail_decomposition.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for r in rows: print(r)

if __name__ == "__main__": main()
