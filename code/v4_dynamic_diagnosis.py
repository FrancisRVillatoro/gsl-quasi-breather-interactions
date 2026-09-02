"""Decompose frozen-to-dynamic drift into isolated ageing and true pair deformation."""
from pathlib import Path
import zipfile, io, csv, math, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from ssf4_gsl import grid, ssf4_step, spectral_derivative, gsl_potential
ZIP=ROOT/'milestones'/'gsl_chirp_aware_phase_experiment_b1_w3.zip'
OUT=ROOT/'data'/'v4_frozen_evolvedsuper_fullpair.csv'
with zipfile.ZipFile(ZIP) as zf:
    with zf.open('chirp_template_spp320.npz') as f:
        d=np.load(io.BytesIO(f.read())); x=d['x'].copy(); CU=d['CU'].copy(); CV=d['CV'].copy(); CUx=d['CUx'].copy(); omega=float(d['omega']); kappa=float(d['kappa'])
b=1.; H=(1,3,5,7); N=x.size; dx=x[1]-x[0]; L=N*dx; ic=N//2
_,_,k,omega_k=grid(N,L); Tstar=2*np.pi/omega

def reconstruct(th):
    u=CU[0].copy(); v=CV[0].copy()
    for n in H:
        j=1+2*H.index(n); u+=CU[j]*np.cos(n*th)+CU[j+1]*np.sin(n*th); v+=CV[j]*np.cos(n*th)+CV[j+1]*np.sin(n*th)
    return u,v

def reconstruct3(th):
    u=CU[0].copy(); v=CV[0].copy(); ux=CUx[0].copy()
    for n in H:
        j=1+2*H.index(n); c=np.cos(n*th); s=np.sin(n*th); u+=CU[j]*c+CU[j+1]*s; v+=CV[j]*c+CV[j+1]*s; ux+=CUx[j]*c+CUx[j+1]*s
    return u,v,ux

def shift(f,a): return np.fft.irfft(np.fft.rfft(f)*np.exp(-1j*k*a),n=N)
def sigma(u,v,ux): return float(gsl_potential(np.array([u[ic]]),b)[0]-.5*v[ic]**2-.5*ux[ic]**2)
def power(u,v,ux): return float(v[ic]*ux[ic])

def frozen(R,delta):
    m=int(round((R/2)/dx)); ip=ic+m; im=ic-m; Fs=[]; Ps=[]
    for th in np.linspace(0,2*np.pi,512,endpoint=False):
        uL,vL,xL=reconstruct3(th-delta/2); uR,vR,xR=reconstruct3(th+delta/2)
        UL,VL,XL=uL[ip],vL[ip],xL[ip]; UR,VR,XR=uR[im],vR[im],xR[im]
        Fs.append(gsl_potential(np.array([UL+UR]),b)[0]-gsl_potential(np.array([UL]),b)[0]-gsl_potential(np.array([UR]),b)[0]-VL*VR-XL*XR)
        Ps.append(VL*XR+VR*XL)
    return float(np.mean(Fs)),float(np.mean(Ps))

def run(target,delta):
    m=max(1,int(round((target/kappa)/(2*dx)))); R=2*m*dx
    uL0,vL0=reconstruct(-delta/2); uR0,vR0=reconstruct(delta/2)
    uL,vL=shift(uL0,-R/2),shift(vL0,-R/2); uR,vR=shift(uR0,R/2),shift(vR0,R/2); up,vp=uL+uR,vL+vR
    dt=Tstar/320; t=[]; Fs=[]; Fp=[]; Ps=[]; Pp=[]
    def collect(tt):
        uxL=spectral_derivative(uL,k); uxR=spectral_derivative(uR,k); uxP=spectral_derivative(up,k); us=uL+uR; vs=vL+vR; uxs=uxL+uxR
        Fs.append(-(sigma(us,vs,uxs)-sigma(uL,vL,uxL)-sigma(uR,vR,uxR))); Fp.append(-(sigma(up,vp,uxP)-sigma(uL,vL,uxL)-sigma(uR,vR,uxR)))
        Ps.append(power(us,vs,uxs)-power(uL,vL,uxL)-power(uR,vR,uxR)); Pp.append(power(up,vp,uxP)-power(uL,vL,uxL)-power(uR,vR,uxR)); t.append(tt)
    collect(0.)
    for n in range(320):
        uL[:],vL[:]=ssf4_step(uL,vL,dt,b,omega_k); uR[:],vR[:]=ssf4_step(uR,vR,dt,b,omega_k); up[:],vp[:]=ssf4_step(up,vp,dt,b,omega_k)
        if (n+1)%2==0: collect((n+1)*dt)
    tt=np.asarray(t); avg=lambda y: float(np.trapezoid(y,tt)/(tt[-1]-tt[0])); Ff,Pf=frozen(R,delta)
    return dict(target_kappaR=target,actual_kappaR=kappa*R,delta=delta,frozen_force_right=-Ff,evolved_superposition_force_right=avg(Fs),full_pair_force_right=avg(Fp),frozen_power_left=Pf,evolved_superposition_power_left=avg(Ps),full_pair_power_left=avg(Pp))
rows=[run(R,d) for R in (6.,7.,8.) for d in (0.,np.pi/2,np.pi)]
with open(OUT,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(OUT)
