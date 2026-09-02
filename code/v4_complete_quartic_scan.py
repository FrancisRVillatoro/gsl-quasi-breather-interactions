"""Validate the complete direct quartic overlap law used in manuscript v4.

Input:
  milestones/gsl_chirp_aware_phase_experiment_b1_w3.zip
Output:
  data/v4_complete_quartic_law_scan.csv
"""
from pathlib import Path
import zipfile, io, math, csv
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ZIP=ROOT/'milestones'/'gsl_chirp_aware_phase_experiment_b1_w3.zip'
OUT=ROOT/'data'/'v4_complete_quartic_law_scan.csv'
H=(1,3,5,7); b=1.0; gamma=(2+3*b*b)/12
with zipfile.ZipFile(ZIP) as zf:
    with zf.open('chirp_template_spp320.npz') as f:
        d=np.load(io.BytesIO(f.read()))
        x=d['x']; CU=d['CU']; CV=d['CV']; CUx=d['CUx']; kappa=float(d['kappa'])
dx=x[1]-x[0]; ic=len(x)//2

def G(u):
    y=1-np.cos(u); return 2*y/(1+np.sqrt(1+b*b*y))

def point(theta,idx,full=True):
    u=v=ux=0.0
    if full: u=float(CU[0,idx]); v=float(CV[0,idx]); ux=float(CUx[0,idx])
    for n in (H if full else (1,)):
        j=1+2*H.index(n); c=math.cos(n*theta); s=math.sin(n*theta)
        u += CU[j,idx]*c+CU[j+1,idx]*s
        v += CV[j,idx]*c+CV[j+1,idx]*s
        ux += CUx[j,idx]*c+CUx[j+1,idx]*s
    return u,v,ux

def cross(L,R,quadratic=False):
    uL,vL,xL=L; uR,vR,xR=R
    if quadratic: return uL*uR-vL*vR-xL*xR
    return float(G(uL+uR)-G(uL)-G(uR)-vL*vR-xL*xR)

def fit(delta,F,nmax=5):
    cols=[np.ones_like(delta)]
    for n in range(1,nmax+1): cols += [np.cos(n*delta),np.sin(n*delta)]
    c,*_=np.linalg.lstsq(np.column_stack(cols),F,rcond=None)
    z={'c0':float(c[0])}
    for n in range(1,nmax+1): z[f'A{n}']=float(c[1+2*(n-1)])
    return z

nd=64; nt=256; deltas=2*np.pi*np.arange(nd)/nd; thetas=2*np.pi*np.arange(nt)/nt
rows=[]
for target in np.arange(6.0,10.01,0.25):
    m=max(1,int(round((target/kappa)/(2*dx)))); ip=ic+m; im=ic-m; s=kappa*2*m*dx
    F=np.empty(nd); Fq=np.empty(nd)
    for j,dlt in enumerate(deltas):
        F[j]=np.mean([cross(point(th-dlt/2,ip),point(th+dlt/2,im)) for th in thetas])
        Fq[j]=np.mean([cross(point(th-dlt/2,ip,False),point(th+dlt/2,im,False),True) for th in thetas])
    ff=fit(deltas,F); fq=fit(deltas,Fq)
    A1q=fq['A1']; dA1=ff['A1']-A1q; A2=ff['A2']; c0=ff['c0']
    Zp=CU[1,ip]-1j*CU[2,ip]; Zm=CU[1,im]-1j*CU[2,im]
    amid=np.sqrt(abs(Zp)*abs(Zm)); A2th=-(3*gamma/16)*amid**4
    rows.append(dict(target_kappaR=target,actual_kappaR=s,c0_full=c0,A1_full=ff['A1'],
        A1_quadratic=A1q,deltaA1_full_minus_quadratic=dA1,A2_full=A2,A2_quartic_local=A2th,
        c0_over_2A2local=c0/(2*A2th),deltaA1_over_4A2local=dA1/(4*A2th),A2_over_local=A2/A2th))
with open(OUT,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(OUT)
