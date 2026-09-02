from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from ssf4_gsl import grid, ssf4_step

HERE=Path(__file__).resolve().parents[1] / 'data'
H=(1,3,5,7)

def initial_qb(x,w):
    q=1/np.sqrt(1+w*w)
    return np.zeros_like(x),4*q/np.cosh(q*x)

def G(u,b):
    y=1-np.cos(u)
    return 2*y/(1+np.sqrt(1+b*b*y))

def phase_design(tau,omega,alpha):
    theta=omega*tau+0.5*alpha*tau*tau
    cols=[np.ones_like(tau),tau]
    for n in H:
        c=np.cos(n*theta); s=np.sin(n*theta)
        cols += [c,s,tau*c,tau*s]
    return np.column_stack(cols)

def fit_chirp(t,y,omega_guess):
    tref=float(np.mean(t)); tau=t-tref; scale=max(abs(tau).max(),1.0)
    def obj(z):
        om=float(z[0]); alpha=float(z[1])/(scale*scale)
        A=phase_design(tau,om,alpha)
        c,*_=np.linalg.lstsq(A,y,rcond=None)
        r=y-A@c
        return float(r@r)
    res=minimize(obj,[omega_guess,0.0],method='Nelder-Mead',
                 options={'xatol':2e-12,'fatol':1e-14,'maxiter':1000})
    om=float(res.x[0]); alpha=float(res.x[1])/(scale*scale)
    return tref,om,alpha,float(np.sqrt(obj(res.x)/len(y)))

def zero_freq(t,y):
    ids=np.where((y[:-1]<=0)&(y[1:]>0))[0]
    tc=[]
    for i in ids:
        a=-y[i]/(y[i+1]-y[i]); tc.append(t[i]+a*(t[i+1]-t[i]))
    tc=np.asarray(tc)
    return float(2*np.pi/np.mean(np.diff(tc)))

def fixed_coeff(t,U,V,tref,omega,alpha):
    tau=t-tref; A=phase_design(tau,omega,alpha)
    CU,*_=np.linalg.lstsq(A,U,rcond=None)
    CV,*_=np.linalg.lstsq(A,V,rcond=None)
    rowsU=[CU[0]]; rowsV=[CV[0]]; j=2
    for n in H:
        rowsU += [CU[j],CU[j+1]]; rowsV += [CV[j],CV[j+1]]; j+=4
    return np.asarray(rowsU),np.asarray(rowsV)

def fit_tail(x,amp,kpred):
    peak=float(amp.max()); candidates=[]
    bands=[(0.45,0.08),(0.4,0.05),(0.3,0.03),(0.2,0.02)]
    for up,lo in bands:
        m=(x>0)&(amp<up*peak)&(amp>lo*peak)
        if m.sum()<12: continue
        rr=x[m]; yy=np.log(amp[m]); sl,it=np.polyfit(rr,yy,1)
        kap=-float(sl)
        if kap<=0: continue
        yf=sl*rr+it; ssr=((yy-yf)**2).sum(); sst=((yy-yy.mean())**2).sum(); r2=1-ssr/sst
        score=r2-0.1*abs(kap-kpred)/max(kpred,1e-12)
        candidates.append((score,r2,kap,float(np.exp(it)),float(rr.min()),float(rr.max()),up,lo))
    if not candidates: return (np.nan,)*7
    candidates.sort(key=lambda z:z[0],reverse=True)
    _,r2,kap,C,rmin,rmax,up,lo=candidates[0]
    return r2,kap,C,rmin,rmax,up,lo

def fit_phase(delta,F,nmax=4):
    cols=[np.ones_like(delta)]
    for n in range(1,nmax+1): cols += [np.cos(n*delta),np.sin(n*delta)]
    A=np.column_stack(cols); c,*_=np.linalg.lstsq(A,F,rcond=None)
    z={}
    for n in range(1,nmax+1): z[f'cos{n}']=float(c[1+2*(n-1)]); z[f'sin{n}']=float(c[2+2*(n-1)])
    return z

def run(b,w,spp=160,L=600,N=4096,age_periods=40,window_periods=10):
    x,dx,k,omega_k=grid(N,L); ic=N//2
    omega0=w/np.sqrt(1+w*w); T0=2*np.pi/omega0; dt=T0/spp
    nsteps=int(round(age_periods*spp)); t0=(age_periods-window_periods)*T0
    u,v=initial_qb(x,w); ts=[]; Us=[]; Vs=[]; yc=[]
    sample_every=2
    for n in range(nsteps):
        u,v=ssf4_step(u,v,dt,b,omega_k); t=(n+1)*dt
        if (n+1)%sample_every==0 and t>=t0:
            ts.append(t); Us.append(u.copy()); Vs.append(v.copy()); yc.append(u[ic])
    ts=np.asarray(ts); Us=np.asarray(Us); Vs=np.asarray(Vs); yc=np.asarray(yc)
    og=zero_freq(ts,yc); tref,omega,alpha,rms=fit_chirp(ts,yc,og)
    CU,CV=fixed_coeff(ts,Us,Vs,tref,omega,alpha)
    CUx=np.fft.irfft(1j*k[None,:]*np.fft.rfft(CU,axis=1),n=N,axis=1)
    kappa=np.sqrt(max(0,1-omega*omega))
    amp1=np.sqrt(CU[1]**2+CU[2]**2)
    tr2,kfit,Cfit,rmin,rmax,tup,tlo=fit_tail(x,amp1,kappa)

    def point(theta,idx):
        uu=float(CU[0,idx]); vv=float(CV[0,idx]); ux=float(CUx[0,idx])
        for n in H:
            j=1+2*H.index(n); c=math.cos(n*theta); s=math.sin(n*theta)
            uu += CU[j,idx]*c+CU[j+1,idx]*s
            vv += CV[j,idx]*c+CV[j+1,idx]*s
            ux += CUx[j,idx]*c+CUx[j+1,idx]*s
        return uu,vv,ux
    def fund(theta,idx):
        c=math.cos(theta); s=math.sin(theta)
        return (float(CU[1,idx]*c+CU[2,idx]*s),float(CV[1,idx]*c+CV[2,idx]*s),float(CUx[1,idx]*c+CUx[2,idx]*s))
    def cross(Lp,Rp):
        uL,vL,xL=Lp; uR,vR,xR=Rp
        return float(G(uL+uR,b)-G(uL,b)-G(uR,b)-vL*vR-xL*xR)
    def qcross(Lp,Rp):
        uL,vL,xL=Lp; uR,vR,xR=Rp
        return float(uL*uR-vL*vR-xL*xR)

    nd=48; nt=192; deltas=2*np.pi*np.arange(nd)/nd; thetas=2*np.pi*np.arange(nt)/nt
    force_rows=[]
    D=2+3*b*b; mu=(41*b**4+28*b*b+4)/(D*D)
    for target in (6.0,7.0,8.0):
        m=max(1,int(round((target/kappa)/(2*dx))))
        R=2*m*dx; sact=kappa*R; ip=ic+m; im=ic-m
        F=np.empty(nd); F1q=np.empty(nd)
        for jd,dlt in enumerate(deltas):
            vf=[]; v1=[]
            for th in thetas:
                vf.append(cross(point(th-dlt/2,ip),point(th+dlt/2,im)))
                v1.append(qcross(fund(th-dlt/2,ip),fund(th+dlt/2,im)))
            F[jd]=np.mean(vf); F1q[jd]=np.mean(v1)
        fp=fit_phase(deltas,F); f1=fit_phase(deltas,F1q)
        A1=fp['cos1']; A2=fp['cos2']; A1loc=f1['cos1']
        Zp=CU[1,ip]-1j*CU[2,ip]; Zm=CU[1,im]-1j*CU[2,im]
        amid=np.sqrt(abs(Zp)*abs(Zm)); A2loc=-(D/64)*amid**4
        R1=D*np.exp(sact)*A1/(128*kappa**4)
        R2=-D*np.exp(2*sact)*A2/(256*kappa**4)
        force_rows.append(dict(b=b,w=w,spp=spp,omega=omega,alpha=alpha,kappa=kappa,kappa_tail=kfit,tail_R2=tr2,C1_tail=Cfit,
                               target_kappaR=target,actual_kappaR=sact,R=R,A1=A1,A2=A2,A1_local=A1loc,A2_local=A2loc,
                               A1_over_local=A1/A1loc,A2_over_local=A2/A2loc,R1_leading=R1,R1_over_1plus_mu_k2=R1/(1+mu*kappa*kappa),
                               R2_leading=R2,mu=mu,center_fit_rms=rms,amid=amid))
    summary=dict(b=b,w=w,spp=spp,omega0=omega0,omega=omega,alpha=alpha,kappa=kappa,kappa_tail=kfit,
                 relative_kappa_tail_error=(kfit-kappa)/kappa,tail_R2=tr2,C1_tail=Cfit,tail_rmin=rmin,tail_rmax=rmax,
                 center_fit_rms=rms,mu=mu)
    tag=f'b{b:g}_w{w:g}_spp{spp}'.replace('.','p')
    np.savez_compressed(HERE/f'm1m2_template_{tag}.npz',x=x,CU=CU,CV=CV,CUx=CUx,omega=omega,alpha=alpha,kappa=kappa,b=b,w=w)
    return summary,force_rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--b',type=float,required=True); ap.add_argument('--w',type=float,required=True); ap.add_argument('--spp',type=int,default=160); ap.add_argument('--age',type=float,default=40); ap.add_argument('--window',type=float,default=10); ap.add_argument('--L',type=float,default=600); ap.add_argument('--N',type=int,default=4096)
    a=ap.parse_args(); summary,rows=run(a.b,a.w,a.spp,L=a.L,N=a.N,age_periods=a.age,window_periods=a.window)
    tag=f'b{a.b:g}_w{a.w:g}_age{a.age:g}_L{a.L:g}_N{a.N}_spp{a.spp}'.replace('.','p')
    with open(HERE/f'm1m2_summary_{tag}.csv','w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['quantity','value']); [wr.writerow([k,v]) for k,v in summary.items()]
    with open(HERE/f'm1m2_forces_{tag}.csv','w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print('CASE',a.b,a.w,'spp',a.spp)
    for k,v in summary.items(): print(f'{k:32s} {v:.12e}' if isinstance(v,float) else k,v)
    for r in rows:
        print(f"s={r['actual_kappaR']:.5f} A1={r['A1']:.7e} A1/loc={r['A1_over_local']:.5f} R1={r['R1_leading']:.5f} R1corr={r['R1_over_1plus_mu_k2']:.5f} A2={r['A2']:.7e} A2/loc={r['A2_over_local']:.5f} R2={r['R2_leading']:.5f}")
if __name__=='__main__': main()
