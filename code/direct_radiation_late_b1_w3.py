from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from ssf4_gsl import grid, ssf4_step, spectral_derivative

HERE = Path(__file__).resolve().parent


def initial_qb(x, w):
    q = 1.0/np.sqrt(1.0+w*w)
    return np.zeros_like(x), 4.0*q/np.cosh(q*x)


def zero_crossing_frequency(t, y):
    idx=np.where((y[:-1]<=0)&(y[1:]>0))[0]
    tc=[]
    for i in idx:
        a=-y[i]/(y[i+1]-y[i])
        tc.append(t[i]+a*(t[i+1]-t[i]))
    tc=np.asarray(tc)
    return float(2*np.pi/np.mean(np.diff(tc)))


def harmonic_fit_frequency(t, y, guess):
    tt=t-t.mean()
    def design(omega):
        cols=[np.ones_like(tt),tt]
        for n in (1,3,5):
            cols += [np.cos(n*omega*tt),np.sin(n*omega*tt)]
        return np.column_stack(cols)
    def obj(omega):
        A=design(omega)
        c,*_=np.linalg.lstsq(A,y,rcond=None)
        r=y-A@c
        return float(r@r)
    res=minimize_scalar(obj,bounds=(0.98*guess,min(0.999999,1.02*guess)),
                        method='bounded',options={'xatol':1e-13})
    omega=float(res.x)
    return omega, float(np.sqrt(obj(omega)/len(y)))


def parabolic_peak(x, y, j):
    if j<=0 or j>=len(y)-1:
        return float(x[j]), float(y[j])
    yy=np.log(np.maximum(y[j-1:j+2],1e-300))
    den=yy[0]-2*yy[1]+yy[2]
    if abs(den)<1e-30:
        return float(x[j]),float(y[j])
    delta=0.5*(yy[0]-yy[2])/den
    dx=x[j+1]-x[j]
    xp=x[j]+delta*dx
    yp=np.exp(yy[1]-0.25*(yy[0]-yy[2])*delta)
    return float(xp),float(yp)


def band_limited_real(signal, dt, omega0, halfwidth):
    n=signal.shape[0]
    om=2*np.pi*np.fft.fftfreq(n,d=dt)
    S=np.fft.fft(signal,axis=0)
    mask=(np.abs(np.abs(om)-omega0)<=halfwidth)
    shape=(slice(None),)+(None,)*(signal.ndim-1)
    Sb=S*mask[shape]
    return np.fft.ifft(Sb,axis=0).real


def write_csv(path,rows):
    with open(path,'w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--spp',type=int,default=160)
    ap.add_argument('--tag',default='spp160')
    args=ap.parse_args()

    b=1.0; w=3.0
    L=1000.0; N=8192
    age_periods=50.0
    measure_periods=12.0
    sample_every=2

    x,dx,k,omega_k=grid(N,L)
    ic=N//2
    omega0=w/np.sqrt(1+w*w)
    T0=2*np.pi/omega0
    dt=T0/args.spp
    sample_dt=sample_every*dt
    age_steps=int(round(age_periods*args.spp))
    meas_steps=int(round(measure_periods*args.spp))
    total_time=(age_steps+meas_steps)*dt
    causal_margin=L/2-total_time

    probe_x=np.array([20.,30.,40.,50.,60.,80.,100.])
    probe_idx=np.array([int(np.argmin(abs(x-xp))) for xp in probe_x])
    probe_x=x[probe_idx]
    far_mask=(x>=20.0)&(x<=100.0)
    xf=x[far_mask]

    u,v=initial_qb(x,w)
    for _ in range(age_steps):
        u,v=ssf4_step(u,v,dt,b,omega_k)

    times=[]; center=[]; pu=[]; pv=[]; pux=[]; far_u=[]
    for n in range(meas_steps):
        u,v=ssf4_step(u,v,dt,b,omega_k)
        if (n+1)%sample_every==0:
            t=(age_steps+n+1)*dt
            ux=spectral_derivative(u,k)
            times.append(t); center.append(u[ic])
            pu.append(u[probe_idx].copy()); pv.append(v[probe_idx].copy())
            pux.append(ux[probe_idx].copy()); far_u.append(u[far_mask].copy())

    times=np.asarray(times); center=np.asarray(center)
    pu=np.asarray(pu); pv=np.asarray(pv); pux=np.asarray(pux); far_u=np.asarray(far_u)

    omega_zc=zero_crossing_frequency(times,center)
    omega_qb,center_fit_rms=harmonic_fit_frequency(times,center,omega_zc)

    nt=len(times); tw=np.hanning(nt)
    om=2*np.pi*np.fft.rfftfreq(nt,d=sample_dt)
    probe_rows=[]; flux_rows=[]; harmonic_targets=(3,5,7)

    for jp,xp in enumerate(probe_x):
        sig=(pu[:,jp]-pu[:,jp].mean())*tw
        S=np.abs(np.fft.rfft(sig))
        total_flux=float(np.mean(-pv[:,jp]*pux[:,jp]))
        row={'tag':args.tag,'x':float(xp),'total_mean_energy_flux':total_flux}
        for nh in harmonic_targets:
            target=nh*omega_qb
            ids=np.where((om>=target-0.20)&(om<=target+0.20))[0]
            j=ids[np.argmax(S[ids])]
            op,amp=parabolic_peak(om,S,j)
            row[f'omega_peak_n{nh}']=op; row[f'spectrum_amp_n{nh}']=amp
            vb=band_limited_real(pv[:,jp],sample_dt,op,0.18)
            xb=band_limited_real(pux[:,jp],sample_dt,op,0.18)
            row[f'flux_n{nh}']=float(np.mean(-vb*xb))
        probe_rows.append(row)

    # 2D spectrum in right far field.
    A=far_u-far_u.mean(axis=0,keepdims=True)
    Aw=A*np.hanning(A.shape[0])[:,None]*np.hanning(A.shape[1])[None,:]
    ntt=2*A.shape[0]; nxx=2*A.shape[1]
    P2=np.abs(np.fft.fft2(Aw,s=(ntt,nxx)))**2
    om2=2*np.pi*np.fft.fftfreq(ntt,d=sample_dt)
    kk2=2*np.pi*np.fft.fftfreq(nxx,d=dx)
    dispersion_rows=[]; peak_points=[]
    for nh in harmonic_targets:
        target=nh*omega_qb
        ommask=(om2>0)&(om2>=target-0.25)&(om2<=target+0.25)
        kmask=np.abs(kk2)>=0.5
        sub=P2[np.ix_(ommask,kmask)]
        ii,jj=np.unravel_index(np.argmax(sub),sub.shape)
        iom=np.where(ommask)[0][ii]; ik=np.where(kmask)[0][jj]
        op=float(om2[iom]); kp_abs=abs(float(kk2[ik]))
        kpred=float(np.sqrt(max(op*op-1.0,0.0)))
        residual=float(op*op-kp_abs*kp_abs-1.0)
        rel_k=(kp_abs-kpred)/kpred if kpred>0 else np.nan
        dispersion_rows.append({
            'tag':args.tag,'n':nh,'omega_center_QB':omega_qb,
            'omega_target_nOmega':target,'omega_peak_2D':op,
            'k_peak_abs_2D':kp_abs,'k_pred_from_peak_omega':kpred,
            'relative_k_error':rel_k,
            'dispersion_residual_omega2_minus_k2_minus1':residual,
        })
        peak_points.append((kp_abs,op,nh))

    for nh in harmonic_targets:
        vals=np.array([r[f'flux_n{nh}'] for r in probe_rows])
        flux_rows.append({
            'tag':args.tag,'n':nh,'mean_flux_over_probes':float(vals.mean()),
            'median_flux_over_probes':float(np.median(vals)),
            'min_flux_over_probes':float(vals.min()),'max_flux_over_probes':float(vals.max()),
            'fraction_positive':float(np.mean(vals>0)),
        })

    write_csv(HERE/f'direct_radiation_probes_{args.tag}.csv',probe_rows)
    write_csv(HERE/f'direct_radiation_dispersion_{args.tag}.csv',dispersion_rows)
    write_csv(HERE/f'direct_radiation_flux_{args.tag}.csv',flux_rows)

    np.savez_compressed(HERE/f'direct_radiation_data_{args.tag}.npz',
        times=times,probe_x=probe_x,probe_u=pu,probe_v=pv,probe_ux=pux,
        x_far=xf,far_u=far_u,omega_qb=omega_qb,sample_dt=sample_dt,dx=dx)

    plt.figure(figsize=(7.8,5.2))
    for jp,xp in enumerate(probe_x[:5]):
        sig=(pu[:,jp]-pu[:,jp].mean())*tw
        S=np.abs(np.fft.rfft(sig)); S=S/max(S.max(),1e-300)
        plt.semilogy(om,S,label=f'x={xp:.1f}')
    for nh in harmonic_targets:
        plt.axvline(nh*omega_qb,ls='--')
    plt.xlim(1.5,6.5); plt.ylim(1e-8,1.2)
    plt.xlabel(r'$\omega$'); plt.ylabel('normalized probe spectrum')
    plt.title('Direct far-field spectra'); plt.grid(True,which='both'); plt.legend(); plt.tight_layout()
    plt.savefig(HERE/f'direct_radiation_probe_spectra_{args.tag}.png',dpi=180); plt.close()

    omplot=(om2>1.5)&(om2<6.5); kplot=(kk2>=0)&(kk2<7.0)
    Z=P2[np.ix_(omplot,kplot)]; Z=Z/max(Z.max(),1e-300)
    plt.figure(figsize=(7.8,5.5))
    plt.imshow(np.log10(np.maximum(Z,1e-12)),origin='lower',aspect='auto',
               extent=[kk2[kplot][0],kk2[kplot][-1],om2[omplot][0],om2[omplot][-1]])
    kval=np.linspace(0.5,7.0,600)
    plt.plot(kval,np.sqrt(1+kval*kval),'--',label=r'$\omega=\sqrt{1+k^2}$')
    for kp,op,nh in peak_points:
        plt.plot(kp,op,'o',label=f'n={nh}')
    plt.xlabel(r'$|k|$'); plt.ylabel(r'$\omega$'); plt.title('Space-time spectrum in the right far field')
    plt.legend(); plt.tight_layout(); plt.savefig(HERE/f'direct_radiation_komega_{args.tag}.png',dpi=180); plt.close()

    plt.figure(figsize=(7.5,5.0))
    plt.plot(probe_x,[r['total_mean_energy_flux'] for r in probe_rows],'o-',label='total')
    for nh in harmonic_targets:
        plt.plot(probe_x,[r[f'flux_n{nh}'] for r in probe_rows],'o-',label=f'band n={nh}')
    plt.xlabel('x'); plt.ylabel(r'$\langle J_E\rangle=\langle-u_tu_x\rangle$')
    plt.title('Outward energy flux measured directly'); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(HERE/f'direct_radiation_flux_{args.tag}.png',dpi=180); plt.close()

    print('DIRECT RADIATION EXPERIMENT')
    print(f'spp={args.spp}')
    print(f'omega_QB={omega_qb:.12e}')
    print(f'omega_zero_crossing={omega_zc:.12e}')
    print(f'center_fit_rms={center_fit_rms:.12e}')
    print(f'total_time={total_time:.6f}')
    print(f'causal_margin_to_boundary={causal_margin:.6f}')
    print('\n2D DISPERSION PEAKS')
    for r in dispersion_rows:
        print(f"n={r['n']} omega={r['omega_peak_2D']:.6f} |k|={r['k_peak_abs_2D']:.6f} "
              f"kpred={r['k_pred_from_peak_omega']:.6f} rel.k.err={r['relative_k_error']:+.3e} "
              f"disp.res={r['dispersion_residual_omega2_minus_k2_minus1']:+.3e}")
    print('\nBAND-LIMITED OUTWARD FLUX')
    for r in flux_rows:
        print(f"n={r['n']} median={r['median_flux_over_probes']:.6e} mean={r['mean_flux_over_probes']:.6e} "
              f"positive_fraction={r['fraction_positive']:.2f}")
    print('\nPROBE FLUXES')
    for r in probe_rows:
        print(f"x={r['x']:.2f} total={r['total_mean_energy_flux']:.6e} "
              f"J3={r['flux_n3']:.6e} J5={r['flux_n5']:.6e} J7={r['flux_n7']:.6e}")

if __name__=='__main__':
    main()
