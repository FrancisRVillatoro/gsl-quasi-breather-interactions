
from pathlib import Path
import zipfile, io, numpy as np, csv, math
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
SRC=HERE/"gsl_chirp_aware_phase_experiment_b1_w3.zip"
H=(1,3,5,7)

def load_npz_from_zip(zf,name):
    with zf.open(name) as f:
        return np.load(io.BytesIO(f.read()))

def G(u,b=1.0):
    y=1-np.cos(u)
    return 2*y/(1+np.sqrt(1+b*b*y))

def fit_delta(delta,F,nmax=4):
    cols=[np.ones_like(delta)]
    for n in range(1,nmax+1):
        cols += [np.cos(n*delta),np.sin(n*delta)]
    A=np.column_stack(cols)
    c,*_=np.linalg.lstsq(A,F,rcond=None)
    fit=A@c
    z={"c0":float(c[0]),"rms":float(np.sqrt(np.mean((F-fit)**2)))}
    for n in range(1,nmax+1):
        z[f"cos{n}"]=float(c[1+2*(n-1)])
        z[f"sin{n}"]=float(c[2+2*(n-1)])
    return z

def analyze(d,tag):
    x=d["x"]; CU=d["CU"]; CV=d["CV"]; CUx=d["CUx"]
    omega=float(d["omega"]); kappa=float(d["kappa"])
    N=x.size; dx=x[1]-x[0]; ic=N//2
    gamma=5.0/12.0

    def point(theta,idx):
        u=float(CU[0,idx]); v=float(CV[0,idx]); ux=float(CUx[0,idx])
        for n in H:
            j=1+2*H.index(n)
            c=math.cos(n*theta); s=math.sin(n*theta)
            u += CU[j,idx]*c+CU[j+1,idx]*s
            v += CV[j,idx]*c+CV[j+1,idx]*s
            ux += CUx[j,idx]*c+CUx[j+1,idx]*s
        return u,v,ux

    def cross(L,R):
        uL,vL,xL=L; uR,vR,xR=R
        return float(G(uL+uR)-G(uL)-G(uR)-vL*vR-xL*xR)

    nd=48; nt=192
    deltas=2*np.pi*np.arange(nd)/nd
    thetas=2*np.pi*np.arange(nt)/nt
    targets=np.arange(6.0,10.01,0.25)
    rows=[]

    for target in targets:
        m=max(1,int(round((target/kappa)/(2*dx))))
        R=2*m*dx; s=kappa*R
        ip=ic+m; im=ic-m

        Zp=CU[1,ip]-1j*CU[2,ip]
        Zm=CU[1,im]-1j*CU[2,im]
        amid=np.sqrt(abs(Zp)*abs(Zm))

        F=np.empty(nd)
        for jd,delta in enumerate(deltas):
            vals=np.empty(nt)
            for jt,th in enumerate(thetas):
                vals[jt]=cross(point(th-delta/2,ip),point(th+delta/2,im))
            F[jd]=vals.mean()

        ff=fit_delta(deltas,F,4)
        A1=ff["cos1"]; A2=ff["cos2"]
        A2local=-(3*gamma/16)*amid**4
        rows.append({
            "tag":tag,"target_kappaR":float(target),"actual_kappaR":float(s),
            "R":float(R),"omega":omega,"kappa":kappa,
            "A1":A1,"A2":A2,"A3":ff["cos3"],"A4":ff["cos4"],
            "A2_over_A1":A2/A1,"amid_fundamental":float(amid),
            "A2_quartic_local":float(A2local),
            "A2_over_local_theory":float(A2/A2local),
            "phase_fit_rms":ff["rms"],
        })
    return rows

def expfit(R,A,mask):
    xx=R[mask]; yy=np.log(np.abs(A[mask]))
    slope,inter=np.polyfit(xx,yy,1)
    yf=slope*xx+inter
    r2=1-np.sum((yy-yf)**2)/np.sum((yy-yy.mean())**2)
    return -slope,np.exp(inter),r2

def write_csv(path,rows):
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def main():
    with zipfile.ZipFile(SRC) as zf:
        rows160=analyze(load_npz_from_zip(zf,"chirp_template_spp160.npz"),"spp160")
        rows320=analyze(load_npz_from_zip(zf,"chirp_template_spp320.npz"),"spp320")

    rows=rows160+rows320
    write_csv(HERE/"second_harmonic_distance_b1_w3.csv",rows)

    s=np.array([r["actual_kappaR"] for r in rows320])
    R=np.array([r["R"] for r in rows320])
    A1=np.array([r["A1"] for r in rows320])
    A2=np.array([r["A2"] for r in rows320])
    kappa=rows320[0]["kappa"]

    fitrows=[]
    for lo,hi in [(6,10),(6.5,9),(7,9),(7,8.5),(7,8)]:
        m=(s>=lo)&(s<=hi)
        e1,c1,r21=expfit(R,A1,m)
        e2,c2,r22=expfit(R,A2,m)
        er,cr,r2r=expfit(R,np.abs(A2/A1),m)
        fitrows.append({
            "kappaR_min":lo,"kappaR_max":hi,
            "A1_exponent":e1,"A1_exponent_over_kappa":e1/kappa,"A1_R2":r21,
            "A2_exponent":e2,"A2_exponent_over_kappa":e2/kappa,"A2_R2":r22,
            "ratio_exponent":er,"ratio_exponent_over_kappa":er/kappa,"ratio_R2":r2r,
        })
    write_csv(HERE/"second_harmonic_exponential_fits_b1_w3.csv",fitrows)

    plt.figure(figsize=(7.4,5.1))
    plt.semilogy(s,np.abs(A1),"o-",label=r"$|A_1|$")
    plt.semilogy(s,np.abs(A2),"o-",label=r"$|A_2|$")
    j=np.argmin(abs(s-7.5))
    plt.semilogy(s,abs(A1[j])*np.exp(-(s-s[j])),"--",label=r"$e^{-\kappa R}$")
    plt.semilogy(s,abs(A2[j])*np.exp(-2*(s-s[j])),"--",label=r"$e^{-2\kappa R}$")
    plt.xlabel(r"$\kappa R$"); plt.ylabel("phase-force coefficient")
    plt.grid(True,which="both"); plt.legend(); plt.tight_layout()
    plt.savefig(HERE/"second_harmonic_distance_scaling.png",dpi=180); plt.close()

    plt.figure(figsize=(7.4,5.1))
    q=np.array([r["A2_over_local_theory"] for r in rows320])
    plt.plot(s,q,"o-"); plt.axhline(1,ls="--")
    plt.xlabel(r"$\kappa R$")
    plt.ylabel(r"$A_2/[-(3\gamma/16)a_{\rm mid}^4]$")
    plt.grid(True); plt.tight_layout()
    plt.savefig(HERE/"second_harmonic_local_quartic_test.png",dpi=180); plt.close()

    print("Done.")
    for r in fitrows:
        print(r)

if __name__=="__main__":
    main()
