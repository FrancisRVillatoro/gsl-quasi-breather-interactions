"""
third_harmonic_order6_b1_w3.py
==============================

Order-six analysis of the cos(3 delta) interaction-force harmonic for two
graphene-superlattice quasi-breathers.

Input dependency:
    gsl_chirp_aware_phase_experiment_b1_w3.zip

Theory
------
Write the local fundamental/third-harmonic tail as
    u = a cos(theta) + alpha a^3 cos(3 theta) + ...
with
    alpha = -gamma/32,
    gamma = (2+3 b^2)/12,
    eta   = (4+30 b^2+45 b^4)/480.

At O(a^6), the cos(3 delta) coefficient has three contributions:

  sextic potential:
      A3^(6) = (5 eta/48) a^6,

  quartic potential with one induced third harmonic:
      A3^(4x3) = -(alpha gamma/4) a^6,

  quadratic stress of the induced third harmonic:
      A3^(3x3) = alpha^2(-4+9 kappa^2) a^6.

Hence
    A3_local =
      [5 eta/48 + gamma^2(4+9 kappa^2)/1024] a^6.

For b=1, w=3 the script also evaluates these three pieces directly from
the measured chirp-aware n=1 and n=3 template coefficients and compares
their sum against the exact phase-averaged stress harmonic A3.

Expected distance law for the *pure local evanescent contribution*:
    A3_local ~ exp(-3 kappa R).
"""

from pathlib import Path
import zipfile, io, csv, math
import numpy as np

HERE=Path(__file__).resolve().parent
SRC=HERE/"gsl_chirp_aware_phase_experiment_b1_w3.zip"

H=(1,3,5,7)
bpar=1.0
gamma=(2+3*bpar*bpar)/12.0
eta=(4+30*bpar*bpar+45*bpar**4)/480.0

def load_npz_zip(zf,name):
    with zf.open(name) as f:
        return np.load(io.BytesIO(f.read()))

def G2(u): return 0.5*u*u
def G4(u): return -gamma*u**4/4.0
def G6(u): return eta*u**6/6.0

def Gexact(u):
    y=1-np.cos(u)
    return 2*y/(1+np.sqrt(1+bpar*bpar*y))

def fit_phase(delta,F,nmax=5):
    cols=[np.ones_like(delta)]
    for n in range(1,nmax+1):
        cols += [np.cos(n*delta),np.sin(n*delta)]
    A=np.column_stack(cols)
    c,*_=np.linalg.lstsq(A,F,rcond=None)
    z={"c0":float(c[0])}
    for n in range(1,nmax+1):
        z[f"cos{n}"]=float(c[1+2*(n-1)])
        z[f"sin{n}"]=float(c[2+2*(n-1)])
    return z

def analyze(d,tag):
    x=d["x"]; CU=d["CU"]; CV=d["CV"]; CUx=d["CUx"]
    omega=float(d["omega"]); kappa=float(d["kappa"])
    N=x.size; dx=x[1]-x[0]; ic=N//2

    def comp(n,theta,idx):
        j=1+2*H.index(n)
        c=math.cos(n*theta); s=math.sin(n*theta)
        return (float(CU[j,idx]*c+CU[j+1,idx]*s),
                float(CV[j,idx]*c+CV[j+1,idx]*s),
                float(CUx[j,idx]*c+CUx[j+1,idx]*s))

    def full(theta,idx):
        u=float(CU[0,idx]); v=float(CV[0,idx]); ux=float(CUx[0,idx])
        for n in H:
            q=comp(n,theta,idx)
            u+=q[0]; v+=q[1]; ux+=q[2]
        return u,v,ux

    def add(A,B):
        return (A[0]+B[0],A[1]+B[1],A[2]+B[2])

    def cross(L,R,Gfun,include_deriv=True):
        uL,vL,xL=L; uR,vR,xR=R
        ans=Gfun(uL+uR)-Gfun(uL)-Gfun(uR)
        if include_deriv:
            ans -= vL*vR+xL*xR
        return float(ans)

    nd=64; nt=256
    deltas=2*np.pi*np.arange(nd)/nd
    thetas=2*np.pi*np.arange(nt)/nt
    targets=np.arange(6.0,9.01,0.25)
    rows=[]

    for target in targets:
        m=max(1,int(round((target/kappa)/(2*dx))))
        R=2*m*dx; s=kappa*R; ip=ic+m; im=ic-m

        Z1p=CU[1,ip]-1j*CU[2,ip]
        Z1m=CU[1,im]-1j*CU[2,im]
        a1=np.sqrt(abs(Z1p)*abs(Z1m))

        Ffull=np.empty(nd); Fsext=np.empty(nd)
        Fquart13=np.empty(nd); Fquart1=np.empty(nd); Fquad3=np.empty(nd)

        for jd,delta in enumerate(deltas):
            vf=[]; vs=[]; vq13=[]; vq1=[]; v3=[]
            for th in thetas:
                thL=th-delta/2; thR=th+delta/2
                FL=full(thL,ip); FR=full(thR,im)
                L1=comp(1,thL,ip); R1=comp(1,thR,im)
                L3=comp(3,thL,ip); R3=comp(3,thR,im)
                L13=add(L1,L3); R13=add(R1,R3)

                vf.append(cross(FL,FR,Gexact,True))
                vs.append(cross(L1,R1,G6,False))
                vq13.append(cross(L13,R13,G4,False))
                vq1.append(cross(L1,R1,G4,False))
                v3.append(cross(L3,R3,G2,True))

            Ffull[jd]=np.mean(vf)
            Fsext[jd]=np.mean(vs)
            Fquart13[jd]=np.mean(vq13)
            Fquart1[jd]=np.mean(vq1)
            Fquad3[jd]=np.mean(v3)

        Fquartcorr=Fquart13-Fquart1
        Forder6=Fsext+Fquartcorr+Fquad3

        ff=fit_phase(deltas,Ffull)
        fs=fit_phase(deltas,Fsext)
        fq=fit_phase(deltas,Fquartcorr)
        f3=fit_phase(deltas,Fquad3)
        fo=fit_phase(deltas,Forder6)

        B3=5*eta/48 + gamma**2*(4+9*kappa*kappa)/1024
        A3local=B3*a1**6

        rows.append({
            "tag":tag,"target_kappaR":float(target),"actual_kappaR":float(s),
            "R":float(R),"omega":omega,"kappa":kappa,
            "a1_mid":float(a1),
            "A3_full":ff["cos3"],
            "A3_sextic_direct":fs["cos3"],
            "A3_quartic_via_u3":fq["cos3"],
            "A3_quadratic_u3":f3["cos3"],
            "A3_order6_sum":fo["cos3"],
            "A3_asym_local":float(A3local),
            "full_over_order6":ff["cos3"]/fo["cos3"],
            "full_over_asym_local":ff["cos3"]/A3local,
        })
    return rows

def main():
    with zipfile.ZipFile(SRC) as zf:
        rows=analyze(load_npz_zip(zf,"chirp_template_spp320.npz"),"spp320")

    with open(HERE/"third_harmonic_components_b1_w3.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("gamma =",gamma)
    print("eta   =",eta)
    print("B3    =",5*eta/48+gamma**2*(4+9*rows[0]["kappa"]**2)/1024)
    for r in rows:
        if abs(r["target_kappaR"]-round(r["target_kappaR"]))<1e-12:
            print(r)

if __name__=="__main__":
    main()
