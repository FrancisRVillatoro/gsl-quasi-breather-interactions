#!/usr/bin/env python3
"""Phase-resolved first-period two-body deformation diagnostics.

Reproduces the sign-consistent convention used in the manuscript:

    Delta F_def = F_L,full - F_L,sup,

where F_L is the self-subtracted left momentum flux sigma(x_m,t).
The script evaluates the exact sine-Gordon benchmark and the production
GSL frozen template at kappa R ~ 7, Fourier fits the phase dependence,
and writes CSV tables.
"""
from pathlib import Path
import argparse, csv
import numpy as np

from ssf4_gsl import grid, ssf4_step, spectral_derivative, gsl_potential


def fit_fourier(delta, y, nmax=4):
    cols = [np.ones_like(delta)]
    for n in range(1, nmax + 1):
        cols += [np.cos(n * delta), np.sin(n * delta)]
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A, np.asarray(y), rcond=None)
    out = {"c0": float(c[0])}
    for n in range(1, nmax + 1):
        out[f"A{n}"] = float(c[1 + 2 * (n - 1)])
        out[f"S{n}"] = float(c[2 + 2 * (n - 1)])
    return out


def avg_period(t, y):
    return float(np.trapezoid(np.asarray(y), np.asarray(t)) / (t[-1] - t[0]))


def sg_scan(target=7.0, omega=0.9, spp=160, nphase=16, N=2048, L=240.0):
    kappa = np.sqrt(1.0 - omega**2)
    T = 2 * np.pi / omega
    R = target / kappa
    x, dx, k, omega_k = grid(N, L)
    ic = N // 2
    delta = 2 * np.pi * np.arange(nphase) / nphase

    def breather(center, theta):
        X = kappa * (x - center)
        S = 1 / np.cosh(X)
        z = (kappa / omega) * np.sin(theta) * S
        den = 1 + z * z
        return 4 * np.arctan(z), 4 * kappa * np.cos(theta) * S / den

    def sigma(u, v, ux):
        return float(1 - np.cos(u[ic]) - 0.5 * v[ic] ** 2 - 0.5 * ux[ic] ** 2)

    Fsup, Ffull = [], []
    for d in delta:
        uL, vL = breather(-R / 2, -d / 2)
        uR, vR = breather(+R / 2, +d / 2)
        u, v = uL + uR, vL + vR
        dt = T / spp
        tt, fs, ff = [], [], []
        for j in range(spp + 1):
            if j % 4 == 0 or j == spp:
                uxL = spectral_derivative(uL, k)
                uxR = spectral_derivative(uR, k)
                ux = spectral_derivative(u, k)
                fs.append(sigma(uL + uR, vL + vR, uxL + uxR) - sigma(uL, vL, uxL) - sigma(uR, vR, uxR))
                ff.append(sigma(u, v, ux) - sigma(uL, vL, uxL) - sigma(uR, vR, uxR))
                tt.append(j * dt)
            if j < spp:
                uL, vL = ssf4_step(uL, vL, dt, 0.0, omega_k)
                uR, vR = ssf4_step(uR, vR, dt, 0.0, omega_k)
                u, v = ssf4_step(u, v, dt, 0.0, omega_k)
        Fsup.append(avg_period(tt, fs))
        Ffull.append(avg_period(tt, ff))
    return delta, np.asarray(Fsup), np.asarray(Ffull), kappa, R


def gsl_scan(template_path, target=7.0, spp=160, nphase=16):
    d = np.load(template_path)
    x = d["x"].copy(); CU = d["CU"].copy(); CV = d["CV"].copy()
    omega = float(d["omega"]); kappa = float(d["kappa"])
    H = (1, 3, 5, 7)
    N = x.size; dx = x[1] - x[0]; L = N * dx
    _, _, k, omega_k = grid(N, L); ic = N // 2
    T = 2 * np.pi / omega
    m = max(1, int(round((target / kappa) / (2 * dx))))
    R = 2 * m * dx
    delta = 2 * np.pi * np.arange(nphase) / nphase

    def reconstruct(theta):
        u = CU[0].copy(); v = CV[0].copy()
        for n in H:
            j = 1 + 2 * H.index(n)
            u += CU[j] * np.cos(n * theta) + CU[j + 1] * np.sin(n * theta)
            v += CV[j] * np.cos(n * theta) + CV[j + 1] * np.sin(n * theta)
        return u, v

    def shift(f, a):
        return np.fft.irfft(np.fft.rfft(f) * np.exp(-1j * k * a), n=N)

    def sigma(u, v, ux):
        return float(gsl_potential(np.array([u[ic]]), 1.0)[0] - 0.5 * v[ic] ** 2 - 0.5 * ux[ic] ** 2)

    Fsup, Ffull = [], []
    for dlt in delta:
        uL0, vL0 = reconstruct(-dlt / 2); uR0, vR0 = reconstruct(+dlt / 2)
        uL, vL = shift(uL0, -R / 2), shift(vL0, -R / 2)
        uR, vR = shift(uR0, +R / 2), shift(vR0, +R / 2)
        u, v = uL + uR, vL + vR
        dt = T / spp
        tt, fs, ff = [], [], []
        for j in range(spp + 1):
            if j % 4 == 0 or j == spp:
                uxL = spectral_derivative(uL, k); uxR = spectral_derivative(uR, k); ux = spectral_derivative(u, k)
                fs.append(sigma(uL + uR, vL + vR, uxL + uxR) - sigma(uL, vL, uxL) - sigma(uR, vR, uxR))
                ff.append(sigma(u, v, ux) - sigma(uL, vL, uxL) - sigma(uR, vR, uxR))
                tt.append(j * dt)
            if j < spp:
                uL, vL = ssf4_step(uL, vL, dt, 1.0, omega_k)
                uR, vR = ssf4_step(uR, vR, dt, 1.0, omega_k)
                u, v = ssf4_step(u, v, dt, 1.0, omega_k)
        Fsup.append(avg_period(tt, fs)); Ffull.append(avg_period(tt, ff))
    return delta, np.asarray(Fsup), np.asarray(Ffull), kappa, R


def write_case(name, delta, sup, full, kappa, R, outdir):
    deform = full - sup
    qs, qf, qd = fit_fourier(delta, sup), fit_fourier(delta, full), fit_fourier(delta, deform)
    row = {"case": name, "kappa": kappa, "R": R, "kappaR": kappa * R}
    for p, q in [("sup", qs), ("full", qf), ("def", qd)]:
        for key, val in q.items(): row[f"{p}_{key}"] = val
    row["def_c0_over_2A2"] = qd["c0"] / (2 * qd["A2"])
    row["def_A1_over_4A2"] = qd["A1"] / (4 * qd["A2"])
    with open(outdir / f"v8_phase_fourier_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    with open(outdir / f"v8_phase_curve_{name}.csv", "w", newline="") as f:
        fields = ["delta", "delta_over_pi", "F_sup", "F_full", "DeltaF_def"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for d, a, b, c in zip(delta, sup, full, deform):
            w.writerow({"delta": d, "delta_over_pi": d / np.pi, "F_sup": a, "F_full": b, "DeltaF_def": c})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    ap.add_argument("--spp", type=int, default=160)
    ap.add_argument("--nphase", type=int, default=16)
    args = ap.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    d, s, f, k, R = sg_scan(spp=args.spp, nphase=args.nphase)
    print(write_case("SG", d, s, f, k, R, args.data_dir))
    template = args.data_dir / "v3_allharmonic_template_spp320.npz"
    d, s, f, k, R = gsl_scan(template, spp=args.spp, nphase=args.nphase)
    print(write_case("GSL", d, s, f, k, R, args.data_dir))


if __name__ == "__main__":
    main()
