#!/usr/bin/env python3
"""Sine-Gordon first-period deformation A2_def/A2^(4) versus omega at kappa R=7."""
from pathlib import Path
import csv, numpy as np
from v8_phase_resolved_deformation import sg_scan, fit_fourier


def exact_local_U1(omega, s=7.0, nt=8192):
    kappa = np.sqrt(1 - omega**2)
    theta = 2 * np.pi * np.arange(nt) / nt
    S = 1 / np.cosh(s / 2)
    z = (kappa / omega) * np.sin(theta) * S
    u = 4 * np.arctan(z)
    return 2 * np.mean(u * np.exp(-1j * theta))


def main():
    out = Path(__file__).resolve().parents[1] / "data"
    rows = []
    for omega in (0.80, 0.90, 0.95, 0.98):
        delta, sup, full, kappa, R = sg_scan(omega=omega, target=7.0, spp=120, nphase=12, N=1024, L=180.0)
        qsup = fit_fourier(delta, sup, 3); qfull = fit_fourier(delta, full, 3); qdef = fit_fourier(delta, full - sup, 3)
        A2dir = -(1 / 32) * abs(exact_local_U1(omega))**4
        rows.append({
            "omega": omega, "kappa": kappa, "kappaR": kappa * R,
            "A2_direct_quartic": A2dir,
            "def_c0": qdef["c0"], "def_A1": qdef["A1"], "def_A2": qdef["A2"], "def_A3": qdef["A3"],
            "def_A2_over_A2direct": qdef["A2"] / A2dir,
            "def_c0_over_2A2def": qdef["c0"] / (2 * qdef["A2"]),
            "def_A1_over_4A2def": qdef["A1"] / (4 * qdef["A2"]),
            "sup_A1": qsup["A1"], "sup_A2": qsup["A2"],
            "full_A1": qfull["A1"], "full_A2": qfull["A2"],
            "total_A2_over_direct": qfull["A2"] / A2dir,
        })
    with open(out / "v8_sg_deformation_vs_omega.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows: print(r)

if __name__ == "__main__": main()
