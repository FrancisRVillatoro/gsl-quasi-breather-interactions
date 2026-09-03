#!/usr/bin/env python3
"""Regenerate the publication-quality figure set for the v1.0.1 archive.

Outputs are written to ``publication_figures/``. Line/scatter figures are
vector PDF. The k-omega density map is intentionally raster PNG because it
represents a 2-D spectral field.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "publication_figures"
OUT.mkdir(exist_ok=True)


def readcsv(name):
    with open(DATA / name, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        q = {}
        for k, v in r.items():
            try: q[k] = float(v)
            except (TypeError, ValueError): q[k] = v
        out.append(q)
    return out


def finish(name, xlabel, ylabel, yscale=None, legend=True):
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if yscale: plt.yscale(yscale)
    plt.grid(True, alpha=0.25)
    if legend: plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.pdf")
    plt.close()


def main():
    # Isolated fundamental tail
    d = np.load(DATA / "v3_allharmonic_template_spp320.npz")
    x, CU, kap = d["x"], d["CU"], float(d["kappa"])
    amp = np.hypot(CU[1], CU[2]); m = (x >= 0) & (x <= 20)
    plt.figure(figsize=(6.4, 4.3)); plt.semilogy(x[m], amp[m], label=r"$|U_1(x)|$")
    i = np.argmin(abs(x - 4.0)); plt.semilogy(x[m], amp[i] * np.exp(-kap * (x[m] - x[i])), "--", label=rf"$\propto e^{{-\kappa x}}$, $\kappa={kap:.3f}$")
    finish("fig_isolated_tail", r"$x$", "fundamental amplitude")

    # All-harmonic amplitudes
    rows = [r for r in readcsv("v3_allharmonic_amplitudes.csv") if r["spp"] == 320]
    plt.figure(figsize=(6.4, 4.3))
    for xx, marker in zip(sorted(set(r["x"] for r in rows)), ["o", "s", "^", "D"]):
        rr = sorted([r for r in rows if r["x"] == xx], key=lambda r: r["n"])
        plt.plot([r["n"] for r in rr], [r["amp_over_n1"] for r in rr], marker=marker, label=rf"$x={xx:.1f}$")
    finish("fig_even_harmonics", r"temporal harmonic $n$", r"$|U_n|/|U_1|$", yscale="log")

    # Integrator cross-check
    rows = [r for r in readcsv("m3_ssf4_vs_etdrk4_b1_w3.csv") if r["spp"] == 320]
    labels = [r"$\Omega$", r"$\kappa$", r"$A_1$", r"$A_2$", r"$F(0)$", r"$F(\pi/2)$"]
    keys = ["omega", "kappa", "A1", "A2", "F0", "Fpi2"]
    plt.figure(figsize=(6.6, 4.2))
    for target, marker in [(7.0, "o"), (8.0, "s")]:
        vals = [[r for r in rows if r["target_kappaR"] == target and r["quantity"] == k0][0]["relative_difference"] for k0 in keys]
        plt.plot(range(len(keys)), vals, marker=marker, label=rf"$\kappa R\simeq {target:g}$")
    plt.xticks(range(len(keys)), labels)
    finish("fig_integrator_crosscheck", "", "relative SSF4--ETDRK4 difference", yscale="log")

    # Exact SG closure
    m8 = readcsv("m8_sg_exact_local_scan.csv"); rr = [r for r in m8 if abs(r["omega"] - .9) < 1e-12]
    plt.figure(figsize=(6.5, 4.3))
    for vals, label, marker in [
        ([r["A1q_over_Manton"] for r in rr], r"$A_1^{(2)}/A_1^{\rm Manton}$", "o"),
        ([r["B1q_over_Manton"] for r in rr], r"$B_1/B_1^{\rm ad}$", "s"),
        ([r["c0_over_2A2quartic"] for r in rr], r"$c_0/(2A_2^{(4)})$", "^"),
        ([r["deltaA1_over_4A2quartic"] for r in rr], r"$\Delta A_1/(4A_2^{(4)})$", "D"),
        ([r["A2_over_quartic"] for r in rr], r"$A_2/A_2^{(4)}$", "v")]:
        plt.plot([r["kappaR"] for r in rr], vals, marker=marker, label=label)
    plt.axhline(1, ls="--", lw=1)
    finish("fig_m8_sg_integrable", r"$\kappa R$", "measured / asymptotic prediction")

    # Frozen phase force
    rows = readcsv("m6_phase_force_energy_scan.csv"); summary = readcsv("m6_phase_force_energy_summary.csv")
    plt.figure(figsize=(6.4, 4.3))
    for target, marker in [(6.0, "o"), (7.0, "s"), (8.0, "^")]:
        rr = sorted([r for r in rows if r["target_kappaR"] == target], key=lambda r: r["delta"])
        A1 = [s["A1_force_cos1"] for s in summary if s["target_kappaR"] == target][0]
        plt.plot([r["delta_over_pi"] for r in rr], [r["Fbar"] / A1 for r in rr], marker=marker, markevery=8, label=rf"$\kappa R\simeq {target:g}$")
    xx = np.linspace(0, 2, 300); plt.plot(xx, np.cos(np.pi * xx), "--", label=r"$\cos\delta$")
    finish("fig_phase_force", r"$\delta/\pi$", r"$\overline{F}/A_1$")

    # SG quartic fundamental correction
    plt.figure(figsize=(6.4, 4.3))
    for om, marker in zip((.8, .9, .95, .98), ("o", "s", "^", "D")):
        rr = [r for r in m8 if abs(r["omega"] - om) < 1e-12]
        plt.plot([r["kappaR"] for r in rr], [r["deltaA1_over_4A2quartic"] for r in rr], marker=marker, label=rf"$\omega={om:.2f}$")
    plt.axhline(1, ls="--", lw=1)
    finish("supp_m8_quartic", r"$\kappa R$", r"$(A_1-A_1^{(2)})/(4A_2^{(4)})$")

    # A3 decomposition
    rows = readcsv("third_harmonic_components_b1_w3.csv"); plt.figure(figsize=(6.5, 4.3))
    for key, label, marker in [("A3_sextic_direct", "sextic direct", "o"), ("A3_quartic_via_u3", r"quartic via $u_3$", "s"), ("A3_quadratic_u3", r"quadratic $u_3$", "^"), ("A3_order6_sum", "order-six sum", "D")]:
        plt.plot([r["actual_kappaR"] for r in rows], [abs(r[key]) for r in rows], marker=marker, label=label)
    finish("supp_A3", r"$\kappa R$", r"$|A_3|$ contribution", yscale="log")

    # Age/nonadiabaticity and age laws
    rows = readcsv("m5_adiabaticity_aggregate.csv")
    plt.figure(figsize=(6.4, 4.3))
    for w, marker in [(5, "o"), (8, "s")]:
        rr = sorted([r for r in rows if r["w"] == w], key=lambda r: r["age_T0"])
        plt.plot([r["age_T0"] for r in rr], [r["lambda_real_over_kappa"] for r in rr], marker=marker, label=rf"$\Re\Lambda/\kappa$, $w={w}$")
        plt.plot([r["age_T0"] for r in rr], [r["lambda_imag_over_kappa"] for r in rr], marker=marker, ls="--", label=rf"$\Im\Lambda/\kappa$, $w={w}$")
    finish("fig_age", r"age $/T_0$", "normalized local logarithmic derivative")

    plt.figure(figsize=(6.4, 4.3))
    for w, marker in [(5, "o"), (8, "s")]:
        rr = sorted([r for r in rows if r["w"] == w], key=lambda r: r["age_T0"])
        plt.plot([r["age_T0"] for r in rr], [r["A1_over_local"] for r in rr], marker=marker, label=rf"$A_1/A_1^{{(2)}}$, $w={w}$")
        plt.plot([r["age_T0"] for r in rr], [r["A2_over_local"] for r in rr], marker=marker, ls="--", label=rf"$A_2/A_2^{{(4)}}$, $w={w}$")
    plt.axhline(1, ls=":", lw=1)
    finish("supp_age_laws", r"age $/T_0$", "measured / local prediction")

    # Radiation dispersion
    disp = readcsv("direct_radiation_dispersion_spp320.csv")
    kk = np.linspace(0, 6.2, 400); plt.figure(figsize=(6.3, 4.3))
    plt.plot(kk, np.sqrt(1 + kk**2), "--", label=r"$\omega^2=1+k^2$")
    plt.scatter([r["k_peak_abs_2D"] for r in disp], [r["omega_peak_2D"] for r in disp], marker="o", label="measured odd-harmonic peaks")
    for r in disp: plt.annotate(rf"$n={int(r['n'])}$", (r["k_peak_abs_2D"], r["omega_peak_2D"]), xytext=(5, 5), textcoords="offset points")
    finish("fig_radiation", r"$|k|$", r"$\omega$")

    # A1/A2 separation scaling and local A2 test
    rows = readcsv("second_harmonic_distance_b1_w3.csv"); s = np.array([r["actual_kappaR"] for r in rows]); a1 = np.abs([r["A1"] for r in rows]); a2 = np.abs([r["A2"] for r in rows])
    plt.figure(figsize=(6.4, 4.3)); plt.plot(s, a1, "o-", label=r"$|A_1|$"); plt.plot(s, a2, "s-", label=r"$|A_2|$")
    plt.plot(s, a1[0] * np.exp(-(s - s[0])), "--", label=r"$\propto e^{-\kappa R}$"); plt.plot(s, a2[0] * np.exp(-2 * (s - s[0])), ":", label=r"$\propto e^{-2\kappa R}$")
    finish("supp_A2_scaling", r"$\kappa R$", "phase-harmonic amplitude", yscale="log")
    plt.figure(figsize=(6.4, 4.3)); plt.plot(s, [r["A2_over_local_theory"] for r in rows], "o-"); plt.axhline(1, ls="--", lw=1)
    finish("fig_A2_test", r"$\kappa R$", r"$A_2/A_2^{(4)}$", legend=False)

    # Complete direct quartic overlap
    rows = readcsv("v4_complete_quartic_law_scan.csv"); rr = [r for r in rows if r["actual_kappaR"] <= 8.15]
    plt.figure(figsize=(6.5, 4.3))
    plt.plot([r["actual_kappaR"] for r in rr], [r["c0_over_2A2local"] for r in rr], "o-", label=r"$c_0/(2A_2^{(4)})$")
    plt.plot([r["actual_kappaR"] for r in rr], [r["deltaA1_over_4A2local"] for r in rr], "s-", label=r"$(A_1-A_1^{(2)})/(4A_2^{(4)})$")
    plt.plot([r["actual_kappaR"] for r in rr], [r["A2_over_local"] for r in rr], "^-", label=r"$A_2/A_2^{(4)}$")
    plt.axhline(1, ls="--", lw=1)
    finish("fig_quartic_complete", r"$\kappa R$", "frozen-superposition ratio")

    # Force-energy quadrature circle
    rows = readcsv("m6_phase_force_energy_scan.csv"); summ = readcsv("m6_phase_force_energy_summary.csv"); plt.figure(figsize=(5.1, 5.1))
    for target, marker in [(6.0, "o"), (7.0, "s"), (8.0, "^")]:
        rr = sorted([r for r in rows if r["target_kappaR"] == target], key=lambda r: r["delta"]); sm = [r for r in summ if r["target_kappaR"] == target][0]
        A1 = sm["A1_force_cos1"]; scale = sm["kappa"] / sm["omega"] / A1
        plt.plot([r["Fbar"] / A1 for r in rr], [r["Pi_left"] * scale for r in rr], marker=marker, markevery=8, label=rf"$\kappa R\simeq {target:g}$")
    plt.axhline(0, lw=.7); plt.axvline(0, lw=.7); plt.gca().set_aspect("equal", adjustable="box")
    finish("fig_quadratures", r"$\overline{F}/A_1$", r"$(\kappa/\Omega)\,\overline{\Pi}_L/A_1$")

    # GSL frozen/evolved/full diagnostics
    rows = readcsv("v4_frozen_evolvedsuper_fullpair.csv"); rr = [r for r in rows if abs(r["delta_over_pi"] - .5) < 1e-12]
    plt.figure(figsize=(6.3, 4.2)); plt.plot([r["actual_kappaR"] for r in rr], [r["super_vs_frozen_power"] for r in rr], "o-", label="evolved isolated superposition / frozen")
    plt.plot([r["actual_kappaR"] for r in rr], [r["pair_vs_super_power"] for r in rr], "s-", label="full pair / evolved superposition"); plt.axhline(1, ls="--", lw=1)
    finish("fig_dynamic_diagnosis", r"$\kappa R$", "energy-transfer ratio")

    plt.figure(figsize=(6.4, 4.3))
    for dpi, marker in [(0.0, "o"), (1.0, "s")]:
        rr = [r for r in rows if abs(r["delta_over_pi"] - dpi) < 1e-12]; lab = r"$\delta=0$" if dpi == 0 else r"$\delta=\pi$"
        plt.plot([r["actual_kappaR"] for r in rr], [r["super_vs_frozen_force"] for r in rr], marker=marker, label="evolved/frozen, " + lab)
        plt.plot([r["actual_kappaR"] for r in rr], [r["pair_vs_super_force"] for r in rr], marker=marker, ls="--", label="full/evolved, " + lab)
    plt.axhline(1, ls=":", lw=1)
    finish("supp_dynamic_force_diagnosis", r"$\kappa R$", "force ratio")

    # Exact SG no-ageing dynamics
    rows = readcsv("m8_sg_dynamic_benchmark.csv"); plt.figure(figsize=(6.4, 4.3))
    for dpi, marker in [(0.0, "o"), (1.0, "s")]:
        rr = [r for r in rows if abs(r["delta_over_pi"] - dpi) < 1e-12]; lab = r"$\delta=0$" if dpi == 0 else r"$\delta=\pi$"
        plt.plot([r["kappaR"] for r in rr], [r["super_over_frozen_force"] for r in rr], marker=marker, label="isolated/frozen, " + lab)
        plt.plot([r["kappaR"] for r in rr], [r["pair_over_super_force"] for r in rr], marker=marker, ls="--", label="full/isolated, " + lab)
    plt.axhline(1, ls=":", lw=1)
    finish("supp_m8_dynamic_force", r"$\kappa R$", "force ratio")

    rr = [r for r in rows if abs(r["delta_over_pi"] - .5) < 1e-12]; plt.figure(figsize=(6.2, 4.2))
    plt.plot([r["kappaR"] for r in rr], [r["super_over_frozen_power"] for r in rr], "o-", label="isolated/frozen")
    plt.plot([r["kappaR"] for r in rr], [r["pair_over_super_power"] for r in rr], "s-", label="full/isolated"); plt.axhline(1, ls="--", lw=1)
    finish("supp_m8_dynamic_power", r"$\kappa R$", "energy-transfer ratio")

    # Published instantaneous Manton comparison
    om = .9; kap = np.sqrt(1 - om**2); th = np.linspace(0, 2 * np.pi, 1200, endpoint=False); plt.figure(figsize=(6.4, 4.3))
    for ss in (7.0, 8.0):
        S = 1 / np.cosh(ss / 2); z = kap / om * np.sin(th) * S; den = 1 + z * z; u = 4 * np.arctan(z); v = 4 * kap * np.cos(th) * S / den
        tanh = np.tanh(ss / 2); xL = -4 * kap * kap / om * np.sin(th) * S * tanh / den; xR = -xL
        F = (1 - np.cos(2 * u)) - 2 * (1 - np.cos(u)) - v * v - xL * xR; pref = 64 * kap**4 / om**2 * np.exp(-ss); FM = pref * (1 - np.cos(2 * th) / kap**2)
        plt.plot(th / (2 * np.pi), F, label=rf"exact tails, $\kappa R={ss:g}$"); plt.plot(th / (2 * np.pi), FM, "--", label=rf"Manton, $\kappa R={ss:g}$")
    finish("supp_m8_manton_instantaneous", r"breather phase $/(2\pi)$", r"$F_L(t)$")

    # Momentum balance
    d = np.load(DATA / "m4_momentum_flux_validation_timeseries.npz"); plt.figure(figsize=(6.5, 4.3))
    for key, lab in [("k6_d0", r"$\kappa R\simeq6$"), ("k7_d0", r"$\kappa R\simeq7$")]:
        t = d["t_" + key]; T = t[-1]; plt.plot(t / T, d["dP_" + key], label=r"$\Delta P_L$: " + lab); plt.plot(t / T, d["I_" + key], "--", label=r"$\int F_L\,dt$: " + lab)
    finish("fig_momentum_flux", r"$t/T$", "interaction momentum / impulse")

    # Energy balance
    d = np.load(DATA / "m6_dynamic_energy_transfer_timeseries.npz"); plt.figure(figsize=(6.5, 4.3))
    for key, lab in [("k6_d1", r"$\kappa R\simeq6$"), ("k7_d1", r"$\kappa R\simeq7$")]:
        t = d["t_" + key]; T = t[-1]; plt.plot(t / T, d["dE_" + key], label=r"$\Delta E_L$: " + lab); plt.plot(t / T, d["I_" + key], "--", label=r"$\int \Pi_L\,dt$: " + lab)
    finish("supp_energy_flux", r"$t/T$", "interaction energy / flux impulse")

    # w=8 controls
    rows = readcsv("v4_w8_filtered_restart.csv"); plt.figure(figsize=(6.3, 4.2))
    for lab, disp, marker in [("unfiltered", "unfiltered", "o"), ("farfield_cleaned", "far-field cleaned", "s")]:
        rr = sorted([r for r in rows if r["restart"] == lab], key=lambda r: r["restart_age_T0"]); plt.plot([r["restart_age_T0"] for r in rr], [r["lambda_mismatch_over_kappa"] for r in rr], marker=marker, label=disp)
    finish("fig_w8_filtered_restart", r"time after restart $/T_0$", r"$|\Lambda-\kappa_{\rm core}|/\kappa_{\rm core}$")

    rows = readcsv("v4_w8_seed_memory_summary.csv"); plt.figure(figsize=(6.3, 4.2))
    for seed, disp, marker in [("SG", "SG-type seed", "o"), ("GSL_e3", r"GSL $O(\epsilon^3)$ seed", "s")]:
        rr = sorted([r for r in rows if r["seed"] == seed], key=lambda r: r["age_T0"]); plt.plot([r["age_T0"] for r in rr], [r["lambda_mismatch_over_kappa"] for r in rr], marker=marker, label=disp)
    finish("fig_w8_seed_history", r"age $/T_0$", r"$|\Lambda-\kappa_{\rm core}|/\kappa_{\rm core}$")

    # Frozen versus dynamic Pi
    fd = readcsv("m6_frozen_vs_dynamic_energy_transfer.csv"); plt.figure(figsize=(6.3, 4.2))
    plt.semilogy([r["actual_kappaR"] for r in fd], [r["frozen_Pi_pi2"] for r in fd], "o-", label="frozen template")
    plt.semilogy([r["actual_kappaR"] for r in fd], [r["dynamic_mean_Pi_pi2"] for r in fd], "s-", label="full PDE, one-period mean")
    finish("fig_frozen_dynamic_energy", r"$\kappa R$", r"$\overline{\Pi}_L(\pi/2)$")

    # ETDRK4 convergence
    rows = readcsv("benchmark_etdrk4_sg_results_extended.csv"); dt = np.array([r["dt"] for r in rows]); err = np.array([r["rel_u_l2"] for r in rows])
    plt.figure(figsize=(6.2, 4.2)); plt.loglog(dt, err, "o-", label=r"relative $L^2$ field error"); plt.loglog(dt, err[-1] * (dt / dt[-1])**4, "--", label=r"$\propto\Delta t^4$")
    finish("supp_etdrk4", r"$\Delta t$", "relative error")

    # Near-threshold radiation fraction
    rows = readcsv("v4_w8_nearthreshold_komega.csv"); plt.figure(figsize=(5.6, 4.1))
    labels = ["SG-type seed" if r["seed"] == "SG" else r"GSL $O(\epsilon^3)$ seed" for r in rows]
    plt.bar(range(len(rows)), [r["branch_power_fraction_of_lowband"] for r in rows]); plt.xticks(range(len(rows)), labels, rotation=10)
    finish("fig_w8_nearthreshold", "", "KG-branch / low-band power", legend=False)

    # k-omega density map (raster)
    d = np.load(DATA / "direct_radiation_data_spp320.npz"); Y = d["far_u"].copy(); Y -= Y.mean(axis=0, keepdims=True); Y *= np.hanning(Y.shape[0])[:, None]; Y *= np.hanning(Y.shape[1])[None, :]
    F = np.fft.fftshift(np.fft.fft2(Y)); P = np.abs(F)**2
    omgrid = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Y.shape[0], d=float(d["sample_dt"]))); kgrid = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Y.shape[1], d=float(d["dx"])))
    so = (omgrid > .8) & (omgrid < 3.1); sk = (kgrid > -3.1) & (kgrid < 3.1); Z = np.log10(P[np.ix_(so, sk)] / P.max() + 1e-18)
    plt.figure(figsize=(6.3, 4.3)); plt.imshow(Z, origin="lower", aspect="auto", extent=[kgrid[sk][0], kgrid[sk][-1], omgrid[so][0], omgrid[so][-1]], rasterized=True)
    kv = np.linspace(-3, 3, 400); plt.plot(kv, np.sqrt(1 + kv**2), "--", lw=1); Omega = float(d["omega_qb"]); plt.axhline(2 * Omega, ls=":", lw=1); plt.axhline(3 * Omega, ls=":", lw=1)
    plt.xlabel(r"$k$"); plt.ylabel(r"$\omega$"); plt.tight_layout(); plt.savefig(OUT / "fig_evenharm_komega.png", dpi=300); plt.close()

    # SG deformation frequency scan
    rows = readcsv("v8_sg_deformation_vs_omega.csv"); plt.figure(figsize=(6.2, 4.2))
    plt.plot([r["omega"] for r in rows], [r["def_A2_over_A2direct"] for r in rows], "o-"); plt.axhline(-4, ls="--", lw=1, label=r"$-4$")
    finish("supp_sg_deformation_vs_omega", r"$\omega$", r"$A_{2,\mathrm{def}}/A_2^{(4)}$")

    print(f"Wrote publication figures to {OUT}")

if __name__ == "__main__":
    main()
