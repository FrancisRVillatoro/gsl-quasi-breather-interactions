"""
benchmark_sg.py
===============

First validation benchmark for ssf4_gsl.py.

Tests the fourth-order split-step Fourier solver against the exact
sine-Gordon breather (b=0).

Outputs
-------
benchmark_sg_results.csv
benchmark_spatial_check.csv
benchmark_convergence.png
benchmark_energy_longrun.png
"""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ssf4_gsl import (
    grid,
    ssf4_step,
    energy,
    momentum,
    sine_gordon_breather,
    sine_gordon_breather_energy,
)


HERE = Path(__file__).resolve().parent


def l2_norm(z, dx):
    return np.sqrt(dx * np.sum(z * z))


def run_case(
    steps_per_period,
    *,
    w=2.0,
    L=120.0,
    N=2048,
    nperiods=10.25,
    diagnostic_samples_per_period=10,
    collect_history=False,
):
    x, dx, k, omega_k = grid(N, L)

    q = 1.0 / np.sqrt(1.0 + w * w)
    omega = w / np.sqrt(1.0 + w * w)
    T = 2.0 * np.pi / omega
    dt = T / steps_per_period

    nsteps = int(round(nperiods * steps_per_period))
    tfinal = nsteps * dt

    u, v = sine_gordon_breather(x, 0.0, w)
    H0 = energy(u, v, dx, k, b=0.0)

    sample_stride = max(
        1, steps_per_period // diagnostic_samples_per_period
    )

    max_rel_dH = 0.0
    max_abs_P = 0.0

    times = []
    rel_energy_errors = []

    for n in range(nsteps):
        u, v = ssf4_step(u, v, dt, b=0.0, omega_k=omega_k)

        if (n + 1) % sample_stride == 0 or n + 1 == nsteps:
            H = energy(u, v, dx, k, b=0.0)
            P = momentum(u, v, dx, k)

            rel_dH = abs(H - H0) / abs(H0)
            max_rel_dH = max(max_rel_dH, rel_dH)
            max_abs_P = max(max_abs_P, abs(P))

            if collect_history:
                times.append((n + 1) * dt / T)  # in periods
                rel_energy_errors.append((H - H0) / H0)

    ue, ve = sine_gordon_breather(x, tfinal, w)

    err_u_inf = float(np.max(np.abs(u - ue)))
    rel_u_l2 = float(l2_norm(u - ue, dx) / l2_norm(ue, dx))

    phase_err = np.sqrt(
        l2_norm(u - ue, dx) ** 2 + l2_norm(v - ve, dx) ** 2
    )
    phase_ref = np.sqrt(
        l2_norm(ue, dx) ** 2 + l2_norm(ve, dx) ** 2
    )
    rel_phase_l2 = float(phase_err / phase_ref)

    result = {
        "steps_per_period": int(steps_per_period),
        "dt": float(dt),
        "N": int(N),
        "L": float(L),
        "periods": float(nperiods),
        "tfinal": float(tfinal),
        "err_u_inf": err_u_inf,
        "rel_u_l2": rel_u_l2,
        "rel_phase_l2": rel_phase_l2,
        "max_rel_energy_error": float(max_rel_dH),
        "max_abs_momentum": float(max_abs_P),
        "H0_numeric": float(H0),
        "H_exact": float(sine_gordon_breather_energy(w)),
    }

    if collect_history:
        return result, np.asarray(times), np.asarray(rel_energy_errors)
    return result


def add_observed_orders(rows, key):
    rows[0]["p_" + key] = ""
    for j in range(1, len(rows)):
        e0 = rows[j - 1][key]
        e1 = rows[j][key]
        rows[j]["p_" + key] = float(np.log(e0 / e1) / np.log(2.0))


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    # ---------------------------------------------------------------
    # Benchmark A: fourth-order temporal convergence.
    # ---------------------------------------------------------------
    temporal_rows = [
        run_case(spp)
        for spp in (20, 40, 80, 160)
    ]

    for key in (
        "err_u_inf",
        "rel_u_l2",
        "rel_phase_l2",
        "max_rel_energy_error",
    ):
        add_observed_orders(temporal_rows, key)

    write_csv(HERE / "benchmark_sg_results.csv", temporal_rows)

    # Convergence plot: relative L2 error in u and O(dt^4) reference.
    dt = np.asarray([r["dt"] for r in temporal_rows])
    err = np.asarray([r["rel_u_l2"] for r in temporal_rows])

    reference = err[-1] * (dt / dt[-1]) ** 4

    plt.figure(figsize=(7.0, 5.0))
    plt.loglog(dt, err, "o-", label=r"$\|u-u_{\rm ex}\|_2/\|u_{\rm ex}\|_2$")
    plt.loglog(dt, reference, "--", label=r"$O(\Delta t^4)$")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("error")
    plt.title("SSF4 temporal convergence: sine-Gordon breather")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HERE / "benchmark_convergence.png", dpi=180)
    plt.close()

    # ---------------------------------------------------------------
    # Benchmark B: simple spatial-resolution check.
    # Temporal error is deliberately held fixed at 160 steps/period.
    # ---------------------------------------------------------------
    spatial_rows = []
    for N in (512, 1024, 2048, 4096):
        spatial_rows.append(
            run_case(
                160,
                w=2.0,
                L=120.0,
                N=N,
                nperiods=10.25,
            )
        )
    write_csv(HERE / "benchmark_spatial_check.csv", spatial_rows)

    # ---------------------------------------------------------------
    # Benchmark C: 100.25-period long-time diagnostic.
    # ---------------------------------------------------------------
    long_runs = []
    for spp in (80, 160):
        result, periods, rel_dH = run_case(
            spp,
            w=2.0,
            L=120.0,
            N=2048,
            nperiods=100.25,
            diagnostic_samples_per_period=10,
            collect_history=True,
        )

        # Ten diagnostic samples are stored per period.  Use the maximum
        # |dH/H| over each full period as a clean long-time envelope.
        samples_per_period = 10
        ngroups = len(rel_dH) // samples_per_period
        env = np.asarray([
            np.max(np.abs(
                rel_dH[j * samples_per_period:(j + 1) * samples_per_period]
            ))
            for j in range(ngroups)
        ])
        env_t = np.arange(1, ngroups + 1, dtype=float)
        drift_slope = float(np.polyfit(env_t, env, 1)[0])
        long_runs.append((spp, result, env_t, env, drift_slope))

    plt.figure(figsize=(7.0, 5.0))
    for spp, result, env_t, env, drift_slope in long_runs:
        plt.semilogy(
            env_t,
            env,
            label=f"{spp} steps/period",
        )
    plt.xlabel("breather period")
    plt.ylabel(r"max per period $|H(t)-H(0)|/H(0)$")
    plt.title("SSF4 long-time energy-error envelope")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(HERE / "benchmark_energy_longrun.png", dpi=180)
    plt.close()

    # Human-readable summary.
    print("Temporal convergence benchmark")
    print(
        "spp        dt          relL2(u)       p       "
        "phase-L2        p       max rel dH      p        max |P|"
    )
    for r in temporal_rows:
        pu = r["p_rel_u_l2"]
        pp = r["p_rel_phase_l2"]
        pe = r["p_max_rel_energy_error"]

        pu_s = "   -" if pu == "" else f"{pu:6.3f}"
        pp_s = "   -" if pp == "" else f"{pp:6.3f}"
        pe_s = "   -" if pe == "" else f"{pe:6.3f}"

        print(
            f'{r["steps_per_period"]:3d} '
            f'{r["dt"]:12.6e} '
            f'{r["rel_u_l2"]:12.6e} {pu_s} '
            f'{r["rel_phase_l2"]:12.6e} {pp_s} '
            f'{r["max_rel_energy_error"]:12.6e} {pe_s} '
            f'{r["max_abs_momentum"]:12.6e}'
        )

    print()
    print("Energy check")
    print(f'  H_numeric(t=0) = {temporal_rows[0]["H0_numeric"]:.16e}')
    print(f'  H_exact         = {temporal_rows[0]["H_exact"]:.16e}')
    print(
        "  relative difference = "
        f'{abs(temporal_rows[0]["H0_numeric"] - temporal_rows[0]["H_exact"]) / temporal_rows[0]["H_exact"]:.3e}'
    )

    print()
    print("Spatial-resolution check at 160 steps/period")
    print("N          relL2(u)          max rel dH")
    for r in spatial_rows:
        print(
            f'{r["N"]:4d}   '
            f'{r["rel_u_l2"]:14.7e}   '
            f'{r["max_rel_energy_error"]:14.7e}'
        )

    print()
    print("Long run: 100.25 periods")
    print(
        "spp      relL2(u)       phase-space relL2   "
        "max rel dH      max |P|        envelope slope/period"
    )
    for spp, result, env_t, env, drift_slope in long_runs:
        print(
            f"{spp:3d}   "
            f'{result["rel_u_l2"]:12.6e}   '
            f'{result["rel_phase_l2"]:12.6e}   '
            f'{result["max_rel_energy_error"]:12.6e}   '
            f'{result["max_abs_momentum"]:12.6e}   '
            f"{drift_slope:12.6e}"
        )

    long_order = np.log(
        long_runs[0][1]["rel_phase_l2"] /
        long_runs[1][1]["rel_phase_l2"]
    ) / np.log(2.0)
    print(f"  observed long-time phase-space order = {long_order:.6f}")


if __name__ == "__main__":
    main()
