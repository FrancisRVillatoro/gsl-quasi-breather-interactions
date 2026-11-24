"""Revised Wave Motion figures from the isolated-state and interaction diagnostics."""
import csv

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


@plt.rc_context()
def plot_isolated_tail(data_dir, output):
    """Fit the 160-step-per-period isolated-state tail on the positive half of its grid."""
    with np.load(data_dir / "isolated_qb_data_spp160.npz") as archived:
        positive_half = archived["x"] >= 0
        x = archived["x"][positive_half]
        fundamental = archived["U1"][positive_half]
        amplitude = np.hypot(fundamental.real, fundamental.imag)
    with (data_dir / "isolated_qb_summary_spp160.csv").open(newline="") as handle:
        omega = next(float(row["value"]) for row in csv.DictReader(handle)
                     if row["quantity"] == "omega_fit")
    kappa_core = np.sqrt(1.0 - omega**2)

    peak = amplitude.max()
    selected = (x > 0) & (amplitude > 0.05 * peak) & (amplitude < 0.40 * peak)
    x_fit = x[selected]
    y_log = np.log(amplitude[selected])
    slope, intercept = np.polyfit(x_fit, y_log, 1)
    kappa_fit = -slope
    residuals = y_log - (intercept + slope * x_fit)
    r_squared = 1.0 - (residuals @ residuals) / np.sum((y_log - y_log.mean())**2)
    print(f"n = {len(x_fit)}, interval = [{x_fit[0]:g}, {x_fit[-1]:g}]")
    print(f"C1 = {np.exp(intercept):.10f}, kappa_fit = {kappa_fit:.10f}, "
          f"kappa_core = {kappa_core:.10f}, R^2 = {r_squared:.9f}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (full, detail) = plt.subplots(2, 1, figsize=(6.2, 4.9),
                                      gridspec_kw={"height_ratios": (1.3, 1)},
                                      constrained_layout=True)
    ink = "#24475F"
    accent = "#A54618"
    shown = x <= 20
    full.axvspan(x_fit[0], x_fit[-1], color="#E8EEF2")
    full.semilogy(x[shown], amplitude[shown], color=ink, linewidth=1.6,
                  label=r"Isolated $|U_1|$")
    full.semilogy(x_fit, np.exp(intercept - kappa_fit * x_fit),
                  linestyle="--", color=accent, linewidth=2.1,
                  label=r"Exponential fit")
    full.set(xlim=(0, 20), ylim=(5e-4, 3.0), ylabel=r"$|U_1(x)|$", xlabel=r"$x$")
    full.legend(loc="upper right", frameon=False, fontsize=9)
    full.text(0.012, 0.05, "(a)", transform=full.transAxes)

    detail.axhline(0.0, color="0.60", linewidth=0.7)
    detail.plot(x_fit, 100.0 * np.expm1(residuals), "o", color=ink, markersize=3.2,
                label="Measured profile")
    detail.plot(x_fit, 100.0 * np.expm1((kappa_fit - kappa_core) * x_fit),
                "--", color=accent, linewidth=1.7, label="Core slope, same intercept")
    detail.set(xlim=(x_fit[0]-0.08, x_fit[-1]+0.08),
               ylabel="Deviation from fit (%)", xlabel=r"$x$")
    detail.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    detail.legend(loc="lower center", frameon=False, fontsize=9)
    detail.text(0.012, 0.85, "(b)", transform=detail.transAxes)
    fig.savefig(output)
    plt.close(fig)


def plot_momentum_balance(data_dir, output):
    """Compare measured right-half momentum changes with the interaction impulse."""
    figure, ax = plt.subplots(figsize=(6.5, 4.3))
    with np.load(data_dir / "m4_momentum_flux_validation_timeseries.npz") as series:
        for target, color, marker in [(6, "#1f77b4", "o"), (7, "#d4550b", "s")]:
            key = f"k{target}_d0"
            t = series[f"t_{key}"]
            suffix = rf", $\kappa R\simeq{target}$"
            ax.plot(t / t[-1], series[f"dP_{key}"], color=color,
                    label=r"$\Delta P_R^{\rm int}$" + suffix)
            ax.plot(t / t[-1], series[f"I_{key}"], ls="none", color=color, marker=marker,
                    markevery=10, ms=4, markerfacecolor="white",
                    label=r"$\int F_R^{\rm int}\,dt$" + suffix, linewidth=1.9)
    ax.set(xlabel=r"$t/T$", ylabel="Right-half interaction momentum / impulse")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


@plt.rc_context()
def plot_quadratures(data_dir, output):
    """Plot each phase scan and its radial deviation in measured-coefficient units."""
    scan = np.genfromtxt(data_dir / "m6_phase_force_energy_scan.csv",
                         delimiter=",", names=True)
    summary = np.genfromtxt(data_dir / "m6_phase_force_energy_summary.csv",
                            delimiter=",", names=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "mathtext.fontset": "dejavusans", "axes.linewidth": 0.8,
        "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42,
    })
    fig, (phase_ax, radial_ax) = plt.subplots(
        1, 2, figsize=(6.75, 3.34), gridspec_kw={"width_ratios": [1, 1.10]}
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.955, bottom=0.27, wspace=0.41)
    for ax, panel in ((phase_ax, "(a)"), (radial_ax, "(b)")):
        ax.text(0.02, 0.985, panel, transform=ax.transAxes, va="top")
        ax.tick_params(length=3)
    phase_ax.axhline(0, lw=0.5, color="0.82")
    phase_ax.axvline(0, lw=0.5, color="0.82")
    radial_ax.axhline(0, lw=0.6, color="0.55")
    phase_ax.set_aspect("equal", adjustable="box")
    phase_ax.set(xlim=(-1.12, 1.12), ylim=(-1.12, 1.12),
                 xticks=[-1, 0, 1], yticks=[-1, 0, 1],
                 xlabel=r"$\overline{F}/A_1$", ylabel=r"$\overline{\Pi}_L/B_1$")
    radial_ax.set(xlim=(0, 2), ylim=(-4.25, 4.65), xticks=[0, 0.5, 1, 1.5, 2],
                  xticklabels=["0", "0.5", "1", "1.5", "2"], yticks=[-4, -2, 0, 2, 4],
                  xlabel=r"$\delta/\pi$", ylabel=r"$100(r-1)$ (%)")
    colors = ["#0072B2", "#D55E00", "#00865B"]
    markers = ["o", "s", "^"]
    legend_handles = []
    angle = np.linspace(0, 2*np.pi, 361)
    for row, color, marker in zip(summary, colors, markers):
        data = np.sort(scan[scan["target_kappaR"] == row["target_kappaR"]], order="delta")
        x = data["Fbar"] / row["A1_force_cos1"]
        y = data["Pi_left"] / row["B1_power_sin1"]
        residual = 100 * (np.hypot(x, y) - 1)
        print(f"kappa R = {row['actual_kappaR']:.8f}, phases = {len(data)}, "
              f"RMS radial deviation = {np.sqrt(np.mean(residual**2)):.8f}%")
        # Append the first point only to close the displayed periodic curves.
        style = dict(color=color, marker=marker, markersize=3.8, linewidth=1.35,
                     markevery=8, markerfacecolor="white", markeredgewidth=0.85)
        phase_ax.plot(np.append(x, x[0]), np.append(y, y[0]), **style)
        line, = radial_ax.plot(np.append(data["delta_over_pi"], 2),
                               np.append(residual, residual[0]),
                               label=rf"$\kappa R={row['actual_kappaR']:.2f}$", **style)
        legend_handles.append(line)
    guide, = phase_ax.plot(np.cos(angle), np.sin(angle), linestyle=(0, (3.2, 3.8)),
                           lw=0.95, color="0.16", label="Unit circle")
    fig.legend(handles=[guide, *legend_handles], loc="lower center", ncol=4,
               frameon=False, fontsize=9, columnspacing=1.3, handlelength=2.2)
    fig.savefig(output)
    plt.close(fig)
