#!/usr/bin/env python3
"""Recompute the Wave Motion diagnostics in a temporary directory (requires NumPy >= 2)."""
import csv
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


CODE = Path(__file__).resolve().parent
DATA = CODE.parent / "data"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def compare_columns(reference, actual, columns, expected_count):
    reference_rows = read_rows(reference)
    actual_rows = read_rows(actual)
    if len(reference_rows) != expected_count or len(actual_rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows in {reference.name} and {actual.name}.")
    for reference_column, actual_column in columns:
        expected = [float(row[reference_column]) for row in reference_rows]
        observed = [float(row[actual_column]) for row in actual_rows]
        np.testing.assert_allclose(observed, expected, rtol=2e-8, atol=5e-13,
                                   equal_nan=False, err_msg=f"{actual.name}: {actual_column}")


def check_phase_and_drift(outdir):
    for filename in ("v8_phase_resolved_deformation.py", "v8_template_tail_decomposition.py"):
        subprocess.run([sys.executable, "-B", str(CODE / filename), "--output-dir", str(outdir)],
                       stdout=subprocess.DEVNULL, check=True)

    fourier = [("kappaR", "kappaR")]
    for reference, computed in (("super", "sup"), ("pair", "full"), ("deform", "def")):
        coefficients = ["c0"] + [f"{part}{i}" for i in range(1, 5) for part in ("A", "S")]
        fourier.extend((f"{reference}_{c}", f"{computed}_{c}") for c in coefficients)
    fourier.extend((("deform_c0_over_2A2", "def_c0_over_2A2"),
                    ("deform_A1_over_4A2", "def_A1_over_4A2")))
    for case in ("SG", "GSL"):
        compare_columns(DATA / f"final_dynamic_phase_fourier_{case}.csv",
                        outdir / f"v8_phase_fourier_{case}.csv", fourier, 1)
    compare_columns(DATA / "final_dynamic_phase_curve_GSL.csv", outdir / "v8_phase_curve_GSL.csv",
                    [("delta", "delta"), ("delta_over_pi", "delta_over_pi"),
                     ("F_evolved_superposition", "F_sup"), ("F_full_pair", "F_full"),
                     ("DeltaF_deformation", "DeltaF_def")], 16)
    drift = "v8_template_tail_decomposition.csv"
    columns = [(k, k) for k in read_rows(DATA / drift)[0]]
    compare_columns(DATA / drift, outdir / drift, columns, 3)
    print("PASS  SG and GSL phase coefficients, 16-phase GSL curve, three tail-drift rows")


def check_tail_fit():
    reference = {r["quantity"]: float(r["value"])
                 for r in read_rows(DATA / "isolated_qb_summary_spp160.csv")}
    with np.load(DATA / "isolated_qb_data_spp160.npz", allow_pickle=False) as saved:
        x, amplitude = saved["x"], np.abs(saved["U1"])
    peak = amplitude.max()
    mask = (x > 0) & (amplitude > 0.05 * peak) & (amplitude < 0.4 * peak)
    xx, yy = x[mask], np.log(amplitude[mask])
    if len(xx) != 24:
        raise ValueError(f"Figure 2 fit contains {len(xx)} points, expected 24.")
    slope, intercept = np.polyfit(xx, yy, 1)
    residual = yy - (slope * xx + intercept)
    r2 = 1 - np.sum(residual**2) / np.sum((yy - yy.mean())**2)
    actual = [xx.min(), xx.max(), -slope, np.exp(intercept), r2, np.sqrt(1-reference["omega_fit"]**2)]
    expected = [reference[k] for k in ("tail_rmin", "tail_rmax", "kappa_fit", "C1_fit", "tail_R2", "kappa_pred")]
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-13, equal_nan=False)
    print(f"PASS  Figure 2: 24 points, kappa_fit={-slope:.10f}, R^2={r2:.9f}")


def check_radial_rms():
    scan = np.genfromtxt(DATA / "m6_phase_force_energy_scan.csv", delimiter=",", names=True)
    summary = np.genfromtxt(DATA / "m6_phase_force_energy_summary.csv", delimiter=",", names=True)
    summary = np.sort(summary, order="target_kappaR")
    np.testing.assert_array_equal(summary["target_kappaR"], [6, 7, 8])
    values = []
    for r in summary:
        phases = np.sort(scan[scan["target_kappaR"] == r["target_kappaR"]], order="delta")
        np.testing.assert_allclose(phases["delta"], np.arange(64) * (2*np.pi/64), rtol=0, atol=2e-12)
        x = phases["Fbar"] / r["A1_force_cos1"]
        y = phases["Pi_left"] / r["B1_power_sin1"]
        values.append(100 * np.sqrt(np.mean((np.hypot(x, y) - 1)**2)))
    np.testing.assert_allclose(values, [2.25323106, 0.77765052, 0.36978282], rtol=0, atol=1e-7)
    print("PASS  Figure 8: radial RMS " + ", ".join(f"{v:.5f}%" for v in values))


def main():
    if np.lib.NumpyVersion(np.__version__) < "2.0.0":
        raise SystemExit("The archived diagnostics require NumPy >= 2.0 (numpy.trapezoid).")
    check_tail_fit()
    check_radial_rms()
    with TemporaryDirectory(prefix="gsl-wamot-") as temporary:
        check_phase_and_drift(Path(temporary))


if __name__ == "__main__":
    main()
