# Forces and energy transfer between graphene-superlattice quasi-breathers

Reproducibility package for the Wave Motion manuscript by Francisca Martin-Vergara,
Francisco Rus, and Francisco R. Villatoro.

## Release status

The revised manuscript is supported by [v1.0.2](https://github.com/FrancisRVillatoro/gsl-quasi-breather-interactions/releases/tag/v1.0.2).
See [the release notes](RELEASE_NOTES_v1.0.2.md) for the changes relative to v1.0.1.
The [Zenodo record for all versions](https://doi.org/10.5281/zenodo.22257368)
lists the version-specific records.

## Contents

- `code/` — SSF4/ETDRK4 solvers and analysis scripts.
- `data/` — processed CSV/NPZ products supporting manuscript results.
- `publication_figures/` — publication-quality figure exports used in the
  revised manuscript and supplementary material.
- `milestones/` — archived M1–M8 analysis packages.
- `MANIFEST.tsv` — file-size manifest.
- `CHANGELOG.md` — release history.

## Principal production state

For the main `b=1, w=3` interaction calculations:

- domain length: `L = 600`
- Fourier modes: `N = 4096`
- `dx = 0.146484375`
- seed period: `T0 = 2*pi/(3/sqrt(10))`
- production step: `dt = T0/320 = 0.0206970589`
- frozen-template fitting window: `[30 T0, 40 T0]`
- reference age: `35 T0`
- `Omega* = 0.8463557897`
- `kappa* = 0.5326179468`

## Main numerical methods

`code/ssf4_gsl.py` implements the fourth-order symmetric split-step Fourier
method used for production calculations.

`code/etdrk4_gsl.py` implements the independent Cox–Matthews ETDRK4
cross-check.

The exact sine-Gordon/Manton closure tests are implemented in
`code/m8_sg_integrable_benchmark.py` and the deformation controls in
`code/v8_*.py`.

## Sign convention for two-body deformation

The phase-resolved deformation tables use the manuscript convention

```text
F_L = [sigma_pair - sigma_1 - sigma_2] at x_m
Delta F_def = F_L,full - F_L,sup
```

where `sup` means the two isolated constituents evolved independently and
then superposed, and `full` means the fully interacting pair.

This explicit convention supersedes the ambiguous/right-object sign used in
intermediate diagnostic files before v1.0.1.

## Reproduction map

The repository supports:

- local complex-tail force and energy-transfer laws;
- complete **direct** quartic `2:4:1` overlap structure;
- phase-resolved two-body deformation at the same exponential overlap order;
- third-harmonic continuum/radiation diagnostics;
- SSF4/ETDRK4 cross-validation;
- momentum- and energy-flux conservation-law checks;
- parameter and age dependence;
- finite-age seed/history dependence;
- exact sine-Gordon/Manton closure benchmark;
- frozen-template tail-decomposition controls.

### Publication figure → analysis/script map

The publication exports are in `publication_figures/`. Run
`code/make_publication_figures_v8.py` to regenerate them. For the revised article,
this script calls `code/revised_wave_motion_figures.py` for Figures 2, 6 and 8 and
copies the supplied final artwork for Figure 3.

| Publication figure file | Main analysis source |
|---|---|
| `fig_isolated_tail.pdf` (Fig. 2) | `code/analyze_isolated_qb.py`; `data/isolated_qb_data_spp160.npz` and `data/isolated_qb_summary_spp160.csv` |
| `fig_radiation.pdf` (Fig. 3) | Final artwork in `data/artwork/fig_radiation.pdf`; radiation diagnostics in `direct_radiation_b1_w3.py` and `data/direct_radiation_*` |
| `fig_m8_sg_integrable.pdf` | `m8_sg_integrable_benchmark.py` |
| `fig_phase_force.pdf` | `m6_interaction_quadratures_b1_w3.py` |
| `fig_A2_test.pdf` | `second_harmonic_distance_b1_w3.py` |
| `fig_quartic_complete.pdf` | `v4_complete_quartic_scan.py` |
| `fig_even_harmonics.pdf` | all-harmonic frozen-template products |
| `fig_integrator_crosscheck.pdf` | `etdrk4_gsl.py`; ETDRK4 benchmark products |
| `fig_momentum_flux.pdf` (Fig. 6) | Right-half balance from `data/m4_momentum_flux_validation_timeseries.npz` |
| `fig_dynamic_diagnosis.pdf` | `v4_dynamic_diagnosis.py` |
| `fig_quadratures.pdf` (Fig. 8) | `data/m6_phase_force_energy_scan.csv` and `data/m6_phase_force_energy_summary.csv`; measured first-harmonic coefficients for each separation |
| `fig_age.pdf` | `m5_phase_spline_age.py` |
| `fig_w8_seed_history.pdf` | `m5_phase_spline_age.py`; seed-history controls |
| `fig_w8_nearthreshold.pdf` | `direct_radiation_b1_w3.py`; low-band spectral controls |
| `fig_w8_filtered_restart.pdf` | filtered-restart control data |
| `fig_evenharm_komega.png` | `direct_radiation_b1_w3.py`; raw `direct_radiation_data_spp320.npz` |
| `fig_frozen_dynamic_energy.pdf` | `m6_interaction_quadratures_b1_w3.py`; dynamic energy-transfer products |
| `supp_A2_scaling.pdf` | `second_harmonic_distance_b1_w3.py` |
| `supp_A3.pdf` | `third_harmonic_order6_b1_w3.py` |
| `supp_age_laws.pdf` | `m5_phase_spline_age.py` |
| `supp_dynamic_force_diagnosis.pdf` | `v4_dynamic_diagnosis.py` |
| `supp_energy_flux.pdf` | `m6_interaction_quadratures_b1_w3.py` |
| `supp_etdrk4.pdf` | `benchmark_etdrk4_sg.py` |
| `supp_m8_dynamic_force.pdf` | `m8_sg_integrable_benchmark.py` |
| `supp_m8_dynamic_power.pdf` | `m8_sg_integrable_benchmark.py` |
| `supp_m8_manton_instantaneous.pdf` | `m8_sg_integrable_benchmark.py` |
| `supp_m8_quartic.pdf` | `m8_sg_integrable_benchmark.py` |
| `supp_sg_deformation_vs_omega.pdf` | `v8_sg_deformation_frequency_scan.py` |

Run the publication renderer from the repository root with:

```bash
python code/make_publication_figures_v8.py
```

## Reproducing the revised checks

From the repository root, run the checks after installing the Python dependencies.

```bash
python3 -B code/validate_revision_diagnostics.py
```

The validator writes its intermediate files in a temporary directory. It
recomputes the 24-point fit in Figure 2 and the three radial RMS deviations
in Figure 8, then runs both corrected drivers and compares their outputs with
the archived SG and GSL coefficients, the 16-phase GSL curve and all three
tail-drift rows.

To repeat the isolated simulation underlying Figure 2, use
`python3 code/analyze_isolated_qb.py --spp 160 --output-dir /tmp/gsl-isolated-160`.
The fit uses the isolated-state observation window from 50 to 66 seed periods.
The pair diagnostics instead use the 320-step production fit over seed periods
30–40. Both drivers below read `chirp_template_spp320.npz` directly from
`milestones/gsl_chirp_aware_phase_experiment_b1_w3.zip`. It contains the
mean and cosine/sine pairs for harmonics 1, 3, 5 and 7. The separate
`v3_allharmonic_template_spp320.npz` remains available for the even-harmonic
control. Use `--output-dir DIR` on either corrected driver to save new tables
in a separate directory.

## Diagnostic commands

### Phase-resolved two-body deformation

```bash
python code/v8_phase_resolved_deformation.py
```

writes sign-consistent SG and GSL first-period deformation tables.

### Sine-Gordon frequency dependence

```bash
python code/v8_sg_deformation_frequency_scan.py
```

writes `data/v8_sg_deformation_vs_omega.csv`.

### Multi-period stationarity check

```bash
python code/v8_sg_deformation_multiperiod.py
```

shows that the deformation coefficients evolve over periods 1–3.

### Frozen-template tail decomposition

```bash
python code/v8_template_tail_decomposition.py
```

writes the decomposition of the frozen/evolved discrepancy into `|U|^2`,
`Lambda`, `Gamma`, and the resulting local `A1^(2)` ratio.

## Python environment

The analysis uses Python 3 with:

```text
numpy>=2.0
scipy
matplotlib
```

Install with:

```bash
python -m pip install -r requirements.txt
```

## Citation

[![DOI for all versions](https://zenodo.org/badge/DOI/10.5281/zenodo.22257368.svg)](https://doi.org/10.5281/zenodo.22257368)

To cite the revised calculations, select **v1.0.2** in the Zenodo version
list and use the DOI shown on that individual record. The badge above
identifies the full version series. The earlier records are
[v1.0.1](https://doi.org/10.5281/zenodo.22274849) and
[v1.0.0](https://doi.org/10.5281/zenodo.22257369).

## Licensing

- Source code under `code/`: MIT License (`LICENSE`).
- Data, figures, and processed numerical outputs under `data/`,
  `publication_figures/`, and `milestones/`: Creative Commons Attribution
  4.0 International (`LICENSE-DATA`).

## Authors

- Francisca Martin-Vergara
- Francisco Rus
- Francisco R. Villatoro
