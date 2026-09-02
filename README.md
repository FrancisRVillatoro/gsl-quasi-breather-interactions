# Local complex-tail interactions of graphene-superlattice quasi-breathers

Reproducibility repository for the manuscript

**Local complex-tail interactions of graphene-superlattice quasi-breathers:
force, energy transfer, and finite-age nonadiabatic tails**

by Francisca Martin-Vergara, Francisco Rus, and Francisco R. Villatoro.

Repository: https://github.com/FrancisRVillatoro/gsl-quasi-breather-interactions

## Contents

- `code/` — SSF4/ETDRK4 solvers and principal analysis scripts.
- `data/` — processed CSV/NPZ/PNG outputs supporting manuscript results.
- `milestones/` — archived M1–M8 analysis packages.
- `MANIFEST.tsv` — file manifest.

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

The exact sine-Gordon benchmark and the M8 integrable closure tests are
included in the repository.

## Reproduction map

The repository contains the processed numerical products used to support:

- leading local complex-tail force and energy-transfer laws;
- complete direct quartic `2:4:1` overlap structure;
- third-harmonic continuum/radiation diagnostics;
- SSF4/ETDRK4 cross-validation;
- momentum- and energy-flux conservation-law checks;
- parameter and age dependence;
- finite-age seed/history dependence;
- exact sine-Gordon/Manton closure benchmark;
- dynamic full-pair versus evolved-isolated-superposition controls.

See `MANIFEST.tsv` and the source manuscript for the precise mapping of
figures and numerical statements.

## Python environment

The analysis uses Python 3 with:

```text
numpy
scipy
matplotlib
```

Install with:

```bash
python -m pip install -r requirements.txt
```

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22257369.svg)](https://doi.org/10.5281/zenodo.22257369)

Please cite the associated manuscript and the archived reproducibility release:

- Version `v1.0.0` DOI: https://doi.org/10.5281/zenodo.22257369
- Concept DOI (all versions): https://doi.org/10.5281/zenodo.22257368
- GitHub repository: https://github.com/FrancisRVillatoro/gsl-quasi-breather-interactions

For exact reproducibility of the submitted manuscript, cite the version-specific
DOI `10.5281/zenodo.22257369`.

## Licensing

- Source code under `code/`: MIT License (`LICENSE`).
- Data, figures, and processed numerical outputs under `data/` and
  `milestones/`: Creative Commons Attribution 4.0 International
  (`LICENSE-DATA`).

Unless a file explicitly states otherwise, these are the applicable licenses.

## Authors

- Francisca Martin-Vergara
- Francisco Rus
- Francisco R. Villatoro
