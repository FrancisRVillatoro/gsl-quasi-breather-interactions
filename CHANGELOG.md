# Changelog

## v1.0.1 — 2026-09-03

Pre-submission reproducibility update.

- Fixed the force-sign convention in the phase-resolved two-body deformation
  tables. The archived convention is now explicitly
  `Delta F_def = F_L,full - F_L,sup`, with `F_L = sigma(x_m,t)`.
- Added the sine-Gordon frequency scan of
  `A2_def / A2_direct^(4)` at fixed `kappa R = 7`.
- Added the three-period sine-Gordon control showing that the deformation
  coefficients are a short-time response and are not stationary effective
  two-body-potential coefficients.
- Added the frozen-template tail decomposition into `|U|^2`, `Lambda`,
  `Gamma`, and the resulting local `A1^(2)` ratio.
- Added publication-quality figure exports: vector PDF for line/scatter plots
  and a 300-dpi raster for the k-omega density map.
- Added a reproducible publication-figure renderer and a figure-to-script map.
- Added raw radiation and energy-balance products needed by the renderer.

## v1.0.0 — 2026-09-02

Initial archived reproducibility release.
Zenodo DOI: 10.5281/zenodo.22257369.
