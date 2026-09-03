# v1.0.1 — Pre-submission reproducibility update

This release updates the reproducibility archive for the manuscript:

**Local complex-tail interactions of graphene-superlattice quasi-breathers: force, energy transfer, and finite-age nonadiabatic tails**

## Changes since v1.0.0

- Corrects and explicitly documents the force convention for phase-resolved
  two-body deformation:
  `Delta F_def = F_L,full - F_L,sup`, with `F_L = sigma(x_m,t)`.
- Adds coefficient-by-coefficient full/superposition/deformation phase fits.
- Adds the sine-Gordon scan of `A2_def/A2_direct^(4)` versus frequency at
  fixed `kappa R = 7`.
- Adds a three-period sine-Gordon control showing that deformation
  coefficients are a short-time response rather than stationary effective
  two-body-potential coefficients.
- Adds the frozen-template tail decomposition into `|U|^2`, `Lambda`,
  `Gamma`, and the local `A1^(2)` ratio.
- Adds publication-quality figure exports and a tested publication-figure
  renderer.
- Adds a figure-to-script reproduction map in the README.

Code remains under the MIT License. Data, processed outputs, and publication
figures are under CC BY 4.0.
