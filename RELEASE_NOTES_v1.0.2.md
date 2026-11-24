# v1.0.2 — Wave Motion revision

Reproducibility update for *Forces and energy transfer between graphene-superlattice quasi-breathers*.

- Figure 2 now uses the dedicated isolated-state simulation at 160 steps per seed period. The package includes the simulation driver, fitted spatial profile, summary and plotting code. The fit uses 24 points on the positive half-grid and gives `kappa_fit = 0.5352974` and `R² = 0.999674`.
- Figure 3 includes the authors’ revised vector artwork. The publication renderer copies this PDF exactly. The radiation measurements and their analysis code are included in the repository.
- Figure 6 identifies the plotted balance as the **right-half** interaction momentum and impulse. Figure 8 uses the measured force and energy-transfer coefficients at each separation and shows the radial residual. Its RMS values are 2.25%, 0.78% and 0.37% in order of increasing separation.
- The phase-resolved deformation and tail-drift drivers now select the production template with odd harmonics 1, 3, 5 and 7 from the archived milestone. Both reproduce the reference tables. For the full graphene-superlattice pair, the second cosine and sine coefficients are +7.3370e-7 and +7.3245e-6, respectively.
- Run `python3 -B code/validate_revision_diagnostics.py` to check the tail fit, the radial residuals, the phase coefficients and the three drift rows against the archived results.
