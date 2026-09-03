"""
m4_dynamic_validation_b1_w3.py
==============================

Final M4 validation protocol.

Inputs
------
  ssf4_gsl.py
  gsl_chirp_aware_phase_experiment_b1_w3.zip

Part A: calibrate the quasi-breather inertial mass by small Lorentz boosts
of the frozen-age chirp-aware template and fit
    P(v) = M_eff v + c3 v^3.

Part B: for kappa R ~ 6,7 and delta=0,pi, evolve
    pair, isolated-left, isolated-right
with SSF4 and verify the invariant conservation-law identity
    Delta P_R^int(t) = integral_0^t F_R^int(s) ds,
where
    F_L^int = sigma_pair(0)-sigma_L(0)-sigma_R(0),
    F_R^int = -F_L^int,
and P_R^int is the analogous renormalized momentum on x>=0.

The production run reported in the associated CSV files used 320 steps
per quasi-breather period.
"""

# This file documents the exact final protocol used in M4.
# The numerical outputs are included in the same ZIP package.
