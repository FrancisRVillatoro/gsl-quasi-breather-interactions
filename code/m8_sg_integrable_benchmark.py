"""
M8 — Integrable sine-Gordon closure benchmark
===============================================

This script documents/reproduces the M8 benchmark used in the Physica D
submission package.

It compares the local complex-tail theory against exact sine-Gordon
breathers and the published Manton asymptotic result of

P. G. Kevrekidis, A. Khare, A. Saxena,
Phys. Rev. E 70, 057603 (2004).

Key analytical formulas
-----------------------
Exact one-breather:
    u_B = 4 atan[(kappa/omega) sin(theta) sech(kappa x)],
    kappa = sqrt(1-omega^2).

Published Manton mean force:
    A1_Manton = 64 kappa^4 / omega^2 * exp(-kappa R).

M7 adiabatic energy quadrature:
    B1 = (omega/kappa) A1.

For sine-Gordon, gamma=1/6, so the direct quartic local coefficient is
    A2^(4) = -|U1|^4/32,
and the complete direct quartic phase law is
    Fbar^(4) = A2^(4)[2 + 4 cos(delta) + cos(2 delta)].

Production M8 outputs
---------------------
  m8_sg_exact_local_scan.csv
  m8_sg_manton_instantaneous.csv
  m8_sg_dynamic_benchmark.csv
  m8_summary_metrics.csv
  m8_dynamic_deformation_fit.csv

Figures
-------
  m8_sg_integrable_summary.png
  m8_sg_manton_instantaneous.png
  m8_sg_dynamic_force.png
  m8_sg_dynamic_power.png

The exact-field frozen scan uses vectorized phase quadrature.
The dynamic benchmark evolves exact isolated sine-Gordon breathers and
their full interacting superposition with the same fourth-order SSF4 solver
used in the GSL study. Because the isolated SG breathers are exactly
periodic, isolated-evolution/frozen ratios diagnose numerical error rather
than ageing.
"""
