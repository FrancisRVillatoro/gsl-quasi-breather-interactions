"""
M6: force/energy-transfer quadratures for GSL quasi-breathers
==============================================================

Required inputs
---------------
  ssf4_gsl.py
  gsl_chirp_aware_phase_experiment_b1_w3.zip

Production parameters
---------------------
  b=1, w=3
  frozen chirp-aware template: SSF4, 320 steps/period
  relative-phase scan: 64 delta values x 256 common phases
  separations: kappa R ~ 6,7,8
  dynamic PDE validation: kappa R ~ 7, delta=+/-pi/2,
                          160 and 320 steps/period.

Definitions
-----------
Momentum-transfer force on the left half:
    F_L^int = [G(u)-u_t^2/2-u_x^2/2]_pair
              - same quantity for isolated left and right fields.

Energy-transfer rate into the left half:
    Pi_L^int = [u_t u_x]_pair
               - [u_t u_x]_left - [u_t u_x]_right.

For an adiabatic fundamental tail
    u_j ~ C1 exp(-kappa |x-X_j|) cos(Omega t+phi_j),
one obtains
    Fbar_L = C1^2 kappa^2 exp(-kappa R) cos(delta),
    Pibar_L = C1^2 Omega kappa exp(-kappa R) sin(delta),
and hence
    B1/A1 = Omega/kappa.

Important higher-order result
-----------------------------
The exact spatial energy flux is quadratic.  Because the quasi-breather has
odd temporal harmonics, phase averaging generates sin(delta), sin(3 delta),
sin(5 delta), ... but no sin(2 delta).  Numerically B2 is zero to roundoff,
even though the force has a nonzero A2 cos(2 delta) quartic correction.

Dynamic conservation-law check
------------------------------
For the left half-domain,
    Delta E_L^int(t)
      = integral_0^t [Pi_mid^int(s)-Pi_left_boundary^int(s)] ds.
At kappa R~7, delta=pi/2, this identity is verified over one period to
~6e-6 relative endpoint error in the 320-step/period run.

The numerical CSV and PNG files in the M6 package contain the complete
production outputs.
"""
