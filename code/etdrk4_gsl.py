
"""
etdrk4_gsl.py
=============

Cox-Matthews fourth-order exponential time-differencing Runge-Kutta
integrator for the graphene-superlattice Klein-Gordon equation

    u_tt - u_xx + f_b(u) = 0,
    f_b(u)=sin(u)/sqrt(1+b^2(1-cos u)).

Write y=(u,v)^T, v=u_t,

    y_t = L y + N(y),

where for Fourier mode k,

    L_k = [[0,1],[-(1+k^2),0]],

and

    N(y) = [0, -(f_b(u)-u)]^T.

Because L_k^2=-omega_k^2 I, every matrix function f(h L_k) can be
applied analytically as a_k I + b_k L_k.
"""

from __future__ import annotations
import numpy as np

from ssf4_gsl import gsl_force, nonlinear_remainder


def _phi_series(z: np.ndarray, m: int, nterms: int = 40) -> np.ndarray:
    """
    phi_m(z) = sum_{j>=0} z^j/(j+m)!.
    Stable for modest |z|; used only near the origin.
    """
    z = np.asarray(z, dtype=np.complex128)
    # first term 1/m!
    fact = 1.0
    for j in range(2, m + 1):
        fact *= j
    if m == 0:
        fact = 1.0
    term = np.ones_like(z) / fact
    s = term.copy()
    for j in range(1, nterms):
        term = term * z / (j + m)
        s += term
        if np.max(np.abs(term)) < 1e-18:
            break
    return s


def phi_functions(z: np.ndarray):
    """Return phi1, phi2, phi3 for complex z, stably near z=0."""
    z = np.asarray(z, dtype=np.complex128)
    p1 = np.empty_like(z)
    p2 = np.empty_like(z)
    p3 = np.empty_like(z)

    small = np.abs(z) < 0.5
    large = ~small

    if np.any(small):
        zs = z[small]
        p1[small] = _phi_series(zs, 1)
        p2[small] = _phi_series(zs, 2)
        p3[small] = _phi_series(zs, 3)

    if np.any(large):
        zl = z[large]
        p1l = np.expm1(zl) / zl
        p2l = (p1l - 1.0) / zl
        p3l = (p2l - 0.5) / zl
        p1[large], p2[large], p3[large] = p1l, p2l, p3l

    return p1, p2, p3


def _ab_from_scalar_function(values: np.ndarray, omega: np.ndarray):
    """
    If f(L)=a I + b L and f(i omega)=values, then
        a=Re(values), b=Im(values)/omega.
    """
    return values.real, values.imag / omega


class ETDRK4Coefficients:
    """Precomputed mode-by-mode Cox-Matthews ETDRK4 coefficients."""

    def __init__(self, omega_k: np.ndarray, h: float):
        self.omega = np.asarray(omega_k, dtype=float)
        self.h = float(h)

        z = 1j * self.omega * self.h
        z2 = 0.5 * z

        # Exact propagators.
        self.E_a = np.cos(self.omega * self.h)
        self.E_b = np.sin(self.omega * self.h) / self.omega
        hh = 0.5 * self.h
        self.E2_a = np.cos(self.omega * hh)
        self.E2_b = np.sin(self.omega * hh) / self.omega

        p1, p2, p3 = phi_functions(z)
        p1h, _, _ = phi_functions(z2)

        # Q = h/2 phi1(hL/2)
        Qvals = 0.5 * self.h * p1h

        # Cox-Matthews final weights:
        # f1 = h(phi1 - 3phi2 + 4phi3)
        # f2 = h(phi2 - 2phi3), used as 2*f2*(Na+Nb)
        # f3 = h(-phi2 + 4phi3)
        f1vals = self.h * (p1 - 3.0*p2 + 4.0*p3)
        f2vals = self.h * (p2 - 2.0*p3)
        f3vals = self.h * (-p2 + 4.0*p3)

        self.Q_a, self.Q_b = _ab_from_scalar_function(Qvals, self.omega)
        self.f1_a, self.f1_b = _ab_from_scalar_function(f1vals, self.omega)
        self.f2_a, self.f2_b = _ab_from_scalar_function(f2vals, self.omega)
        self.f3_a, self.f3_b = _ab_from_scalar_function(f3vals, self.omega)


def _apply_ab(uh, vh, a, b, omega):
    """Apply (a I + b L_k) to Fourier pair (uh,vh)."""
    out_u = a*uh + b*vh
    out_v = a*vh - b*(omega*omega)*uh
    return out_u, out_v


def _nonlinear_hat(u: np.ndarray, bpar: float):
    """
    FFT of N(y)=[0,-(f_b(u)-u)].
    Returns the Fourier pair (0, Nv_hat).
    """
    Nv = -nonlinear_remainder(u, bpar)
    return np.zeros(u.size//2 + 1, dtype=np.complex128), np.fft.rfft(Nv)


def etdrk4_step(
    u: np.ndarray,
    v: np.ndarray,
    bpar: float,
    coeff: ETDRK4Coefficients,
):
    """One Cox-Matthews ETDRK4 step."""
    N = u.size
    om = coeff.omega

    uh = np.fft.rfft(u)
    vh = np.fft.rfft(v)

    # N1
    n1u, n1v = _nonlinear_hat(u, bpar)

    # a = E2*y + Q*N1
    e2u, e2v = _apply_ab(uh, vh, coeff.E2_a, coeff.E2_b, om)
    qu, qv = _apply_ab(n1u, n1v, coeff.Q_a, coeff.Q_b, om)
    ah, avh = e2u + qu, e2v + qv
    a = np.fft.irfft(ah, n=N)
    n2u, n2v = _nonlinear_hat(a, bpar)

    # bstage = E2*y + Q*N2
    qu, qv = _apply_ab(n2u, n2v, coeff.Q_a, coeff.Q_b, om)
    bh, bvh = e2u + qu, e2v + qv
    bstage = np.fft.irfft(bh, n=N)
    n3u, n3v = _nonlinear_hat(bstage, bpar)

    # c = E2*a + Q*(2*N3-N1)
    e2au, e2av = _apply_ab(ah, avh, coeff.E2_a, coeff.E2_b, om)
    combu = 2.0*n3u - n1u
    combv = 2.0*n3v - n1v
    qu, qv = _apply_ab(combu, combv, coeff.Q_a, coeff.Q_b, om)
    ch, cvh = e2au + qu, e2av + qv
    c = np.fft.irfft(ch, n=N)
    n4u, n4v = _nonlinear_hat(c, bpar)

    # y_{n+1} = E*y + f1*N1 + 2*f2*(N2+N3) + f3*N4
    eu, ev = _apply_ab(uh, vh, coeff.E_a, coeff.E_b, om)

    t1u, t1v = _apply_ab(n1u, n1v, coeff.f1_a, coeff.f1_b, om)
    s23u, s23v = n2u+n3u, n2v+n3v
    t2u, t2v = _apply_ab(s23u, s23v, coeff.f2_a, coeff.f2_b, om)
    t4u, t4v = _apply_ab(n4u, n4v, coeff.f3_a, coeff.f3_b, om)

    un_h = eu + t1u + 2.0*t2u + t4u
    vn_h = ev + t1v + 2.0*t2v + t4v

    un = np.fft.irfft(un_h, n=N)
    vn = np.fft.irfft(vn_h, n=N)
    return un, vn
