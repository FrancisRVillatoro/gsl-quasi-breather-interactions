"""
ssf4_gsl.py
===========

Fourth-order symmetric split-step Fourier solver for

    u_tt - u_xx + f_b(u) = 0,

    f_b(u) = sin(u) / sqrt(1 + b^2 (1 - cos(u))).

The splitting is

    A: u_t = v,       v_t = u_xx - u
    B: u_t = 0,       v_t = -(f_b(u) - u),

with both subflows integrated exactly.  A is diagonalized in Fourier
space and B is a pointwise kick.  Fourth order is obtained with the
three-stage Yoshida composition of Strang splitting.

No spectral filtering/dealiasing is applied by default.
"""

from __future__ import annotations

import numpy as np


# Yoshida coefficients
YOSHIDA_G1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
YOSHIDA_G0 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))


def grid(N: int, L: float):
    """Periodic grid x in [-L/2,L/2), spacing dx, and rFFT wave numbers."""
    if N % 2:
        raise ValueError("N must be even.")
    dx = L / N
    x = (np.arange(N) - N // 2) * dx
    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)
    omega_k = np.sqrt(1.0 + k * k)
    return x, dx, k, omega_k


def gsl_force(u: np.ndarray, b: float) -> np.ndarray:
    """f_b(u) = G'(u)."""
    y = 1.0 - np.cos(u)
    return np.sin(u) / np.sqrt(1.0 + (b * b) * y)


def gsl_potential(u: np.ndarray, b: float) -> np.ndarray:
    """
    Exact potential G(u), normalized by G(0)=0.

    Stable rationalized form:
        G = 2(1-cos u)/(1 + sqrt(1+b^2(1-cos u))).

    At b=0 this reduces exactly to 1-cos(u).
    """
    y = 1.0 - np.cos(u)
    return 2.0 * y / (1.0 + np.sqrt(1.0 + (b * b) * y))


def nonlinear_remainder(u: np.ndarray, b: float) -> np.ndarray:
    """N_b(u) = f_b(u) - u."""
    return gsl_force(u, b) - u


def linear_flow(
    u: np.ndarray,
    v: np.ndarray,
    h: float,
    omega_k: np.ndarray,
):
    """
    Exact flow of u_t=v, v_t=u_xx-u for time h.

    Uses real FFTs.  omega_k = sqrt(1+k^2).
    """
    N = u.size
    uh = np.fft.rfft(u)
    vh = np.fft.rfft(v)

    c = np.cos(omega_k * h)
    s = np.sin(omega_k * h)

    u_new = np.fft.irfft(c * uh + (s / omega_k) * vh, n=N)
    v_new = np.fft.irfft(-omega_k * s * uh + c * vh, n=N)
    return u_new, v_new


def nonlinear_flow(
    u: np.ndarray,
    v: np.ndarray,
    h: float,
    b: float,
):
    """Exact nonlinear kick for time h."""
    return u, v - h * nonlinear_remainder(u, b)


def strang_step(
    u: np.ndarray,
    v: np.ndarray,
    h: float,
    b: float,
    omega_k: np.ndarray,
):
    """Second-order symmetric Strang step B(h/2) A(h) B(h/2)."""
    u, v = nonlinear_flow(u, v, 0.5 * h, b)
    u, v = linear_flow(u, v, h, omega_k)
    u, v = nonlinear_flow(u, v, 0.5 * h, b)
    return u, v


def ssf4_step(
    u: np.ndarray,
    v: np.ndarray,
    h: float,
    b: float,
    omega_k: np.ndarray,
):
    """
    Fourth-order symmetric Yoshida composition:
        S4(h) = S2(g1 h) S2(g0 h) S2(g1 h).

    Adjacent nonlinear half-kicks are merged.  Thus one macro-step uses
    three exact Fourier linear flows and four nonlinear force evaluations.
    """
    a1 = YOSHIDA_G1
    a0 = YOSHIDA_G0
    b1 = 0.5 * a1
    b2 = 0.5 * (a1 + a0)

    u, v = nonlinear_flow(u, v, b1 * h, b)
    u, v = linear_flow(u, v, a1 * h, omega_k)

    u, v = nonlinear_flow(u, v, b2 * h, b)
    u, v = linear_flow(u, v, a0 * h, omega_k)

    u, v = nonlinear_flow(u, v, b2 * h, b)
    u, v = linear_flow(u, v, a1 * h, omega_k)

    u, v = nonlinear_flow(u, v, b1 * h, b)
    return u, v


def spectral_derivative(u: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Periodic spectral derivative u_x."""
    return np.fft.irfft(1j * k * np.fft.rfft(u), n=u.size)


def energy(
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    k: np.ndarray,
    b: float,
) -> float:
    """Hamiltonian H = integral [v^2/2 + u_x^2/2 + G(u)] dx."""
    ux = spectral_derivative(u, k)
    density = 0.5 * v * v + 0.5 * ux * ux + gsl_potential(u, b)
    return float(dx * np.sum(density))


def momentum(
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    k: np.ndarray,
) -> float:
    """P = - integral v u_x dx."""
    ux = spectral_derivative(u, k)
    return float(-dx * np.sum(v * ux))


def stress(
    u: np.ndarray,
    v: np.ndarray,
    k: np.ndarray,
) -> np.ndarray:
    """
    Placeholder linear part is not enough for GSL; use stress_gsl below.
    Kept separate to make accidental use difficult.
    """
    raise RuntimeError("Use stress_gsl(u,v,k,b).")


def stress_gsl(
    u: np.ndarray,
    v: np.ndarray,
    k: np.ndarray,
    b: float,
) -> np.ndarray:
    """
    Momentum-flux/stress quantity

        sigma = G(u) - v^2/2 - u_x^2/2,

    so dP_L/dt = sigma(X,t) for
        P_L = - integral_{-infty}^X v u_x dx.
    """
    ux = spectral_derivative(u, k)
    return gsl_potential(u, b) - 0.5 * v * v - 0.5 * ux * ux


def sine_gordon_breather(x: np.ndarray, t: float, w: float):
    """
    Exact sine-Gordon breather (b=0) with phase chosen so that

        u(x,0)=0,
        u_t(x,0)=4 q sech(q x),

    where
        q     = 1/sqrt(1+w^2),
        omega = w/sqrt(1+w^2).
    """
    q = 1.0 / np.sqrt(1.0 + w * w)
    omega = w / np.sqrt(1.0 + w * w)

    sech = 1.0 / np.cosh(q * x)
    z = (q / omega) * np.sin(omega * t) * sech

    u = 4.0 * np.arctan(z)
    v = 4.0 * q * np.cos(omega * t) * sech / (1.0 + z * z)
    return u, v


def sine_gordon_breather_energy(w: float) -> float:
    """Exact rest energy E_B=16 q."""
    q = 1.0 / np.sqrt(1.0 + w * w)
    return 16.0 * q
