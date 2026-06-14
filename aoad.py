"""
aoad.py
-------
Main simulator for the Anisotropic Adjoint Orbit Alignment Dynamics (AOAD) system:

    L_G' = -alpha * [L_G, [L_G, L_J]]
    L_J' = -beta  * [L_J, [L_J, L_G]]

Reference:
    Jerina Jeneth C. Francisco
    "Coupled Double-Bracket Gradient Flows on Adjoint Orbits:
     Commutator Energy Dissipation, Algebraic Confinement,
     and Asymptotic Alignment" (2026)
    Submitted to SIAM Journal on Matrix Analysis and Applications
"""

import numpy as np
from scipy.integrate import solve_ivp


def commutator(A, B):
    """Return [A, B] = AB - BA."""
    return A @ B - B @ A


def double_bracket(L, N):
    """Return [L, [L, N]]."""
    return commutator(L, commutator(L, N))


def commutator_energy(LG, LJ):
    """Return E = 0.5 * ||[L_G, L_J]||_F^2."""
    C = commutator(LG, LJ)
    return 0.5 * np.linalg.norm(C, 'fro') ** 2


def aoad_rhs(t, y, n, alpha, beta):
    """
    Right-hand side of the AOAD system, flattened for scipy.integrate.

    Parameters
    ----------
    t     : float         current time (unused; autonomous system)
    y     : (2*n*n,)      flattened [L_G, L_J]
    n     : int           matrix dimension
    alpha : float         weight for L_G equation
    beta  : float         weight for L_J equation

    Returns
    -------
    dy : (2*n*n,) flattened time derivatives
    """
    LG = y[:n*n].reshape(n, n)
    LJ = y[n*n:].reshape(n, n)

    dLG = -alpha * double_bracket(LG, LJ)
    dLJ = -beta  * double_bracket(LJ, LG)

    return np.concatenate([dLG.ravel(), dLJ.ravel()])


def run_aoad(LG0, LJ0, alpha=1.0, beta=1.0, t_span=(0, 15), t_eval=None,
             rtol=1e-9, atol=1e-11):
    """
    Integrate the AOAD system from initial conditions (LG0, LJ0).

    Parameters
    ----------
    LG0    : (n, n) real symmetric matrix
    LJ0    : (n, n) real symmetric matrix
    alpha  : float, weight for L_G equation (default 1.0)
    beta   : float, weight for L_J equation (default 1.0)
    t_span : (t0, tf) integration interval
    t_eval : array of times at which to store solution (optional)
    rtol   : relative tolerance for ODE solver
    atol   : absolute tolerance for ODE solver

    Returns
    -------
    sol    : OdeSolution object from scipy
    t      : (N,) time array
    LG_t   : (N, n, n) array of L_G matrices along the trajectory
    LJ_t   : (N, n, n) array of L_J matrices along the trajectory
    E_t    : (N,) commutator energy along the trajectory
    """
    assert LG0.shape == LJ0.shape, "LG0 and LJ0 must have the same shape."
    n = LG0.shape[0]

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)

    y0 = np.concatenate([LG0.ravel(), LJ0.ravel()])

    sol = solve_ivp(
        aoad_rhs,
        t_span,
        y0,
        args=(n, alpha, beta),
        t_eval=t_eval,
        method='RK45',
        rtol=rtol,
        atol=atol,
        dense_output=False
    )

    t = sol.t
    LG_t = sol.y[:n*n, :].T.reshape(-1, n, n)
    LJ_t = sol.y[n*n:, :].T.reshape(-1, n, n)
    E_t  = np.array([commutator_energy(LG_t[i], LJ_t[i]) for i in range(len(t))])

    return sol, t, LG_t, LJ_t, E_t


def verify_isospectrality(LG0, LG_t, LJ0, LJ_t, tol=1e-6):
    """
    Check that eigenvalues of L_G and L_J are preserved along the trajectory.

    Returns True if max deviation is below tol, prints a summary.
    """
    eig_G0 = np.sort(np.linalg.eigvalsh(LG0))
    eig_J0 = np.sort(np.linalg.eigvalsh(LJ0))

    max_dev_G = max(
        np.max(np.abs(np.sort(np.linalg.eigvalsh(LG_t[i])) - eig_G0))
        for i in range(len(LG_t))
    )
    max_dev_J = max(
        np.max(np.abs(np.sort(np.linalg.eigvalsh(LJ_t[i])) - eig_J0))
        for i in range(len(LJ_t))
    )

    print(f"Isospectrality check:")
    print(f"  max eigenvalue deviation L_G: {max_dev_G:.2e}")
    print(f"  max eigenvalue deviation L_J: {max_dev_J:.2e}")
    passed = (max_dev_G < tol) and (max_dev_J < tol)
    print(f"  {'PASSED' if passed else 'FAILED'} (tolerance {tol:.0e})")
    return passed
