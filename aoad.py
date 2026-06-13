"""
AOAD (Anisotropic Orbit Adjoint Dynamics) Simulator
=====================================================

Main simulator for the coupled double-bracket flow system:

    dL_G/dt = -α [L_G, [L_G, L_J]]
    dL_J/dt = -β [L_J, [L_J, L_G]]

Provides high-level functions:
  - run_aoad()              : Run the AOAD simulation
  - commutator_energy()     : Compute Frobenius norm of commutator
  - verify_isospectrality() : Verify spectral properties and dissipation

Reference:
    Francisco, J.J.C. (2026). Anisotropic Orbit Adjoint Dynamics.

Requirements:
    numpy, scipy
    Install via: pip install numpy scipy matplotlib
"""

import numpy as np
from scipy.integrate import solve_ivp
from aoad.dynamics import aoad_vector_field, compute_commutator_energy
from aoad.analysis import extract_asymptotic_spectral_gap


__all__ = [
    'run_aoad',
    'commutator_energy',
    'verify_isospectrality',
]


def run_aoad(
    LG0: np.ndarray,
    LJ0: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    t_max: float = 20.0,
    max_step: float = 0.01,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    num_points: int = None,
) -> dict:
    """
    Run the anisotropic orbit adjoint dynamics (AOAD) simulation.

    Parameters
    ----------
    LG0 : np.ndarray
        Initial condition for L_G (n × n symmetric matrix).
    LJ0 : np.ndarray
        Initial condition for L_J (n × n symmetric matrix).
    alpha : float
        Anisotropy parameter for L_G dynamics. Default: 1.0.
    beta : float
        Anisotropy parameter for L_J dynamics. Default: 1.0.
    t_max : float
        Final integration time. Default: 20.0.
    max_step : float
        Maximum step size for ODE solver. Default: 0.01.
    rtol : float
        Relative tolerance for ODE solver. Default: 1e-9.
    atol : float
        Absolute tolerance for ODE solver. Default: 1e-12.
    num_points : int, optional
        Number of evaluation points. If None, use adaptive stepping.

    Returns
    -------
    dict
        Dictionary containing:
          - 't'       : Time array
          - 'LG_traj' : Trajectory of L_G (N, n, n)
          - 'LJ_traj' : Trajectory of L_J (N, n, n)
          - 'energy'  : Commutator energy over time (N,)
    """
    n = LG0.shape[0]
    if LG0.shape != (n, n) or LJ0.shape != (n, n):
        raise ValueError("LG0 and LJ0 must be square matrices of the same shape")

    y0 = np.concatenate([LG0.flatten(), LJ0.flatten()])

    t_eval = None
    if num_points is not None:
        t_eval = np.linspace(0, t_max, num_points)

    sol = solve_ivp(
        aoad_vector_field,
        [0, t_max],
        y0,
        args=(n, alpha, beta),
        method='RK45',
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
    )

    LG_traj = sol.y[:n*n].T.reshape(-1, n, n)
    LJ_traj = sol.y[n*n:].T.reshape(-1, n, n)

    energies = np.array([
        commutator_energy(LG_traj[i], LJ_traj[i])
        for i in range(len(sol.t))
    ])

    return {
        't': sol.t,
        'LG_traj': LG_traj,
        'LJ_traj': LJ_traj,
        'energy': energies,
    }


def commutator_energy(LG: np.ndarray, LJ: np.ndarray) -> float:
    """
    Compute the commutator energy: 0.5 * ||[L_G, L_J]||_F^2.

    Parameters
    ----------
    LG : np.ndarray
        Operator L_G (n × n).
    LJ : np.ndarray
        Operator L_J (n × n).

    Returns
    -------
    float
        Frobenius norm squared of the commutator, scaled by 0.5.
    """
    return compute_commutator_energy(LG, LJ)


def verify_isospectrality(
    LG_traj: np.ndarray,
    LJ_traj: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    tolerance: float = 1e-6,
) -> dict:
    """
    Verify isospectrality, monotone dissipation, and convergence properties.

    Parameters
    ----------
    LG_traj : np.ndarray
        Trajectory of L_G (N, n, n).
    LJ_traj : np.ndarray
        Trajectory of L_J (N, n, n).
    alpha : float
        Anisotropy parameter for L_G. Default: 1.0.
    beta : float
        Anisotropy parameter for L_J. Default: 1.0.
    tolerance : float
        Tolerance for near-commutativity check. Default: 1e-6.

    Returns
    -------
    dict
        Dictionary containing verification results:
          - 'is_monotone_dissipation' : bool, energy monotone decreasing
          - 'is_nearly_commuting'     : bool, final commutator small
          - 'energy_decrease'         : float, total energy decrease
          - 'final_commutator_norm'   : float, ||[L_G, L_J]|| at final time
          - 'gamma_min'               : float, minimal spectral gap (if available)
          - 'convergence_rate'        : float, asymptotic decay rate estimate
    """
    n = LG_traj.shape[1]
    N = LG_traj.shape[0]

    # Compute energy trajectory
    energies = np.array([
        commutator_energy(LG_traj[i], LJ_traj[i])
        for i in range(N)
    ])

    # Check monotone dissipation
    energy_diffs = np.diff(energies)
    is_monotone = np.all(energy_diffs <= tolerance)

    # Check near-commutativity at final time
    def comm(A, B):
        return A @ B - B @ A

    final_comm = comm(LG_traj[-1], LJ_traj[-1])
    final_comm_norm = np.linalg.norm(final_comm, 'fro')
    is_nearly_commuting = final_comm_norm < tolerance

    # Energy decrease
    energy_decrease = energies[0] - energies[-1]

    # Compute gamma_min for asymptotic rate
    gamma_min = None
    try:
        gamma_min = extract_asymptotic_spectral_gap(
            LG_traj[-1],
            LJ_traj[-1],
            alpha,
            beta,
        )
    except Exception:
        gamma_min = None

    # Estimate convergence rate from final energies
    convergence_rate = None
    if N > 1 and energies[-1] > 1e-10 and energies[-2] > 1e-10:
        # Approximate exponential decay rate
        rate = -np.log(energies[-1] / energies[-2])
        convergence_rate = rate

    return {
        'is_monotone_dissipation': bool(is_monotone),
        'is_nearly_commuting': bool(is_nearly_commuting),
        'energy_decrease': float(energy_decrease),
        'final_commutator_norm': float(final_comm_norm),
        'gamma_min': float(gamma_min) if gamma_min is not None else None,
        'convergence_rate': convergence_rate,
    }
