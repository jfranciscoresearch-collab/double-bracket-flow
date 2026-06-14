"""AOAD package entrypoint and convenience simulation helpers."""

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import aoad_matrix_vector_field, aoad_vector_field, compute_commutator_energy
from .analysis import extract_asymptotic_spectral_gap
from scipy.integrate import solve_ivp
import numpy as _np

__all__ = [
    'aoad_matrix_vector_field',
    'compute_commutator_energy',
    'extract_asymptotic_spectral_gap',
    'simulate_global_dynamics',
]


def commutator_energy(LG, LJ):
    """Compatibility wrapper for examples: compute commutator energy."""
    return compute_commutator_energy(LG, LJ)


def run_aoad(LG0, LJ0, alpha=1.0, beta=1.0, t_max=10.0, num_points=300, rtol=1e-10, atol=1e-13):
    """Run AOAD simulation and return a dict matching example expectations."""
    n = LG0.shape[0]
    y0 = _np.concatenate([LG0.flatten(), LJ0.flatten()])
    t_span = (0.0, float(t_max))
    t_eval = _np.linspace(t_span[0], t_span[1], int(num_points))

    sol = solve_ivp(
        aoad_vector_field,
        t_span,
        y0,
        args=(n, float(alpha), float(beta)),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    LG_traj = sol.y[0 : n * n].T.reshape(-1, n, n)
    LJ_traj = sol.y[n * n :].T.reshape(-1, n, n)

    energies = _np.array([
        compute_commutator_energy(LG_traj[i], LJ_traj[i]) for i in range(len(sol.t))
    ])

    return {
        't': sol.t,
        'energy': energies,
        'LG_traj': LG_traj,
        'LJ_traj': LJ_traj,
    }


def verify_isospectrality(LG_traj, LJ_traj, alpha=1.0, beta=1.0):
    """Basic verification utilities used by the examples."""
    energies = _np.array([
        compute_commutator_energy(LG_traj[i], LJ_traj[i]) for i in range(len(LG_traj))
    ])
    energy_decrease = energies[0] - energies[-1]
    final_comm_norm = compute_commutator_energy(LG_traj[-1], LJ_traj[-1])
    is_monotone = _np.all(_np.diff(energies) <= 1e-6)
    is_nearly_commuting = final_comm_norm < 1e-6

    return {
        'is_monotone_dissipation': bool(is_monotone),
        'is_nearly_commuting': bool(is_nearly_commuting),
        'energy_decrease': float(energy_decrease),
        'final_commutator_norm': float(final_comm_norm),
    }


def global_initial_conditions(n: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical global initial conditions for AOAD examples."""
    if n != 2:
        raise ValueError('Global example only supports n=2 for the current script.')

    LG0 = np.array([[1.0, 0.0], [0.0, -1.0]])
    LJ0 = np.array([[0.0, 1.0], [1.0, 0.0]])
    return LG0, LJ0


def simulate_global_dynamics(
    n: int = 2,
    alpha: float = 1.0,
    beta: float = 1.0,
    t_max: float = 20.0,
    max_step: float = 0.01,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the global relaxation AOAD simulation and return the trajectory.

    Returns (times, LG_traj, LJ_traj).
    """
    LG0, LJ0 = global_initial_conditions(n)
    y0 = np.concatenate([LG0.flatten(), LJ0.flatten()])

    def rhs(t, y):
        return aoad_matrix_vector_field(
            LG=y[: n * n].reshape((n, n)),
            LJ=y[n * n :].reshape((n, n)),
            alpha=alpha,
            beta=beta,
        )

    sol = solve_ivp(
        rhs,
        [0, t_max],
        y0,
        method='RK45',
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )

    LG_traj = sol.y[: n * n].T.reshape(-1, n, n)
    LJ_traj = sol.y[n * n :].T.reshape(-1, n, n)
    return sol.t, LG_traj, LJ_traj
