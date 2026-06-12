"""AOAD package entrypoint and convenience simulation helpers."""

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import aoad_matrix_vector_field, compute_commutator_energy
from .analysis import extract_asymptotic_spectral_gap

__all__ = [
    'aoad_matrix_vector_field',
    'compute_commutator_energy',
    'extract_asymptotic_spectral_gap',
    'simulate_global_dynamics',
]


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
