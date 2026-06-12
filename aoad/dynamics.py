"""Core dynamics for the anisotropic orbit adjoint dynamics (AOAD)."""

import numpy as np


def comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix commutator [A, B] = AB - BA."""
    return A @ B - B @ A


def aoad_vector_field(t: float, y: np.ndarray, n: int, alpha: float, beta: float) -> np.ndarray:
    r"""AOAD right-hand side for flattened state vectors.

    The state y is the concatenation of the flattened operators LG and LJ.

        \dot{L}_G = -\alpha [L_G, [L_G, L_J]]
        \dot{L}_J = -\beta  [L_J, [L_J, L_G]]
    """
    LG = y[: n * n].reshape((n, n))
    LJ = y[n * n :].reshape((n, n))

    C = comm(LG, LJ)
    dLG = -alpha * comm(LG, C)
    dLJ = -beta * comm(LJ, C)

    return np.concatenate([dLG.flatten(), dLJ.flatten()])


def aoad_matrix_vector_field(
    LG: np.ndarray,
    LJ: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """AOAD vector field for matrix-valued operators."""
    C = comm(LG, LJ)
    dLG = -alpha * comm(LG, C)
    dLJ = -beta * comm(LJ, C)
    return dLG, dLJ


def compute_commutator_energy(LG: np.ndarray, LJ: np.ndarray) -> float:
    """Evaluate the Frobenius norm squared energy: 0.5 * ||[LG, LJ]||_F^2."""
    C = comm(LG, LJ)
    return 0.5 * np.sum(C**2)
