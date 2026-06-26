"""Core dynamics for the Anisotropic Orbit Adjoint Dynamics (AOAD).

Reference:
    Jerina Jeneth C. Francisco
    'Coupled Double-Bracket Gradient Flows on Adjoint Orbits:
     Commutator Energy, Alignment Dissipation, and Asymptotic
     Confinement' (corrected version)

Note: this module implements the vector field only, which is unchanged
from the original manuscript. The corrected Lyapunov/dissipation theory
built on top of this vector field is in aoad/lyapunov.py and
aoad/analysis.py; see those modules' docstrings for what changed and why.

The system is:
    dL_G/dt = -alpha * [L_G, [L_G, L_J]]
    dL_J/dt = -beta  * [L_J, [L_J, L_G]]

Sign note:
    [L_J, L_G] = -[L_G, L_J] = -C  where C = [L_G, L_J]
    so [L_J, [L_J, L_G]] = [L_J, -C] = -[L_J, C]
    therefore dL_J = -beta * [L_J, [L_J, L_G]] = +beta * [L_J, C]
"""

import numpy as np


def comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix commutator [A, B] = AB - BA."""
    return A @ B - B @ A


def aoad_vector_field(
    t: float, y: np.ndarray, n: int, alpha: float, beta: float
) -> np.ndarray:
    r"""AOAD right-hand side for flattened state vectors.

    The state y is the concatenation of the flattened operators LG and LJ.

        \dot{L}_G = -\alpha [L_G, [L_G, L_J]]  =  -alpha * comm(LG, C)
        \dot{L}_J = -\beta  [L_J, [L_J, L_G]]  =  +beta  * comm(LJ, C)

    where C = [L_G, L_J].
    """
    LG = y[: n * n].reshape((n, n))
    LJ = y[n * n :].reshape((n, n))

    C = comm(LG, LJ)
    dLG = -alpha * comm(LG, C)
    dLJ = +beta  * comm(LJ, C)   # NOTE: positive sign, see module docstring

    return np.concatenate([dLG.flatten(), dLJ.flatten()])


def aoad_matrix_vector_field(
    LG: np.ndarray,
    LJ: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple:
    """AOAD vector field returning (dLG, dLJ) as a matrix pair.

    Implements:
        dLG = -alpha * [LG, [LG, LJ]]  =  -alpha * comm(LG, C)
        dLJ = -beta  * [LJ, [LJ, LG]]  =  +beta  * comm(LJ, C)

    where C = [LG, LJ].
    """
    C = comm(LG, LJ)
    dLG = -alpha * comm(LG, C)
    dLJ = +beta  * comm(LJ, C)   # NOTE: positive sign, see module docstring
    return dLG, dLJ


def compute_commutator_energy(LG: np.ndarray, LJ: np.ndarray) -> float:
    """Evaluate the commutator energy: E = 0.5 * ||[LG, LJ]||_F^2."""
    C = comm(LG, LJ)
    return 0.5 * float(np.sum(C ** 2))
