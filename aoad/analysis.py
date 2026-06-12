"""Analysis utilities for asymptotic spectral gap extraction."""

import numpy as np


def extract_asymptotic_spectral_gap(LG_star: np.ndarray, LJ_star: np.ndarray, alpha: float, beta: float) -> float:
    """Compute the minimal spectral gap for the linearised AOAD operator.

    The linearised gap is obtained in the shared eigenbasis of the terminal
    matrix LG_star, and uses Theorem 8.2 to compute element-wise contributions.
    """
    n = LG_star.shape[0]
    if LG_star.shape != (n, n) or LJ_star.shape != (n, n):
        raise ValueError("LG_star and LJ_star must be square matrices of the same shape")

    vals_G, vecs_G = np.linalg.eigh(LG_star)
    paired_J = np.diag(vecs_G.T @ LJ_star @ vecs_G)

    gamma_elements = []
    for i in range(n):
        for j in range(i + 1, n):
            g = alpha * (vals_G[i] - vals_G[j])**2 + beta * (paired_J[i] - paired_J[j])**2
            gamma_elements.append(g)

    if not gamma_elements:
        raise ValueError("Matrix dimension must be at least 2 to compute a spectral gap")

    return min(gamma_elements)
