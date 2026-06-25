"""Analysis utilities for asymptotic spectral gap extraction.

CORRECTED VERSION. The original implementation computed
    gamma_ij = alpha*(lam_i-lam_j)^2 + beta*(mu_i-mu_j)^2
following the original (incorrect) Theorem 8.2 / Lemma 8.1 of the AOAD
manuscript. That identity does not hold: the correct linearization gives
    rho_ij = -(alpha+beta) * (lam_i - lam_j) * (mu_i - mu_j)
which is sign-indefinite rather than a sum of squares. In particular, a
regular commuting equilibrium is locally stable iff rho_ij > 0 for ALL
i != j, which holds iff the eigenvalues of LG_star and LJ_star are sorted
in OPPOSITE relative order ("anti-monotone"); if they are sorted in the
SAME order ("co-monotone"), some rho_ij < 0 and the equilibrium is a
saddle (unstable in at least one mode).

See the corrected manuscript, Lemma 8.1 / Theorem 8.2 (renumbered), for the
derivation and numerical verification.
"""

import numpy as np


def extract_local_rates(LG_star: np.ndarray, LJ_star: np.ndarray, alpha: float, beta: float):
    """Compute the corrected pairwise linearized rates rho_ij at a regular
    commuting equilibrium (LG_star, LJ_star).

    Returns
    -------
    rho_min : float
        The minimal pairwise rate, min_{i!=j} rho_ij. If rho_min > 0, the
        equilibrium is locally exponentially stable (modulo the joint-
        conjugation zero mode) with asymptotic rate
        E(t) <= K * exp(-2 * rho_min * t).
        If rho_min <= 0, the equilibrium has at least one unstable
        (growing) mode and is a saddle point of the dynamics.
    rho_elements : list of float
        All pairwise rates rho_ij for i < j, for inspection.
    is_stable : bool
        True iff every rho_ij > 0, i.e. the eigenvalues of LG_star and
        LJ_star are anti-monotonically ordered (opposite relative order).

    Notes
    -----
    LG_star and LJ_star are assumed to (approximately) commute; eigenvalues
    are read off LG_star directly and LJ_star is diagonalized in LG_star's
    eigenbasis (paired_J below).
    """
    n = LG_star.shape[0]
    if LG_star.shape != (n, n) or LJ_star.shape != (n, n):
        raise ValueError("LG_star and LJ_star must be square matrices of the same shape")

    vals_G, vecs_G = np.linalg.eigh(LG_star)
    paired_J = np.diag(vecs_G.T @ LJ_star @ vecs_G)

    rho_elements = []
    for i in range(n):
        for j in range(i + 1, n):
            rho = -(alpha + beta) * (vals_G[i] - vals_G[j]) * (paired_J[i] - paired_J[j])
            rho_elements.append(rho)

    if not rho_elements:
        raise ValueError("Matrix dimension must be at least 2 to compute a local rate")

    rho_min = min(rho_elements)
    is_stable = all(r > 0 for r in rho_elements)

    return rho_min, rho_elements, is_stable


def extract_asymptotic_spectral_gap(LG_star: np.ndarray, LJ_star: np.ndarray, alpha: float, beta: float) -> float:
    """DEPRECATED: kept only so old scripts fail loudly instead of silently
    using the wrong formula. Use `extract_local_rates` instead.
    """
    raise NotImplementedError(
        "extract_asymptotic_spectral_gap implemented the incorrect rate "
        "formula gamma_ij = alpha*(li-lj)^2 + beta*(mi-mj)^2 from the "
        "original (incorrect) Theorem 8.2. Use extract_local_rates(...) "
        "instead, which implements the corrected rate "
        "rho_ij = -(alpha+beta)*(li-lj)*(mi-mj)."
    )
