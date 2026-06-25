"""Corrected Lyapunov function for the AOAD system.

The original manuscript claimed the commutator energy
    E(LG, LJ) = 0.5 * ||[LG, LJ]||_F^2
decreases monotonically along every trajectory (Theorem 4.2), via a
claimed gradient-flow structure (Lemma 2.1). Both claims are false: the
system is not a gradient flow of E under any constant-weighted Frobenius
metric (including alpha=beta), and E is sign-indefinite in time -- it can,
and does, increase substantially along real trajectories.

The correct, unconditional Lyapunov identity is for the cross-alignment
    W(LG, LJ) = <LG, LJ> = tr(LG @ LJ)
which satisfies, exactly and for all alpha, beta > 0:
    dW/dt = -(alpha + beta) * ||[LG, LJ]||_F^2 = -2*(alpha+beta)*E <= 0.

This module implements W and the dissipation check, and provides a
drop-in replacement for any script that previously checked monotonicity
of E.
"""

import numpy as np


def comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix commutator [A, B] = AB - BA."""
    return A @ B - B @ A


def compute_alignment(LG: np.ndarray, LJ: np.ndarray) -> float:
    """W(LG, LJ) = <LG, LJ> = tr(LG @ LJ). The corrected Lyapunov function."""
    return float(np.sum(LG * LJ))


def compute_commutator_energy(LG: np.ndarray, LJ: np.ndarray) -> float:
    """E = 0.5 * ||[LG, LJ]||_F^2. NOT monotone; retained for diagnostics
    and for reproducing the (corrected) figures, which show E alongside W.
    """
    C = comm(LG, LJ)
    return 0.5 * float(np.sum(C ** 2))


def verify_alignment_dissipation(LG_t: np.ndarray, LJ_t: np.ndarray, tol: float = 1e-8) -> bool:
    """Check that W(t) = <LG(t), LJ(t)> is non-increasing along a computed
    trajectory, to within numerical tolerance `tol`.

    Parameters
    ----------
    LG_t, LJ_t : (N, n, n) arrays of matrices along the trajectory.
    tol : allowed numerical-noise tolerance for "non-increasing".

    Returns
    -------
    bool : True if W(t) is non-increasing (within tol) at every step.
    """
    W_t = np.array([compute_alignment(LG_t[i], LJ_t[i]) for i in range(len(LG_t))])
    diffs = np.diff(W_t)
    n_violations = int(np.sum(diffs > tol))
    print("Alignment (W) monotonicity check:")
    print(f"  W(0)   = {W_t[0]:.6f}")
    print(f"  W(end) = {W_t[-1]:.6f}")
    print(f"  Increasing steps: {n_violations} / {len(diffs)}")
    print(f"  {'PASSED' if n_violations == 0 else 'FAILED'} (tolerance {tol:.0e})")
    return n_violations == 0
