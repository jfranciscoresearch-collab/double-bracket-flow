"""
plot2_local_rates.py
---------------------
CORRECTED VERSION. The original script used
`extract_asymptotic_spectral_gap`, implementing the incorrect formula
gamma_ij = alpha*(li-lj)^2 + beta*(mi-mj)^2 from the original (incorrect)
Theorem 8.2. It also used the equilibrium pairing lam=(2.5, 0.0, -2.5),
mu=(-1.5, 0.5, 1.0), which (descending vs. ascending) happens to be
anti-monotone and therefore IS genuinely stable under the corrected
theory too -- so the original figure's qualitative shape (decay) was not
wrong, but the rate annotation and the underlying formula were.

This script uses `extract_local_rates` (the corrected formula) for the
rate annotation, and keeps the same example equilibrium since it is a
genuine, verified instance of the stable (anti-monotone) case.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aoad.dynamics import aoad_vector_field, compute_commutator_energy
from aoad.analysis import extract_local_rates


def main():
    n = 3
    alpha, beta = 1.2, 0.8

    Lambda = np.diag([2.5, 0.0, -2.5])
    M = np.diag([-1.5, 0.5, 1.0])
    perturbation = np.array([[0, 1e-3, 0], [1e-3, 0, 1e-3], [0, 1e-3, 0]])

    L_G0 = Lambda + perturbation
    L_J0 = M + perturbation

    t_span = (0, 4.0)
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    y0 = np.concatenate([L_G0.flatten(), L_J0.flatten()])

    sol = solve_ivp(
        aoad_vector_field,
        t_span,
        y0,
        args=(n, alpha, beta),
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-13,
    )

    energies = np.array([
        compute_commutator_energy(
            sol.y[0 : n * n, i].reshape((n, n)),
            sol.y[n * n :, i].reshape((n, n)),
        )
        for i in range(len(sol.t))
    ])

    rho_min, rho_elements, is_stable = extract_local_rates(
        sol.y[0 : n * n, -1].reshape((n, n)),
        sol.y[n * n :, -1].reshape((n, n)),
        alpha,
        beta,
    )
    print(f"Corrected local rate: rho_min = {rho_min:.4f}  (stable = {is_stable})")
    print(f"All pairwise rates: {[round(r, 3) for r in rho_elements]}")

    t_tail = sol.t
    reference_slope = energies[0] * np.exp(-2 * rho_min * t_tail)

    plt.figure(figsize=(7, 4.5))
    plt.semilogy(sol.t, energies, 'g-', linewidth=2, label='Local Empirical Energy $E(t)$')
    plt.semilogy(
        t_tail,
        reference_slope,
        'k--',
        linewidth=1.5,
        label=fr'Asymptotic rate $e^{{-2\rho_{{\min}}t}}$, $\rho_{{\min}}={rho_min:.2f}$',
    )
    plt.title('Local Asymptotic Rate Verification (corrected)')
    plt.xlabel('Time ($t$)')
    plt.ylabel('Commutator Energy (Log Scale)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot2_local_rates.png', dpi=150, bbox_inches='tight')
    print('Saved plot2_local_rates.png')


if __name__ == '__main__':
    main()
