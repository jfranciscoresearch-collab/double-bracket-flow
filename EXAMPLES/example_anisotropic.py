"""
example_anisotropic.py
----------------------
Demonstrates local stability/instability and the corrected convergence
rate near a regular commuting equilibrium of the AOAD system, for
several (alpha, beta) pairs.

CORRECTED VERSION. The original script used the formula

    gamma_min = min_{i!=j} [ alpha*(lambda_i-lambda_j)^2 + beta*(mu_i-mu_j)^2 ]

from the original (incorrect) Lemma 8.1 / Theorem 8.2, which is always
positive and so always predicts local stability. This is false: the
correct linearization gives

    rho_ij = -(alpha+beta) * (lambda_i - lambda_j) * (mu_i - mu_j)

which is sign-indefinite. A regular commuting equilibrium is locally
stable iff rho_ij > 0 for ALL i != j, which holds iff the eigenvalues of
Lambda and M are sorted in OPPOSITE relative order ("anti-monotone"); if
sorted in the SAME order ("co-monotone"), the equilibrium is a saddle.

This script demonstrates BOTH regimes explicitly, using the SAME
eigenvalues lambda, paired in two different orders against the same mu
values, so the only difference between the two runs is the pairing.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aoad import run_aoad
from aoad.analysis import extract_local_rates

n = 4
lam = np.array([3.0, 1.0, -1.0, -3.0])          # fixed, descending

# Same multiset of mu-values, two different pairings against lam:
mu_anti = np.array([-2.0, -0.5, 0.5, 2.0])      # ascending -> OPPOSITE order to lam -> anti-monotone (stable)
mu_co   = np.array([2.0, 0.5, -0.5, -2.0])      # descending -> SAME order as lam -> co-monotone (unstable)

np.random.seed(7)


def small_perturbation(L, eps=0.01):
    n = L.shape[0]
    dS = np.random.randn(n, n) * eps
    dS = 0.5 * (dS + dS.T)
    return L + dS


alpha, beta = 1.3, 0.7

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

print("=" * 70)
print("Anisotropic AOAD: stability depends on relative eigenvalue ordering")
print("=" * 70)

for ax, mu, label, color in [
    (axes[0], mu_anti, "anti-monotone (opposite order) -> predicted STABLE", "#2ca02c"),
    (axes[1], mu_co,   "co-monotone (same order) -> predicted UNSTABLE", "#d62728"),
]:
    Lambda = np.diag(lam)
    M = np.diag(mu)

    rho_min, rho_elements, is_stable = extract_local_rates(Lambda, M, alpha, beta)
    print(f"\n{label}")
    print(f"  mu = {mu.tolist()}")
    print(f"  rho_ij values: {[round(r, 3) for r in rho_elements]}")
    print(f"  rho_min = {rho_min:.4f}   predicted stable = {is_stable}")

    np.random.seed(7)
    LG0 = small_perturbation(Lambda)
    LJ0 = small_perturbation(M)

    t_eval = np.linspace(0, 3.0, 1500)
    _, t, _, _, E_t = run_aoad(LG0, LJ0, alpha=alpha, beta=beta, t_span=(0, 3.0), t_eval=t_eval)

    ax.semilogy(t, E_t, color=color, lw=2, label=f'Numerical $E(t)$')
    if is_stable:
        E_theory = E_t[0] * np.exp(-2 * rho_min * t)
        ax.semilogy(t, E_theory, 'k--', lw=1.5, label=fr'$e^{{-2\rho_{{\min}}t}}$, $\rho_{{\min}}={rho_min:.2f}$')
    ax.set_xlabel('Time $t$', fontsize=11)
    ax.set_ylabel(r'Commutator Energy $E(t)$', fontsize=11)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

print("\n" + "=" * 70)
print("Note: with eps=0.01 the unstable case will eventually leave the")
print("linear regime and may still decay at LATE times, as the nonlinear")
print("dynamics redirects the trajectory toward a different, stable")
print("equilibrium elsewhere in the commuting variety (see corrected")
print("manuscript, Theorem 8.2 discussion). The early-time GROWTH is the")
print("signature to look for, not the eventual long-run fate.")

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/anisotropic.png', dpi=150, bbox_inches='tight')
print("\nFigure saved to figures/anisotropic.png")
plt.show()
