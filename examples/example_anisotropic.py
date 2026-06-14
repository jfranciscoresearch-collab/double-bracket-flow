"""
example_anisotropic.py
----------------------
Demonstrates the effect of anisotropic weighting (alpha != beta)
on convergence rate in the AOAD system.

The linearized theory (Lemma 6.1) predicts that the convergence rate
near a regular commuting equilibrium is:

    gamma_min = min_{i != j} [ alpha*(lambda_i - lambda_j)^2
                              + beta*(mu_i - mu_j)^2 ]

This example runs the same initial data under several (alpha, beta) pairs
and compares empirical decay rates to the theoretical prediction.

Reference:
    Jerina Jeneth C. Francisco
    "Coupled Double-Bracket Gradient Flows on Adjoint Orbits" (2026)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aoad import run_aoad

# -----------------------------------------------------------------------
# Initial conditions: small perturbation of a commuting equilibrium
# -----------------------------------------------------------------------
Lambda = np.diag([3.0, 1.0, -1.0, -3.0])
M      = np.diag([2.0, 0.5, -0.5, -2.0])

np.random.seed(7)

def small_perturbation(L, eps=0.05):
    n = L.shape[0]
    dS = np.random.randn(n, n) * eps
    dS = 0.5 * (dS + dS.T)
    return L + dS

LG0 = small_perturbation(Lambda)
LJ0 = small_perturbation(M)

n = 4

# -----------------------------------------------------------------------
# Theoretical gamma_min for given (alpha, beta)
# -----------------------------------------------------------------------
lam = np.diag(Lambda)
mu  = np.diag(M)

def gamma_min_theory(alpha, beta):
    gammas = []
    for i, j in combinations(range(n), 2):
        g = alpha * (lam[i] - lam[j])**2 + beta * (mu[i] - mu[j])**2
        gammas.append(g)
    return min(gammas)

# -----------------------------------------------------------------------
# Run for multiple (alpha, beta) pairs
# -----------------------------------------------------------------------
param_pairs = [
    (1.0, 1.0),
    (2.0, 1.0),
    (1.0, 2.0),
    (3.0, 0.5),
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

t_eval = np.linspace(0, 5, 800)

print("=" * 60)
print("Anisotropic weighting: predicted vs empirical gamma_min")
print("=" * 60)
print(f"{'(alpha, beta)':<18} {'Theory 2*gamma_min':>20} {'Empirical rate':>16}")
print("-" * 60)

fig, ax = plt.subplots(figsize=(9, 6))

for (alpha, beta), color in zip(param_pairs, colors):
    _, t, _, _, E_t = run_aoad(
        LG0.copy(), LJ0.copy(),
        alpha=alpha, beta=beta,
        t_span=(0, 5),
        t_eval=t_eval
    )

    # Empirical rate from log-linear fit in late regime
    mask = t > 2.0
    log_E = np.log(E_t[mask] + 1e-30)
    coeffs = np.polyfit(t[mask], log_E, 1)
    emp_rate = -coeffs[0]

    theory_rate = 2 * gamma_min_theory(alpha, beta)
    label = f'α={alpha}, β={beta}  [pred={theory_rate:.1f}, emp={emp_rate:.1f}]'

    ax.semilogy(t, E_t, color=color, lw=2, label=label)
    print(f"  ({alpha}, {beta}){'':10} {theory_rate:>20.4f} {emp_rate:>16.4f}")

print("=" * 60)

ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel(r'Commutator Energy $E(t)$', fontsize=12)
ax.set_title('Effect of Anisotropic Weighting on Convergence Rate', fontsize=13)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/anisotropic.png', dpi=150, bbox_inches='tight')
print("\nFigure saved to figures/anisotropic.png")
plt.show()
