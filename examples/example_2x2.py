"""
example_2x2.py
--------------
Reproduces the explicit 2x2 example from Section 8 of:

    Jerina Jeneth C. Francisco
    "Coupled Double-Bracket Gradient Flows on Adjoint Orbits:
     Commutator Energy Dissipation, Algebraic Confinement,
     and Asymptotic Alignment" (2026)

System:
    L_G = diag(1, -1),   L_J = [[a, b], [b, -a]]

Theory predicts (alpha = beta = 1):
    b(t) = b0 * exp(-4t)
    E(t) = 4 * b0^2 * exp(-8t)

with gamma_min = 4, so E(t) <= K * exp(-2 * gamma_min * t) = K * exp(-8t).

This script:
1. Integrates the AOAD system numerically.
2. Plots E(t) against the theoretical prediction.
3. Verifies isospectrality.
4. Prints the empirical vs predicted decay rate.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aoad import run_aoad, verify_isospectrality

# -----------------------------------------------------------------------
# Initial conditions (Section 8)
# -----------------------------------------------------------------------
a0 = 0.0
b0 = 1.0          # perturbation magnitude; theory predicts b(t) = b0 exp(-4t)

LG0 = np.array([[1.0,  0.0],
                [0.0, -1.0]])

LJ0 = np.array([[a0,  b0],
                [b0, -a0]])

alpha = 1.0
beta  = 1.0

# -----------------------------------------------------------------------
# Integrate
# -----------------------------------------------------------------------
t_span = (0, 3.0)
t_eval = np.linspace(0, 3.0, 1000)

sol, t, LG_t, LJ_t, E_t = run_aoad(
    LG0, LJ0,
    alpha=alpha, beta=beta,
    t_span=t_span,
    t_eval=t_eval
)

# -----------------------------------------------------------------------
# Theoretical prediction
# -----------------------------------------------------------------------
gamma_min  = alpha * (1 - (-1))**2 + beta * 0**2   # = 4
E0_theory  = 4 * b0**2                              # = 4
E_theory   = E0_theory * np.exp(-2 * gamma_min * t) # = 4 exp(-8t)
b_theory   = b0 * np.exp(-4 * t)

# -----------------------------------------------------------------------
# Isospectrality check
# -----------------------------------------------------------------------
print("=" * 55)
print("Section 8: Explicit 2x2 AOAD Example")
print("=" * 55)
verify_isospectrality(LG0, LG_t, LJ0, LJ_t)

# -----------------------------------------------------------------------
# Empirical decay rate (late-time log-linear fit)
# -----------------------------------------------------------------------
mask = t > 1.0
log_E = np.log(E_t[mask] + 1e-30)
coeffs = np.polyfit(t[mask], log_E, 1)
empirical_rate = -coeffs[0]
print(f"\nDecay rate verification:")
print(f"  Predicted  2*gamma_min = {2*gamma_min:.4f}")
print(f"  Empirical  rate        = {empirical_rate:.4f}")
print(f"  Relative error         = {abs(empirical_rate - 2*gamma_min)/(2*gamma_min)*100:.4f}%")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: E(t) log scale with theoretical overlay
ax = axes[0]
ax.semilogy(t, E_t,      color='#2ca02c', lw=2,   label=r'Numerical $E(t)$')
ax.semilogy(t, E_theory, color='black',   lw=1.5,
            linestyle='--', label=r'Theory: $4b_0^2 e^{-8t}$')
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel(r'Commutator Energy $E(t)$', fontsize=12)
ax.set_title('Section 8: Energy Decay (log scale)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.text(0.05, 0.08,
        r'$\gamma_{\min}=4$, predicted rate $e^{-8t}$',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Right: b(t) comparison
b_t = np.array([LJ_t[i, 0, 1] for i in range(len(t))])
ax2 = axes[1]
ax2.plot(t, b_t,      color='#1f77b4', lw=2,   label=r'Numerical $b(t)$')
ax2.plot(t, b_theory, color='black',   lw=1.5,
         linestyle='--', label=r'Theory: $b_0 e^{-4t}$')
ax2.set_xlabel('Time $t$', fontsize=12)
ax2.set_ylabel(r'Off-diagonal entry $b(t)$', fontsize=12)
ax2.set_title(r'Off-diagonal decay: $b(t) = b_0 e^{-4t}$', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/section8_2x2.png', dpi=150, bbox_inches='tight')
print("\nFigure saved to figures/section8_2x2.png")
plt.show()
