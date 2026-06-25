"""
example_2x2.py
--------------
Reproduces the corrected explicit 2x2 example from the corrected manuscript.

CORRECTED VERSION. The original script claimed L_G(t) stays fixed at
diag(1, -1) for all time, with only a single off-diagonal entry b(t) of
L_J evolving according to b_dot = -4b. This does NOT solve the stated
system: direct substitution shows dL_G/dt != 0 for the claimed solution
(L_G genuinely rotates under the flow). This script instead uses the
corrected angle-reduction:

    L_G = r_G * (cos(a) * sigma_z + sin(a) * sigma_x)
    L_J = r_J * (cos(b) * sigma_z + sin(b) * sigma_x)

with r_G, r_J constant (isospectrality) and phi := a - b satisfying the
single scalar ODE

    dphi/dt = 4 * r_G * r_J * (alpha + beta) * sin(phi)

with exact solution tan(phi(t)/2) = tan(phi(0)/2) * exp(k*t),
k = 4*r_G*r_J*(alpha+beta), and E(t) = 4 * r_G^2 * r_J^2 * sin(phi(t))^2.

This script verifies the closed form against direct numerical
integration of the full 2x2 matrix system.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aoad import run_aoad, verify_isospectrality

# -----------------------------------------------------------------------
# Initial conditions (corrected example)
# -----------------------------------------------------------------------
rG, rJ = 1.0, 1.0
a0, b0 = 0.0, 1.0          # initial angles; phi0 = a0 - b0 = -1.0
alpha, beta = 1.3, 0.7

sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])


def angle_to_matrix(r, theta):
    return r * (np.cos(theta) * sigma_z + np.sin(theta) * sigma_x)


LG0 = angle_to_matrix(rG, a0)
LJ0 = angle_to_matrix(rJ, b0)

# -----------------------------------------------------------------------
# Integrate the FULL 2x2 matrix system
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
# Corrected closed-form (scalar angle ODE) prediction
# -----------------------------------------------------------------------
phi0 = a0 - b0
k = 4 * rG * rJ * (alpha + beta)
phi_theory = 2 * np.arctan(np.tan(phi0 / 2) * np.exp(k * t))
E_theory = 4 * (rG * rJ * np.sin(phi_theory)) ** 2

# -----------------------------------------------------------------------
# Isospectrality check
# -----------------------------------------------------------------------
print("=" * 60)
print("Corrected 2x2 AOAD Example (angle reduction)")
print("=" * 60)
verify_isospectrality(LG0, LG_t, LJ0, LJ_t)

# -----------------------------------------------------------------------
# Compare full-system E(t) against the closed-form prediction
# -----------------------------------------------------------------------
rel_error = np.max(np.abs(E_t - E_theory)) / (np.max(E_theory) + 1e-30)
print(f"\nClosed-form verification:")
print(f"  max |E_numeric - E_theory| / max(E_theory) = {rel_error:.3e}")
print(f"  {'PASSED' if rel_error < 1e-6 else 'FAILED'} (expected agreement to integrator precision)")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogy(t, E_t, color='#2ca02c', lw=2, label=r'Numerical $E(t)$ (full $2\times2$ system)')
ax.semilogy(t, E_theory, color='black', lw=1.5, linestyle='--',
            label=r'Closed form: $4r_G^2r_J^2\sin^2\phi(t)$')
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel(r'Commutator Energy $E(t)$', fontsize=12)
ax.set_title('Corrected $2\\times2$ Example: Energy Decay', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/section8_2x2.png', dpi=150, bbox_inches='tight')
print("\nFigure saved to figures/section8_2x2.png")
plt.show()
