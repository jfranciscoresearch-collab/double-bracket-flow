"""
example_random.py
-----------------
Verifies the main structural results of the AOAD paper
for random real symmetric initial data.

Checks:
1. Monotone decay of commutator energy E(t)
2. Isospectrality of L_G and L_J
3. Convergence toward a commuting equilibrium [L_G*, L_J*] ~ 0

Reference:
    Jerina Jeneth C. Francisco
    "Coupled Double-Bracket Gradient Flows on Adjoint Orbits" (2026)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aoad import run_aoad, verify_isospectrality, commutator_energy

np.random.seed(42)

def random_symmetric(n, scale=1.0):
    A = np.random.randn(n, n) * scale
    return 0.5 * (A + A.T)

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------
n     = 4
alpha = 1.0
beta  = 1.0

LG0 = random_symmetric(n)
LJ0 = random_symmetric(n)

print("=" * 55)
print(f"Random {n}x{n} AOAD experiment (alpha={alpha}, beta={beta})")
print("=" * 55)
print(f"Initial energy E(0) = {commutator_energy(LG0, LJ0):.6f}")

# -----------------------------------------------------------------------
# Integrate
# -----------------------------------------------------------------------
_, t, LG_t, LJ_t, E_t = run_aoad(
    LG0, LJ0,
    alpha=alpha, beta=beta,
    t_span=(0, 20),
    t_eval=np.linspace(0, 20, 1000)
)

print(f"Final   energy E(T) = {E_t[-1]:.2e}")

# -----------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------
verify_isospectrality(LG0, LG_t, LJ0, LJ_t)

# Monotonicity check
diffs = np.diff(E_t)
n_violations = np.sum(diffs > 1e-12)
print(f"\nMonotonicity check:")
print(f"  Energy non-decreasing steps: {n_violations} / {len(diffs)}")
print(f"  {'PASSED' if n_violations == 0 else 'FAILED (numerical noise expected near zero)'}")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.semilogy(t, E_t, color='#2ca02c', lw=2)
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel(r'Commutator Energy $E(t)$', fontsize=12)
ax.set_title(f'Energy Decay — Random {n}×{n} Symmetric Matrices', fontsize=13)
ax.grid(True, which='both', linestyle='--', alpha=0.5)

ax2 = axes[1]
eig_G = np.array([np.sort(np.linalg.eigvalsh(LG_t[i])) for i in range(len(t))])
for k in range(n):
    ax2.plot(t, eig_G[:, k], lw=1.5, label=f'$\\lambda_{k+1}$')
ax2.set_xlabel('Time $t$', fontsize=12)
ax2.set_ylabel('Eigenvalue', fontsize=12)
ax2.set_title(r'Eigenvalues of $L_G(t)$ (should be constant)', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/random_nxn.png', dpi=150, bbox_inches='tight')
print("\nFigure saved to figures/random_nxn.png")
plt.show()
