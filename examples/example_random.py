"""
example_random.py
-----------------
Verifies the main structural results of the AOAD system for random real
symmetric initial data.

CORRECTED VERSION. The original script checked monotone decay of the
commutator energy E(t), following the original (incorrect) Theorem 4.2.
That claim is false: E(t) is sign-indefinite and can increase
substantially along real trajectories. This script instead checks the
corrected, unconditional Lyapunov identity for the alignment
    W(t) = <LG(t), LJ(t)>
which IS monotone non-increasing for all alpha, beta > 0. It also reports
E(t)'s behavior honestly, including any transient increases, so the
contrast with the original (false) claim is visible directly in the
output and the figure.

Checks:
1. Monotone decay of the alignment W(t)  [corrected Lyapunov function]
2. Isospectrality of L_G and L_J
3. Convergence toward a commuting equilibrium [L_G*, L_J*] ~ 0
4. (For comparison) non-monotonicity of E(t), reported honestly

Reference:
    Jerina Jeneth C. Francisco
    "Coupled Double-Bracket Gradient Flows on Adjoint Orbits" (corrected version)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aoad import run_aoad, verify_isospectrality, commutator_energy
from aoad.lyapunov import compute_alignment

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

print("=" * 60)
print(f"Random {n}x{n} AOAD experiment (alpha={alpha}, beta={beta})")
print("=" * 60)
print(f"Initial energy    E(0) = {commutator_energy(LG0, LJ0):.6f}")
print(f"Initial alignment W(0) = {compute_alignment(LG0, LJ0):.6f}")

# -----------------------------------------------------------------------
# Integrate
# -----------------------------------------------------------------------
_, t, LG_t, LJ_t, E_t = run_aoad(
    LG0, LJ0,
    alpha=alpha, beta=beta,
    t_span=(0, 20),
    t_eval=np.linspace(0, 20, 1000)
)

W_t = np.array([compute_alignment(LG_t[i], LJ_t[i]) for i in range(len(t))])

print(f"Final   energy    E(T) = {E_t[-1]:.2e}")
print(f"Final   alignment W(T) = {W_t[-1]:.6f}")

# -----------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------
verify_isospectrality(LG0, LG_t, LJ0, LJ_t)

# Corrected monotonicity check: W, not E
diffs_W = np.diff(W_t)
n_violations_W = np.sum(diffs_W > 1e-10)
print(f"\nAlignment W(t) monotonicity check (corrected Lyapunov function):")
print(f"  W non-increasing steps violated: {n_violations_W} / {len(diffs_W)}")
print(f"  {'PASSED' if n_violations_W == 0 else 'FAILED'} (this should always pass, for any alpha,beta>0)")

# For contrast: report E(t)'s actual (non-monotone) behavior honestly
diffs_E = np.diff(E_t)
n_increasing_E = np.sum(diffs_E > 1e-10)
print(f"\nCommutator energy E(t) behavior (NOT claimed to be monotone):")
print(f"  E increasing steps: {n_increasing_E} / {len(diffs_E)}")
if n_increasing_E > 0:
    print(f"  Max increase in a single step: {diffs_E.max():.6f}")
print(f"  (Some increase is expected and is NOT a bug -- see corrected manuscript, Section 4.)")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.semilogy(t, E_t, color='#2ca02c', lw=2)
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel(r'Commutator Energy $E(t)$', fontsize=12)
ax.set_title(f'$E(t)$ — NOT monotone (log scale)', fontsize=12)
ax.grid(True, which='both', linestyle='--', alpha=0.5)

ax1 = axes[1]
ax1.plot(t, W_t, color='#9467bd', lw=2)
ax1.set_xlabel('Time $t$', fontsize=12)
ax1.set_ylabel(r'Alignment $W(t)=\langle L_G,L_J\rangle$', fontsize=12)
ax1.set_title(r'$W(t)$ — monotone non-increasing (corrected)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = axes[2]
eig_G = np.array([np.sort(np.linalg.eigvalsh(LG_t[i])) for i in range(len(t))])
for k in range(n):
    ax2.plot(t, eig_G[:, k], lw=1.5, label=f'$\\lambda_{k+1}$')
ax2.set_xlabel('Time $t$', fontsize=12)
ax2.set_ylabel('Eigenvalue', fontsize=12)
ax2.set_title(r'Eigenvalues of $L_G(t)$ (constant)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/random_nxn.png', dpi=150, bbox_inches='tight')
print("\nFigure saved to figures/random_nxn.png")
plt.show()
