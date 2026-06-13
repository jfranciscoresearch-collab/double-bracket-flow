"""
Example 1: Section 8 Artifact (2×2 System)
===========================================

Reproduces the explicit solution for the 2×2 case with:
  - b(t) = b₀ e^{-4t}    (off-diagonal element)
  - E(t) = 4b₀² e^{-8t}  (commutator energy)

This example demonstrates exact agreement between the empirical trajectory
and the predicted theoretical rates, providing a clean validation of the
AOAD dynamics for the simplest nontrivial case.

Reference:
    Francisco, J.J.C. (2026). Section 8: Explicit 2×2 Verification.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure Python can find the aoad package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import aoad


def main():
    """Run 2×2 example and verify predicted rates."""
    # Initial conditions: canonical 2×2 from Section 7
    LG0 = np.array([[1.0, 0.0], [0.0, -1.0]])
    LJ0 = np.array([[0.0, 1.0], [1.0, 0.0]])

    b0 = LJ0[0, 1]  # Initial off-diagonal element
    E0 = aoad.commutator_energy(LG0, LJ0)

    print("=" * 70)
    print("EXAMPLE 1: Section 8 Artifact (2×2 System)")
    print("=" * 70)
    print(f"Initial off-diagonal element: b₀ = {b0:.6f}")
    print(f"Initial commutator energy: E₀ = {E0:.6f}")
    print(f"Predicted: E(t) = 4b₀² e^{{-8t}} = {4 * b0**2:.6f} e^{{-8t}}")
    print()

    # Run simulation with α = β = 1 for the canonical case
    result = aoad.run_aoad(
        LG0, LJ0,
        alpha=1.0,
        beta=1.0,
        t_max=5.0,
        num_points=500,
    )

    t = result['t']
    energies = result['energy']
    LG_traj = result['LG_traj']
    LJ_traj = result['LJ_traj']

    # Extract off-diagonal elements b(t) from LJ
    b_traj = np.array([LJ_traj[i, 0, 1] for i in range(len(t))])

    # Theoretical predictions
    b_theory = b0 * np.exp(-4.0 * t)
    E_theory = 4.0 * b0**2 * np.exp(-8.0 * t)

    # Compute errors
    b_error = np.abs(b_traj - b_theory)
    E_error = np.abs(energies - E_theory)

    print("Empirical vs Predicted Decay Rates")
    print("-" * 70)
    print(f"{'Time':>8} | {'b(t) Empirical':>14} | {'b(t) Theory':>14} | {'Error':>12}")
    print("-" * 70)
    for i in [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]:
        print(f"{t[i]:8.3f} | {b_traj[i]:14.6e} | {b_theory[i]:14.6e} | {b_error[i]:12.2e}")

    print()
    print(f"{'Time':>8} | {'E(t) Empirical':>14} | {'E(t) Theory':>14} | {'Error':>12}")
    print("-" * 70)
    for i in [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]:
        print(f"{t[i]:8.3f} | {energies[i]:14.6e} | {E_theory[i]:14.6e} | {E_error[i]:12.2e}")

    print()
    print(f"Max relative error in b(t):  {np.max(b_error / (np.abs(b_theory) + 1e-15)):.2e}")
    print(f"Max relative error in E(t):  {np.max(E_error / (np.abs(E_theory) + 1e-15)):.2e}")
    print()

    # Verify isospectrality
    verify_result = aoad.verify_isospectrality(
        LG_traj, LJ_traj,
        alpha=1.0, beta=1.0,
    )

    print("Verification Results")
    print("-" * 70)
    print(f"Monotone dissipation:        {verify_result['is_monotone_dissipation']}")
    print(f"Nearly commuting (final):    {verify_result['is_nearly_commuting']}")
    print(f"Total energy decrease:       {verify_result['energy_decrease']:.6e}")
    print(f"Final commutator norm:       {verify_result['final_commutator_norm']:.6e}")
    print()

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # Plot b(t)
    axes[0].semilogy(t, np.abs(b_traj), 'b-', linewidth=2.0, label='Empirical $b(t)$')
    axes[0].semilogy(t, np.abs(b_theory), 'r--', linewidth=1.5, label='Theory: $b_0 e^{-4t}$')
    axes[0].set_ylabel('$|b(t)|$', fontsize=12)
    axes[0].set_title('Section 8 Artifact: 2×2 Canonical Example', fontsize=13, fontweight='bold')
    axes[0].grid(True, which='both', linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=11)

    # Plot E(t)
    axes[1].semilogy(t, energies, 'g-', linewidth=2.0, label='Empirical $E(t)$')
    axes[1].semilogy(t, E_theory, 'r--', linewidth=1.5, label='Theory: $4b_0^2 e^{-8t}$')
    axes[1].set_xlabel('Time ($t$)', fontsize=12)
    axes[1].set_ylabel('Commutator Energy $E(t)$', fontsize=12)
    axes[1].grid(True, which='both', linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('example_2x2.png', dpi=150, bbox_inches='tight')
    print(f"Figure saved: example_2x2.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
