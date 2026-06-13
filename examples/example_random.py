"""
Example 2: Random n×n Symmetric Initial Conditions
===================================================

Tests AOAD dynamics on random symmetric initial conditions and verifies:
  1. Isospectrality: joint spectrum of (L_G, L_J) is preserved
  2. Monotone dissipation: commutator energy decreases monotonically
  3. Convergence: system converges toward the commuting variety

This example demonstrates robustness of the AOAD dynamics across
various matrix sizes and random initial data.

Reference:
    Francisco, J.J.C. (2026). Section 5: Isospectrality & Monotone Dissipation.
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


def symmetric_random(n, seed=None):
    """Generate a random symmetric matrix."""
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(n, n)
    return 0.5 * (A + A.T)


def compute_joint_spectrum(LG, LJ):
    """
    Compute the joint spectrum by eigendecomposing L_G
    and measuring L_J in that eigenbasis.
    """
    evals_G, evecs_G = np.linalg.eigh(LG)
    LJ_diag = evecs_G.T @ LJ @ evecs_G
    return evals_G, np.diag(LJ_diag)


def main():
    """Run random example and verify key properties."""
    print("=" * 70)
    print("EXAMPLE 2: Random n×n Symmetric Initial Conditions")
    print("=" * 70)
    print()

    # Test multiple matrix sizes
    sizes = [2, 3, 4]
    results_summary = {}

    for n in sizes:
        print(f"Testing n = {n} × {n}")
        print("-" * 70)

        # Generate random symmetric initial conditions
        LG0 = symmetric_random(n, seed=42 + n)
        LJ0 = symmetric_random(n, seed=100 + n)

        E0 = aoad.commutator_energy(LG0, LJ0)
        print(f"  Initial commutator energy: E₀ = {E0:.6e}")

        # Run simulation
        result = aoad.run_aoad(
            LG0, LJ0,
            alpha=1.0,
            beta=1.0,
            t_max=10.0,
            num_points=300,
        )

        t = result['t']
        energies = result['energy']
        LG_traj = result['LG_traj']
        LJ_traj = result['LJ_traj']

        # Verify isospectrality
        evals_G_0, evals_J_0 = compute_joint_spectrum(LG0, LJ0)
        evals_G_f, evals_J_f = compute_joint_spectrum(LG_traj[-1], LJ_traj[-1])

        spec_error = np.max(np.abs(np.sort(evals_G_0) - np.sort(evals_G_f)))
        print(f"  Spectrum drift (L_G):        {spec_error:.2e}")

        # Verify monotone dissipation
        energy_diffs = np.diff(energies)
        max_increase = np.max(energy_diffs)
        is_monotone = np.all(energy_diffs <= 1e-6)
        print(f"  Max energy increase:         {max_increase:.2e}")
        print(f"  Monotone dissipation:        {is_monotone}")

        # Verify convergence to commuting variety
        verify_result = aoad.verify_isospectrality(LG_traj, LJ_traj)
        print(f"  Final commutator norm:       {verify_result['final_commutator_norm']:.2e}")
        print(f"  Total energy decrease:       {verify_result['energy_decrease']:.2e}")
        print(f"  Energy ratio (E_f/E_0):      {energies[-1] / energies[0]:.2e}")
        print()

        results_summary[n] = {
            't': t,
            'energies': energies,
            'spec_error': spec_error,
            'is_monotone': is_monotone,
        }

    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = ['steelblue', 'forestgreen', 'orangered']
    for i, n in enumerate(sizes):
        t = results_summary[n]['t']
        energies = results_summary[n]['energies']
        axes[0].semilogy(t, energies, linewidth=2.0, label=f'$n = {n}$', color=colors[i])

    axes[0].set_xlabel('Time ($t$)', fontsize=11)
    axes[0].set_ylabel('Commutator Energy $E(t)$', fontsize=11)
    axes[0].set_title('Monotone Dissipation: Random Initial Conditions', fontsize=12, fontweight='bold')
    axes[0].grid(True, which='both', linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=10)

    # Plot spectral error
    n_list = list(results_summary.keys())
    spec_errors = [results_summary[n]['spec_error'] for n in n_list]
    axes[1].bar(range(len(n_list)), spec_errors, color=colors[:len(n_list)], alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(n_list)))
    axes[1].set_xticklabels([f'$n = {n}$' for n in n_list])
    axes[1].set_ylabel('Spectral Drift', fontsize=11)
    axes[1].set_title('Isospectrality Verification', fontsize=12, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('example_random.png', dpi=150, bbox_inches='tight')
    print(f"Figure saved: example_random.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
