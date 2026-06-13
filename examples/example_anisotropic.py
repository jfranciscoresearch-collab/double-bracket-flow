"""
Example 3: Anisotropic (α,β) Parameter Sweep
============================================

Runs the same initial data under multiple (α, β) anisotropy pairs
and compares empirical decay rates to theoretical predictions from
Lemma 6.1 (γ_min, the minimal spectral gap).

This example validates the rate prediction for linearized dynamics
and demonstrates how the anisotropy parameters modulate decay rates.

Reference:
    Francisco, J.J.C. (2026). Lemma 6.1: Spectral Gap Analysis.
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
from aoad.analysis import extract_asymptotic_spectral_gap


def symmetric_random(n, seed=None):
    """Generate a random symmetric matrix."""
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(n, n)
    return 0.5 * (A + A.T)


def extract_decay_rate(t, energies, start_frac=0.5):
    """
    Extract empirical decay rate from tail of energy trajectory.
    Fits E(t) ≈ E₀ e^{-2γt} to the final portion of the data.
    """
    idx_start = int(len(t) * start_frac)
    if idx_start >= len(t) - 1:
        return None

    t_tail = t[idx_start:]
    E_tail = energies[idx_start:]

    # Remove near-zero values
    valid = E_tail > 1e-15
    if np.sum(valid) < 2:
        return None

    t_tail = t_tail[valid]
    E_tail = E_tail[valid]

    # Linear fit to log energy: log(E) ≈ log(E₀) - 2γ*t
    log_E = np.log(E_tail)
    coeffs = np.polyfit(t_tail, log_E, 1)
    decay_rate = -coeffs[0] / 2  # Recover γ from slope = -2γ

    return max(decay_rate, 0)


def main():
    """Run anisotropic sweep and compare to theory."""
    print("=" * 80)
    print("EXAMPLE 3: Anisotropic (α,β) Parameter Sweep and Rate Prediction")
    print("=" * 80)
    print()

    # Fixed initial condition
    n = 3
    LG0 = symmetric_random(n, seed=42)
    LJ0 = symmetric_random(n, seed=100)

    E0 = aoad.commutator_energy(LG0, LJ0)
    print(f"Fixed initial condition: n = {n}")
    print(f"Initial commutator energy: E₀ = {E0:.6e}")
    print()

    # Parameter sweep
    alpha_vals = [0.5, 1.0, 1.5]
    beta_vals = [0.5, 1.0, 1.5]

    print("Parameter Sweep Results")
    print("-" * 80)
    print(f"{'(α, β)':>12} | {'γ_min':>12} | {'Empirical γ':>14} | {'Ratio':>10} | "
          f"{'Decay Rate':>14}")
    print("-" * 80)

    results = {}
    for alpha in alpha_vals:
        for beta in beta_vals:
            # Run simulation
            result = aoad.run_aoad(
                LG0, LJ0,
                alpha=alpha,
                beta=beta,
                t_max=8.0,
                num_points=400,
            )

            t = result['t']
            energies = result['energy']
            LG_traj = result['LG_traj']
            LJ_traj = result['LJ_traj']

            # Compute theoretical γ_min
            try:
                gamma_min = extract_asymptotic_spectral_gap(
                    LG_traj[-1], LJ_traj[-1],
                    alpha, beta
                )
            except Exception:
                gamma_min = np.nan

            # Extract empirical decay rate
            empirical_gamma = extract_decay_rate(t, energies, start_frac=0.6)
            if empirical_gamma is None:
                empirical_gamma = np.nan

            ratio = empirical_gamma / gamma_min if (not np.isnan(gamma_min) and not np.isnan(empirical_gamma)) else np.nan

            # Final decay rate from slope
            final_slope = extract_decay_rate(t, energies, start_frac=0.7)

            results[(alpha, beta)] = {
                't': t,
                'energies': energies,
                'gamma_min': gamma_min,
                'empirical_gamma': empirical_gamma,
            }

            print(f"({alpha:.1f}, {beta:.1f})  | {gamma_min:12.6e} | {empirical_gamma:14.6e} | "
                  f"{ratio:10.4f} | {final_slope:14.6e}")

    print()

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    # Plot energies for different α, β pairs
    plot_idx = 0
    for i, alpha in enumerate([0.5, 1.5]):
        for j, beta in enumerate([0.5, 1.5]):
            ax = axes[i * 2 + j]
            key = (alpha, beta)
            t = results[key]['t']
            energies = results[key]['energies']
            gamma_min = results[key]['gamma_min']

            # Empirical trajectory
            ax.semilogy(t, energies, 'b-', linewidth=2.5, label='Empirical $E(t)$')

            # Theoretical prediction from γ_min
            if not np.isnan(gamma_min):
                E_theory = energies[0] * np.exp(-2 * gamma_min * t)
                ax.semilogy(t, E_theory, 'r--', linewidth=2, label=f'Theory: $e^{{-2\\gamma_{{min}}t}}$')

            ax.set_xlabel('Time ($t$)', fontsize=11)
            ax.set_ylabel('Energy $E(t)$', fontsize=11)
            ax.set_title(f'$(\\alpha, \\beta) = ({alpha}, {beta})$, $\\gamma_{{min}} = {gamma_min:.4e}$',
                        fontsize=11, fontweight='bold')
            ax.grid(True, which='both', linestyle='--', alpha=0.6)
            ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    plt.savefig('example_anisotropic.png', dpi=150, bbox_inches='tight')
    print(f"Figure saved: example_anisotropic.png")
    print("=" * 80)


if __name__ == '__main__':
    main()
