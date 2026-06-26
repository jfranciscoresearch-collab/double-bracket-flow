import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Ensure Python can look backward and find the 'aoad' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aoad.dynamics import aoad_vector_field, compute_commutator_energy
from aoad.lyapunov import compute_alignment


def main():
    n = 3
    alpha, beta = 1.2, 0.8
    np.random.seed(42)

    A, B = np.random.randn(n, n), np.random.randn(n, n)
    L_G0 = 0.5 * (A + A.T)
    L_J0 = 0.5 * (B + B.T)

    t_span = (0, 15.0)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
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

    energies = []
    alignments = []
    for i in range(len(sol.t)):
        LG = sol.y[0 : n * n, i].reshape((n, n))
        LJ = sol.y[n * n :, i].reshape((n, n))
        energies.append(compute_commutator_energy(LG, LJ))
        alignments.append(compute_alignment(LG, LJ))

    # Note: E(t) is NOT claimed to be monotone (see corrected manuscript,
    # Section 4); it is plotted here only to show overall relaxation to
    # the commuting variety. The corrected, genuinely monotone Lyapunov
    # function is W(t) = <LG(t), LJ(t)>, plotted alongside for contrast.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(sol.t, energies, 'b-', linewidth=2, label='Empirical Energy $E(t)$')
    axes[0].set_title('Global Relaxation to Commuting Variety')
    axes[0].set_xlabel('Time ($t$)')
    axes[0].set_ylabel('Commutator Energy $E(t)$ (not claimed monotone)')
    axes[0].grid(True, ls='--')
    axes[0].legend()

    axes[1].plot(sol.t, alignments, color='#9467bd', linewidth=2, label=r'Alignment $W(t)=\langle L_G,L_J\rangle$')
    axes[1].set_title('Corrected Lyapunov Function (monotone)')
    axes[1].set_xlabel('Time ($t$)')
    axes[1].set_ylabel('$W(t)$')
    axes[1].grid(True, ls='--')
    axes[1].legend()

    plt.tight_layout()
    import os
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/plot1_global_dynamics.png', dpi=150, bbox_inches='tight')
    print('Saved plot1_global_dynamics.png')


if __name__ == '__main__':
    main()
