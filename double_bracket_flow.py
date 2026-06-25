"""
Simulation: Invariant Lie Algebra Confinement in a Symmetric Double-Bracket Flow
=================================================================================
Numerically integrates the coupled system:

    dL_G/dt = [L_G, [L_G, L_J]]
    dL_J/dt = [L_J, [L_J, L_G]]

for n=2 with initial conditions from the paper's explicit example (Section 7).

Produces:
    - trajectory.png  : projected trajectory in h_0-basis coordinates
    - verification of Frobenius norm conservation to 6 decimal places

Reference:
    Francisco, J.J.C. (2026). Invariant Lie Algebra Confinement in a
    Symmetric Double-Bracket Flow. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

Requirements:
    numpy, scipy, matplotlib
    Install via: pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ── Commutator ────────────────────────────────────────────────────────────────

def comm(A, B):
    """Matrix commutator [A, B] = AB - BA."""
    return A @ B - B @ A


# ── Right-hand side of the ODE ─────────────────────────────────────────────────

def rhs(t, y):
    """
    Flattened state vector y = [L_G (4,), L_J (4,)] for n=2.
    Returns dy/dt.
    """
    LG = y[:4].reshape(2, 2)
    LJ = y[4:].reshape(2, 2)
    dLG = comm(LG, comm(LG, LJ))
    dLJ = comm(LJ, comm(LJ, LG))
    return np.concatenate([dLG.flatten(), dLJ.flatten()])


# ── Initial conditions (from Section 7 of the paper) ──────────────────────────

LG0 = np.array([[1.0,  0.0],
                [0.0, -1.0]])   # diag(1, -1)

LJ0 = np.array([[0.0, 1.0],
                [1.0, 0.0]])   # off-diagonal symmetric

y0 = np.concatenate([LG0.flatten(), LJ0.flatten()])


# ── Integration ────────────────────────────────────────────────────────────────

T_MAX = 20.0

sol = solve_ivp(
    rhs,
    [0, T_MAX],
    y0,
    method='RK45',
    max_step=0.01,
    rtol=1e-9,
    atol=1e-12,
)

LG_traj = sol.y[:4].T.reshape(-1, 2, 2)   # shape (N, 2, 2)
LJ_traj = sol.y[4:].T.reshape(-1, 2, 2)


# ── Norm conservation check ────────────────────────────────────────────────────

H_G_start = 0.5 * np.trace(LG_traj[0]  @ LG_traj[0])
H_G_end   = 0.5 * np.trace(LG_traj[-1] @ LG_traj[-1])
H_J_start = 0.5 * np.trace(LJ_traj[0]  @ LJ_traj[0])
H_J_end   = 0.5 * np.trace(LJ_traj[-1] @ LJ_traj[-1])

print("Frobenius norm conservation check")
print(f"  H_G: {H_G_start:.6f} → {H_G_end:.6f}  (drift = {abs(H_G_end - H_G_start):.2e})")
print(f"  H_J: {H_J_start:.6f} → {H_J_end:.6f}  (drift = {abs(H_J_end - H_J_start):.2e})")


# ── Project L_G(t) onto orthonormal h_0-basis ──────────────────────────────────

# Gram-Schmidt on {LG0, LJ0}
b1 = LG0 / np.linalg.norm(LG0, 'fro')
b2 = LJ0 - np.tensordot(LJ0, b1, axes=2) * b1
b2 = b2  / np.linalg.norm(b2,  'fro')

x = np.array([np.tensordot(M, b1, axes=2) for M in LG_traj])
y = np.array([np.tensordot(M, b2, axes=2) for M in LG_traj])


# ── Plot ───────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
})

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.plot(x, y, linewidth=1.8, color='steelblue', label="Trajectory")
ax.scatter(x[0],  y[0],  color='green', zorder=5, s=60, label=r"$t = 0$")
ax.scatter(x[-1], y[-1], color='red',   zorder=5, s=60, label=r"$t = t_{\max}$")

ax.set_title("Projected Lie-Confined Double-Bracket Flow", fontsize=12)
ax.set_xlabel(r"$\mathfrak{h}_0$ Basis Coordinate 1", fontsize=11)
ax.set_ylabel(r"$\mathfrak{h}_0$ Basis Coordinate 2", fontsize=11)
ax.legend(frameon=True)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("trajectory.png", dpi=150, bbox_inches='tight')
print("Figure saved as trajectory.png")
