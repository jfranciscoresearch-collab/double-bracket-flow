# Coupled Double-Bracket Gradient Flows on Adjoint Orbits (AOAD)

Reference implementation accompanying the paper:

> Jerina Jeneth C. Francisco
> **Coupled Double-Bracket Gradient Flows on Adjoint Orbits: Commutator Energy,
> Alignment Dissipation, and Asymptotic Confinement** (corrected version)

---

## Important: this repository was corrected

An earlier version of both the manuscript and this repository claimed that
the commutator energy `E = 0.5*||[L_G, L_J]||_F^2` decreases monotonically
along every trajectory (the original Theorem 4.2), via a claimed gradient-flow
structure (the original Lemma 2.1), and that local convergence rates near a
regular commuting equilibrium are always positive (the original Lemma
8.1 / Theorem 8.2). **All three claims are false**, including in the isotropic
case α = β. On reworking the proofs in full, both the manuscript and this code
were corrected:

- The system is **not** a gradient flow of `E` under any constant-weighted
  Frobenius metric.
- `E(t)` is **sign-indefinite** and can increase substantially along real
  trajectories (see `examples/example_anisotropic.py`, right panel, for an
  explicit case where `E` grows by five orders of magnitude before eventually
  decaying).
- The correct, **unconditionally monotone** Lyapunov function is the
  alignment `W(L_G, L_J) = <L_G, L_J> = tr(L_G @ L_J)`, satisfying
  `dW/dt = -(alpha+beta) * ||[L_G,L_J]||_F^2 <= 0` for all α, β > 0. See
  `aoad/lyapunov.py`.
- Local stability of a regular commuting equilibrium `(Λ, M)` is **not**
  automatic. It depends on the relative ordering of the eigenvalues of Λ and
  M: stable (exponential convergence) when they are **anti-monotone**
  (opposite relative order), and unstable (saddle) when **co-monotone** (same
  relative order). See `aoad/analysis.py` (`extract_local_rates`) and
  `examples/example_anisotropic.py`, which demonstrates both regimes
  explicitly using the same eigenvalues in two different pairings.

The isospectrality, algebraic confinement, and Frobenius-norm-conservation
results (and their code) are **unaffected** by these corrections and are
unchanged.

If you previously used this repository's `extract_asymptotic_spectral_gap`
function, `aoad/__init__.py`'s old `verify_isospectrality`'s
`is_monotone_dissipation` field, or the original `example_2x2.py`,
`example_random.py`, `example_anisotropic.py`, or `plot2_local_rates.py`
scripts, please update: those implemented or asserted the incorrect formulas
above and have been replaced.

---

## Overview

This repository contains simulation code illustrating and investigating the
numerical behavior of the coupled isospectral gradient system studied in the
AOAD paper.

The implementation supports the theoretical analysis in the manuscript and
should not be interpreted as an independent proof of the mathematical
results; conversely, where the manuscript's claims were found to be
incorrect, this code was used as part of the process of finding the
corrected statements (every corrected theorem in the manuscript was verified
numerically here before being written up).

---

## Implemented Dynamics

All simulations in this repository implement the coupled AOAD system exactly
as stated in the manuscript:

```
L_G' = -α [L_G, [L_G, L_J]]
L_J' = -β [L_J, [L_J, L_G]]
```

with α, β > 0, on products of adjoint orbits of real symmetric n×n matrices.
**This vector field itself is unchanged from the original manuscript** — the
corrections concern only the claimed Lyapunov structure and convergence
theory built on top of it, not the dynamics being studied.

The core implementation is in `aoad/dynamics.py`.

---

## Reproducibility Artifacts

### Corrected 2×2 Example

The primary reproducibility artifact is a corrected explicit 2×2 system. The
original manuscript's worked example claimed `L_G(t)` stays fixed at
`diag(1,-1)` for all time, with only a single entry `b(t)` of `L_J` evolving
according to `b_dot = -4b`. This does **not** solve the stated system —
direct substitution shows `dL_G/dt != 0` for the claimed solution. The
corrected example instead uses an angle reduction,

```
L_G = r_G (cos a · σ_z + sin a · σ_x),   L_J = r_J (cos b · σ_z + sin b · σ_x)
```

which reduces the full system to a single scalar equation for `φ := a - b`:

```
dφ/dt = 4 r_G r_J (α+β) sin(φ)
```

with exact closed-form solution `tan(φ(t)/2) = tan(φ(0)/2) e^{kt}`. This is
verified directly against the full 2×2 matrix system in
`examples/example_2x2.py`, agreeing to integrator precision (~1e-9).

### Stability depends on eigenvalue ordering

`examples/example_anisotropic.py` runs the *same* eigenvalues λ, paired
against the *same* multiset of μ values in two different orders, and shows
the predicted stable/unstable outcome in each case, matching the corrected
formula

```
ρ_ij = -(α+β)(λ_i - λ_j)(μ_i - μ_j)
```

implemented in `aoad/analysis.py::extract_local_rates`.

### General n×n Simulations

The implementation also supports arbitrary real symmetric initial data and
illustrates:

- Spectral preservation along the flow (isospectrality) — unaffected by the corrections
- Evolution of the commutator energy `E = ½ ‖[L_G, L_J]‖_F²` — **not monotone**, reported honestly
- Evolution of the corrected Lyapunov function `W = <L_G, L_J>` — monotone, for any α, β
- Approach toward commuting configurations
- Dependence of local stability on the relative ordering of eigenvalues (not just on regularity)
- Confinement inside the Lie algebra generated by initial conditions — unaffected by the corrections

---

## Repository Structure

```
double-bracket-flow/
│
├── aoad/                          # Core AOAD package
│   ├── __init__.py                # run_aoad(), verify_isospectrality() — unified, see note below
│   ├── dynamics.py                # Implements the AOAD vector field (unchanged)
│   ├── lyapunov.py                # NEW: corrected Lyapunov function W and dissipation check
│   └── analysis.py                # Corrected local-rate computation (extract_local_rates)
│
├── examples/
│   ├── example_2x2.py             # Corrected explicit 2x2 example (primary artifact)
│   ├── example_random.py          # Random symmetric matrix experiment (W monotone, E reported honestly)
│   ├── example_anisotropic.py     # Stable vs. unstable pairing, same eigenvalues
│   ├── plot1_global_dynamics.py   # Global relaxation illustration (E and W side by side)
│   └── plot2_local_rates.py       # Local rate verification (corrected formula)
│
├── figures/                       # Pre-generated figures
│
├── aoad.py                        # Standalone simulator (canonical run_aoad/verify_isospectrality)
├── double_bracket_flow.py         # Companion: Lie confinement paper (separate, unweighted system; unaffected)
├── requirements.txt               # Python dependencies
└── README.md
```

**Note on `aoad/__init__.py`:** an earlier version of this file defined a
*second*, incompatible `run_aoad` and a `verify_isospectrality` that returned
an `is_monotone_dissipation` field based on the incorrect energy claim. Because
Python resolves the `aoad/` package before the sibling `aoad.py` file, this
silently shadowed the top-level module for anyone writing `from aoad import
run_aoad`. This has been fixed: `aoad/__init__.py` now re-exports the
canonical implementations from `aoad.py` directly, so there is exactly one
`run_aoad` and one `verify_isospectrality`, and the `is_monotone_dissipation`
field has been removed (use `aoad.lyapunov.verify_alignment_dissipation`
instead, which checks `W`, not `E`).

---

## Running the Code

Install dependencies:

```bash
pip install numpy scipy matplotlib
```

Reproduce the corrected explicit example:

```bash
python examples/example_2x2.py
```

Run a random symmetric matrix experiment (reports both W and E honestly):

```bash
python examples/example_random.py
```

Run the stability/instability comparison:

```bash
python examples/example_anisotropic.py
```

Reproduce the local rate verification (corrected formula):

```bash
python examples/plot2_local_rates.py
```

---

## Companion Paper

This repository also contains `double_bracket_flow.py`, the simulation code
for the companion work:

> Jerina Jeneth C. Francisco
> **Invariant Lie Algebra Confinement in a Symmetric Double-Bracket Flow** (2026)
> DOI: [10.5281/zenodo.20325811](https://zenodo.org/records/20325811)

The companion paper studies confinement and norm conservation for the
unweighted system with the **opposite** vector-field sign convention
(`L_G' = +[L_G,[L_G,L_J]]`, a non-dissipative/conservative flow, consistent
with the closed-orbit behavior shown in its figure). It does not study energy
dissipation or convergence, and is unaffected by the corrections described
above. The AOAD paper is now treated as a self-contained result; it does not
depend on the companion paper's confinement theorem, which it re-proves
independently for the (α, β)-weighted setting (see the AOAD manuscript,
Theorem on algebraic confinement).

---

## License

MIT License.
