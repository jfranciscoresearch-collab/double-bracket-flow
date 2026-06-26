"""AOAD package entrypoint and convenience simulation helpers.

CORRECTED VERSION. The previous version of this file defined a
`verify_isospectrality` helper that returned a field named
`is_monotone_dissipation`, computed by checking whether the commutator
energy E(t) is non-increasing. This baked the original manuscript's
incorrect Theorem 4.2 (false monotone energy dissipation) directly into
the package API. That field has been removed.

The previous version also defined a SECOND, incompatible `run_aoad`
function here that silently shadowed the one in the top-level `aoad.py`
module (because Python resolves the `aoad/` package before the
sibling `aoad.py` file), with a different signature (`t_max`/`num_points`
instead of `t_span`/`t_eval`) and a different return type (dict vs.
tuple). This caused `from aoad import run_aoad` to behave differently
from what the top-level module's own docstring promised, independent of
any of the mathematical corrections elsewhere in this package. This
version removes the duplicate and re-exports the top-level
implementation directly, so there is exactly one `run_aoad`.
"""

import os
import sys

import numpy as np

# Re-export the canonical implementations from the top-level aoad.py module
# (the sibling file, not this package) rather than maintaining a second,
# divergent copy here.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import importlib.util as _ilu

_toplevel_path = os.path.join(_repo_root, "aoad.py")
_spec = _ilu.spec_from_file_location("_aoad_toplevel", _toplevel_path)
_toplevel = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_toplevel)

run_aoad = _toplevel.run_aoad
commutator_energy = _toplevel.commutator_energy
commutator = _toplevel.commutator
double_bracket = _toplevel.double_bracket

from .dynamics import aoad_matrix_vector_field, aoad_vector_field, compute_commutator_energy
from .lyapunov import compute_alignment, verify_alignment_dissipation
from .analysis import extract_local_rates

__all__ = [
    "run_aoad",
    "commutator_energy",
    "commutator",
    "double_bracket",
    "aoad_matrix_vector_field",
    "aoad_vector_field",
    "compute_commutator_energy",
    "compute_alignment",
    "verify_alignment_dissipation",
    "extract_local_rates",
    "verify_isospectrality",
]


def verify_isospectrality(LG0, LG_t, LJ0, LJ_t, tol=1e-6):
    """Check that eigenvalues of L_G and L_J are preserved along the
    trajectory. Returns True if max deviation is below tol, prints a
    summary.

    NOTE: this signature matches the top-level aoad.py's
    `verify_isospectrality` (LG0, LG_t, LJ0, LJ_t), NOT the previous
    version of this file's `verify_isospectrality(LG_traj, LJ_traj,
    alpha, beta)`, which also returned an `is_monotone_dissipation`
    field based on the incorrect Theorem 4.2. That field has been
    removed; use `aoad.lyapunov.verify_alignment_dissipation` for the
    corrected (and actually correct) monotonicity check, which checks
    W(t) = <LG(t), LJ(t)>, not E(t).
    """
    return _toplevel.verify_isospectrality(LG0, LG_t, LJ0, LJ_t, tol=tol)


def global_initial_conditions(n: int = 2):
    """Return the canonical global initial conditions for AOAD examples."""
    if n != 2:
        raise ValueError("Global example only supports n=2 for the current script.")
    LG0 = np.array([[1.0, 0.0], [0.0, -1.0]])
    LJ0 = np.array([[0.0, 1.0], [1.0, 0.0]])
    return LG0, LJ0
