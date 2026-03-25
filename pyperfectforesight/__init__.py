"""
pyperfectforesight: A minimal Dynare-style perfect foresight solver in Python

This package provides tools for solving perfect foresight dynamic economic models.
"""

from pyperfectforesight.__version__ import __version__
from pyperfectforesight.core import (
    # Utilities
    v,

    # Model processing
    lead_lag_incidence,
    is_static,
    eliminate_static,
    local_blocks,
    process_model,

    # Auxiliary variables
    solve_auxiliary_nested,
    compute_auxiliary_variables,

    # Steady state
    compute_steady_state_numerical,

    # Solver components
    residual,
    sparse_jacobian,
    append_terminal_conditions,

    # High-level solver
    solve_perfect_foresight,
    solve_perfect_foresight_homotopy,
    solve_perfect_foresight_expectation_errors,

    # Initial guess helper
    make_initial_guess,
)

__all__ = [
    "__version__",
    # Utilities
    "v",
    # Model processing
    "lead_lag_incidence",
    "is_static",
    "eliminate_static",
    "local_blocks",
    "process_model",
    # Auxiliary variables
    "solve_auxiliary_nested",
    "compute_auxiliary_variables",
    # Steady state
    "compute_steady_state_numerical",
    # Solver components
    "residual",
    "sparse_jacobian",
    "append_terminal_conditions",
    # High-level solver
    "solve_perfect_foresight",
    "solve_perfect_foresight_homotopy",
    "solve_perfect_foresight_expectation_errors",
    # Initial guess helper
    "make_initial_guess",
]
