"""Test that pyperfectforesight replicates Dynare's perfect foresight RBC solution.

Model: basic RBC with CRRA utility, Cobb-Douglas production, one-time TFP shock.
Source .mod file: https://git.dynare.org/JohannesPfeifer/dynare/-/blob/6.x/examples/perfect_foresight_rbc.mod

Dynare .mod equations (Dynare end-of-period capital convention, i.e. k(-1) = k_{t-1}):
    c + k = z*k(-1)^alpha + (1-delta)*k(-1)        [resource constraint]
    c^(-sigma) = beta*(alpha*z(+1)*k^(alpha-1) + 1-delta)*c(+1)^(-sigma)  [Euler]

Parameters: alpha=0.5, sigma=0.5, delta=0.02, beta=1/1.05
Shock: z=1.2 in Dynare period 1 (Python index t=0); z=1 otherwise (initval and terminal).
Simulation length: T=200 periods.
"""

import os
import numpy as np
import pytest

from pyperfectforesight import v, process_model, solve_perfect_foresight

# ---------------------------------------------------------------------------
# Parameters (baked in numerically — no SymPy parameter symbols needed)
# ---------------------------------------------------------------------------
ALPHA = 0.5
SIGMA = 0.5
DELTA = 0.02
BETA = 1 / 1.05

# ---------------------------------------------------------------------------
# Steady state at z=1
# ---------------------------------------------------------------------------
Z_SS = 1.0
K_SS = ((1 / BETA - (1 - DELTA)) / (Z_SS * ALPHA)) ** (1 / (ALPHA - 1))
C_SS = Z_SS * K_SS**ALPHA - DELTA * K_SS

SS = np.array([C_SS, K_SS])

# ---------------------------------------------------------------------------
# Model equations (Dynare lag notation)
# ---------------------------------------------------------------------------
EQ_RESOURCE = (
    v("c", 0) + v("k", 0)
    - v("z", 0) * v("k", -1) ** ALPHA
    - (1 - DELTA) * v("k", -1)
)
EQ_EULER = (
    v("c", 0) ** (-SIGMA)
    - BETA
    * (ALPHA * v("z", 1) * v("k", 0) ** (ALPHA - 1) + 1 - DELTA)
    * v("c", 1) ** (-SIGMA)
)

VARS_DYN = ["c", "k"]
VARS_EXO = ["z"]
PARAMS = {}

T = 200

# ---------------------------------------------------------------------------
# Dynare reference path (produced by running perfect_foresight_rbc.mod in
# Dynare 6.2 and exporting oo_.endo_simul columns 2..201 to a CSV file).
# Shape: (200, 2) — columns are [c, k], rows are periods 1..200.
# ---------------------------------------------------------------------------
_REF_PATH = os.path.join(os.path.dirname(__file__), "dynare_ref_output", "perfect_foresight_rbc_output.csv")


@pytest.fixture(scope="module")
def dynare_ref():
    if not os.path.exists(_REF_PATH):
        pytest.fail(f"Dynare reference file not found: {_REF_PATH}")
    ref = np.loadtxt(_REF_PATH, delimiter=",")
    if ref.shape != (T, 2):
        pytest.fail(f"Unexpected shape {ref.shape} in {_REF_PATH}; expected ({T}, 2).")
    return ref  # shape (T, 2): [c, k]


@pytest.fixture(scope="module")
def model():
    return process_model([EQ_RESOURCE, EQ_EULER], VARS_DYN, vars_exo=VARS_EXO)


@pytest.fixture(scope="module")
def exog_path():
    """z=1.2 in Dynare period 1 (Python index t=0), z=1.0 thereafter."""
    path = np.ones((T, 1)) * Z_SS
    path[0, 0] = 1.2
    return path


@pytest.fixture(scope="module")
def solution(model, exog_path):
    X0 = np.tile(SS, (T, 1))
    k_neg1 = np.array([K_SS])  # k_{-1}: capital at pre-shock steady state
    return solve_perfect_foresight(
        T, PARAMS, SS, model, VARS_DYN, X0,
        exog_path=exog_path,
        initial_state=k_neg1,
        stock_var_indices=[1],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_solver_converges(solution):
    """Python solver finds a solution."""
    assert solution.success
    assert np.linalg.norm(solution.fun) < 1e-6


def test_initial_period_response(solution):
    """Positive TFP shock raises both c and k in period 1."""
    path = solution.x.reshape(T, -1)
    c0, k0 = path[0, 0], path[0, 1]
    assert c0 > C_SS, "consumption should rise on impact"
    assert k0 > K_SS, "capital should rise on impact"


def test_convergence_to_steady_state(solution):
    """Path converges back to the original steady state by the end."""
    path = solution.x.reshape(T, -1)
    np.testing.assert_allclose(path[-1], SS, atol=1e-3)


def test_matches_dynare_reference(solution, dynare_ref):
    """Python solution matches Dynare 6.2 output to within 1e-4."""
    path = solution.x.reshape(T, -1)  # (200, 2): [c, k]
    np.testing.assert_allclose(path, dynare_ref, atol=1e-4,
                               err_msg="Python path diverges from Dynare reference")
