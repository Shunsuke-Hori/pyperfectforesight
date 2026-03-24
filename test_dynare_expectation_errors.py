"""Test that pyperfectforesight replicates Dynare's perfect foresight with
expectation errors solution.

Model: basic RBC with CRRA utility, Cobb-Douglas production (same as
       perfect_foresight_rbc.mod), simulated with three surprise shocks:

  shocks block (learnt_in=1):  z=1.2 in period 1, z=1 otherwise
  shocks(learnt_in=2):         z=1.1 in period 3 (surprise revealed at t=2)
  endval(learnt_in=5):         z=1.05 permanently (surprise at t=5,
                                new terminal steady state)

Dynare .mod equations:
    c + k = z*k(-1)^alpha + (1-delta)*k(-1)        [resource constraint]
    c^(-sigma) = beta*(alpha*z(+1)*k^(alpha-1) + 1-delta)*c(+1)^(-sigma)  [Euler]

Parameters: alpha=0.5, sigma=0.5, delta=0.02, beta=1/1.05
Simulation length: T=200 periods.

Dynare reference: dynare_ref_output/perfect_foresight_expectation_errors_output.csv
  Shape: (200, 3) — columns are [c, k, z], rows are periods 1..200.
"""

import os
import numpy as np
import pytest

from dynare_python import v, process_model, solve_perfect_foresight_expectation_errors

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ALPHA = 0.5
SIGMA = 0.5
DELTA = 0.02
BETA = 1 / 1.05

# ---------------------------------------------------------------------------
# Steady states
# ---------------------------------------------------------------------------
Z1 = 1.0   # initial z
Z2 = 1.05  # permanent z from t=5

K_SS  = ((1/BETA - (1 - DELTA)) / (Z1 * ALPHA)) ** (1 / (ALPHA - 1))
C_SS  = Z1  * K_SS**ALPHA  - DELTA * K_SS

K_SS2 = ((1/BETA - (1 - DELTA)) / (Z2 * ALPHA)) ** (1 / (ALPHA - 1))
C_SS2 = Z2 * K_SS2**ALPHA - DELTA * K_SS2

SS_Z1 = np.array([C_SS,  K_SS])   # terminal SS used by segments 1 & 2
SS_Z2 = np.array([C_SS2, K_SS2])  # terminal SS used from segment 3 onward

# ---------------------------------------------------------------------------
# Model equations (identical to perfect_foresight_rbc.mod)
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
# Dynare reference path
# ---------------------------------------------------------------------------
_REF_PATH = os.path.join(
    os.path.dirname(__file__),
    "dynare_ref_output",
    "perfect_foresight_expectation_errors_output.csv",
)


@pytest.fixture(scope="module")
def dynare_ref():
    if not os.path.exists(_REF_PATH):
        pytest.fail(f"Dynare reference file not found: {_REF_PATH}")
    ref = np.loadtxt(_REF_PATH, delimiter=",")
    if ref.shape != (T, 3):
        pytest.fail(f"Unexpected shape {ref.shape}; expected ({T}, 3).")
    return ref  # columns: [c, k, z]


# ---------------------------------------------------------------------------
# Python model & solution fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    return process_model([EQ_RESOURCE, EQ_EULER], VARS_DYN, vars_exo=VARS_EXO)


@pytest.fixture(scope="module")
def solution(model):
    """
    Replicate Dynare's three-segment expectation-errors simulation.

    Segment 1 (learnt_in=1):  z=1.2 at t=1, z=1 for t=2..T
    Segment 2 (learnt_in=2):  z=1.0 at t=2, z=1.1 at t=3, z=1 for t=4..T
    Segment 3 (learnt_in=5):  z=1.05 permanently  → new terminal SS = SS_Z2
    """
    # Segment 1 belief: z=1.2 at row 0 (period 1), z=1 thereafter
    exog_1 = np.ones((T, 1)) * Z1
    exog_1[0, 0] = 1.2

    # Segment 2 belief (from t=2 onward):
    #   row 0 → period 2: z=1.0
    #   row 1 → period 3: z=1.1  (surprise learned at t=2)
    #   row 2+ → z=1.0
    exog_2 = np.ones((T, 1)) * Z1
    exog_2[1, 0] = 1.1

    # Segment 3 belief (from t=5 onward): z=1.05 permanently
    exog_3 = np.ones((T, 1)) * Z2

    news_shocks = [
        (1, exog_1),          # terminal SS still SS_Z1
        (2, exog_2),          # terminal SS still SS_Z1
        (5, exog_3, SS_Z2),   # endval override: permanent shock → SS_Z2
    ]

    X0 = np.tile(SS_Z1, (T, 1))
    k_neg1 = np.array([K_SS])  # k_{-1}: pre-simulation capital at z=1 SS

    return solve_perfect_foresight_expectation_errors(
        T, X0, PARAMS, SS_Z1, model, VARS_DYN,
        news_shocks=news_shocks,
        initial_state=k_neg1,
        stock_var_indices=[1],
        constant_simulation_length=False,  # Dynare default: shrinking window
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_solver_converges(solution):
    assert solution.success
    assert len(solution.sub_results) == 3
    for sr in solution.sub_results:
        assert sr.success


def test_convergence_to_new_steady_state(solution):
    """Path must converge to the z=1.05 steady state by period 200."""
    path = solution.x.reshape(T, -1)
    np.testing.assert_allclose(path[-1], SS_Z2, atol=1e-3)


def test_period1_shock_raises_c_and_k(solution):
    """Positive TFP shock at t=1 raises both c and k relative to SS_Z1."""
    path = solution.x.reshape(T, -1)
    assert path[0, 0] > C_SS,  "c should rise on impact of z=1.2 shock"
    assert path[0, 1] > K_SS,  "k should rise on impact"


def test_matches_dynare_reference(solution, dynare_ref):
    """Python solution must match Dynare 6.2 output to within 1e-4."""
    path = solution.x.reshape(T, -1)   # (200, 2): [c, k]
    dynare_ck = dynare_ref[:, :2]      # drop the z column
    np.testing.assert_allclose(
        path, dynare_ck, atol=1e-4,
        err_msg="Python path diverges from Dynare reference",
    )
