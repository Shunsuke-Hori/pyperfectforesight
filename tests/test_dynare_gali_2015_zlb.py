"""
test_dynare_gali_2015_zlb.py

Regression test: pyperfectforesight vs Dynare 6.x/7.x for the Gali (2015)
Chapter 5.4.2 optimal monetary policy at the ZLB under commitment.

Model variables (deviations from SS except interest rates):
  pi    — inflation
  x     — output gap
  i     — nominal interest rate (quarterly %)
  p     — price level
  xi_1  — Lagrange multiplier on NKPC
  xi_2  — Lagrange multiplier on IS curve

Exogenous:
  r_nat — natural rate of interest (quarterly %)

Shock: r_nat drops from 1 to −1 for Dynare periods 1:6 (pypf periods 0:5),
       then returns to 1. T = 50 periods.

ZLB formulation:
  Dynare: [mcp = 'i > 0'] xi_2/siggma = 0   solved with lmmcp
  pypf:   sp.Min(xi_2/siggma, i) = 0          (NCP / Fischer-min)

Reference CSV: tests/dynare_ref_output/gali_2015_zlb_output.csv
  Shape (50, 6), columns: [pi, x, i, p, xi_1, xi_2]
  Rows: Dynare periods 1..50  ==  pypf periods 0..49

Generate with:
  dynare gali_2015_zlb   (in tests/dynare_ref_output/)
  then copy the output CSV to tests/dynare_ref_output/gali_2015_zlb_output.csv

Tolerance: 1e-4 (NCP vs lmmcp formulations may differ by O(1e-5)).
"""

import os
import numpy as np
import pytest
import sympy as sp

from pyperfectforesight import p, v, process_model, solve_perfect_foresight

# ── Reference data ────────────────────────────────────────────────────────────

_REF_CSV = os.path.join(
    os.path.dirname(__file__), "dynare_ref_output", "gali_2015_zlb_output.csv"
)

T = 50
VARS_DYN = ['pi', 'x', 'i', 'p', 'xi_1', 'xi_2']
VARS_EXO = ['r_nat']
VARS_PARAMS = ["alppha", "betta", "siggma", "varphi", "epsilon", "theta"]

# ── Parameters ────────────────────────────────────────────────────────────────

alppha_s  = p("alppha")
betta_s   = p("betta")
siggma_s  = p("siggma")
varphi_s  = p("varphi")
epsilon_s = p("epsilon")
theta_s   = p("theta")

PARAMS = {
    alppha_s:  0.25,
    betta_s:   0.99,
    siggma_s:  1.0,
    varphi_s:  5.0,
    epsilon_s: 9.0,
    theta_s:   0.75,
}

# ── Model-local definitions ───────────────────────────────────────────────────

Omega    = (1 - alppha_s) / (1 - alppha_s + alppha_s * epsilon_s)
lambda_e = (1 - theta_s) * (1 - betta_s * theta_s) / theta_s * Omega
kappa    = lambda_e * (siggma_s + (varphi_s + alppha_s) / (1 - alppha_s))
vartheta = kappa / epsilon_s

# ── Variable symbols ──────────────────────────────────────────────────────────

pi_0,   pi_p   = v('pi',   0), v('pi',   1)
x_0,    x_p    = v('x',    0), v('x',    1)
i_0            = v('i',    0)
p_0,    p_m    = v('p',    0), v('p',   -1)
xi_1_0, xi_1_m = v('xi_1', 0), v('xi_1', -1)
xi_2_0, xi_2_m = v('xi_2', 0), v('xi_2', -1)
r_nat_0        = v('r_nat', 0)

# ── Equations ─────────────────────────────────────────────────────────────────

EQUATIONS = [
    pi_0 - betta_s * pi_p - kappa * x_0,                                 # NKPC
    x_0 - x_p + 1/siggma_s * (i_0 - pi_p - r_nat_0),                    # IS
    pi_0 + xi_1_0 - xi_1_m - 1/(betta_s * siggma_s) * xi_2_m,           # FOC pi
    vartheta * x_0 - kappa * xi_1_0 + xi_2_0 - 1/betta_s * xi_2_m,     # FOC x
    sp.Min(xi_2_0 / siggma_s, i_0),                                       # ZLB (NCP)
    p_0 - p_m - pi_0,                                                     # price level
]

# ── Steady state ──────────────────────────────────────────────────────────────
# pi=0, x=0, i=r_nat=1 (quarterly %), p=0, xi_1=0, xi_2=0

SS = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

# ── Exogenous path ────────────────────────────────────────────────────────────

R_NAT_SS = 1.0
_exog_path = np.full((T, 1), R_NAT_SS)
_exog_path[0:6, 0] = -1.0   # shock at pypf periods 0-5 == Dynare periods 1-6


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    return process_model(
        EQUATIONS, VARS_DYN,
        vars_exo=VARS_EXO,
        vars_params=VARS_PARAMS,
    )


@pytest.fixture(scope="module")
def solution(model):
    initial_state = SS[[3, 4, 5]]   # p, xi_1, xi_2 at pre-period-0 (all zero)
    sol = solve_perfect_foresight(
        T, PARAMS, SS, model, VARS_DYN,
        exog_path=_exog_path,
        initial_state=initial_state,
    )
    assert sol.success, f"Solver failed: {sol.message}"
    return sol.x.reshape(T, len(VARS_DYN))


@pytest.fixture(scope="module")
def dynare_ref():
    if not os.path.exists(_REF_CSV):
        pytest.fail(
            f"Dynare reference file not found: {_REF_CSV}\n"
            "Run `dynare gali_2015_zlb` in tests/dynare_ref_output/ to generate it."
        )
    ref = np.loadtxt(_REF_CSV, delimiter=",")
    if ref.shape != (T, len(VARS_DYN)):
        pytest.fail(
            f"Unexpected shape {ref.shape}; expected ({T}, {len(VARS_DYN)})."
        )
    return ref   # shape (50, 6): columns [pi, x, i, p, xi_1, xi_2]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_solver_converges(solution):
    """Solver finds a converged solution."""
    # solution fixture already asserts success; this makes it an explicit test
    assert solution is not None


def test_zlb_binds_on_impact(solution):
    """Nominal rate is at the zero lower bound on impact."""
    i_path = solution[:, 2]
    assert i_path[0] < 1e-6, f"ZLB should bind at t=0; got i={i_path[0]:.6f}"


def test_zlb_not_binding_after_shock(solution):
    """Nominal rate is positive well after the ZLB episode ends."""
    i_path = solution[:, 2]
    # By period 15 the economy should have left the ZLB
    assert i_path[15] > 0.01, (
        f"Expected i > 0 by t=15; got i={i_path[15]:.6f}"
    )


def test_convergence_to_steady_state(solution):
    """Stationary variables converge back to the steady state by the end of the horizon.

    p (price level) is a unit root in this model — it has no lead in any
    equation, so the BVP terminal condition never constrains it.  p is
    excluded from the convergence check.
    """
    # Indices: 0=pi, 1=x, 2=i, 4=xi_1, 5=xi_2  (skip 3=p)
    stationary_idx = [0, 1, 2, 4, 5]
    np.testing.assert_allclose(
        solution[-1, stationary_idx], SS[stationary_idx], atol=1e-3,
        err_msg="Final period must be near the steady state (excluding price level p)",
    )


def test_commitment_premium(solution):
    """Verify key qualitative features of the Gali (2015) Ch.5.4.2 commitment solution.

    Under commitment the planner front-loads the make-up promise: it announces
    positive inflation IN FUTURE PERIODS WHILE THE ZLB IS STILL BINDING (t=1..6).
    This raises expected inflation, lowers the real rate, and partially offsets
    the demand shortfall.  Key features:

      1. On impact (t=0): deflation and recession (pi < 0, x < 0, i = 0).
      2. Positive inflation during ZLB (t=1..6) — the commitment premium.
      3. ZLB binds longer than the natural-rate shock (>6 periods): the planner
         keeps rates at zero even after r_nat recovers to stimulate further.
    """
    pi_path = solution[:, 0]
    x_path  = solution[:, 1]
    i_path  = solution[:, 2]

    # 1. On impact: deflation, recession, ZLB binding
    assert pi_path[0] < 0, f"Inflation negative on impact; got {pi_path[0]:.4f}"
    assert x_path[0]  < 0, f"Output gap negative on impact; got {x_path[0]:.4f}"
    assert i_path[0] == pytest.approx(0.0, abs=1e-6), "ZLB binds on impact"

    # 2. Positive inflation during ZLB after t=0 (commitment make-up)
    assert pi_path[1] > 0, (
        f"Commitment: inflation should be positive at t=1 while ZLB binds; "
        f"got {pi_path[1]:.4f}"
    )

    # 3. ZLB binds for more than 6 periods (natural-rate shock is 6 periods)
    zlb_duration = sum(1 for t in range(T) if i_path[t] < 1e-6)
    assert zlb_duration > 6, (
        f"ZLB should bind > 6 periods under commitment; got {zlb_duration}"
    )


def test_matches_dynare_reference(solution, dynare_ref):
    """pypf solution matches Dynare lmmcp output to within 1e-4.

    The NCP min-reformulation (pypf) and the lmmcp complementarity solver
    (Dynare) are equivalent formulations; numerical solutions may differ by
    O(Newton tolerance).
    """
    # Check all variables across all periods
    np.testing.assert_allclose(
        solution, dynare_ref, atol=1e-4,
        err_msg="pypf path diverges from Dynare lmmcp reference",
    )


def test_matches_dynare_selected_periods(solution, dynare_ref):
    """Spot-check key periods against Dynare reference (tighter output for debugging)."""
    tol = 1e-4
    check_periods = [1, 2, 3, 4, 5, 6, 7, 10, 15, 20, 30, 50]
    var_names = VARS_DYN
    for dynare_t in check_periods:
        pypf_t = dynare_t - 1   # Dynare 1-indexed → pypf 0-indexed
        for j, var in enumerate(var_names):
            val = solution[pypf_t, j]
            ref = dynare_ref[pypf_t, j]
            assert abs(val - ref) < tol, (
                f"t={dynare_t}, {var}: pypf={val:.8f} vs Dynare={ref:.8f} "
                f"(diff={abs(val-ref):.2e})"
            )
