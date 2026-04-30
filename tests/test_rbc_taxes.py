"""
test_rbc_taxes.py

Regression test for the RBC model with government spending shocks.
Reference data: Dynare 7.0 perfect foresight simulation (T=200).

Model: rbc_taxes_pf_dynare_clean.mod
  - g = linspace(0.1, 0, 10) at Dynare periods 1-10, then g=0
  - initval: distorted SS at g=0.1
  - endval:  baseline SS at g=0

Timing convention:
  Dynare period t  <->  pyperfectforesight period (t-1)
  exog_path[t]     =    g at pypf period t  =  g at Dynare period (t+1)

Tolerance: 1e-6 (Dynare reference values stored at 15 significant digits).
"""

import os
import numpy as np
import pytest
import sympy as sp

from pyperfectforesight import v, p, process_model, solve_perfect_foresight

# ── Reference data ────────────────────────────────────────────────────────────

_REF_CSV = os.path.join(os.path.dirname(__file__), "dynare_ref_output", "rbc_taxes_output.csv")


def _load_ref():
    """Load Dynare reference as dict {dynare_t: {var: value}}."""
    data = np.loadtxt(_REF_CSV, delimiter=",", skiprows=1)
    # columns: dynare_t, k, g, i, n
    return {int(row[0]): {"k": row[1], "g": row[2], "i": row[3], "n": row[4]}
            for row in data}


# ── Model definition ──────────────────────────────────────────────────────────

beta_s  = p("beta")
delta_s = p("delta")
alpha_s = p("alpha")
rho_s   = p("rho")
sigma_s = p("sigma")
eta_s   = p("eta")
chi_s   = p("chi")
zbar_s  = p("zbar")

VARS_PARAMS = ["beta", "delta", "alpha", "rho", "sigma", "eta", "chi", "zbar"]

VARS_DYN = ["z", "k", "g", "i", "n"]
VARS_EXO = ["e_g"]

z_0, k_0, g_0, i_0, n_0 = [v(x, 0) for x in VARS_DYN]
z_m, k_m, i_m = v("z", -1), v("k", -1), v("i", -1)
z_p, k_p, g_p, i_p, n_p = [v(x, 1) for x in VARS_DYN]
e_g_0 = v("e_g", 0)


def _rk(z, k, n):
    return alpha_s * z * (n / k) ** (1 - alpha_s)


def _c(z, k, g, i, n):
    return z * k**alpha_s * n ** (1 - alpha_s) - i - g


def _w(z, k, n):
    return (1 - alpha_s) * z * (k / n) ** alpha_s


EQUATIONS = [
    1 - beta_s * (_c(z_0, k_0, g_0, i_0, n_0) / _c(z_p, k_p, g_p, i_p, n_p)) ** sigma_s
      * (1 - delta_s + _rk(z_p, k_p, n_p)),
    chi_s * n_0**eta_s * _c(z_0, k_0, g_0, i_0, n_0) ** sigma_s - _w(z_0, k_0, n_0),
    z_0 - (1 - rho_s) * zbar_s - rho_s * z_m,
    k_0 - (1 - delta_s) * k_m - i_m,
    g_0 - e_g_0,
]

# ── Calibration ───────────────────────────────────────────────────────────────

BETA_V, DELTA_V, ALPHA_V = 0.99, 0.025, 0.33
RHO_V, SIGMA_V, ETA_V, ZBAR_V = 0.80, 1.0, 1.0, 1.0

_rk_ss = 1 / BETA_V - 1 + DELTA_V
_n_ss  = 0.33
_h     = (_rk_ss / ALPHA_V) ** (1 / (1 - ALPHA_V))
_k_ss  = _n_ss / _h
_y_ss  = ZBAR_V * _k_ss**ALPHA_V * _n_ss ** (1 - ALPHA_V)
_i_ss  = DELTA_V * _k_ss
_c_ss  = _y_ss - _i_ss
_w_ss  = (1 - ALPHA_V) * ZBAR_V * (_k_ss / _n_ss) ** ALPHA_V
CHI_V  = _w_ss / (_c_ss**SIGMA_V * _n_ss**ETA_V)

# Baseline SS (e_g = 0)
SS_BASELINE = np.array([ZBAR_V, _k_ss, 0.0, _i_ss, _n_ss])

PARAMS = {
    beta_s: BETA_V, delta_s: DELTA_V, alpha_s: ALPHA_V,
    rho_s: RHO_V, sigma_s: SIGMA_V, eta_s: ETA_V,
    chi_s: CHI_V, zbar_s: ZBAR_V,
}


def _ss_with_g(g_val):
    """Analytical steady state at exogenous government spending g_val."""
    A = ZBAR_V / _h**ALPHA_V
    B = DELTA_V / _h
    a = CHI_V * (A - B)
    b = -CHI_V * g_val
    c = -(1 - ALPHA_V) * A
    n = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    k = n / _h
    return np.array([ZBAR_V, k, g_val, DELTA_V * k, n])


# ── Shared simulation ─────────────────────────────────────────────────────────

T = 200
G_INIT = 0.1
_exo_g = np.linspace(G_INIT, 0.0, 10)   # 10 values: 0.1, 8/90, ..., 0


def _run_sim(model):
    exog_path = np.zeros((T, 1))
    exog_path[:10, 0] = _exo_g           # periods 0-9; 10+ stay at 0

    ss_init = _ss_with_g(G_INIT)

    return solve_perfect_foresight(
        T, PARAMS, model, VARS_DYN,
        endval=SS_BASELINE,
        exog_path=exog_path,
        ss_initial=ss_init,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    return process_model(EQUATIONS, VARS_DYN, vars_exo=VARS_EXO, vars_params=VARS_PARAMS)


@pytest.fixture(scope="module")
def solution(model):
    sol = _run_sim(model)
    assert sol.success, f"Solver failed: {sol.message}"
    return sol.x.reshape(T, len(VARS_DYN))


@pytest.fixture(scope="module")
def dynare_ref():
    return _load_ref()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_equation_count(model):
    """process_model must keep the processed dynamic system square."""
    assert len(model["dynamic_eqs"]) == len(model["vars_dyn"])


def test_solver_converges(model):
    sol = _run_sim(model)
    assert sol.success, sol.message


def test_matches_dynare_reference(solution, dynare_ref):
    """
    Compare against Dynare 7.0 at selected periods.
    Timing: Dynare period t == pypf period (t-1) == solution row (t-1).
    Tolerance: 1e-6.
    """
    tol = 1e-6
    # Check a spread of periods through the simulation
    check_periods = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200]
    for dynare_t in check_periods:
        pypf_t = dynare_t - 1          # convert Dynare 1-indexed to pypf 0-indexed
        row = solution[pypf_t]
        ref = dynare_ref[dynare_t]
        _, k, g, i, n = row
        assert abs(k - ref["k"]) < tol, f"t={dynare_t}: k={k:.8f} vs Dynare {ref['k']:.8f}"
        assert abs(g - ref["g"]) < tol, f"t={dynare_t}: g={g:.8f} vs Dynare {ref['g']:.8f}"
        assert abs(i - ref["i"]) < tol, f"t={dynare_t}: i={i:.8f} vs Dynare {ref['i']:.8f}"
        assert abs(n - ref["n"]) < tol, f"t={dynare_t}: n={n:.8f} vs Dynare {ref['n']:.8f}"


def test_terminal_convergence(solution):
    """Final period must be near the baseline SS (g=0)."""
    _, k_f, g_f, i_f, n_f = solution[-1]
    assert abs(g_f) < 1e-10
    assert abs(k_f - _k_ss) < 1e-4
    assert abs(n_f - _n_ss) < 1e-4
