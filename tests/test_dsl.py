"""Tests for the declaration DSL (endog / exog / params).

Injection via PyFrame_LocalsToFast only works at module level (where f_locals
IS the module dict).  Tests that verify injection behaviour use exec() in an
explicit global namespace, which is equivalent.  All other tests use the
returned objects directly.
"""

import numpy as np
import sympy as sp
import pytest

from pyperfectforesight import (
    endog, exog, params, reset_registry, Model,
    v, p,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rbc_ss(alpha, beta):
    k = (alpha * beta) ** (1 / (1 - alpha))
    c = k ** alpha - k
    return k, c


def _exec(code, extra=None):
    """Run code in a fresh global namespace that includes package imports."""
    ns = {
        'endog': endog, 'exog': exog, 'params': params,
        'reset_registry': reset_registry, 'Model': Model,
        'v': v, 'sp': sp, 'np': np,
    }
    if extra:
        ns.update(extra)
    exec(code, ns)
    return ns


# ===========================================================================
# 1.  _Var bracket notation
# ===========================================================================

def test_var_bracket_returns_correct_symbols():
    reset_registry()
    k_var = endog("k")
    assert k_var[-1] == v("k", -1)
    assert k_var[0]  == v("k",  0)
    assert k_var[1]  == v("k",  1)


def test_var_bare_equals_lag0_in_arithmetic():
    reset_registry()
    k_var = endog("k")
    assert k_var + 1   == k_var[0] + 1
    assert 2 * k_var   == 2 * k_var[0]
    assert k_var ** 2  == k_var[0] ** 2
    assert -k_var      == -k_var[0]


def test_var_sympy_integration():
    """_Var works inside sp.exp and sp.sympify."""
    reset_registry()
    z_var = exog("z")
    assert sp.exp(z_var[0]) == sp.exp(v("z", 0))


def test_var_single_name_returns_var_not_list():
    reset_registry()
    result = endog("k")
    # single name → _Var, not a list
    assert result[-1] == v("k", -1)


def test_var_multi_name_returns_list():
    reset_registry()
    result = endog("k c")
    assert len(result) == 2
    k_v, c_v = result
    assert k_v[0] == v("k", 0)
    assert c_v[-1] == v("c", -1)


def test_params_single_name_returns_symbol():
    reset_registry()
    sym = params("rho")
    assert sym == sp.Symbol("rho")


def test_params_multi_name_returns_list():
    reset_registry()
    result = params("alpha beta")
    assert len(result) == 2
    assert result[0] == sp.Symbol("alpha")
    assert result[1] == sp.Symbol("beta")


# ===========================================================================
# 2.  Frame injection — tested via exec in a global namespace
# ===========================================================================

def test_endog_injects_into_module_scope():
    ns = _exec("""
reset_registry()
endog("x y")
result_x = x[0]
result_y = y[-1]
""")
    assert ns['result_x'] == v("x", 0)
    assert ns['result_y'] == v("y", -1)


def test_exog_injects_into_module_scope():
    ns = _exec("""
reset_registry()
exog("z")
result_z = z[0]
""")
    assert ns['result_z'] == v("z", 0)


def test_params_injects_symbols_into_module_scope():
    ns = _exec("""
reset_registry()
params("alpha beta")
result_alpha = alpha
result_beta  = beta
""")
    assert ns['result_alpha'] == sp.Symbol("alpha")
    assert ns['result_beta']  == sp.Symbol("beta")


# ===========================================================================
# 3.  Registry accumulation and reset
# ===========================================================================

def test_registry_accumulates_across_calls():
    reset_registry()
    endog("k")
    endog("c")
    assert set(endog.__module__ and __import__('pyperfectforesight.core',
               fromlist=['_registry'])._registry['endog']) == {"k", "c"}


def test_reset_registry_clears_state():
    reset_registry()
    endog("k")
    reset_registry()
    with pytest.raises(ValueError, match="endog registry is empty"):
        Model([v("k", 0)])


def test_model_reads_vars_dyn_from_registry():
    reset_registry()
    endog("k c")
    eq_euler = v("c", 0)**(-1) - 0.99 * 0.36 * v("k", 0)**(0.36-1) * v("c", 1)**(-1)
    eq_kacc  = v("k", 0) - v("k", -1)**0.36 + v("c", 0)
    model = Model([eq_euler, eq_kacc])
    assert set(model.vars_dyn) == {"k", "c"}


def test_model_reads_vars_exo_from_registry():
    reset_registry()
    endog("k c")
    exog("z")
    eq_euler = v("c", 0)**(-1) - 0.99 * 0.36 * v("k", 0)**(0.36-1) * v("c", 1)**(-1)
    eq_kacc  = v("k", 0) - sp.exp(v("z", 0)) * v("k", -1)**0.36 + v("c", 0)
    model = Model([eq_euler, eq_kacc])
    assert "z" in model.vars_exo


def test_model_reads_vars_params_from_registry():
    reset_registry()
    endog("k c")
    params("alpha beta")
    eq_euler = v("c", 0)**(-1) - p("beta") * p("alpha") * v("k", 0)**(p("alpha")-1) * v("c", 1)**(-1)
    eq_kacc  = v("k", 0) - v("k", -1)**p("alpha") + v("c", 0)
    model = Model([eq_euler, eq_kacc])
    assert set(model.vars_params) == {"alpha", "beta"}


def test_model_explicit_vars_dyn_overrides_registry():
    reset_registry()
    endog("k c")
    eq_euler = v("c", 0)**(-1) - 0.99 * 0.36 * v("k", 0)**(0.36-1) * v("c", 1)**(-1)
    eq_kacc  = v("k", 0) - v("k", -1)**0.36 + v("c", 0)
    model = Model([eq_euler, eq_kacc], vars_dyn=["c", "k"])
    assert model.vars_dyn[0] == "c"


# ===========================================================================
# 4.  End-to-end: DSL with return values → Model → solve
# ===========================================================================

def test_dsl_rbc_solve():
    """Return-value form of the DSL → equations → Model → solve."""
    reset_registry()
    k_v, c_v = endog("k c")
    alpha_s, beta_s = params("alpha beta")

    eq_euler = c_v[0]**(-1) - beta_s * alpha_s * k_v[0]**(alpha_s - 1) * c_v[1]**(-1)
    eq_kacc  = k_v[0] - k_v[-1]**alpha_s + c_v[0]

    PARAMS = {alpha_s: 0.36, beta_s: 0.99}
    K_SS, C_SS = _rbc_ss(0.36, 0.99)
    ss = np.array([K_SS, C_SS])

    model = Model([eq_euler, eq_kacc])
    T = 80
    k_neg1 = np.array([K_SS * 1.1])
    sol = model.solve(T, PARAMS, initial_state=k_neg1, stock_var_indices=[0], endval=ss)

    assert sol.success
    X = sol.x.reshape(T, -1)
    np.testing.assert_allclose(X[-1], ss, atol=1e-3)


def test_dsl_steady_state_with_good_initial_guess():
    reset_registry()
    k_v, c_v = endog("k c")
    alpha_s, beta_s = params("alpha beta")

    eq_euler = c_v[0]**(-1) - beta_s * alpha_s * k_v[0]**(alpha_s - 1) * c_v[1]**(-1)
    eq_kacc  = k_v[0] - k_v[-1]**alpha_s + c_v[0]

    PARAMS = {alpha_s: 0.36, beta_s: 0.99}
    K_SS, C_SS = _rbc_ss(0.36, 0.99)

    model = Model([eq_euler, eq_kacc])
    ss_num = model.steady_state(PARAMS, initial_guess=np.array([K_SS * 0.9, C_SS * 0.9]))
    np.testing.assert_allclose(np.array(ss_num), [K_SS, C_SS], atol=1e-6)


def test_dsl_with_exog_variable():
    reset_registry()
    k_v, c_v = endog("k c")
    z_v = exog("z")
    alpha_s, beta_s = params("alpha beta")

    eq_euler = c_v[0]**(-1) - beta_s * alpha_s * sp.exp(z_v[1]) * k_v[0]**(alpha_s - 1) * c_v[1]**(-1)
    eq_kacc  = k_v[0] - sp.exp(z_v[0]) * k_v[-1]**alpha_s + c_v[0]

    PARAMS = {alpha_s: 0.36, beta_s: 0.99}
    K_SS, C_SS = _rbc_ss(0.36, 0.99)
    ss = np.array([K_SS, C_SS])

    model = Model([eq_euler, eq_kacc])
    assert "z" in model.vars_exo

    T = 50
    k_neg1 = np.array([K_SS * 1.05])
    sol = model.solve(T, PARAMS,
                      initial_state=k_neg1, stock_var_indices=[0],
                      exog_path=np.zeros((T, 1)), endval=ss)
    assert sol.success
    X = sol.x.reshape(T, -1)
    np.testing.assert_allclose(X[-1], ss, atol=1e-2)


def test_symbol_keyed_params_solve():
    """{Symbol: float} PARAMS dict works end-to-end."""
    reset_registry()
    k_v, c_v = endog("k c")
    alpha_s, beta_s = params("alpha beta")

    eq_euler = c_v[0]**(-1) - beta_s * alpha_s * k_v[0]**(alpha_s - 1) * c_v[1]**(-1)
    eq_kacc  = k_v[0] - k_v[-1]**alpha_s + c_v[0]

    K_SS, C_SS = _rbc_ss(0.36, 0.99)
    ss = np.array([K_SS, C_SS])
    model = Model([eq_euler, eq_kacc])

    T = 50
    k_neg1 = np.array([K_SS * 1.1])
    sol = model.solve(T, {alpha_s: 0.36, beta_s: 0.99},
                      initial_state=k_neg1, stock_var_indices=[0], endval=ss)

    assert sol.success
    X = sol.x.reshape(T, -1)
    np.testing.assert_allclose(X[-1], ss, atol=1e-3)


# ===========================================================================
# 5.  End-to-end injection form via exec (module-level analogue)
# ===========================================================================

def test_dsl_injection_full_workflow():
    """Injection form (module-level equivalent) → Model → solve."""
    K_SS, C_SS = _rbc_ss(0.36, 0.99)
    ns = _exec("""
reset_registry()
endog("k c")
params("alpha beta")

eq_euler = c[0]**(-1) - beta * alpha * k[0]**(alpha - 1) * c[1]**(-1)
eq_kacc  = k[0] - k[-1]**alpha + c[0]

PARAMS = {alpha: 0.36, beta: 0.99}
model  = Model([eq_euler, eq_kacc])
""", extra={'K_SS': K_SS, 'C_SS': C_SS})

    model  = ns['model']
    PARAMS = ns['PARAMS']
    ss = np.array([K_SS, C_SS])
    T = 80
    k_neg1 = np.array([K_SS * 1.1])
    sol = model.solve(T, PARAMS, initial_state=k_neg1, stock_var_indices=[0], endval=ss)
    assert sol.success
    X = sol.x.reshape(T, -1)
    np.testing.assert_allclose(X[-1], ss, atol=1e-3)
