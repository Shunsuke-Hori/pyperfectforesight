"""Tests for the Model builder DSL (m.endog / m.exog / m.params / m.build)."""

import numpy as np
import sympy as sp
import pytest

from pyperfectforesight import Model, v


def _rbc_ss(alpha, beta):
    k = (alpha * beta) ** (1 / (1 - alpha))
    c = k ** alpha - k
    return k, c


# ===========================================================================
# 1.  _Var proxy — bracket notation and arithmetic
# ===========================================================================

def test_var_bracket_notation():
    m = Model()
    m.endog("k c")
    assert m.k[-1] == v("k", -1)
    assert m.k[0]  == v("k",  0)
    assert m.k[1]  == v("k",  1)
    assert m.c[0]  == v("c",  0)


def test_var_bare_is_lag0_in_arithmetic():
    m = Model()
    m.endog("k")
    assert m.k + 1  == m.k[0] + 1
    assert 2 * m.k  == 2 * m.k[0]
    assert m.k ** 2 == m.k[0] ** 2
    assert -m.k     == -m.k[0]


def test_var_sympy_function_integration():
    m = Model()
    m.exog("z")
    assert sp.exp(m.z[0]) == sp.exp(v("z", 0))


def test_params_returns_sympy_symbol():
    m = Model()
    m.params("alpha beta")
    assert m.alpha == sp.Symbol("alpha")
    assert m.beta  == sp.Symbol("beta")


# ===========================================================================
# 2.  Builder method chaining
# ===========================================================================

def test_builder_methods_return_self():
    m = Model()
    assert m.endog("k c") is m
    assert m.params("alpha") is m
    assert m.exog("z") is m


def test_build_returns_self():
    m = Model()
    m.endog("k c")
    eq = m.k[0] - m.k[-1]**0.36 + m.c[0]
    euler = m.c[0]**(-1) - 0.99 * 0.36 * m.k[0]**(0.36-1) * m.c[1]**(-1)
    result = m.build([euler, eq])
    assert result is m


# ===========================================================================
# 3.  Builder-mode guards
# ===========================================================================

def test_builder_methods_raise_after_build():
    m = Model()
    m.endog("k c")
    m.build([m.k[0] - m.k[-1]**0.36 + m.c[0],
             m.c[0]**(-1) - 0.99 * 0.36 * m.k[0]**(0.36-1) * m.c[1]**(-1)])
    with pytest.raises(RuntimeError, match="only available before build"):
        m.endog("x")


def test_build_without_endog_raises():
    m = Model()
    with pytest.raises(ValueError, match="No endogenous variables"):
        m.build([v("k", 0)])


def test_getattr_unknown_symbol_raises():
    m = Model()
    m.endog("k")
    with pytest.raises(AttributeError, match="no declared symbol"):
        _ = m.z   # 'z' was never declared


# ===========================================================================
# 4.  vars_dyn / vars_exo / vars_params populated after build
# ===========================================================================

def test_vars_dyn_after_build():
    m = Model()
    m.endog("k c")
    eq_euler = m.c[0]**(-1) - 0.99 * 0.36 * m.k[0]**(0.36-1) * m.c[1]**(-1)
    eq_kacc  = m.k[0] - m.k[-1]**0.36 + m.c[0]
    m.build([eq_euler, eq_kacc])
    assert set(m.vars_dyn) == {"k", "c"}


def test_vars_exo_after_build():
    m = Model()
    m.endog("k c")
    m.exog("z")
    eq_euler = m.c[0]**(-1) - 0.99 * 0.36 * m.k[0]**(0.36-1) * m.c[1]**(-1)
    eq_kacc  = m.k[0] - sp.exp(m.z[0]) * m.k[-1]**0.36 + m.c[0]
    m.build([eq_euler, eq_kacc])
    assert "z" in m.vars_exo


def test_vars_params_after_build():
    m = Model()
    m.endog("k c")
    m.params("alpha beta")
    eq_euler = m.c[0]**(-1) - m.beta * m.alpha * m.k[0]**(m.alpha-1) * m.c[1]**(-1)
    eq_kacc  = m.k[0] - m.k[-1]**m.alpha + m.c[0]
    m.build([eq_euler, eq_kacc])
    assert set(m.vars_params) == {"alpha", "beta"}


def test_symbols_accessible_after_build():
    """m.alpha etc. remain usable after build() for PARAMS dicts."""
    m = Model()
    m.endog("k c")
    m.params("alpha beta")
    eq_euler = m.c[0]**(-1) - m.beta * m.alpha * m.k[0]**(m.alpha-1) * m.c[1]**(-1)
    eq_kacc  = m.k[0] - m.k[-1]**m.alpha + m.c[0]
    m.build([eq_euler, eq_kacc])
    # Symbols are still accessible
    assert m.alpha == sp.Symbol("alpha")
    assert m.k[-1] == v("k", -1)


# ===========================================================================
# 5.  Classic API still works
# ===========================================================================

def test_classic_api_unchanged():
    eq_euler = v("c", 0)**(-1) - 0.99 * 0.36 * v("k", 0)**(0.36-1) * v("c", 1)**(-1)
    eq_kacc  = v("k", 0) - v("k", -1)**0.36 + v("c", 0)
    model = Model([eq_euler, eq_kacc], ["c", "k"])
    assert set(model.vars_dyn) == {"c", "k"}
    assert model.vars_params == []


# ===========================================================================
# 6.  End-to-end: builder → solve
# ===========================================================================

def test_builder_rbc_solve():
    m = Model()
    m.endog("k c")
    m.params("alpha beta")

    eq_euler = m.c[0]**(-1) - m.beta * m.alpha * m.k[0]**(m.alpha - 1) * m.c[1]**(-1)
    eq_kacc  = m.k[0] - m.k[-1]**m.alpha + m.c[0]
    m.build([eq_euler, eq_kacc])

    PARAMS = {m.alpha: 0.36, m.beta: 0.99}
    K_SS, C_SS = _rbc_ss(0.36, 0.99)
    ss = np.array([K_SS, C_SS])

    T = 80
    k_neg1 = np.array([K_SS * 1.1])
    sol = m.solve(T, PARAMS, initial_state=k_neg1, stock_var_indices=[0], endval=ss)

    assert sol.success
    X = sol.x.reshape(T, -1)
    np.testing.assert_allclose(X[-1], ss, atol=1e-3)


def test_builder_steady_state():
    m = Model()
    m.endog("k c")
    m.params("alpha beta")

    eq_euler = m.c[0]**(-1) - m.beta * m.alpha * m.k[0]**(m.alpha - 1) * m.c[1]**(-1)
    eq_kacc  = m.k[0] - m.k[-1]**m.alpha + m.c[0]
    m.build([eq_euler, eq_kacc])

    PARAMS = {m.alpha: 0.36, m.beta: 0.99}
    K_SS, C_SS = _rbc_ss(0.36, 0.99)

    ss_num = m.steady_state(PARAMS, initial_guess=np.array([K_SS * 0.9, C_SS * 0.9]))
    np.testing.assert_allclose(np.array(ss_num), [K_SS, C_SS], atol=1e-6)


def test_builder_with_exog():
    m = Model()
    m.endog("k c")
    m.exog("z")
    m.params("alpha beta")

    eq_euler = m.c[0]**(-1) - m.beta * m.alpha * sp.exp(m.z[1]) * m.k[0]**(m.alpha - 1) * m.c[1]**(-1)
    eq_kacc  = m.k[0] - sp.exp(m.z[0]) * m.k[-1]**m.alpha + m.c[0]
    m.build([eq_euler, eq_kacc])

    PARAMS = {m.alpha: 0.36, m.beta: 0.99}
    K_SS, C_SS = _rbc_ss(0.36, 0.99)
    ss = np.array([K_SS, C_SS])

    T = 50
    sol = m.solve(T, PARAMS,
                  initial_state=np.array([K_SS * 1.05]), stock_var_indices=[0],
                  exog_path=np.zeros((T, 1)), endval=ss)
    assert sol.success
    np.testing.assert_allclose(sol.x.reshape(T, -1)[-1], ss, atol=1e-2)


def test_builder_homotopy():
    m = Model()
    m.endog("k c")
    m.params("alpha beta")

    eq_euler = m.c[0]**(-1) - m.beta * m.alpha * m.k[0]**(m.alpha - 1) * m.c[1]**(-1)
    eq_kacc  = m.k[0] - m.k[-1]**m.alpha + m.c[0]
    m.build([eq_euler, eq_kacc])

    PARAMS = {m.alpha: 0.36, m.beta: 0.99}
    K_SS, C_SS = _rbc_ss(0.36, 0.99)
    ss = np.array([K_SS, C_SS])

    sol = m.solve_homotopy(80, PARAMS,
                           initial_state=np.array([K_SS * 1.5]), stock_var_indices=[0],
                           endval=ss, n_steps=5)
    assert sol.success
    np.testing.assert_allclose(sol.x.reshape(80, -1)[-1], ss, atol=1e-3)
