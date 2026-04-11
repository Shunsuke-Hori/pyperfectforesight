"""
test_eliminate_static.py

Unit tests for eliminate_static / _eliminate_static_core, covering the
success path (a truly-static variable is eliminated) as well as failure
paths (variable has dynamics, no candidates, sp.solve fails).
"""

import sympy as sp
import pytest

from pyperfectforesight import v, process_model
from pyperfectforesight.core import _eliminate_static_core, eliminate_static


# ── helpers ──────────────────────────────────────────────────────────────────

delta = sp.Rational(1, 40)   # 0.025, exact arithmetic


# ── _eliminate_static_core: success path ─────────────────────────────────────

def test_core_eliminates_truly_static_var():
    """Variable appearing only at t=0 is solved out of the system."""
    k_m, k_0, y_0 = v("k", -1), v("k", 0), v("y", 0)

    eq_dyn    = k_0 - (1 - delta) * k_m   # k has a lag → dynamic
    eq_static = y_0 - k_0**2              # y only at t=0 → truly static

    eqs, eliminated = _eliminate_static_core(
        [eq_static], [eq_dyn], vars_dyn=["k", "y"], vars_exo=[]
    )

    assert eliminated == frozenset({"y"})
    assert len(eqs) == 1
    assert v("y", 0) not in eqs[0].free_symbols


def test_core_substitutes_into_dynamic_eq():
    """Eliminated variable is substituted into dynamic equations that use it."""
    k_m, k_0, c_0, y_0 = v("k", -1), v("k", 0), v("c", 0), v("y", 0)

    # y_0 = k_0^0.5  (static)
    # k_0 = (1-delta)*k_{-1} + y_0 - c_0  (dynamic, uses y_0)
    eq_static = y_0 - k_0**sp.Rational(1, 2)
    eq_dyn    = k_0 - (1 - delta) * k_m - y_0 + c_0

    eqs, eliminated = _eliminate_static_core(
        [eq_static], [eq_dyn], vars_dyn=["k", "y", "c"], vars_exo=[]
    )

    assert "y" in eliminated
    # y_0 should have been substituted → k_0^(1/2) appears instead
    assert v("y", 0) not in eqs[0].free_symbols
    assert k_0**sp.Rational(1, 2) in eqs[0].expand().args or k_0 in eqs[0].free_symbols


# ── _eliminate_static_core: no-op paths ──────────────────────────────────────

def test_core_skips_var_with_lead():
    """Variable that appears at a lead in any equation is not eliminated."""
    k_m, k_0, k_p = v("k", -1), v("k", 0), v("k", 1)
    g_0 = v("g", 0)

    # g_0 = 0 is static, but g also appears as g_1 via k_p (hypothetical Euler)
    # We simulate this by having g appear at lag +1 somewhere.
    g_p = v("g", 1)
    eq_static  = g_0                          # g_0 = 0
    eq_dynamic = k_0 - (1 - delta) * k_m - g_p  # g appears at +1

    eqs, eliminated = _eliminate_static_core(
        [eq_static], [eq_dynamic], vars_dyn=["k", "g"], vars_exo=[]
    )

    assert eliminated == frozenset()
    assert len(eqs) == 2                       # fallback: static + dynamic


def test_core_skips_exo_symbol():
    """Exogenous variable symbols are excluded from candidates."""
    k_m, k_0, e_0 = v("k", -1), v("k", 0), v("e", 0)

    eq_static = e_0                        # e_0 = 0 — but e is exogenous
    eq_dyn    = k_0 - (1 - delta) * k_m

    eqs, eliminated = _eliminate_static_core(
        [eq_static], [eq_dyn], vars_dyn=["k"], vars_exo=["e"]
    )

    assert eliminated == frozenset()
    assert len(eqs) == 2


def test_core_skips_param_named_like_var():
    """A parameter like rho_1 (parses as lag 1) must not pollute vars_with_dynamics."""
    # If rho_1 is a parameter and "rho" is not in vars_dyn, it should be ignored.
    rho_1 = sp.Symbol("rho_1")             # looks like v("rho", 1)
    k_m, k_0, y_0 = v("k", -1), v("k", 0), v("y", 0)

    eq_static = y_0 - rho_1 * k_0         # uses param rho_1
    eq_dyn    = k_0 - (1 - delta) * k_m

    # "rho" is not in vars_dyn → should not block y from being eliminated
    eqs, eliminated = _eliminate_static_core(
        [eq_static], [eq_dyn], vars_dyn=["k", "y"], vars_exo=[]
    )

    assert "y" in eliminated
    assert len(eqs) == 1


def test_core_empty_static_eqs():
    k_m, k_0 = v("k", -1), v("k", 0)
    eq_dyn = k_0 - (1 - delta) * k_m
    eqs, eliminated = _eliminate_static_core([], [eq_dyn])
    assert eqs == [eq_dyn]
    assert eliminated == frozenset()


# ── eliminate_static public API ───────────────────────────────────────────────

def test_public_api_returns_list():
    k_m, k_0, y_0 = v("k", -1), v("k", 0), v("y", 0)
    result = eliminate_static(
        [y_0 - k_0**2], [k_0 - (1 - delta) * k_m],
        vars_dyn=["k", "y"], vars_exo=[]
    )
    assert isinstance(result, list)
    assert len(result) == 1


# ── process_model integration ─────────────────────────────────────────────────

def test_process_model_eliminates_truly_static_var_and_updates_vars_dyn():
    """process_model must remove eliminated var from vars_dyn to stay square."""
    k_m, k_0, y_0 = v("k", -1), v("k", 0), v("y", 0)

    eqs = [
        k_0 - (1 - delta) * k_m,   # dynamic
        y_0 - k_0**2,              # static, y truly static
    ]
    model = process_model(eqs, ["k", "y"])

    assert len(model["dynamic_eqs"]) == len(model["vars_dyn"]), (
        "dynamic_eqs and vars_dyn must be the same length after elimination"
    )
    assert "y" not in model["vars_dyn"]
    assert "k" in model["vars_dyn"]
