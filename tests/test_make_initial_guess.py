"""Tests for make_initial_guess helper."""

import numpy as np
import pytest
from pyperfectforesight import make_initial_guess


SS0 = np.array([1.0, 2.0, 3.0])
SS1 = np.array([2.0, 4.0, 6.0])
T = 20


# ---------------------------------------------------------------------------
# constant method
# ---------------------------------------------------------------------------

def test_constant_shape():
    X0 = make_initial_guess(T, SS0, SS1, method='constant')
    assert X0.shape == (T, 3)


def test_constant_all_rows_equal_terminal():
    X0 = make_initial_guess(T, SS0, SS1, method='constant')
    assert np.allclose(X0, SS1[None, :])


# ---------------------------------------------------------------------------
# linear method
# ---------------------------------------------------------------------------

def test_linear_shape():
    X0 = make_initial_guess(T, SS0, SS1, method='linear')
    assert X0.shape == (T, 3)


def test_linear_first_row_is_ss_initial():
    X0 = make_initial_guess(T, SS0, SS1, method='linear')
    assert np.allclose(X0[0], SS0)


def test_linear_last_row_is_ss_terminal():
    X0 = make_initial_guess(T, SS0, SS1, method='linear')
    assert np.allclose(X0[-1], SS1)


def test_linear_monotone():
    """Each variable should move monotonically from SS0 to SS1."""
    X0 = make_initial_guess(T, SS0, SS1, method='linear')
    diffs = np.diff(X0, axis=0)
    assert np.all(diffs >= 0)   # SS0 < SS1 for all variables


def test_linear_t1_is_constant_when_ss_equal():
    ss = np.array([1.5, 2.5])
    X0 = make_initial_guess(T, ss, ss, method='linear')
    assert np.allclose(X0, ss[None, :])


# ---------------------------------------------------------------------------
# exponential method
# ---------------------------------------------------------------------------

def test_exponential_shape():
    X0 = make_initial_guess(T, SS0, SS1, method='exponential')
    assert X0.shape == (T, 3)


def test_exponential_first_row_is_ss_initial():
    X0 = make_initial_guess(T, SS0, SS1, method='exponential')
    assert np.allclose(X0[0], SS0)


def test_exponential_converges_to_terminal():
    """Last row should be very close to ss_terminal for small decay."""
    X0 = make_initial_guess(T, SS0, SS1, method='exponential', decay=0.5)
    assert np.allclose(X0[-1], SS1, atol=1e-4)


def test_exponential_faster_than_linear_early():
    """Exponential should close more of the gap in the first period than linear."""
    X0_lin = make_initial_guess(T, SS0, SS1, method='linear')
    X0_exp = make_initial_guess(T, SS0, SS1, method='exponential', decay=0.9)
    gap_lin = np.linalg.norm(SS1 - X0_lin[1])
    gap_exp = np.linalg.norm(SS1 - X0_exp[1])
    assert gap_exp < gap_lin


def test_exponential_monotone():
    X0 = make_initial_guess(T, SS0, SS1, method='exponential')
    diffs = np.diff(X0, axis=0)
    assert np.all(diffs >= 0)


def test_exponential_decay_parameter():
    """Smaller decay → faster convergence → smaller gap at t=1."""
    X0_fast = make_initial_guess(T, SS0, SS1, method='exponential', decay=0.5)
    X0_slow = make_initial_guess(T, SS0, SS1, method='exponential', decay=0.95)
    gap_fast = np.linalg.norm(SS1 - X0_fast[1])
    gap_slow = np.linalg.norm(SS1 - X0_slow[1])
    assert gap_fast < gap_slow


# ---------------------------------------------------------------------------
# edge cases and validation
# ---------------------------------------------------------------------------

def test_T_less_than_2_raises():
    with pytest.raises(ValueError, match="T must be >= 2"):
        make_initial_guess(1, SS0, SS1)


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method must be"):
        make_initial_guess(T, SS0, SS1, method='quadratic')


def test_invalid_decay_raises():
    with pytest.raises(ValueError, match="decay must be in"):
        make_initial_guess(T, SS0, SS1, method='exponential', decay=1.5)


def test_invalid_decay_zero_raises():
    with pytest.raises(ValueError, match="decay must be in"):
        make_initial_guess(T, SS0, SS1, method='exponential', decay=0.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        make_initial_guess(T, SS0, np.array([1.0, 2.0]))


def test_invalid_T_raises():
    with pytest.raises(ValueError, match="T must be >= 2"):
        make_initial_guess(0, SS0, SS1)


def test_float_T_raises():
    with pytest.raises(TypeError, match="integer"):
        make_initial_guess(2.5, SS0, SS1)


def test_numpy_integer_T_accepted():
    """np.integer types are valid integers and should not raise."""
    X0 = make_initial_guess(np.int64(T), SS0, SS1)
    assert X0.shape == (T, 3)


def test_default_method_is_linear():
    X0_default = make_initial_guess(T, SS0, SS1)
    X0_linear = make_initial_guess(T, SS0, SS1, method='linear')
    assert np.allclose(X0_default, X0_linear)


def test_accepts_lists():
    X0 = make_initial_guess(T, list(SS0), list(SS1), method='linear')
    assert X0.shape == (T, 3)
