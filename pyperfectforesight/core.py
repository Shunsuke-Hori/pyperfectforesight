"""
main.py
A minimal Dynare-style perfect foresight solver framework in Python

This module provides the core infrastructure for solving perfect foresight
models. For usage examples, see demo.py.
"""

import sympy as sp
import numpy as np
from scipy.sparse import lil_matrix, vstack
from scipy.sparse.linalg import spsolve, lsmr

# ============================================================
# 1. Utilities
# ============================================================

def v(name, lag):
    """Time-indexed symbolic variable"""
    return sp.Symbol(f"{name}_{lag}")

def _parse_time_symbol(sym_name):
    """Parse a time-indexed symbol name like 'c_0' or 'k_-1'.

    Returns (var_name, lag) for time-indexed symbols, or None for plain names
    and parameters that happen to contain underscores (e.g. 'rho_g').
    """
    if "_" not in sym_name:
        return None
    parts = sym_name.rsplit("_", 1)
    try:
        return parts[0], int(parts[1])
    except ValueError:
        return None


def _resolve_lag_sets(all_syms, vars_dyn, vars_exo, endo_lags, exo_lags):
    """Return (endo_lags, exo_lags), computing any missing sets from all_syms."""
    if endo_lags is None and exo_lags is None:
        return _compute_lag_sets(all_syms, vars_dyn, vars_exo)
    if endo_lags is None:
        endo_lags, _ = _compute_lag_sets(all_syms, vars_dyn, vars_exo)
    elif exo_lags is None:
        _, exo_lags = _compute_lag_sets(all_syms, vars_dyn, vars_exo)
    return endo_lags, exo_lags


def _compute_lag_sets(all_syms, vars_dyn, vars_exo):
    """Return sorted (endo_lags, exo_lags) by scanning all_syms.

    endo_lags : sorted list of integer lags that appear for any variable in vars_dyn.
    exo_lags  : sorted list of integer lags that appear for any variable in vars_exo.
    """
    endo_set = set(vars_dyn)
    exo_set  = set(vars_exo)
    endo_lags = set()
    exo_lags  = set()
    for s in all_syms:
        p = _parse_time_symbol(s.name)
        if p is not None:
            vn, lg = p
            if vn in endo_set:
                endo_lags.add(lg)
            elif vn in exo_set:
                exo_lags.add(lg)
    return sorted(endo_lags), sorted(exo_lags)

# ============================================================
# 2. Lead / lag detection (Dynare lead_lag_incidence)
# ============================================================

def lead_lag_incidence(equations, known_vars=None):
    """
    Detect which variables appear at which time lags in the equations

    Parameters:
    -----------
    equations : list
        List of sympy equations
    known_vars : set, optional
        Set of declared variable base names (vars_dyn + vars_exo + vars_aux).
        When provided, only symbols whose base name is in known_vars are
        recorded, preventing parameters like ``rho_1`` from appearing in the
        incidence table.  When None, every parseable ``name_<int>`` symbol is
        included (legacy behaviour).

    Returns:
    --------
    dict : Dictionary mapping variable names to sets of lags
    """
    inc = {}
    for eq in equations:
        for s in eq.free_symbols:
            parsed = _parse_time_symbol(s.name)
            if parsed is None:
                continue
            var_name, lag = parsed
            if known_vars is not None and var_name not in known_vars:
                continue
            inc.setdefault(var_name, set()).add(lag)
    return inc

# ============================================================
# 3. Static equation detection & elimination
# ============================================================

def is_static(eq, known_vars=None):
    """
    Check if an equation contains only current period variables

    Parameters:
    -----------
    eq : sympy expression
        Equation to check
    known_vars : set, optional
        Set of known variable base names (vars_dyn + vars_exo + vars_aux).
        When provided, only symbols whose base name is in known_vars are
        considered time-indexed, avoiding false positives for parameters
        that happen to end in an integer (e.g. ``rho_1``).
        When None, any symbol whose name parses as ``name_<int>`` with a
        nonzero integer lag is treated as a lead/lag (legacy behaviour).

    Returns:
    --------
    bool : True if equation is static (no leads/lags)
    """
    for s in eq.free_symbols:
        parsed = _parse_time_symbol(s.name)
        if parsed is None:
            continue
        var_name, lag = parsed
        if lag == 0:
            continue
        if known_vars is None or var_name in known_vars:
            return False
    return True

def eliminate_static(static_eqs, dynamic_eqs):
    """
    Eliminate static variables from dynamic equations

    Parameters:
    -----------
    static_eqs : list
        List of static equations
    dynamic_eqs : list
        List of dynamic equations

    Returns:
    --------
    list : Dynamic equations with static variables substituted out
    """
    if not static_eqs:
        return dynamic_eqs

    static_vars = sorted(
        {s for eq in static_eqs for s in eq.free_symbols},
        key=lambda s: s.name
    )

    sol = sp.solve(static_eqs, static_vars, dict=True)
    if not sol:
        return dynamic_eqs

    sol = sol[0]
    return [eq.subs(sol) for eq in dynamic_eqs]

# ============================================================
# 4. Local Jacobian blocks (symbolic)
# ============================================================

def local_blocks(equations, variables):
    """
    Compute symbolic Jacobian blocks for all lags present in equations

    Parameters:
    -----------
    equations : list
        List of equations
    variables : list
        List of variable names

    Returns:
    --------
    dict : Dictionary mapping lags to Jacobian matrices
    """
    # Auto-detect which lags actually appear in the equations
    all_lags = set()
    for eq in equations:
        for s in eq.free_symbols:
            parsed = _parse_time_symbol(s.name)
            if parsed is not None and parsed[0] in variables:
                all_lags.add(parsed[1])

    blocks = {}
    for lag in sorted(all_lags):
        cols = [v(var, lag) for var in variables]
        J = sp.Matrix(equations).jacobian(cols)
        if not J.is_zero_matrix:
            blocks[lag] = J
    return blocks

# ============================================================
# 5. Residual function
# ============================================================

def residual(X, params, all_syms, residual_funcs, vars_dyn, dynamic_eqs, vars_exo=None, exog_path=None,
             endo_lags=None, exo_lags=None):
    """
    Evaluate residuals of the dynamic equations

    Parameters:
    -----------
    X : ndarray
        State path (T x n_endo)
    params : dict
        Parameter values
    all_syms : list
        All symbols in the equations
    residual_funcs : list
        Compiled residual functions
    vars_dyn : list
        List of endogenous variable names
    dynamic_eqs : list
        List of dynamic equations
    vars_exo : list, optional
        List of exogenous variable names
    exog_path : ndarray, optional
        Exogenous variable path (T x n_exo)
    endo_lags : list of int, optional
        Sorted list of integer lags that appear for endogenous variables.
        If None (or if exo_lags is also None), derived automatically from
        all_syms via _compute_lag_sets.  Pass the precomputed value from
        model_funcs['endo_lags'] to avoid rescanning all_syms on every call.
    exo_lags : list of int, optional
        Sorted list of integer lags that appear for exogenous variables.
        Same semantics as endo_lags.  Out-of-range indices are clamped to
        [0, T-1] (boundary replication).

    Returns:
    --------
    ndarray : Flattened residual vector of length (T-1)*neq
    """
    T, n = X.shape
    neq = len(dynamic_eqs)

    if vars_exo is None:
        vars_exo = []
    if exog_path is None:
        exog_path = np.zeros((T, len(vars_exo)))
    endo_lags, exo_lags = _resolve_lag_sets(all_syms, vars_dyn, vars_exo, endo_lags, exo_lags)

    # Standard mode: evaluate T-1 periods with boundary clamping.
    F = np.zeros((T-1, neq))
    for t in range(T-1):
        subs = {}
        # Endogenous variables
        for i, var in enumerate(vars_dyn):
            for lag in endo_lags:
                tt = min(max(t+lag, 0), T-1)
                subs[v(var, lag)] = X[tt, i]

        # Exogenous variables
        for i, var in enumerate(vars_exo):
            for lag in exo_lags:
                tt = min(max(t+lag, 0), T-1)
                subs[v(var, lag)] = exog_path[tt, i]

        subs.update(params)

        # Use compiled functions
        vals = [subs[s] for s in all_syms]
        for i, func in enumerate(residual_funcs):
            F[t, i] = func(*vals)

    return F.ravel()


def _residual_bvp(X, params, all_syms, residual_funcs, vars_dyn, dynamic_eqs,
                  vars_exo, exog_path, initval, endval, endo_lags=None, exo_lags=None):
    """Evaluate T BVP residuals on the T+2-row augmented path [initval, X, endval].

    Internal helper for the stock/jump BVP branch of solve_perfect_foresight.
    Endogenous variables index the augmented path as ``X_aug[t + lag + 1]``,
    clamped to ``[0, T+1]`` so boundary rows serve as pre-sample and
    terminal history.  This means that for ``|lag| > 1``, values beyond the
    single initval/endval row are assumed equal to those boundary values
    (e.g. ``k_{-2} = k_{-1} = initval``).  Exogenous variables are padded
    symmetrically at the endpoints.  Returns a flattened vector of length T*neq.
    """
    T, n = X.shape
    neq = len(dynamic_eqs)
    if exog_path is None:
        exog_path = np.zeros((T, len(vars_exo)))
    X_aug = np.empty((T + 2, n), dtype=float)
    X_aug[0] = initval
    X_aug[1:-1] = X
    X_aug[-1] = endval
    # Exogenous: pad boundary rows so lag/lead at t=0 and t=T-1 are well-defined.
    n_exo = exog_path.shape[1]
    if n_exo > 0:
        exog_aug = np.empty((T + 2, n_exo), dtype=float)
        exog_aug[0] = exog_path[0]
        exog_aug[1:-1] = exog_path
        exog_aug[-1] = exog_path[-1]
    else:
        exog_aug = exog_path
    endo_lags, exo_lags = _resolve_lag_sets(all_syms, vars_dyn, vars_exo, endo_lags, exo_lags)

    F = np.zeros((T, neq))
    for t in range(T):
        subs = {}
        for i, var in enumerate(vars_dyn):
            for lag in endo_lags:
                aug_idx = max(0, min(t + lag + 1, T + 1))
                subs[v(var, lag)] = X_aug[aug_idx, i]
        for i, var in enumerate(vars_exo):
            for lag in exo_lags:
                aug_idx = max(0, min(t + lag + 1, T + 1))
                subs[v(var, lag)] = exog_aug[aug_idx, i]
        subs.update(params)
        vals = [subs[s] for s in all_syms]
        for i, func in enumerate(residual_funcs):
            F[t, i] = func(*vals)
    return F.ravel()

# ============================================================
# 6. Sparse block Jacobian
# ============================================================

def sparse_jacobian(X, params, all_syms, block_funcs, vars_dyn, dynamic_eqs, vars_exo=None, exog_path=None,
                    endo_lags=None, exo_lags=None):
    """
    Build sparse Jacobian matrix using block structure

    Parameters:
    -----------
    X : ndarray
        State path (T x n_endo)
    params : dict
        Parameter values
    all_syms : list
        All symbols in the equations
    block_funcs : dict
        Compiled Jacobian block functions
    vars_dyn : list
        List of endogenous variable names
    dynamic_eqs : list
        List of dynamic equations
    vars_exo : list, optional
        List of exogenous variable names
    exog_path : ndarray, optional
        Exogenous variable path (T x n_exo)
    endo_lags : list of int, optional
        Sorted list of integer lags that appear for endogenous variables.
        If None (or if exo_lags is also None), derived automatically from
        all_syms via _compute_lag_sets.  Pass model_funcs['endo_lags'] to
        avoid rescanning all_syms on every Newton iteration.  Jacobian blocks
        for out-of-range time indices are clamped to [0, T-1] and accumulated
        into the clamped column, consistent with residual() boundary handling.
    exo_lags : list of int, optional
        Sorted list of integer lags for exogenous variables.  Same semantics
        as endo_lags.

    Returns:
    --------
    sparse matrix : Sparse Jacobian in CSR format of shape (neq*(T-1), n*T)
        where neq = len(dynamic_eqs).
    """
    T, n = X.shape
    neq = len(dynamic_eqs)

    if vars_exo is None:
        vars_exo = []
    if exog_path is None:
        exog_path = np.zeros((T, len(vars_exo)))
    endo_lags, exo_lags = _resolve_lag_sets(all_syms, vars_dyn, vars_exo, endo_lags, exo_lags)

    # Standard mode: (T-1) equation-periods with boundary clamping.
    J = lil_matrix((neq*(T-1), n*T))

    for t in range(T-1):
        subs = {}
        # Endogenous variables
        for i, var in enumerate(vars_dyn):
            for lag in endo_lags:
                tt = min(max(t+lag, 0), T-1)
                subs[v(var, lag)] = X[tt, i]

        # Exogenous variables
        for i, var in enumerate(vars_exo):
            for lag in exo_lags:
                tt = min(max(t+lag, 0), T-1)
                subs[v(var, lag)] = exog_path[tt, i]

        subs.update(params)
        vals = [subs[s] for s in all_syms]

        # Clamp column to [0, T-1], consistent with residual() boundary handling.
        # Accumulate all blocks for the same col_t into a dense buffer before
        # writing to the sparse matrix — avoids repeated sparse slice reads when
        # multiple lags clamp to the same column.
        col_blocks = {}
        for lag, f in block_funcs.items():
            col_t = min(max(t + lag, 0), T - 1)
            B = np.asarray(f(*vals))
            if col_t in col_blocks:
                col_blocks[col_t] += B
            else:
                col_blocks[col_t] = B.copy()

        r0 = t * neq
        for col_t, B_sum in col_blocks.items():
            J[r0:r0+neq, col_t*n:col_t*n+n] = B_sum

    return J.tocsr()


def _jacobian_bvp(X, params, all_syms, block_funcs, vars_dyn, dynamic_eqs,
                  vars_exo, exog_path, initval, endval, endo_lags=None, exo_lags=None):
    """Build the (T*neq × T*n) BVP Jacobian on the T+2-row augmented path.

    Internal helper for the stock/jump BVP branch of solve_perfect_foresight.
    Columns referencing the fixed boundary rows (initval/endval) are skipped
    because those values are not unknowns — their derivatives do not enter the
    Newton system.  This is consistent with _residual_bvp, which reads the same
    boundary rows as fixed inputs (including clamped pre-sample values for
    ``|lag| > 1``).  When neq == n (as enforced by the caller), the result is
    a square matrix.
    """
    T, n = X.shape
    neq = len(dynamic_eqs)
    if exog_path is None:
        exog_path = np.zeros((T, len(vars_exo)))
    X_aug = np.empty((T + 2, n), dtype=float)
    X_aug[0] = initval
    X_aug[1:-1] = X
    X_aug[-1] = endval
    n_exo = exog_path.shape[1]
    if n_exo > 0:
        exog_aug = np.empty((T + 2, n_exo), dtype=float)
        exog_aug[0] = exog_path[0]
        exog_aug[1:-1] = exog_path
        exog_aug[-1] = exog_path[-1]
    else:
        exog_aug = exog_path
    endo_lags, exo_lags = _resolve_lag_sets(all_syms, vars_dyn, vars_exo, endo_lags, exo_lags)

    J = lil_matrix((neq * T, n * T))
    for t in range(T):
        subs = {}
        for i, var in enumerate(vars_dyn):
            for lag in endo_lags:
                aug_idx = max(0, min(t + lag + 1, T + 1))
                subs[v(var, lag)] = X_aug[aug_idx, i]
        for i, var in enumerate(vars_exo):
            for lag in exo_lags:
                aug_idx = max(0, min(t + lag + 1, T + 1))
                subs[v(var, lag)] = exog_aug[aug_idx, i]
        subs.update(params)
        vals = [subs[s] for s in all_syms]
        for lag, f in block_funcs.items():
            col_t = t + lag  # index into X (0-based); skip fixed boundary rows
            if not (0 <= col_t < T):
                continue
            B = f(*vals)
            r0 = t * neq
            c0 = col_t * n
            J[r0:r0+neq, c0:c0+n] = B
    return J.tocsr()

# ============================================================
# 7. Terminal steady-state conditions
# ============================================================

def append_terminal_conditions(F, J, X, ss):
    """
    Append terminal conditions to enforce convergence to steady state

    Parameters:
    -----------
    F : ndarray
        Residual vector
    J : sparse matrix
        Jacobian matrix
    X : ndarray
        State path (T x n)
    ss : ndarray
        Steady state values

    Returns:
    --------
    tuple : (F_augmented, J_augmented) with terminal conditions added
    """
    T, n = X.shape

    # Expand J to accommodate terminal conditions
    terminal_rows = lil_matrix((n, n*T))

    for i in range(n):
        F = np.append(F, X[-1,i] - ss[i])
        col = n*(T-1) + i
        terminal_rows[i, col] = 1.0

    J = vstack([J, terminal_rows.tocsr()])

    return F, J

# ============================================================
# 8. Steady state computation
# ============================================================

def compute_steady_state_numerical(equations, vars_dyn, params_dict, initial_guess=None):
    """
    Compute steady state numerically by solving the system where all
    time-indexed variables are set to their steady-state values

    Parameters:
    -----------
    equations : list
        List of model equations
    vars_dyn : list
        List of dynamic variable names
    params_dict : dict
        Parameter values
    initial_guess : ndarray, optional
        Initial guess for steady state values (default: ones)

    Returns:
    --------
    ndarray : Steady state values for each variable in vars_dyn
    """
    from scipy.optimize import fsolve

    # Create steady state symbols
    ss_syms = [sp.Symbol(f"{var}_ss") for var in vars_dyn]

    # At steady state: x_{t-1} = x_t = x_{t+1} = x_ss
    # Build substitution map
    subs_map = {}

    # Auto-detect all lags present
    all_lags = set()
    for eq in equations:
        for s in eq.free_symbols:
            parsed = _parse_time_symbol(s.name)
            if parsed is not None and parsed[0] in vars_dyn:
                all_lags.add(parsed[1])

    # Map all time-indexed variables to steady state
    for var, ss_sym in zip(vars_dyn, ss_syms):
        for lag in all_lags:
            subs_map[v(var, lag)] = ss_sym

    # Add parameters
    subs_map.update(params_dict)

    # Create steady state equations
    ss_equations = [eq.subs(subs_map) for eq in equations]

    # Compile to numeric function
    ss_funcs = [sp.lambdify(ss_syms, eq, "numpy") for eq in ss_equations]

    def residual_ss(x):
        return np.array([float(f(*x)) for f in ss_funcs])

    # Initial guess
    if initial_guess is None:
        initial_guess = np.ones(len(vars_dyn))

    # Solve
    solution = fsolve(residual_ss, initial_guess)

    return solution

# ============================================================
# 9. Model processing pipeline
# ============================================================

def process_model(equations, vars_dyn, vars_exo=None, vars_aux=None, aux_method='auto', eliminate_static_vars=True, compiler='lambdify'):
    """
    Process model equations and compile to numeric functions

    Parameters:
    -----------
    equations : list
        List of model equations
    vars_dyn : list
        List of endogenous (dynamic) variable names
    vars_exo : list, optional
        List of exogenous variable names (default: None)
    vars_aux : list, optional
        List of auxiliary variable names - static variables to be determined
        from dynamic and exogenous variables (default: None).

    aux_method : str, optional
        Method for handling auxiliary variables (default: ``'auto'``):

        - ``'auto'``: Try analytical first; if SymPy can't solve, treat as dynamic.
          Best default — fast when possible, robust when needed.
          (Similar to Dynare: eliminate if possible, else keep in system.)
        - ``'analytical'``: Force analytical method only. Faster but will fail if
          SymPy cannot solve the auxiliary equations symbolically.
        - ``'nested'``: Force post-solve numerical solving of auxiliary equations.
          After the main solver converges, auxiliary variables are solved
          period-by-period using ``scipy.optimize.root`` with warm starting.
          Requires a square auxiliary system and auxiliary variables appearing
          only in auxiliary equations; raises ``ValueError`` otherwise.
          Use when analytical solve fails and these structural conditions hold;
          otherwise prefer ``'dynamic'``.
        - ``'dynamic'``: Treat auxiliary variables as dynamic. Auxiliary equations
          included in main system. Single optimization, higher dimension.

    eliminate_static_vars : bool
        Whether to eliminate non-auxiliary static variables (default: True)
    compiler : str
        Compilation method: 'lambdify' (default), or other backends

    Returns:
    --------
    dict : Dictionary containing:
        - 'dynamic_eqs': Processed dynamic equations
        - 'blocks': Symbolic Jacobian blocks
        - 'all_syms': Sorted list of all symbols
        - 'block_funcs': Compiled Jacobian block functions
        - 'residual_funcs': Compiled residual functions
        - 'incidence': Lead/lag incidence information
        - 'vars_dyn': List of dynamic variable names
        - 'vars_exo': List of exogenous variable names
        - 'vars_aux': List of auxiliary variable names
        - 'aux_method': Method used for auxiliary variables
        - 'aux_eqs': Symbolic auxiliary equations
        - 'aux_eqs_funcs': Compiled residual functions for auxiliary equations (nested method)
        - 'aux_eqs_syms': Sorted list of all free symbols appearing in auxiliary equations (nested method)
        - 'aux_sols': Analytical solutions for auxiliary variables (analytical method)
        - 'aux_funcs': Compiled evaluation functions (analytical method)

    Notes:
    ------
    Recommended usage:

    - ``'auto'`` (default): Best for most cases — tries analytical, falls back to dynamic.
      Like Dynare: eliminate if possible, else keep in system.
    - ``'analytical'``: When equations are simple (e.g. ``i = y - c - g``) and
      guaranteed fast performance with no fallback is desired.
    - ``'nested'``: When nested post-solve optimization is explicitly preferred (rare).
    - ``'dynamic'``: To skip the analytical attempt and treat aux vars as dynamic.

    The analytical method solves equations symbolically then evaluates (fastest).
    The dynamic method includes auxiliary equations in main system (Dynare-style).
    The nested method solves auxiliary variables post-solve, period-by-period.
    """
    if vars_exo is None:
        vars_exo = []
    if vars_aux is None:
        vars_aux = []

    # Precompute the full set of declared model variable names.
    # Used by lead_lag_incidence and is_static to avoid false positives for
    # parameters whose names happen to end in an integer (e.g. rho_1).
    known_vars = set(vars_dyn) | set(vars_exo) | set(vars_aux)

    # Lead/lag detection
    incidence = lead_lag_incidence(equations, known_vars=known_vars)

    # Process auxiliary equations based on chosen method
    aux_eqs = []
    aux_eqs_funcs = []  # For nested method
    aux_eqs_syms = []  # For nested method
    aux_sols = {}  # For analytical method
    aux_funcs = {}  # For analytical method
    aux_method_used = aux_method

    # Always identify auxiliary equations first (needed for all methods)
    remaining_eqs = []
    if vars_aux:
        # Identify equations that define auxiliary variables
        # These should be static equations involving auxiliary variables
        aux_var_syms = {v(name, 0) for name in vars_aux}

        aux_eqs = []
        remaining_eqs = []

        for eq in equations:
            eq_vars = eq.free_symbols
            # Check if this equation defines an auxiliary variable
            if any(aux_var in eq_vars for aux_var in aux_var_syms) and is_static(eq, known_vars):
                aux_eqs.append(eq)
            else:
                remaining_eqs.append(eq)

        if not aux_eqs:
            raise ValueError(
                f"No static equations found for auxiliary variables {vars_aux}. "
                f"Each auxiliary variable must appear in at least one static equation "
                f"(an equation with no leads or lags of declared model variables). "
                f"If these variables are genuinely dynamic, declare them in vars_dyn "
                f"instead, or use aux_method='dynamic' and include them in vars_dyn."
            )

    # Now process based on method (but skip if method is 'dynamic')
    if vars_aux and aux_method != 'dynamic' and aux_eqs:

        # Process based on method
        if aux_method in ['auto', 'analytical']:
            # Try ANALYTICAL METHOD: Solve symbolically
            aux_var_list = [v(name, 0) for name in vars_aux]
            solved_successfully = False
            try:
                aux_sols_list = sp.solve(aux_eqs, aux_var_list, dict=True)
                if aux_sols_list:
                    if len(aux_sols_list) > 1:
                        raise ValueError(
                            f"SymPy found multiple analytical solution branches for "
                            f"auxiliary variables {vars_aux}. Selecting a branch "
                            f"arbitrarily may give wrong results. Use "
                            f"aux_method='dynamic' or aux_method='nested' instead."
                        )
                    aux_sols = aux_sols_list[0]
                    # Check that all auxiliary variables were solved
                    if all(v(name, 0) in aux_sols for name in vars_aux):
                        solved_successfully = True
            except Exception as e:
                solved_successfully = False
                _sympy_solve_exc = e  # preserved for error/warning message below
            else:
                _sympy_solve_exc = None

            if solved_successfully and aux_sols:
                # Successfully solved analytically - compile evaluation functions
                for var_name in vars_aux:
                    var_sym = v(var_name, 0)
                    if var_sym in aux_sols:
                        expr = aux_sols[var_sym]
                        expr_syms = sorted(expr.free_symbols, key=lambda s: s.name)
                        if compiler == 'lambdify':
                            aux_funcs[var_name] = {
                                'expr': expr,
                                'func': sp.lambdify(expr_syms, expr, "numpy"),
                                'syms': expr_syms
                            }
                # Substitute aux solutions into remaining equations (handles any
                # aux variable that may appear in a non-aux equation)
                equations = [eq.subs(aux_sols) for eq in remaining_eqs]
                aux_method_used = 'analytical'
            else:
                # Analytical solve failed
                _exc_detail = f": {_sympy_solve_exc}" if _sympy_solve_exc is not None else ""
                if aux_method == 'analytical':
                    # User forced analytical only - raise error
                    raise ValueError(
                        f"Could not solve auxiliary equations analytically for {vars_aux}{_exc_detail}. "
                        f"Consider using aux_method='auto' (tries analytical then dynamic) "
                        f"or aux_method='dynamic' (include auxiliary equations in system)."
                    ) from _sympy_solve_exc
                else:  # aux_method == 'auto'
                    # Auto fallback to dynamic (Dynare-style)
                    import warnings
                    warnings.warn(
                        f"Could not solve auxiliary equations analytically for {vars_aux}{_exc_detail}. "
                        f"Treating auxiliary variables as dynamic (keeping equations in system). "
                        f"This follows Dynare's approach when analytical elimination fails.",
                        UserWarning
                    )
                    aux_method_used = 'dynamic'
                    aux_sols = {}
                    aux_funcs = {}

        if aux_eqs and (aux_method == 'nested' or aux_method_used == 'nested'):
            # NESTED METHOD: Compile residual functions for numerical solving
            if len(aux_eqs) != len(vars_aux):
                raise ValueError(
                    f"In aux_method='nested' the auxiliary system must be square "
                    f"(one equation per variable). Got {len(aux_eqs)} equations for "
                    f"{len(vars_aux)} auxiliary variables: {vars_aux}."
                )
            # Get all symbols that appear in auxiliary equations
            aux_eqs_syms = sorted(
                {s for eq in aux_eqs for s in eq.free_symbols},
                key=lambda s: s.name
            )

            if compiler == 'lambdify':
                # Compile each auxiliary equation as a residual function
                aux_eqs_funcs = [sp.lambdify(aux_eqs_syms, eq, "numpy") for eq in aux_eqs]

            # Remove auxiliary equations from dynamic system.
            # Aux variables must not appear in the remaining dynamic equations at any
            # lead/lag — they have no values during Newton iterations in nested mode.
            aux_var_names = set(vars_aux or [])

            def _is_aux_sym(s):
                """Return True if symbol s references an auxiliary variable at any lag."""
                name = s.name
                if '_' in name:
                    parts = name.rsplit('_', 1)
                    try:
                        int(parts[1])
                        return parts[0] in aux_var_names
                    except ValueError:
                        pass
                return name in aux_var_names

            leaking = [s for eq in remaining_eqs for s in eq.free_symbols if _is_aux_sym(s)]
            if leaking:
                raise ValueError(
                    f"Auxiliary variable(s) {[s.name for s in set(leaking)]} appear in "
                    f"non-auxiliary equations, which is unsupported in nested mode. "
                    f"Auxiliary variables must only appear in their defining aux equations."
                )
            equations = remaining_eqs

    # Track which equations were auxiliary (so we don't eliminate them in dynamic mode)
    aux_eqs_set = set(aux_eqs) if aux_eqs else set()

    if (vars_aux and aux_method_used == 'dynamic') or (vars_aux and not aux_eqs):
        # DYNAMIC METHOD: Treat auxiliary variables as regular dynamic variables
        # Keep auxiliary equations in the system (Dynare-style)
        vars_dyn = vars_dyn + vars_aux
        vars_aux = []
        # Don't clear aux_eqs yet - we need them to avoid elimination
        aux_eqs_funcs = []
        aux_sols = {}
        aux_funcs = {}
        # Keep all equations including auxiliary ones

    # Static equation elimination for non-auxiliary static variables
    # Use known_vars so parameters like rho_1 don't trigger false dynamic classification.
    # At this point vars_dyn may have been extended (dynamic method), so recompute.
    known_vars_final = set(vars_dyn) | set(vars_exo) | set(vars_aux)
    if eliminate_static_vars:
        static_eqs = [eq for eq in equations if is_static(eq, known_vars_final) and eq not in aux_eqs_set]
        dynamic_eqs = [eq for eq in equations if not is_static(eq, known_vars_final)]

        # For dynamic method, add auxiliary equations to dynamic_eqs
        if aux_method_used == 'dynamic' and aux_eqs:
            dynamic_eqs = dynamic_eqs + aux_eqs

        # Eliminate non-auxiliary static equations
        if static_eqs:
            dynamic_eqs = eliminate_static(static_eqs, dynamic_eqs)
    else:
        dynamic_eqs = equations

    # Now clear aux_eqs for dynamic method
    if aux_method_used == 'dynamic':
        aux_eqs = []

    # Local Jacobian blocks
    blocks = local_blocks(dynamic_eqs, vars_dyn)

    # Compile numeric functions
    all_syms = sorted(
        {s for eq in dynamic_eqs for s in eq.free_symbols},
        key=lambda s: s.name
    )

    if compiler == 'lambdify':
        block_funcs = {
            lag: sp.lambdify(all_syms, J, "numpy")
            for lag, J in blocks.items()
        }
        residual_funcs = [sp.lambdify(all_syms, eq, "numpy") for eq in dynamic_eqs]
    else:
        raise ValueError(f"Unsupported compiler: {compiler}")

    # Recompute incidence from the final dynamic equations so it reflects any
    # auxiliary-variable substitutions / removals that happened during processing.
    incidence = lead_lag_incidence(dynamic_eqs, known_vars=known_vars_final)

    # Precompute lag sets so solvers don't rescan all_syms on every Newton call.
    endo_lags, exo_lags = _compute_lag_sets(all_syms, vars_dyn, vars_exo)

    return {
        'dynamic_eqs': dynamic_eqs,
        'blocks': blocks,
        'all_syms': all_syms,
        'block_funcs': block_funcs,
        'residual_funcs': residual_funcs,
        'incidence': incidence,
        'endo_lags': endo_lags,
        'exo_lags': exo_lags,
        'vars_dyn': vars_dyn,
        'vars_exo': vars_exo,
        'vars_aux': vars_aux,
        'aux_method': aux_method_used,
        'aux_eqs': aux_eqs,
        'aux_eqs_funcs': aux_eqs_funcs,  # For nested method
        'aux_eqs_syms': aux_eqs_syms if aux_eqs and aux_method_used == 'nested' else [],  # Symbols for nested
        'aux_sols': aux_sols,  # For analytical method
        'aux_funcs': aux_funcs  # For analytical method
    }

def solve_auxiliary_nested(X_dyn_t, params_dict, model_funcs, vars_dyn, exog_t=None, aux_guess=None):
    """
    Solve auxiliary equations numerically for a single time period (nested method)

    Parameters:
    -----------
    X_dyn_t : array (n_dyn,)
        Dynamic variable values at time t
    params_dict : dict
        Parameter values
    model_funcs : dict
        Model functions from process_model
    vars_dyn : list
        List of dynamic variable names
    exog_t : array (n_exo,), optional
        Exogenous variable values at time t
    aux_guess : array (n_aux,), optional
        Initial guess for auxiliary variables

    Returns:
    --------
    array (n_aux,) : Solution for auxiliary variables at time t
    """
    from scipy.optimize import root

    if not model_funcs.get('aux_eqs_funcs'):
        raise ValueError(
            "'aux_eqs_funcs' is missing from model_funcs. "
            "solve_auxiliary_nested requires a model processed with "
            "aux_method='nested'."
        )

    n_aux = len(model_funcs['vars_aux'])
    if aux_guess is None:
        aux_guess = np.zeros(n_aux)

    # Precompute lookups used inside aux_residual (called many times by scipy.optimize.root)
    _vars_aux = model_funcs['vars_aux']
    _vars_exo = model_funcs.get('vars_exo', [])
    _known_vars = set(vars_dyn) | set(_vars_aux) | set(_vars_exo)
    _dyn_idx = {var: i for i, var in enumerate(vars_dyn)}
    _aux_idx = {var: i for i, var in enumerate(_vars_aux)}
    _exo_idx = {var: i for i, var in enumerate(_vars_exo)}

    # Pre-parse each symbol in aux_eqs_syms once
    _sym_parse = []  # list of (sym, var, lag) triples
    for sym in model_funcs['aux_eqs_syms']:
        parsed = _parse_time_symbol(sym.name)
        if parsed is not None and parsed[0] in _known_vars:
            var, lag = parsed
        elif parsed is not None:
            var, lag = sym.name, 0  # parameter that happens to look like var_N
        else:
            var, lag = sym.name, 0
        if var in _known_vars and lag != 0:
            raise ValueError(
                f"Auxiliary equations should be static (no leads/lags), but found {sym.name}"
            )
        _sym_parse.append((sym, var, lag))

    # Pre-resolve parameter values for each symbol (None = not a parameter)
    _param_vals = []
    for sym, var, lag in _sym_parse:
        if var not in _known_vars:
            if sym in params_dict:
                _param_vals.append(params_dict[sym])
            else:
                val = next(
                    (v for k, v in params_dict.items() if hasattr(k, 'name') and k.name == var),
                    None,
                )
                if val is None:
                    raise ValueError(f"Unknown symbol '{sym.name}' in auxiliary equations")
                _param_vals.append(val)
        else:
            _param_vals.append(None)  # will be filled at runtime

    # Build residual function for auxiliary equations
    def aux_residual(x_aux):
        # Build full argument list for auxiliary equation functions
        # aux_eqs_funcs expects arguments in order of aux_eqs_syms
        args = []
        for (sym, var, lag), param_val in zip(_sym_parse, _param_vals):
            if var in _dyn_idx:
                args.append(X_dyn_t[_dyn_idx[var]])
            elif var in _aux_idx:
                args.append(x_aux[_aux_idx[var]])
            elif var in _exo_idx:
                args.append(exog_t[_exo_idx[var]] if exog_t is not None else 0.0)
            else:
                args.append(param_val)

        # Evaluate all auxiliary equation residuals
        return np.array([func(*args) for func in model_funcs['aux_eqs_funcs']])

    # Solve auxiliary equations
    sol = root(aux_residual, aux_guess, method='hybr')

    if not sol.success:
        import warnings
        warnings.warn(f"Nested auxiliary solver failed: {sol.message}", UserWarning)

    return sol.x


def compute_auxiliary_variables(X_dyn, params_dict, model_funcs, vars_dyn, exog_path=None):
    """
    Compute auxiliary variables from dynamic variable solution

    Handles both analytical and nested methods based on model_funcs['aux_method'].

    Parameters:
    -----------
    X_dyn : array (T, n_dyn)
        Solution path for dynamic variables
    params_dict : dict
        Parameter values
    model_funcs : dict
        Model functions from process_model
    vars_dyn : list
        List of dynamic variable names
    exog_path : array (T, n_exo), optional
        Path for exogenous variables

    Returns:
    --------
    array (T, n_aux) : Solution path for auxiliary variables
    """
    if not model_funcs.get('vars_aux'):
        return None

    aux_method = model_funcs.get('aux_method', 'analytical')

    if aux_method == 'nested':
        # Nested method: solve numerically for each time period
        T = X_dyn.shape[0]
        n_aux = len(model_funcs['vars_aux'])
        X_aux = np.zeros((T, n_aux))
        aux_guess = np.zeros(n_aux)  # Use previous solution as guess

        for t in range(T):
            exog_t = exog_path[t] if exog_path is not None else None
            X_aux[t] = solve_auxiliary_nested(
                X_dyn[t], params_dict, model_funcs, vars_dyn, exog_t, aux_guess
            )
            aux_guess = X_aux[t]  # Warm start next period

        return X_aux

    elif aux_method == 'analytical':
        # Analytical method: evaluate closed-form solutions
        if not model_funcs.get('aux_funcs'):
            return None

        T = X_dyn.shape[0]
        n_aux = len(model_funcs['vars_aux'])
        X_aux = np.zeros((T, n_aux))

        _vars_exo = model_funcs.get('vars_exo', [])
        _known_vars = set(vars_dyn) | set(_vars_exo) | set(model_funcs['vars_aux'])
        _dyn_idx = {var: i for i, var in enumerate(vars_dyn)}
        _exo_idx = {var: i for i, var in enumerate(_vars_exo)}

        # Pre-parse symbols for each aux var and resolve parameter values once
        # Structure: {var_name: [(sym, var_base, lag, param_val_or_None), ...]}
        _sym_info = {}
        for var_name in model_funcs['vars_aux']:
            if var_name not in model_funcs['aux_funcs']:
                continue
            syms = model_funcs['aux_funcs'][var_name]['syms']
            entries = []
            for sym in syms:
                parsed = _parse_time_symbol(sym.name)
                if parsed is not None and parsed[0] in _known_vars:
                    var_base, lag = parsed
                else:
                    var_base, lag = sym.name, 0  # parameter with underscore or no underscore

                if var_base in _known_vars and lag != 0:
                    raise ValueError(
                        f"Auxiliary expressions must be static (lag=0), but '{sym.name}' "
                        f"has lag={lag}. Check your auxiliary equation definitions."
                    )

                # Resolve parameter value (None for runtime variables)
                if var_base not in _known_vars:
                    if sym in params_dict:
                        param_val = params_dict[sym]
                    else:
                        param_val = next(
                            (pv for pk, pv in params_dict.items()
                             if hasattr(pk, 'name') and pk.name == var_base),
                            None,
                        )
                        if param_val is None:
                            raise ValueError(
                                f"Symbol '{sym.name}' in auxiliary expression for '{var_name}' "
                                f"could not be resolved. Ensure all symbols are in vars_dyn, "
                                f"vars_exo, or params_dict."
                            )
                else:
                    param_val = None  # filled at runtime

                entries.append((var_base, lag, param_val))
            _sym_info[var_name] = entries

        # Evaluate per time period
        for t in range(T):
            for i, var_name in enumerate(model_funcs['vars_aux']):
                if var_name not in _sym_info:
                    continue
                func = model_funcs['aux_funcs'][var_name]['func']
                args = []
                for var_base, lag, param_val in _sym_info[var_name]:
                    if var_base in _dyn_idx:
                        args.append(X_dyn[t, _dyn_idx[var_base]])
                    elif var_base in _exo_idx:
                        args.append(exog_path[t, _exo_idx[var_base]] if exog_path is not None else 0.0)
                    else:
                        args.append(param_val)
                X_aux[t, i] = func(*args)

        return X_aux

    else:
        raise ValueError(
            f"Unknown aux_method {aux_method!r}. Supported methods are: "
            "'analytical', 'nested'."
        )

# ============================================================
# 10. Perfect foresight solver
# ============================================================

def _sparse_newton(F_func, J_sparse_func, x0, tol=1e-8, max_iter=50,
                   overdetermined=False, solver_options=None):
    """
    Sparse Newton-Raphson solver with backtracking line search.

    Uses spsolve for square systems (overdetermined=False) and lsmr for
    overdetermined systems. Both avoid forming dense matrices, making this
    feasible for large T (e.g. T=2000 with n=6 → 12000x12000 system).

    Parameters
    ----------
    F_func : callable
        Returns residual vector F(x).
    J_sparse_func : callable
        Returns sparse Jacobian J(x) as a scipy sparse matrix.
    x0 : ndarray
        Initial guess.
    tol : float
        Convergence tolerance on ||F||.
    max_iter : int
        Maximum Newton iterations.
    overdetermined : bool
        If True, uses lsmr (least-squares); else uses spsolve (direct).
    solver_options : dict, optional
        May contain 'maxiter' (overrides max_iter), 'maxfev' (maximum number of
        function evaluations), and 'ftol'/'xtol' (overrides tol).
    """
    from scipy.optimize import OptimizeResult

    if solver_options:
        max_iter = solver_options.get('maxiter', max_iter)
        # 'maxfev' caps total F evaluations (scipy convention); separate from iteration count
        max_nfev = solver_options.get('maxfev', None)
        tol = solver_options.get('ftol', tol)   # f-norm convergence tolerance
        xtol = solver_options.get('xtol', None)  # x-step tolerance (scipy convention)
    else:
        max_nfev = None
        xtol = None

    x = x0.copy()
    nrm = np.inf
    nfev = 0
    njev = 0
    it = -1
    success = False
    msg = f"Did not converge after {max_iter} iterations"
    fun = None  # residual vector at the final x
    F = None    # carry F_try forward to avoid re-evaluating F at the start of each iteration

    for it in range(max_iter):
        if F is None:
            F = F_func(x)
            nfev += 1
        fun = F

        if not np.isfinite(F).all():
            msg = f"Residual F contains non-finite values at iteration {it}"
            break

        nrm = np.linalg.norm(F)

        if nrm < tol:
            success = True
            msg = f"Converged at iteration {it}, ||F|| = {nrm:.2e}"
            break

        J = J_sparse_func(x)
        njev += 1

        try:
            if overdetermined:
                delta, *_ = lsmr(J, -F)
            else:
                delta = spsolve(J, -F)
        except Exception as e:
            msg = f"Linear solve failed at iteration {it}: {e}, ||F|| = {nrm:.2e}"
            break

        if not np.isfinite(delta).all():
            msg = f"Linear solve produced non-finite delta at iteration {it} (singular/ill-conditioned Jacobian), ||F|| = {nrm:.2e}"
            break

        # Backtracking line search
        alpha = 1.0
        improved = False
        maxfev_hit_in_linesearch = False
        for _ in range(30):
            if max_nfev is not None and nfev >= max_nfev:
                maxfev_hit_in_linesearch = True
                break
            x_try = x + alpha * delta
            F_try = F_func(x_try)
            nfev += 1
            if np.isfinite(F_try).all() and np.linalg.norm(F_try) < nrm:
                improved = True
                break
            alpha *= 0.5

        if maxfev_hit_in_linesearch:
            msg = f"Reached maxfev={max_nfev} function evaluations during line search at iteration {it}, ||F|| = {nrm:.2e}"
            break
        if not improved:
            msg = f"Line search failed at iteration {it}, ||F|| = {nrm:.2e}"
            break

        x = x + alpha * delta
        # Reuse the accepted F_try as next iteration's residual — x == x_try,
        # so F_try == F_func(x) and recomputing would be wasteful.
        F = F_try
        fun = F

        if max_nfev is not None and nfev >= max_nfev:
            msg = f"Reached maxfev={max_nfev} function evaluations at iteration {it}, ||F|| = {nrm:.2e}"
            break

        if xtol is not None and np.linalg.norm(alpha * delta) < xtol:
            # Small step — check whether residuals are also small.
            # F is already F_func(x) (reused from the accepted F_try above),
            # so no extra function evaluation is needed here.
            nrm_new = np.linalg.norm(F)
            if nrm_new < tol:
                success = True
                msg = f"Converged at iteration {it}, ||F|| = {nrm_new:.2e}"
            else:
                msg = f"Stagnated at iteration {it} (xtol), ||F|| = {nrm_new:.2e}"
            break

    return OptimizeResult(x=x, fun=fun, success=success, message=msg,
                          status=1 if success else 0, nfev=nfev, njev=njev)


def _infer_stock_var_indices(model_funcs, vars_dyn):
    """Return indices of variables that appear at any negative lag in the model.

    Stock (predetermined) variables are those whose value at t-1 enters the
    model equations — i.e., they appear at lag < 0 in the lead-lag incidence
    table stored in ``model_funcs['incidence']``.  Jump variables (e.g.
    consumption in a standard RBC) have no negative-lag appearances and are
    therefore free to jump at t=0 under the BVP formulation.

    Parameters
    ----------
    model_funcs : dict
        Output of ``process_model()``.
    vars_dyn : list of str
        Dynamic variable names (in column order).

    Returns
    -------
    list of int
        Sorted list of column indices into ``vars_dyn`` for stock variables.
    """
    try:
        incidence = model_funcs['incidence']
    except (TypeError, KeyError) as exc:
        raise ValueError(
            "model_funcs must contain an 'incidence' mapping when inferring "
            "stock variable indices; ensure model_funcs comes from process_model()."
        ) from exc
    if not hasattr(incidence, 'get'):
        raise ValueError(
            "model_funcs['incidence'] must be a dict-like mapping from "
            "variable names to iterables of lags."
        )
    return [i for i, var in enumerate(vars_dyn)
            if any(lag < 0 for lag in incidence.get(var, []))]


def solve_perfect_foresight(T, X0=None, params_dict=None, ss=None, model_funcs=None, vars_dyn=None,
                           exog_path=None, initial_state=None, ss_initial=None,
                           stock_var_indices=None, method='hybr',
                           solver_options=None, *, endval=None,
                           homotopy_fallback=True, homotopy_options=None):
    """
    Solve the perfect foresight problem using an augmented-path BVP formulation.

    The solver always uses the BVP (boundary value problem) formulation:
    a fixed ``initval`` row (pre-period-0 boundary) and a fixed ``endval``
    row (terminal steady state) are prepended/appended to the ``T``-row
    unknown path, giving a square ``T×n`` system solved by sparse Newton.

    Parameters:
    -----------
    T : int
        Number of periods
    X0 : ndarray or None, optional
        Initial guess for endogenous state path (T x n_endo).  If None
        (the default), the path is initialised to the terminal steady state
        (``endval`` if provided, otherwise ``ss``) tiled over all T periods.
    params_dict : dict
        Parameter values
    ss : ndarray
        Default steady-state values used as fallbacks when ``ss_initial`` and
        ``endval`` are not provided.  Specifically: ``ss_initial`` defaults to
        ``ss`` (pre-shock steady state) and ``endval`` defaults to ``ss``
        (terminal boundary).  For permanent shocks where the initial and
        terminal steady states differ, pass the pre-shock values as
        ``ss_initial`` and the post-shock values as ``endval`` (or as ``ss``
        with ``ss_initial`` set explicitly).
    model_funcs : dict
        Dictionary from process_model() containing compiled functions
    vars_dyn : list
        List of endogenous variable names
    exog_path : ndarray, optional
        Exogenous variable path (T x n_exo). If None, no exogenous variables.
    initial_state : ndarray, optional
        Pre-period-0 values of the stock variables (Dynare convention:
        ``k_{-1}``). The BVP formulation prepends this as the ``initval``
        boundary row; k_0 and all jump variables at t=0 are determined
        simultaneously by the model equations at t=0.
        If None, stock variables default to their initial steady-state values
        (``ss_initial[stock_var_indices]``), i.e., the economy starts at the
        initial steady state. When provided, must have the same length as
        ``stock_var_indices``.
    ss_initial : ndarray, optional
        Initial steady state (at exog[0]). If None, uses ss.
    stock_var_indices : list of int, optional
        Indices of stock (predetermined) variables in vars_dyn.  Stock
        variables are those that appear at lag < 0 in the model equations;
        their pre-period-0 values form the left BVP boundary.  Jump variables
        (no negative-lag appearances) are free to respond at t=0.
        If None, inferred automatically from the lead-lag incidence table in
        ``model_funcs['incidence']``.
        Example: vars_dyn=["c","k"], stock_var_indices=[1] means k is stock, c is jump.
    endval : ndarray, optional
        Terminal boundary values for all endogenous variables (the fixed right
        boundary row of the augmented path, appended at ``t = T``).  If None,
        defaults to ``ss``.  Override this for **permanent shock** scenarios
        where the economy converges to a different steady state than ``ss``:
        compute the new steady state (e.g. via
        ``compute_steady_state_numerical``) and pass it as ``endval``.
        Must match the **effective** dynamic variable vector used internally
        by the solver — ``model_funcs['vars_dyn']`` when present (e.g. when
        ``aux_method='dynamic'`` extends the variable list), falling back to
        the ``vars_dyn`` argument otherwise.  Construct ``endval`` consistently
        with ``ss`` and ``X0`` using that same variable ordering.
    method : str
        Deprecated. Previously selected the scipy.optimize.root method. The
        solver now always uses the sparse Newton method (_sparse_newton)
        regardless of this parameter. Kept for backward compatibility.
    solver_options : dict
        Options forwarded to _sparse_newton. Supported keys:
        'maxiter' (max Newton iterations), 'ftol' (f-norm tolerance),
        'xtol' (x-step tolerance), 'maxfev' (max function evaluations budget).
    homotopy_fallback : bool, default True
        If True and the sparse Newton solver fails to converge, automatically
        retries using solve_perfect_foresight_homotopy with the same inputs.
        A UserWarning is emitted when the fallback is triggered. Set to False
        to surface the Newton failure directly (original behaviour).
        Note: fallback is silently skipped when both ``initial_state`` and
        ``exog_path`` are None, because homotopy has nothing to scale in that
        case; the Newton failure result is returned directly.
    homotopy_options : dict, optional
        Options forwarded to ``solve_perfect_foresight_homotopy`` when the
        fallback is triggered. Supported keys:

        * ``n_steps`` (int, default 10): number of homotopy continuation steps.
        * ``verbose`` (bool, default False): if True, print homotopy progress.
        * ``exog_ss`` (ndarray): steady-state exogenous path (lam=0 value).
        * ``solver_options`` (dict): options for the Newton solver at each
          homotopy step (keys: ``'maxiter'``, ``'ftol'``, ``'xtol'``,
          ``'maxfev'``).
        * ``endval`` (ndarray, optional): terminal boundary override for the
          homotopy solver; if provided, it is interpolated from ``ss_initial``
          at lam=0 to this value at lam=1.
        * ``method`` (str, optional): deprecated; forwarded for backward
          compatibility only.

        Ignored when ``homotopy_fallback=False`` or when Newton succeeds.

    Returns:
    --------
    OptimizeResult : Solution with full path including X[0]
    """

    # Guard required args that were given None defaults to keep X0 optional.
    if params_dict is None:
        raise TypeError("solve_perfect_foresight() missing required argument: 'params_dict'")
    if ss is None:
        raise TypeError("solve_perfect_foresight() missing required argument: 'ss'")
    if model_funcs is None:
        raise TypeError("solve_perfect_foresight() missing required argument: 'model_funcs'")
    if vars_dyn is None:
        raise TypeError("solve_perfect_foresight() missing required argument: 'vars_dyn'")

    all_syms = model_funcs['all_syms']
    residual_funcs = model_funcs['residual_funcs']
    block_funcs = model_funcs['block_funcs']
    dynamic_eqs = model_funcs['dynamic_eqs']
    vars_exo = model_funcs.get('vars_exo', [])
    # Use vars_dyn from model_funcs — process_model may have extended it (e.g. dynamic fallback)
    vars_dyn = model_funcs.get('vars_dyn', vars_dyn)
    n = len(vars_dyn)
    # Precomputed lag sets — avoids rescanning all_syms on every Newton iteration.
    endo_lags = model_funcs.get('endo_lags')
    exo_lags  = model_funcs.get('exo_lags')
    endo_lags, exo_lags = _resolve_lag_sets(all_syms, vars_dyn, vars_exo, endo_lags, exo_lags)

    # Only validate X0 when it is explicitly provided. If X0 is None, a default
    # path based on endval (np.tile(endval, (T, 1))) is constructed later after
    # endval is resolved.
    if X0 is not None:
        X0 = np.asarray(X0, dtype=float)
        if X0.ndim != 2 or X0.shape != (T, n):
            raise ValueError(
                f"X0 must be a 2D array with shape (T, n) = ({T}, {n}); "
                f"got shape {X0.shape}. "
                f"If process_model fell back to aux_method='dynamic', "
                f"vars_dyn was extended to include auxiliary variables. "
                f"Reconstruct X0 and ss using model_funcs['vars_dyn']."
            )
    if ss.shape[0] != n:
        raise ValueError(
            f"ss has {ss.shape[0]} elements but the model has {n} dynamic variables "
            f"({vars_dyn}). Reconstruct ss using model_funcs['vars_dyn']."
        )

    if method != 'hybr':
        import warnings
        warnings.warn(
            f"The 'method' parameter is deprecated and ignored. The solver always uses "
            f"the sparse Newton method regardless of method={method!r}.",
            DeprecationWarning,
            stacklevel=2
        )

    if solver_options is None:
        solver_options = {}

    # Save originals before any mutation so the homotopy fallback receives
    # None vs. an explicit array (the distinction matters for homotopy's
    # endval interpolation and its "nothing to scale" guard).
    _orig_initial_state = initial_state
    _orig_endval = endval

    neq = len(dynamic_eqs)
    if neq != n:
        raise ValueError(
            f"Model has {neq} dynamic equation(s) but {n} dynamic variable(s). "
            f"The solver requires a square system (neq == n)."
        )

    # Infer stock_var_indices from the lead-lag incidence table if not provided.
    if stock_var_indices is None:
        stock_var_indices = _infer_stock_var_indices(model_funcs, vars_dyn)
    else:
        if not isinstance(stock_var_indices, (list, tuple, np.ndarray)):
            raise ValueError(
                "stock_var_indices must be a list, tuple, or numpy.ndarray; "
                f"got {type(stock_var_indices).__name__}. "
                "Sets and other unordered iterables are not accepted because "
                "index order must be deterministic."
            )
        stock_var_indices = list(stock_var_indices)
        if not all(isinstance(i, (int, np.integer)) for i in stock_var_indices):
            raise ValueError(
                "stock_var_indices must contain integers; "
                f"got types {[type(i).__name__ for i in stock_var_indices]}."
            )

    if len(set(stock_var_indices)) != len(stock_var_indices):
        raise ValueError("stock_var_indices contains duplicate indices.")
    if any(i < 0 or i >= n for i in stock_var_indices):
        raise ValueError(
            f"stock_var_indices contains out-of-range index. Valid range is [0, {n-1}]."
        )

    # Resolve ss_initial before defaulting initial_state so the default is
    # consistent with initval (which is built from ss_initial, not ss).
    if ss_initial is None:
        ss_initial = ss  # Default: initial SS same as terminal SS
    ss_initial = np.asarray(ss_initial, dtype=float).ravel()
    if len(ss_initial) != n:
        raise ValueError(
            f"ss_initial has {len(ss_initial)} elements but the model has "
            f"{n} dynamic variables."
        )

    if initial_state is not None:
        initial_state = np.asarray(initial_state, dtype=float).ravel()
    else:
        # Default: stock variables start at their initial steady-state values.
        initial_state = ss_initial[stock_var_indices]

    if len(initial_state) != len(stock_var_indices):
        hint = ""
        if len(initial_state) == n and len(stock_var_indices) != n:
            hint = (
                " It looks like you passed a full period-0 state vector "
                f"(length {n}). The legacy 'pin X[0]' standard mode has been "
                "removed; initial_state must now contain only the pre-period-0 "
                "values k_{-1} for each stock variable."
            )
        raise ValueError(
            f"initial_state has {len(initial_state)} element(s) but "
            f"{len(stock_var_indices)} stock variable(s) are expected "
            f"(stock_var_indices={stock_var_indices}). "
            "initial_state must contain one pre-period-0 value k_{{-1}} per "
            "stock variable. "
            "If stock_var_indices was not passed explicitly, it was inferred "
            f"from model_funcs['incidence'].{hint}"
        )

    # Augmented-path BVP formulation (always active).
    #
    # Dynare convention: the law of motion for a stock variable k is written as
    #   k_t = f(k_{t-1}, c_t)   (k appears at lag, not lead)
    # so ``initial_state`` supplies k_{-1} (the pre-period-0 value of k).
    #
    # We prepend an ``initval`` row (= boundary at t=-1) and append an
    # ``endval`` row (= ss) to the T-row unknown path.  The augmented T+2-row
    # path is passed to BVP-mode helpers which evaluate T equation-periods
    # using index t+lag+1, clamped to [0, T+1] so that any lag/lead beyond the
    # single boundary row reuses initval/endval (e.g. k_{-2} = k_{-1} = initval).
    # This yields T*neq equations for T*n unknowns (square since neq == n).
    # Actual non-singularity of the Jacobian depends on the model and calibration.

    # Warn when the model has lags/leads beyond ±1 in BVP mode.
    # The augmented path provides only one pre-sample row (initval) and one
    # post-sample row (endval), so values for |lag| > 1 are clamped to those
    # boundaries (e.g. k_{-2} = k_{-1} = initval).  This applies equally to
    # endogenous and exogenous variables.  The assumption is correct when the
    # economy was at steady state before t=0, but may be wrong otherwise.
    _has_long_endo = bool(endo_lags) and (min(endo_lags) < -1 or max(endo_lags) > 1)
    _has_long_exo  = bool(exo_lags)  and (min(exo_lags)  < -1 or max(exo_lags)  > 1)
    if _has_long_endo or _has_long_exo:
        import warnings
        warnings.warn(
            "BVP mode: the model contains lags/leads with |lag| > 1 "
            f"(endo_lags={endo_lags}, exo_lags={exo_lags}). Pre-sample values "
            "beyond the single initval row and post-sample values beyond endval "
            "are assumed equal to those boundary rows. This is correct when the "
            "economy was at steady state before t=0, but may produce incorrect "
            "dynamics otherwise.",
            UserWarning,
            stacklevel=2,
        )

    # initval row: stock vars at k_{-1} = initial_state;
    # non-stock vars at initial steady state (ss_initial).
    initval = np.asarray(ss_initial, dtype=float).ravel().copy()
    for pos, i in enumerate(stock_var_indices):
        initval[i] = initial_state[pos]
    # endval row: terminal boundary (defaults to ss; override for permanent shocks).
    if endval is None:
        endval = ss
    endval = np.asarray(endval, dtype=float).ravel().copy()
    if len(endval) != n:
        raise ValueError(
            f"endval has {len(endval)} elements but the model has {n} "
            f"dynamic variables. endval must be a full state vector."
        )

    # Default initial guess: terminal steady state tiled over T periods.
    if X0 is None:
        X0 = np.tile(endval, (T, 1))

    def F_bvp(x):
        X = x.reshape(T, n)
        return _residual_bvp(
            X, params_dict, all_syms, residual_funcs, vars_dyn, dynamic_eqs,
            vars_exo, exog_path, initval, endval, endo_lags, exo_lags,
        )

    def J_bvp(x):
        X = x.reshape(T, n)
        return _jacobian_bvp(
            X, params_dict, all_syms, block_funcs, vars_dyn, dynamic_eqs,
            vars_exo, exog_path, initval, endval, endo_lags, exo_lags,
        )

    sol = _sparse_newton(
        F_bvp, J_bvp, X0.ravel(),
        solver_options=solver_options,
    )

    if not sol.success and homotopy_fallback:
        # Homotopy requires at least one thing to scale.  When both are None
        # the fallback would raise ValueError("nothing to homotopy on"), which
        # would change the failure-mode contract from returning OptimizeResult
        # to raising.  Skip fallback in that case and fall through to return
        # the Newton failure result as-is.
        if _orig_initial_state is None and exog_path is None:
            pass
        else:
            import warnings
            warnings.warn(
                f"Standard solver failed to converge ({sol.message}). "
                "Retrying with homotopy (solve_perfect_foresight_homotopy). "
                "To disable this behaviour, pass homotopy_fallback=False.",
                UserWarning,
                stacklevel=2,
            )
            homotopy_opts = dict(homotopy_options) if homotopy_options is not None else {}
            # Do not forward solver_options from the failed Newton attempt: those
            # options (e.g. maxiter=0) caused the failure and would break every
            # homotopy sub-step.  The caller can pass solver_options inside
            # homotopy_options if per-step limits are desired.
            #
            # Pop 'endval' from homotopy_opts so it can be passed as the
            # explicit keyword argument.  If the user supplied it inside
            # homotopy_options, that value takes precedence over _orig_endval;
            # otherwise fall back to _orig_endval.  This avoids a TypeError
            # from duplicate keyword arguments.
            _homotopy_endval = homotopy_opts.pop('endval', _orig_endval)
            try:
                return solve_perfect_foresight_homotopy(
                    T, X0, params_dict, ss, model_funcs, vars_dyn,
                    exog_path=exog_path,
                    initial_state=_orig_initial_state,
                    ss_initial=ss_initial,
                    stock_var_indices=stock_var_indices,
                    endval=_homotopy_endval,
                    **homotopy_opts,
                )
            except RuntimeError as exc:
                from scipy.optimize import OptimizeResult as _OptRes
                # Assign to sol and fall through to the aux-variable block so
                # that x_aux/vars_aux are populated consistently with other
                # failure modes (Newton failure without fallback).
                sol = _OptRes(
                    success=False, status=0,
                    message=f"Homotopy fallback also failed: {exc}",
                    x=sol.x, fun=sol.fun,
                    nit=getattr(sol, 'nit', 0),
                    nfev=getattr(sol, 'nfev', 0),
                    njev=getattr(sol, 'njev', 0),
                )

    # Compute auxiliary variables if they exist.
    # Note: vars_aux is empty when aux_method='dynamic' (auxiliary variables were
    # merged into vars_dyn by process_model), so x_aux will be None in that case.
    # Users can access those variables via sol.x.reshape(T, -1) using the extended
    # vars_dyn from model_funcs['vars_dyn'].
    if model_funcs.get('vars_aux'):
        # Extract solution path for dynamic variables
        # At this point, sol.x is always the full (raveled) solution
        X_dyn_full = sol.x.reshape(T, n)

        # Compute auxiliary variables (handles both analytical and nested methods)
        X_aux = compute_auxiliary_variables(
            X_dyn_full, params_dict, model_funcs, vars_dyn, exog_path
        )

        # Add auxiliary variables to solution object
        sol.x_aux = X_aux
        sol.vars_aux = model_funcs['vars_aux']
    else:
        sol.x_aux = None
        sol.vars_aux = []

    return sol


# ============================================================
# 11. Expectation-errors solver (multiple MIT shocks)
# ============================================================

def solve_perfect_foresight_expectation_errors(
    T, X0=None, params_dict=None, ss=None, model_funcs=None, vars_dyn=None,
    news_shocks=None,
    initial_state=None,
    ss_initial=None,
    stock_var_indices=None,
    constant_simulation_length=False,
    solver_options=None,
    sub_x0=None,
):
    """
    Solve a perfect foresight model with multiple surprise (MIT) shocks.

    Replicates Dynare's ``perfect_foresight_with_expectation_errors_setup`` /
    ``perfect_foresight_with_expectation_errors_solver``.  At each ``learnt_in``
    period agents receive a surprise update to their information set and
    re-solve the perfect foresight problem from that point forward.  The full
    simulation path is stitched from the sub-simulations.

    Parameters
    ----------
    T : int
        Total simulation length (number of periods in the stitched output).
    X0 : ndarray, shape (T, n_endo), or None, optional
        Initial guess for the endogenous path, used as the warm-start for the
        first sub-solve.  Subsequent sub-solves are warm-started from the
        previous sub-solve's solution.  If None (the default), the path is
        initialised to ``ss`` tiled over all T periods.
    params_dict : dict
        Parameter values.
    ss : ndarray, shape (n_endo,)
        Default terminal steady state (initial right BVP boundary), used
        unless overridden by an ``endval`` provided in a ``news_shocks``
        3-tuple.
    model_funcs : dict
        Output of ``process_model()``.
    vars_dyn : list of str
        Endogenous variable names (may be extended by process_model).
    news_shocks : list of tuples
        Each entry is either a 2-tuple ``(learnt_in, exog_path)`` or a 3-tuple
        ``(learnt_in, exog_path, endval)`` where

        * ``learnt_in`` is the period at which agents receive new information
          (1-indexed, must be in ``[1, T]``).
        * ``exog_path`` is either an array with at least ``T_sub`` rows and
          ``n_exo`` columns, representing the agents' full belief about the
          shock path *from* ``learnt_in`` onward (row 0 is the shock at
          period ``learnt_in``, row 1 at period ``learnt_in + 1``, etc.),
          or ``None`` to pass an all-zero exogenous path to the sub-solver.
          **Note:** ``None`` is only correct when the exogenous steady state
          is zero; for level exogenous variables (e.g. ``z = 1``) you must
          supply an explicit path including the steady-state level, otherwise
          the simulation will be incorrect.  Only the first ``T_sub`` rows
          are used internally (``T_sub = T`` when
          ``constant_simulation_length=True``; otherwise
          ``T_sub = T - learnt_in + 1``); extra rows are ignored.
        * ``endval`` (optional) overrides the terminal BVP boundary (right-hand
          steady state) for this sub-solve *and all subsequent ones*.  Use this
          to replicate Dynare's ``endval(learnt_in=k)`` block, which signals a
          **permanent** shock that changes the terminal steady state.  If
          omitted the current ``endval`` (initialised to ``ss``) is reused.

        The list must be sorted by ``learnt_in`` and the first entry must have
        ``learnt_in == 1``.
    initial_state : ndarray, optional
        Pre-period-0 stock variable values (``k_{-1}`` in Dynare notation).
        Defaults to ``ss_initial[stock_var_indices]``.
    ss_initial : ndarray, optional
        Initial steady state.  Defaults to ``ss``.
    stock_var_indices : list of int, optional
        Inferred from incidence if not provided.
    constant_simulation_length : bool, default False
        If False (Dynare's default), each sub-solve uses the shrinking
        remaining horizon ``T - learnt_in + 1``.  If True (Dynare's
        ``constant_simulation_length`` option), every sub-solve runs for
        the full ``T`` periods.
    solver_options : dict, optional
        Forwarded unchanged to each ``solve_perfect_foresight`` sub-solve.
        Recognised keys: ``maxiter``, ``maxfev``, ``ftol``, ``xtol``.
    sub_x0 : list or tuple, optional
        Per-sub-solve initial guesses, one entry per element of ``news_shocks``.
        Each entry is either:

        * ``None`` — use the automatic warm-start (previous sub-solve's tail,
          or ``X0`` for the first sub-solve).
        * an ndarray of shape ``(T_sub, n_endo)`` — use this array as the
          warm-start for that sub-solve.  ``T_sub`` is
          ``T - learnt_in + 1`` (or ``T`` when
          ``constant_simulation_length=True``); the array will be
          trimmed or padded to ``T_sub`` rows if necessary, matching the
          behaviour applied to the automatic warm-start.

        If ``sub_x0`` is ``None`` (the default) the automatic warm-start is
        used for every sub-solve.

    Returns
    -------
    OptimizeResult
        ``sol.x`` : ndarray, shape (T * n_endo,)
            Stitched endogenous path, reshape to ``(T, n_endo)``.
            ``n_endo`` is ``len(model_funcs.get('vars_dyn', vars_dyn))``,
            i.e. the effective variable count after ``process_model``
            (same ordering as ``vars_dyn`` used internally).
        ``sol.success`` : bool
            True if every sub-solve converged.
        ``sol.status`` : int
            1 if all sub-solves converged, 0 otherwise.
        ``sol.message`` : str
            Human-readable summary; lists failing segments on failure.
        ``sol.sub_results`` : list of OptimizeResult
            Per-sub-solve results (one per entry in ``news_shocks``).
        ``sol.x_aux`` : ndarray or None
            Stitched auxiliary variable path, shape ``(T, n_aux)``, or None.
        ``sol.vars_aux`` : list of str
            Names of auxiliary variables (empty list if none).
    """
    from scipy.optimize import OptimizeResult

    # Guard required args that were given None defaults to keep X0 optional.
    if params_dict is None:
        raise TypeError(
            "solve_perfect_foresight_expectation_errors() missing required argument: 'params_dict'"
        )
    if ss is None:
        raise TypeError(
            "solve_perfect_foresight_expectation_errors() missing required argument: 'ss'"
        )
    if model_funcs is None:
        raise TypeError(
            "solve_perfect_foresight_expectation_errors() missing required argument: 'model_funcs'"
        )
    if vars_dyn is None:
        raise TypeError(
            "solve_perfect_foresight_expectation_errors() missing required argument: 'vars_dyn'"
        )
    if news_shocks is None:
        raise TypeError(
            "solve_perfect_foresight_expectation_errors() missing required argument: 'news_shocks'"
        )

    # Use vars_dyn from model_funcs (may be extended by process_model).
    vars_dyn = model_funcs.get('vars_dyn', vars_dyn)
    n = len(vars_dyn)

    # Infer stock_var_indices once, reuse for all sub-solves.
    if stock_var_indices is None:
        stock_var_indices = _infer_stock_var_indices(model_funcs, vars_dyn)
    else:
        if not isinstance(stock_var_indices, (list, tuple, np.ndarray)):
            raise ValueError(
                "stock_var_indices must be a list, tuple, or numpy.ndarray; "
                f"got {type(stock_var_indices).__name__}. "
                "Sets and other unordered iterables are not accepted because "
                "index order must be deterministic."
            )
        stock_var_indices = list(stock_var_indices)
        if not all(isinstance(i, (int, np.integer)) for i in stock_var_indices):
            raise ValueError(
                "stock_var_indices must contain integers; "
                f"got types {[type(i).__name__ for i in stock_var_indices]}."
            )

    if len(set(stock_var_indices)) != len(stock_var_indices):
        raise ValueError("stock_var_indices contains duplicate indices.")
    if any(i < 0 or i >= n for i in stock_var_indices):
        raise ValueError(
            f"stock_var_indices contains out-of-range index. Valid range is [0, {n-1}]."
        )

    # Require a sequence (list or tuple) so that len() is always available.
    if not isinstance(news_shocks, (list, tuple)):
        raise ValueError(
            f"news_shocks must be a list or tuple of tuples; got {type(news_shocks).__name__}."
        )

    # Normalise news_shocks: each entry → (learnt_in, exog_path, endval_override)
    # where endval_override is None if the 3rd element was not supplied.
    def _parse_entry(entry):
        import collections.abc
        if not isinstance(entry, (tuple, list, collections.abc.Sequence)) or isinstance(entry, (str, bytes)):
            raise ValueError(
                f"Each news_shocks entry must be a 2- or 3-tuple "
                f"(learnt_in, exog_path[, endval]); got {type(entry).__name__}."
            )
        if len(entry) == 2:
            return entry[0], entry[1], None
        elif len(entry) == 3:
            return entry[0], entry[1], entry[2]
        else:
            raise ValueError(
                f"Each news_shocks entry must be a 2- or 3-tuple "
                f"(learnt_in, exog_path[, endval]); got length {len(entry)}."
            )
    parsed = [_parse_entry(e) for e in news_shocks]

    # Validate.
    if not parsed:
        raise ValueError("news_shocks must be a non-empty list of (learnt_in, exog_path) tuples.")
    for p in parsed:
        li = p[0]
        if not isinstance(li, (int, np.integer)):
            raise ValueError(
                f"Each learnt_in must be an integer; got {type(li).__name__} ({li!r})."
            )
    learnt_ins = [p[0] for p in parsed]
    if learnt_ins != sorted(learnt_ins):
        raise ValueError("news_shocks must be sorted by learnt_in.")
    if learnt_ins[0] != 1:
        raise ValueError("The first entry in news_shocks must have learnt_in=1.")
    if any(li < 1 or li > T for li in learnt_ins):
        raise ValueError(f"All learnt_in values must be in [1, T={T}].")
    if len(set(learnt_ins)) != len(learnt_ins):
        raise ValueError("news_shocks contains duplicate learnt_in values.")

    # Validate sub_x0.
    if sub_x0 is not None:
        if not isinstance(sub_x0, (list, tuple)):
            raise ValueError(
                f"sub_x0 must be a list or tuple; got {type(sub_x0).__name__}."
            )
        if len(sub_x0) != len(parsed):
            raise ValueError(
                f"sub_x0 has length {len(sub_x0)} but news_shocks has "
                f"{len(parsed)} entries; they must have the same length."
            )
        for idx, entry in enumerate(sub_x0):
            if entry is None:
                continue
            try:
                arr = np.asarray(entry, dtype=float)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"sub_x0[{idx}] could not be converted to a numeric array: {exc}"
                ) from exc
            if arr.ndim != 2 or arr.shape[1] != n:
                raise ValueError(
                    f"sub_x0[{idx}] must be None or a 2D array with shape "
                    f"(T_sub, {n}); got shape {arr.shape}."
                )
            if arr.shape[0] == 0:
                raise ValueError(
                    f"sub_x0[{idx}] must have at least one row; got shape {arr.shape}."
                )

    all_pieces = []
    all_aux_pieces = []
    sub_results = []
    current_initial_state = initial_state
    current_endval = ss   # updated if a 3-tuple provides an endval override
    # Default initial guess: terminal steady state tiled over all T periods.
    X0_sub = X0 if X0 is not None else np.tile(ss, (T, 1))

    for i, (learnt_in, exog_path_i, endval_override) in enumerate(parsed):
        # Apply endval override (persists to subsequent segments).
        if endval_override is not None:
            current_endval = np.asarray(endval_override, dtype=float).ravel()

        # Number of periods to keep from this sub-solve's output.
        next_learnt_in = parsed[i + 1][0] if i + 1 < len(parsed) else T + 1
        n_keep = next_learnt_in - learnt_in

        # Sub-solve horizon.
        T_sub = T if constant_simulation_length else T - learnt_in + 1

        # Override warm-start with caller-supplied per-sub-solve guess if provided.
        if sub_x0 is not None and sub_x0[i] is not None:
            X0_sub = np.asarray(sub_x0[i], dtype=float)

        # Trim/pad X0_sub to T_sub rows.
        if X0_sub.shape[0] < T_sub:
            pad = np.tile(X0_sub[-1:], (T_sub - X0_sub.shape[0], 1))
            X0_sub = np.vstack([X0_sub, pad])
        elif X0_sub.shape[0] > T_sub:
            X0_sub = X0_sub[:T_sub]

        # Validate and trim exog_path_i to T_sub rows.
        exog_sub = None
        if exog_path_i is not None:
            exog_sub = np.asarray(exog_path_i, dtype=float)
            n_exo = len(model_funcs.get('vars_exo', []))
            if exog_sub.ndim != 2:
                raise ValueError(
                    f"news_shocks entry with learnt_in={learnt_in}: exog_path must be "
                    f"a 2D array of shape (T_sub, n_exo), got shape {exog_sub.shape}."
                )
            if exog_sub.shape[1] != n_exo:
                raise ValueError(
                    f"news_shocks entry with learnt_in={learnt_in}: exog_path has "
                    f"{exog_sub.shape[1]} column(s) but the model has {n_exo} "
                    f"exogenous variable(s)."
                )
            if exog_sub.shape[0] < T_sub:
                raise ValueError(
                    f"news_shocks entry with learnt_in={learnt_in}: exog_path has "
                    f"{exog_sub.shape[0]} row(s) but the sub-solve requires at least "
                    f"{T_sub} (T_sub = {'T' if constant_simulation_length else 'T - learnt_in + 1'} = {T_sub})."
                )
            exog_sub = exog_sub[:T_sub]

        sol = solve_perfect_foresight(
            T_sub, X0_sub, params_dict, ss, model_funcs, vars_dyn,
            exog_path=exog_sub,
            initial_state=current_initial_state,
            ss_initial=ss_initial,
            stock_var_indices=stock_var_indices,
            solver_options=solver_options,
            endval=current_endval,
        )
        sub_results.append(sol)

        X_sub = sol.x.reshape(T_sub, n)
        all_pieces.append(X_sub[:n_keep])

        # Auxiliary variables.
        if sol.x_aux is not None:
            all_aux_pieces.append(sol.x_aux[:n_keep])

        # Stock var values at the boundary (period next_learnt_in - 1).
        if i + 1 < len(parsed):
            current_initial_state = X_sub[n_keep - 1, stock_var_indices]

            # Warm-start next sub-solve: rows starting at next_learnt_in (index
            # n_keep), since current_initial_state already carries stocks at
            # next_learnt_in - 1.  Pad/trim to the *next* sub-solve's horizon.
            T_sub_next = T if constant_simulation_length else T - next_learnt_in + 1
            X0_sub = X_sub[n_keep:]
            if X0_sub.shape[0] < T_sub_next:
                pad = np.tile(X0_sub[-1:], (T_sub_next - X0_sub.shape[0], 1))
                X0_sub = np.vstack([X0_sub, pad])
            elif X0_sub.shape[0] > T_sub_next:
                X0_sub = X0_sub[:T_sub_next]

    X_full = np.vstack(all_pieces)  # (T, n)

    x_aux_full = None
    vars_aux = model_funcs.get('vars_aux', [])
    if all_aux_pieces and len(all_aux_pieces) == len(parsed):
        x_aux_full = np.vstack(all_aux_pieces)

    overall_success = all(s.success for s in sub_results)
    if overall_success:
        message = f"All {len(sub_results)} sub-solve(s) converged."
    else:
        failed = [
            f"learnt_in={parsed[i][0]}: {s.message}"
            for i, s in enumerate(sub_results)
            if not s.success
        ]
        message = "One or more sub-solves failed: " + "; ".join(failed)

    return OptimizeResult(
        x=X_full.ravel(),
        success=overall_success,
        status=1 if overall_success else 0,
        message=message,
        sub_results=sub_results,
        x_aux=x_aux_full,
        vars_aux=vars_aux,
    )


# ============================================================
# 12. Homotopy solver
# ============================================================

def solve_perfect_foresight_homotopy(
    T, X0=None, params_dict=None, ss=None, model_funcs=None, vars_dyn=None,
    exog_path=None, initial_state=None, ss_initial=None,
    stock_var_indices=None,
    *,
    endval=None,
    solver_options=None,
    n_steps=10, verbose=False, exog_ss=None,
    method='hybr',
):
    """
    Solve a perfect foresight model using homotopy (parameter continuation).

    When the Newton solver fails to converge from a direct initial guess --
    typically because the shock is large and the model is nonlinear -- homotopy
    gradually scales the shock from zero to its full size, using each
    intermediate solution as a warm start for the next step.

    The homotopy scales two sources of perturbation simultaneously:

    * ``initial_state`` deviation from ``ss_initial``
      (``initial_state_lam = ss_initial + lam * (initial_state - ss_initial)``)
    * ``exog_path`` deviation from ``exog_ss``
      (``exog_path_lam  = exog_ss  + lam * (exog_path  - exog_ss)``)

    Conceptually, ``lam`` varies from 0 to 1, where ``lam=0`` corresponds to
    the unshocked configuration implied by ``ss_initial`` and ``exog_ss``,
    and ``lam=1`` to the fully shocked problem.  The implementation does not
    explicitly solve a separate ``lam=0`` step; instead it uses the provided
    steady-state path (``np.tile(ss_initial, (T, 1))``) as the warm start for
    the first positive ``lam``.  At least one of ``initial_state`` or
    ``exog_path`` must be provided, otherwise there is nothing to scale.

    Parameters
    ----------
    T : int
        Number of periods.
    X0 : ndarray (T, n_dyn) or None, optional
        Unused by the homotopy solver (the actual warm start for step 1 is
        always the steady-state path ``np.tile(ss_initial, (T, 1))``).
        Accepted for API compatibility; pass None to omit.
    params_dict : dict
        Parameter values.
    ss : ndarray (n_dyn,)
        Terminal steady-state values.
    model_funcs : dict
        Output of ``process_model()``.
    vars_dyn : list
        Dynamic variable names.
    exog_path : ndarray (T, n_exo), optional
        Full-shock exogenous path (lam=1 value).
    initial_state : ndarray, optional
        Pre-period-0 values of the stock variables for the full-shock (lam=1)
        problem (Dynare convention: ``k_{-1}``).  The expected shape depends on
        ``stock_var_indices``:

        * If ``stock_var_indices`` is None, stock variables are inferred from
          the model's lead-lag incidence table, and ``initial_state`` must
          contain values for those inferred stock variables only.
        * If ``stock_var_indices`` is provided, ``initial_state`` must contain
          only the stock variable values at ``t=-1``, with shape ``(n_stock,)``.
          The BVP solver then determines all period-0 variables simultaneously.

        If None, defaults to steady-state values of the stock variables.
    ss_initial : ndarray, optional
        Initial steady state. Defaults to ``ss``.
    stock_var_indices : list of int, optional
        Indices of stock (predetermined) variables in ``vars_dyn``.  Non-stock
        variables are free to jump at t=0.  If None, inferred automatically
        from the lead-lag incidence table in ``model_funcs['incidence']``.
    endval : ndarray, optional
        Terminal boundary values (the fixed right boundary row of the augmented
        path).  If None, defaults to ``ss`` and is held fixed at ``ss`` for
        every homotopy step.  For permanent shocks, pass the new terminal
        steady state here; in that case ``endval`` is interpolated from
        ``ss_initial`` at ``lam=0`` to the provided value at ``lam=1``.

    The remaining parameters are **keyword-only** (enforced by ``*`` in
    the signature):

    solver_options : dict, optional
        Options forwarded to ``_sparse_newton`` at each homotopy step.
    n_steps : int
        Number of homotopy steps (default 10). Larger values help for very
        nonlinear models but increase total compute time.
    verbose : bool
        If True, print convergence status at each step.
    exog_ss : ndarray (T, n_exo) or (n_exo,), optional
        Steady-state exogenous path (lam=0 value). Defaults to zeros, which
        is appropriate when ``exog_path`` represents deviations from a
        zero-shock baseline.
    method : str, optional
        Deprecated and ignored; kept only for backward compatibility.
        The solver always uses the internal sparse Newton implementation
        (``_sparse_newton``), regardless of the value passed. Providing
        any non-default value emits a ``DeprecationWarning``. This
        parameter will be removed in a future release.

    Returns
    -------
    OptimizeResult
        Solution object from the final (lam=1) step, identical in structure
        to the output of ``solve_perfect_foresight``.

    Raises
    ------
    ValueError
        If neither ``initial_state`` nor ``exog_path`` is provided.
    RuntimeError
        If the solver fails to converge at any homotopy step, with the step's
        lam value and solver message included.
    """
    # Guard required args that were given None defaults to keep X0 optional.
    if params_dict is None:
        raise TypeError("solve_perfect_foresight_homotopy() missing required argument: 'params_dict'")
    if ss is None:
        raise TypeError("solve_perfect_foresight_homotopy() missing required argument: 'ss'")
    if model_funcs is None:
        raise TypeError("solve_perfect_foresight_homotopy() missing required argument: 'model_funcs'")
    if vars_dyn is None:
        raise TypeError("solve_perfect_foresight_homotopy() missing required argument: 'vars_dyn'")

    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be an int >= 1, got {n_steps!r}.")

    if method != 'hybr':
        import warnings
        warnings.warn(
            f"The 'method' parameter is deprecated and ignored. The solver always "
            f"uses the sparse Newton method regardless of method={method!r}.",
            DeprecationWarning,
            stacklevel=2,
        )

    if initial_state is None and exog_path is None:
        raise ValueError(
            "Homotopy requires at least one of 'initial_state' or 'exog_path' "
            "to scale. Both are None -- there is nothing to homotopy on."
        )

    if initial_state is not None:
        initial_state = np.asarray(initial_state, dtype=float).ravel()

    if ss_initial is None:
        ss_initial = ss
    ss_initial = np.asarray(ss_initial, dtype=float).ravel()

    vars_dyn_eff = model_funcs.get('vars_dyn', vars_dyn)
    n = len(vars_dyn_eff)

    if len(ss_initial) != n:
        raise ValueError(
            f"ss_initial has {len(ss_initial)} elements but the model has "
            f"{n} dynamic variables."
        )

    # Infer stock_var_indices from the lead-lag incidence table if not provided.
    if stock_var_indices is None:
        stock_var_indices = _infer_stock_var_indices(model_funcs, vars_dyn_eff)
    else:
        if not isinstance(stock_var_indices, (list, tuple, np.ndarray)):
            raise ValueError(
                "stock_var_indices must be a list, tuple, or numpy.ndarray; "
                f"got {type(stock_var_indices).__name__}. "
                "Sets and other unordered iterables are not accepted because "
                "index order must be deterministic."
            )
        stock_var_indices = list(stock_var_indices)
        if not all(isinstance(i, (int, np.integer)) for i in stock_var_indices):
            raise ValueError(
                "stock_var_indices must contain integers; "
                f"got types {[type(i).__name__ for i in stock_var_indices]}."
            )
        if len(set(stock_var_indices)) != len(stock_var_indices):
            raise ValueError("stock_var_indices contains duplicate indices.")
        if any(i < 0 or i >= n for i in stock_var_indices):
            raise ValueError(
                f"stock_var_indices contains an out-of-range index. "
                f"Valid range is [0, {n - 1}]."
            )

    # Default initial_state to steady-state stock values when not provided.
    if initial_state is None:
        initial_state = ss_initial[stock_var_indices]

    if len(initial_state) != len(stock_var_indices):
        hint = ""
        if len(initial_state) == n and len(stock_var_indices) != n:
            hint = (
                " It looks like you passed a full period-0 state vector "
                f"(length {n}). The legacy 'pin X[0]' standard mode has been "
                "removed; initial_state must now contain only the pre-period-0 "
                "values k_{-1} for each stock variable."
            )
        raise ValueError(
            f"initial_state has {len(initial_state)} element(s) but "
            f"{len(stock_var_indices)} stock variable(s) are expected "
            f"(stock_var_indices={stock_var_indices}). "
            "initial_state must contain only the pre-period-0 stock values. "
            "If stock_var_indices was not passed explicitly, it was inferred "
            f"from model_funcs['incidence'].{hint}"
        )

    # Validate and coerce exog_path
    if exog_path is not None:
        exog_path = np.asarray(exog_path, dtype=float)
        if exog_path.ndim != 2:
            raise ValueError(
                f"exog_path must be a 2-D array with shape (T, n_exo); "
                f"got shape {exog_path.shape}."
            )
        if exog_path.shape[0] != T:
            raise ValueError(
                f"exog_path has {exog_path.shape[0]} rows but T={T}."
            )
        n_exo_model = len(model_funcs.get('vars_exo', []))
        if n_exo_model == 0:
            raise ValueError(
                "exog_path was provided but the model defines no exogenous "
                "variables. Remove exog_path or add exogenous variables to "
                "the model."
            )
        if exog_path.shape[1] != n_exo_model:
            raise ValueError(
                f"exog_path has {exog_path.shape[1]} columns but the model "
                f"has {n_exo_model} exogenous variable(s)."
            )

    # Track whether exog_path was user-provided before potentially defaulting it.
    _exog_path_user_provided = exog_path is not None

    if exog_ss is not None and not _exog_path_user_provided:
        import warnings
        warnings.warn(
            "exog_ss was provided but exog_path is None, so exog_ss has no "
            "effect and will be ignored.",
            UserWarning,
            stacklevel=2,
        )

    # If the model has exogenous variables but no exog_path was given, default
    # to an all-zero path so the solver does not hit an IndexError in residual().
    if not _exog_path_user_provided:
        n_exo_model = len(model_funcs.get('vars_exo', []))
        if n_exo_model > 0:
            exog_path = np.zeros((T, n_exo_model))

    # Steady-state exogenous baseline (lam=0 value).
    # Only apply exog_ss when exog_path was user-provided; otherwise the path is
    # already zeros and exog_ss was already warned about / ignored above.
    if exog_path is not None:
        if _exog_path_user_provided and exog_ss is not None:
            exog_arr = np.asarray(exog_ss, dtype=float)
            _, n_exo = exog_path.shape
            # Accept a scalar as shorthand for (1,) when the model has one exo var
            if exog_arr.ndim == 0 and n_exo == 1:
                exog_arr = exog_arr.reshape(1)
            if exog_arr.shape == (n_exo,) or exog_arr.shape == exog_path.shape:
                exog_ss = np.broadcast_to(exog_arr, exog_path.shape).copy()
            else:
                raise ValueError(
                    f"exog_ss has shape {exog_arr.shape} but expected either "
                    f"{exog_path.shape} (T periods × {n_exo} exogenous "
                    f"variable(s)) or ({n_exo},) (one steady-state value per "
                    f"exogenous variable)."
                )
        else:
            exog_ss = np.zeros_like(exog_path)

    # Validate X0 shape when provided (X0 itself is not used as the warm start
    # — the steady-state path is — but a shape mismatch usually indicates a
    # caller error worth flagging immediately).
    if X0 is not None:
        X0 = np.asarray(X0, dtype=float)
        if X0.shape != (T, n):
            raise ValueError(
                f"X0 has shape {X0.shape} but expected ({T}, {n}) "
                f"(T periods × {n} dynamic variables). "
                f"If process_model fell back to aux_method='dynamic', vars_dyn was "
                f"extended to include auxiliary variables. "
                f"Reconstruct X0 and ss using model_funcs['vars_dyn']."
            )

    # Warm start for the first step: full steady-state path
    X_warm = np.tile(ss_initial, (T, 1))

    # Validate and resolve endval.  Track whether the caller supplied it
    # explicitly so we can (a) interpolate only when meaningful and
    # (b) emit a clearer error message when the root cause is a wrong-length ss.
    _endval_user_supplied = endval is not None
    if endval is None:
        endval = ss
    endval = np.asarray(endval, dtype=float).ravel().copy()
    if len(endval) != n:
        if _endval_user_supplied:
            raise ValueError(
                f"endval has {len(endval)} elements but the model has {n} "
                f"dynamic variables. endval must be a full state vector."
            )
        else:
            raise ValueError(
                f"ss has {len(endval)} elements but the model has {n} "
                f"dynamic variables. endval was not provided so it defaulted "
                f"to ss; check that ss matches vars_dyn."
            )

    # Baseline (lam=0) for initial_state interpolation: ss values of stock vars.
    ss_initial_stock = ss_initial[stock_var_indices]

    lambdas = np.linspace(0.0, 1.0, n_steps + 1)[1:]  # skip lam=0 (trivial)

    sol = None
    for step, lam in enumerate(lambdas, start=1):
        # Scale perturbations
        initial_state_lam = ss_initial_stock + lam * (initial_state - ss_initial_stock)
        exog_path_lam = (
            exog_ss + lam * (exog_path - exog_ss)
            if exog_path is not None else None
        )
        # Interpolate endval from ss_initial (lam=0) to the user-supplied
        # target (lam=1) only when the caller explicitly provided endval.
        # When endval was not supplied it defaults to ss and is kept fixed
        # every step, preserving backward-compatible behaviour.
        if _endval_user_supplied:
            endval_lam = ss_initial + lam * (endval - ss_initial)
        else:
            endval_lam = endval

        sol = solve_perfect_foresight(
            T, X_warm, params_dict, ss, model_funcs, vars_dyn_eff,
            exog_path=exog_path_lam,
            initial_state=initial_state_lam,
            ss_initial=ss_initial,
            stock_var_indices=stock_var_indices,
            endval=endval_lam,
            solver_options=solver_options,
            method='hybr',  # already warned above; suppress per-step warnings
            homotopy_fallback=False,  # prevent infinite recursion
        )

        if verbose:
            status = "converged" if sol.success else "FAILED"
            print(f"  homotopy step {step}/{n_steps} (lam={lam:.3f}): {status} -- {sol.message}")

        if not sol.success:
            raise RuntimeError(
                f"Homotopy failed to converge at lam={lam:.4f} "
                f"(step {step}/{n_steps}): {sol.message}. "
                f"Try increasing n_steps or adjusting solver_options."
            )

        # Use solution as warm start for the next step
        X_warm = sol.x.reshape(T, n)

    return sol


# ============================================================
# 13. Initial guess helper
# ============================================================

def make_initial_guess(T, ss_initial, ss_terminal, method='linear', decay=0.9):
    """
    Generate an initial guess path for the perfect foresight solver.

    Replicates the spirit of Dynare's ``perfect_foresight_setup`` path
    initialisation and adds an exponential option that better matches the
    saddle-path dynamics typical of DSGE models.

    Parameters
    ----------
    T : int
        Number of periods (rows of the returned array).
    ss_initial : array-like, shape (n,)
        Starting values — typically the initial steady state or the
        period-0 values you expect the path to begin near.
    ss_terminal : array-like, shape (n,)
        Terminal values — typically the terminal steady state ``ss``.
    method : {'linear', 'exponential', 'constant'}, default 'linear'
        How to interpolate between ``ss_initial`` and ``ss_terminal``:

        * ``'linear'`` — evenly spaced from ``ss_initial`` (t=0) to
          ``ss_terminal`` (t=T-1).  Matches Dynare's default
          ``perfect_foresight_setup`` behaviour when both ``initval`` and
          ``endval`` are supplied.
        * ``'exponential'`` — geometric convergence
          ``x(t) = ss_terminal + (ss_initial - ss_terminal) * decay**t``.
          The path closes most of the gap in early periods and flattens
          near ``ss_terminal``, mimicking saddle-path dynamics.  The gap
          never reaches exactly zero; use ``decay`` to control the speed.
        * ``'constant'`` — ``ss_terminal`` repeated for all periods.
          Equivalent to ``np.tile(ss_terminal, (T, 1))``.
    decay : float, default 0.9
        Decay rate for ``method='exponential'``.  Must be in ``(0, 1)``.
        Smaller values converge faster (e.g. ``0.5`` closes half the gap
        each period); values near 1 converge slowly and approach linear.
        Ignored for other methods.

    Returns
    -------
    X0 : ndarray, shape (T, n)
        Initial guess array, ready to pass as ``X0`` to
        ``solve_perfect_foresight`` or
        ``solve_perfect_foresight_expectation_errors``.

    Examples
    --------
    Linear interpolation (Dynare default):

    >>> X0 = make_initial_guess(T, ss_initial=ss, ss_terminal=ss_new)

    Exponential interpolation with faster convergence:

    >>> X0 = make_initial_guess(T, ss_initial=ss, ss_terminal=ss_new,
    ...                         method='exponential', decay=0.85)
    """
    ss_initial = np.asarray(ss_initial, dtype=float).ravel()
    ss_terminal = np.asarray(ss_terminal, dtype=float).ravel()
    if ss_initial.shape != ss_terminal.shape:
        raise ValueError(
            f"ss_initial and ss_terminal must have the same length; "
            f"got {ss_initial.shape} and {ss_terminal.shape}."
        )
    if not isinstance(T, (int, np.integer)):
        raise TypeError(f"T must be an integer; got {type(T).__name__} ({T!r}).")
    if T < 2:
        raise ValueError(f"T must be >= 2; got {T}.")

    if method == 'constant':
        return np.tile(ss_terminal, (T, 1))

    elif method == 'linear':
        # weights: 0 at t=0, 1 at t=T-1 (exact endpoints)
        weights = np.linspace(0.0, 1.0, T)
        return ss_initial[None, :] + weights[:, None] * (ss_terminal - ss_initial)[None, :]

    elif method == 'exponential':
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1); got {decay}.")
        t = np.arange(T, dtype=float)
        # x(t) = ss_terminal + (ss_initial - ss_terminal) * decay**t
        weights = decay ** t          # shape (T,); 1 at t=0, → 0 as t → ∞
        return ss_terminal[None, :] + weights[:, None] * (ss_initial - ss_terminal)[None, :]

    else:
        raise ValueError(
            f"method must be 'linear', 'exponential', or 'constant'; got {method!r}."
        )
