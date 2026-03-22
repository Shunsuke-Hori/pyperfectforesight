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

def residual(X, params, all_syms, residual_funcs, vars_dyn, dynamic_eqs, vars_exo=None, exog_path=None):
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

    Returns:
    --------
    ndarray : Flattened residual vector
    """
    T, n = X.shape
    neq = len(dynamic_eqs)
    F = np.zeros((T-1, neq))

    if vars_exo is None:
        vars_exo = []
    if exog_path is None:
        exog_path = np.zeros((T, 0))

    for t in range(T-1):
        subs = {}
        # Endogenous variables
        for i, var in enumerate(vars_dyn):
            for lag in [-1, 0, 1]:
                tt = min(max(t+lag, 0), T-1)
                subs[v(var, lag)] = X[tt, i]

        # Exogenous variables
        for i, var in enumerate(vars_exo):
            for lag in [-1, 0, 1]:
                tt = min(max(t+lag, 0), T-1)
                subs[v(var, lag)] = exog_path[tt, i]

        subs.update(params)

        # Use compiled functions
        vals = [subs[s] for s in all_syms]
        for i, func in enumerate(residual_funcs):
            F[t, i] = func(*vals)

    return F.ravel()

# ============================================================
# 6. Sparse block Jacobian
# ============================================================

def sparse_jacobian(X, params, all_syms, block_funcs, vars_dyn, dynamic_eqs, vars_exo=None, exog_path=None):
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

    Returns:
    --------
    sparse matrix : Sparse Jacobian in CSR format
    """
    T, n = X.shape
    neq = len(dynamic_eqs)

    if vars_exo is None:
        vars_exo = []
    if exog_path is None:
        exog_path = np.zeros((T, 0))

    J = lil_matrix((neq*(T-1), n*T))

    for t in range(T-1):
        subs = {}
        # Endogenous variables
        for i, var in enumerate(vars_dyn):
            for lag in [-1, 0, 1]:
                tt = min(max(t+lag, 0), T-1)
                subs[v(var, lag)] = X[tt, i]

        # Exogenous variables
        for i, var in enumerate(vars_exo):
            for lag in [-1, 0, 1]:
                tt = min(max(t+lag, 0), T-1)
                subs[v(var, lag)] = exog_path[tt, i]

        subs.update(params)

        for lag, f in block_funcs.items():
            if not (0 <= t+lag < T):
                continue

            B = f(*[subs[s] for s in all_syms])
            r0 = t*neq
            c0 = (t+lag)*n
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
        Method for handling auxiliary variables (default: 'auto'):
        - 'auto': Try analytical first; if SymPy can't solve, treat as dynamic.
                 Best default - fast when possible, robust when needed.
                 (Similar to Dynare: eliminate if possible, else keep in system)
        - 'analytical': Force analytical method only. Faster but will fail if
                       SymPy cannot solve the auxiliary equations symbolically.
        - 'nested': Force post-solve numerical solving of auxiliary equations.
                   After the main solver converges, auxiliary variables are
                   solved period-by-period using scipy.optimize.root with warm
                   starting. This mode requires a square auxiliary system
                   (same number of equations and variables) and that auxiliary
                   variables appear only in auxiliary equations — a ValueError
                   is raised otherwise. Use when analytical solve fails and
                   these structural conditions are satisfied; otherwise prefer
                   'dynamic'.
        - 'dynamic': Treat auxiliary variables as dynamic. Auxiliary equations
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
    - 'auto' (default): Best for most cases - tries analytical, falls back to dynamic.
                       Like Dynare: eliminate if possible, else keep in system.
    - 'analytical': When you know equations are simple (e.g., i = y - c - g) and
                   want guaranteed fast performance with no fallback.
    - 'nested': When you specifically want nested optimization (rare use case).
    - 'dynamic': When you want to skip analytical attempt and go straight to
                treating as dynamic variables.

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

    return {
        'dynamic_eqs': dynamic_eqs,
        'blocks': blocks,
        'all_syms': all_syms,
        'block_funcs': block_funcs,
        'residual_funcs': residual_funcs,
        'incidence': incidence,
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


def solve_perfect_foresight(T, X0, params_dict, ss, model_funcs, vars_dyn,
                           exog_path=None, initial_state=None, ss_initial=None,
                           stock_var_indices=None,
                           method='hybr', use_terminal_conditions=True, solver_options=None):
    """
    Solve the perfect foresight problem

    Parameters:
    -----------
    T : int
        Number of periods
    X0 : ndarray
        Initial guess for endogenous state path (T x n_endo)
    params_dict : dict
        Parameter values
    ss : ndarray
        Terminal steady state values for endogenous variables (at exog[T-1])
    model_funcs : dict
        Dictionary from process_model() containing compiled functions
    vars_dyn : list
        List of endogenous variable names
    exog_path : ndarray, optional
        Exogenous variable path (T x n_exo). If None, no exogenous variables.
    initial_state : ndarray, optional
        Initial state values. Interpretation depends on stock_var_indices:
        - If stock_var_indices is None: all variables (old behavior, all stock)
        - If stock_var_indices provided: only stock variable initial values
    ss_initial : ndarray, optional
        Initial steady state (at exog[0]). If None, uses ss.
    stock_var_indices : list of int, optional
        Indices of stock variables in vars_dyn. Stock variables are predetermined
        (fixed at t=0, free at t=T-1). Jump variables are free at t=0, fixed at t=T-1.
        If None, treats all variables as stock (backward compatible).
        Example: vars_dyn=["c","k"], stock_var_indices=[1] means k is stock, c is jump.
    method : str
        Deprecated. Previously selected the scipy.optimize.root method ('hybr',
        'lm', 'krylov', etc.). The solver now always uses the sparse Newton
        method (_sparse_newton) regardless of this parameter. Kept for backward
        compatibility.
    use_terminal_conditions : bool
        Whether to enforce terminal steady-state conditions (default: True)
    solver_options : dict
        Options forwarded to _sparse_newton. Supported keys:
        'maxiter' (max Newton iterations), 'ftol' (f-norm tolerance),
        'xtol' (x-step tolerance), 'maxfev' (max function evaluations budget).

    Returns:
    --------
    OptimizeResult : Solution with full path including X[0]
    """

    all_syms = model_funcs['all_syms']
    residual_funcs = model_funcs['residual_funcs']
    block_funcs = model_funcs['block_funcs']
    dynamic_eqs = model_funcs['dynamic_eqs']
    vars_exo = model_funcs.get('vars_exo', [])
    # Use vars_dyn from model_funcs — process_model may have extended it (e.g. dynamic fallback)
    vars_dyn = model_funcs.get('vars_dyn', vars_dyn)
    n = len(vars_dyn)

    if X0.shape[1] != n:
        raise ValueError(
            f"X0 has {X0.shape[1]} columns but the model has {n} dynamic variables "
            f"({vars_dyn}). If process_model fell back to aux_method='dynamic', "
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

    if ss_initial is None:
        ss_initial = ss  # Default: initial SS same as terminal SS

    # Handle stock vs jump variables
    if stock_var_indices is not None and initial_state is not None:
        # New behavior: distinguish stock (predetermined) and jump variables
        # Stock variables: fixed at t=0, free at t=T-1
        # Jump variables: free at t=0, fixed at t=T-1
        if len(initial_state) != len(stock_var_indices):
            raise ValueError(
                f"initial_state has {len(initial_state)} elements but stock_var_indices "
                f"has {len(stock_var_indices)} entries. They must match."
            )
        if len(set(stock_var_indices)) != len(stock_var_indices):
            raise ValueError("stock_var_indices contains duplicate indices.")
        if any(i < 0 or i >= n for i in stock_var_indices):
            raise ValueError(
                f"stock_var_indices contains out-of-range index. Valid range is [0, {n-1}]."
            )
        stock_indices = set(stock_var_indices)
        jump_indices = set(range(n)) - stock_indices
        # Precompute position of each stock variable index within stock_var_indices
        # so reconstruct_full_path avoids repeated linear scans during Newton iterations.
        stock_idx_to_pos = {idx: pos for pos, idx in enumerate(stock_var_indices)}

        # Build list of which (time, variable) pairs are unknowns
        # and which are fixed
        def build_solution_vector(X_full):
            """Extract unknowns from full path X_full"""
            x = []
            for i in range(n):
                if i in stock_indices:
                    # Stock variable: X[1:T,i] are unknowns
                    x.extend(X_full[1:, i])
                else:
                    # Jump variable: X[0:T-1,i] are unknowns
                    x.extend(X_full[:-1, i])
            return np.array(x)

        def reconstruct_full_path(x):
            """Reconstruct full path from solution vector x"""
            X_full = np.zeros((T, n))
            idx = 0
            for i in range(n):
                if i in stock_indices:
                    # Stock variable: X[0,i] fixed, X[1:T,i] from solution
                    X_full[0, i] = initial_state[stock_idx_to_pos[i]]
                    X_full[1:, i] = x[idx:idx+T-1]
                    idx += T-1
                else:
                    # Jump variable: X[0:T-1,i] from solution, X[T-1,i] fixed at ss
                    X_full[:-1, i] = x[idx:idx+T-1]
                    X_full[T-1, i] = ss[i]
                    idx += T-1
            return X_full

        def F_full(x):
            X = reconstruct_full_path(x)
            F = residual(X, params_dict, all_syms, residual_funcs, vars_dyn, dynamic_eqs, vars_exo, exog_path)
            return F

        # Precompute column indices for unknown variables (constant across iterations)
        col_indices = [
            t * n + i
            for i in range(n)
            for t in range(T)
            if (i in stock_indices and t > 0) or (i not in stock_indices and t < T - 1)
        ]

        def J_full(x):
            X = reconstruct_full_path(x)
            J_all = sparse_jacobian(X, params_dict, all_syms, block_funcs, vars_dyn, dynamic_eqs, vars_exo, exog_path)
            return J_all[:, col_indices]

        # Initial guess
        x0_solve = build_solution_vector(X0)

        sol = _sparse_newton(
            F_full, J_full, x0_solve,
            solver_options=solver_options
        )

        # Reconstruct full solution
        X_full = reconstruct_full_path(sol.x)
        sol.x = X_full.ravel()

    elif initial_state is not None:
        # Case 1: X[0] = initial_state (given), X[T-1] = ss (terminal condition)
        # Unknowns: X[1:T-1] → (T-2)*n values
        # Equations: ALL dynamics periods 0 to T-2 → (T-1)*n equations
        # System is overdetermined, so use least-squares approach (method='lm')

        def F_full(x):
            # x contains X[1:T-1] (middle path, not including X[0] or X[T-1])
            X_middle = x.reshape(T-2, -1)
            # Reconstruct full path: X[0] (initial) + X[1:T-1] (unknown) + X[T-1] (terminal SS)
            X = np.vstack([initial_state.reshape(1, -1), X_middle, ss.reshape(1, -1)])

            # Compute ALL dynamic equations for periods 0 to T-2
            # This includes the equation linking X[T-2] to X[T-1]=ss
            F = residual(X, params_dict, all_syms, residual_funcs, vars_dyn, dynamic_eqs, vars_exo, exog_path)

            # F has (T-1)*n equations for (T-2)*n unknowns → overdetermined
            # The solver (especially 'lm') will minimize ||F||^2

            return F

        def J_full(x):
            X_middle = x.reshape(T-2, -1)
            X = np.vstack([initial_state.reshape(1, -1), X_middle, ss.reshape(1, -1)])

            J = sparse_jacobian(X, params_dict, all_syms, block_funcs, vars_dyn, dynamic_eqs, vars_exo, exog_path)

            # J has shape ((T-1)*n, T*n)
            # Drop first n columns (X[0], which is fixed)
            # Drop last n columns (X[T-1], which is fixed at ss)
            J = J[:, n:-n]

            return J  # sparse matrix — no .toarray()

        # Initial guess for X[1:T-1]
        x0_solve = X0[1:-1, :].ravel()

        sol = _sparse_newton(
            F_full, J_full, x0_solve,
            overdetermined=True,
            solver_options=solver_options
        )

        # Reconstruct full solution: X[0] + X[1:T-1] + X[T-1]
        X_middle = sol.x.reshape(T-2, -1)
        X_full = np.vstack([initial_state.reshape(1, -1), X_middle, ss.reshape(1, -1)])
        sol.x = X_full.ravel()

    else:
        # Case 2: Solve for all X[0:T], enforce X[0]=ss_initial and X[T-1]=ss
        def F_full(x):
            X = x.reshape(T, -1)
            F = residual(X, params_dict, all_syms, residual_funcs, vars_dyn, dynamic_eqs, vars_exo, exog_path)
            initial_resid = X[0, :] - ss_initial
            if use_terminal_conditions:
                terminal_resid = X[-1, :] - ss
                return np.concatenate([F, initial_resid, terminal_resid])
            return np.concatenate([F, initial_resid])

        def J_full(x):
            from scipy.sparse import lil_matrix, vstack
            X = x.reshape(T, -1)
            J = sparse_jacobian(X, params_dict, all_syms, block_funcs, vars_dyn, dynamic_eqs, vars_exo, exog_path)
            initial_rows = lil_matrix((n, n*T))
            for i in range(n):
                initial_rows[i, i] = 1.0
            J = vstack([J, initial_rows.tocsr()])
            if use_terminal_conditions:
                # Append terminal condition rows directly — avoids recomputing F
                terminal_rows = lil_matrix((n, n*T))
                for i in range(n):
                    terminal_rows[i, n*(T-1) + i] = 1.0
                J = vstack([J, terminal_rows.tocsr()])
            return J

        sol = _sparse_newton(
            F_full, J_full, X0.ravel(),
            overdetermined=use_terminal_conditions,
            solver_options=solver_options
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
# 11. Homotopy solver
# ============================================================

def solve_perfect_foresight_homotopy(
    T, X0, params_dict, ss, model_funcs, vars_dyn,
    exog_path=None, initial_state=None, ss_initial=None,
    stock_var_indices=None,
    use_terminal_conditions=True, solver_options=None,
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
    X0 : ndarray (T, n_dyn)
        Initial guess shape reference; the actual warm start for step 1 is
        the steady-state path.
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
        Initial state at t=0 for the full-shock (lam=1) problem. The expected
        shape depends on ``stock_var_indices``:

        * If ``stock_var_indices`` is None, ``initial_state`` must be a full
          state vector of shape ``(n_dyn,)``.
        * If ``stock_var_indices`` is provided, ``initial_state`` must contain
          only the stock variable values at t=0, with shape ``(n_stock,)``.
    ss_initial : ndarray, optional
        Initial steady state. Defaults to ``ss``.
    stock_var_indices : list of int, optional
        Indices of stock (predetermined) variables in ``vars_dyn``. When
        provided, ``initial_state`` is interpreted as a stock-only vector
        (see above); non-stock variables are free to jump at t=0.
    use_terminal_conditions : bool
        Enforce terminal steady-state conditions (default True).
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
    if not isinstance(n_steps, int) or n_steps < 1:
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

    if initial_state is not None and stock_var_indices is not None:
        if len(initial_state) != len(stock_var_indices):
            raise ValueError(
                f"initial_state has {len(initial_state)} elements but "
                f"stock_var_indices has {len(stock_var_indices)} entries. "
                f"When stock_var_indices is provided, initial_state must "
                f"contain only the stock variable values."
            )

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

    # Validate stock_var_indices before using them to slice ss_initial
    if stock_var_indices is not None:
        if len(set(stock_var_indices)) != len(stock_var_indices):
            raise ValueError("stock_var_indices contains duplicate indices.")
        if any(i < 0 or i >= n for i in stock_var_indices):
            raise ValueError(
                f"stock_var_indices contains an out-of-range index. "
                f"Valid range is [0, {n - 1}]."
            )

    # Validate full initial_state length when no stock_var_indices provided
    if initial_state is not None and stock_var_indices is None:
        if len(initial_state) != n:
            raise ValueError(
                f"initial_state has {len(initial_state)} elements but the "
                f"model has {n} dynamic variables. When stock_var_indices is "
                f"None, initial_state must be a full state vector."
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

    # Validate X0 shape to catch mismatches early (X0 itself is not used as
    # the warm start — the steady-state path is — but a shape mismatch usually
    # indicates a caller error worth flagging immediately).
    X0 = np.asarray(X0, dtype=float)
    if X0.shape != (T, n):
        raise ValueError(
            f"X0 has shape {X0.shape} but expected ({T}, {n}) "
            f"(T periods × {n} dynamic variables)."
        )

    # Warm start for the first step: full steady-state path
    X_warm = np.tile(ss_initial, (T, 1))

    # Baseline (lam=0) for initial_state interpolation.
    # When stock_var_indices is provided, initial_state contains only stock
    # variable values, so we extract the matching slice of ss_initial.
    if initial_state is not None:
        if stock_var_indices is not None:
            ss_initial_stock = ss_initial[stock_var_indices]
        else:
            ss_initial_stock = ss_initial

    lambdas = np.linspace(0.0, 1.0, n_steps + 1)[1:]  # skip lam=0 (trivial)

    sol = None
    for step, lam in enumerate(lambdas, start=1):
        # Scale perturbations
        initial_state_lam = (
            ss_initial_stock + lam * (initial_state - ss_initial_stock)
            if initial_state is not None else None
        )
        exog_path_lam = (
            exog_ss + lam * (exog_path - exog_ss)
            if exog_path is not None else None
        )

        sol = solve_perfect_foresight(
            T, X_warm, params_dict, ss, model_funcs, vars_dyn_eff,
            exog_path=exog_path_lam,
            initial_state=initial_state_lam,
            ss_initial=ss_initial,
            stock_var_indices=stock_var_indices,
            use_terminal_conditions=use_terminal_conditions,
            solver_options=solver_options,
            method='hybr',  # already warned above; suppress per-step warnings
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
