"""
main.py
A minimal Dynare-style perfect foresight solver framework in Python

This module provides the core infrastructure for solving perfect foresight
models. For usage examples, see demo.py.
"""

import sympy as sp
import numpy as np
from scipy.sparse import lil_matrix, vstack

# ============================================================
# 1. Utilities
# ============================================================

def v(name, lag):
    """Time-indexed symbolic variable"""
    return sp.Symbol(f"{name}_{lag}")

# ============================================================
# 2. Lead / lag detection (Dynare lead_lag_incidence)
# ============================================================

def lead_lag_incidence(equations):
    """
    Detect which variables appear at which time lags in the equations

    Parameters:
    -----------
    equations : list
        List of sympy equations

    Returns:
    --------
    dict : Dictionary mapping variable names to sets of lags
    """
    inc = {}
    for eq in equations:
        for s in eq.free_symbols:
            if "_" not in s.name:
                continue  # Skip parameters without time index
            name, lag = s.name.split("_")
            inc.setdefault(name, set()).add(int(lag))
    return inc

# ============================================================
# 3. Static equation detection & elimination
# ============================================================

def is_static(eq):
    """
    Check if an equation contains only current period variables

    Parameters:
    -----------
    eq : sympy expression
        Equation to check

    Returns:
    --------
    bool : True if equation is static (no leads/lags)
    """
    for s in eq.free_symbols:
        if s.name.endswith("_1") or s.name.endswith("_-1"):
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
            if "_" in s.name:
                name, lag = s.name.split("_")
                if name in variables:
                    all_lags.add(int(lag))

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
            if "_" in s.name:
                try:
                    name, lag = s.name.split("_")
                    if name in vars_dyn:
                        all_lags.add(int(lag))
                except:
                    pass

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
        - 'nested': Force nested root finding at each solver iteration. Always
                   works but slower. Use when you specifically want this approach.
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
    The nested method solves numerically at each iteration (specialized use).
    """
    if vars_exo is None:
        vars_exo = []
    if vars_aux is None:
        vars_aux = []

    # Lead/lag detection
    incidence = lead_lag_incidence(equations)

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
            if any(aux_var in eq_vars for aux_var in aux_var_syms) and is_static(eq):
                aux_eqs.append(eq)
            else:
                remaining_eqs.append(eq)

        if not aux_eqs:
            import warnings
            warnings.warn(
                f"No static equations found for auxiliary variables {vars_aux}. "
                f"Treating them as dynamic variables.",
                UserWarning
            )
            aux_method_used = 'dynamic'

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
                    aux_sols = aux_sols_list[0]
                    # Check that all auxiliary variables were solved
                    if all(v(name, 0) in aux_sols for name in vars_aux):
                        solved_successfully = True
            except Exception:
                solved_successfully = False

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
                # Remove auxiliary equations from system
                equations = remaining_eqs
                aux_method_used = 'analytical'
            else:
                # Analytical solve failed
                if aux_method == 'analytical':
                    # User forced analytical only - raise error
                    raise ValueError(
                        f"Could not solve auxiliary equations analytically for {vars_aux}. "
                        f"SymPy failed to find symbolic solutions. "
                        f"Consider using aux_method='auto' (tries analytical then dynamic) "
                        f"or aux_method='dynamic' (include auxiliary equations in system)."
                    )
                else:  # aux_method == 'auto'
                    # Auto fallback to dynamic (Dynare-style)
                    import warnings
                    warnings.warn(
                        f"Could not solve auxiliary equations analytically for {vars_aux}. "
                        f"Treating auxiliary variables as dynamic (keeping equations in system). "
                        f"This follows Dynare's approach when analytical elimination fails.",
                        UserWarning
                    )
                    aux_method_used = 'dynamic'
                    aux_sols = {}
                    aux_funcs = {}

        if aux_eqs and (aux_method == 'nested' or aux_method_used == 'nested'):
            # NESTED METHOD: Compile residual functions for numerical solving
            # Get all symbols that appear in auxiliary equations
            aux_eqs_syms = sorted(
                {s for eq in aux_eqs for s in eq.free_symbols},
                key=lambda s: s.name
            )

            if compiler == 'lambdify':
                # Compile each auxiliary equation as a residual function
                aux_eqs_funcs = [sp.lambdify(aux_eqs_syms, eq, "numpy") for eq in aux_eqs]

            # Remove auxiliary equations from dynamic system
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
    if eliminate_static_vars:
        static_eqs = [eq for eq in equations if is_static(eq) and eq not in aux_eqs_set]
        dynamic_eqs = [eq for eq in equations if not is_static(eq)]

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
        return None

    n_aux = len(model_funcs['vars_aux'])
    if aux_guess is None:
        aux_guess = np.zeros(n_aux)

    # Build residual function for auxiliary equations
    def aux_residual(x_aux):
        # Build full argument list for auxiliary equation functions
        # aux_eqs_funcs expects arguments in order of aux_eqs_syms
        args = []
        for sym in model_funcs['aux_eqs_syms']:
            sym_name = sym.name
            # Parse symbol: var_lag
            if '_' in sym_name:
                parts = sym_name.rsplit('_', 1)
                var = parts[0]
                try:
                    lag = int(parts[1])
                except ValueError:
                    var = sym_name
                    lag = 0
            else:
                var = sym_name
                lag = 0

            # Only lag=0 is valid for static equations
            if lag != 0:
                raise ValueError(f"Auxiliary equations should be static (no leads/lags), but found {sym_name}")

            # Get value
            if var in vars_dyn:
                var_idx = vars_dyn.index(var)
                args.append(X_dyn_t[var_idx])
            elif var in model_funcs['vars_aux']:
                var_idx = model_funcs['vars_aux'].index(var)
                args.append(x_aux[var_idx])
            elif exog_t is not None and var in model_funcs.get('vars_exo', []):
                var_idx = model_funcs['vars_exo'].index(var)
                args.append(exog_t[var_idx])
            elif sym in params_dict:
                args.append(params_dict[sym])
            else:
                # Try to find parameter by name
                found = False
                for param_sym, param_val in params_dict.items():
                    if hasattr(param_sym, 'name') and param_sym.name == var:
                        args.append(param_val)
                        found = True
                        break
                if not found:
                    raise ValueError(f"Unknown symbol {sym_name} in auxiliary equations")

        # Evaluate all auxiliary equation residuals
        residuals = np.array([func(*args) for func in model_funcs['aux_eqs_funcs']])
        return residuals

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

        # For each time period, compute auxiliary variables
        for t in range(T):
            for i, var_name in enumerate(model_funcs['vars_aux']):
                if var_name in model_funcs['aux_funcs']:
                    aux_info = model_funcs['aux_funcs'][var_name]
                    func = aux_info['func']
                    syms = aux_info['syms']

                    # Build argument list for the function
                    args = []
                    for sym in syms:
                        sym_name = sym.name
                        # Parse symbol name: varname_lag
                        if '_' in sym_name:
                            parts = sym_name.rsplit('_', 1)
                            var = parts[0]
                            lag = int(parts[1])
                        else:
                            var = sym_name
                            lag = 0

                        # Get value from appropriate source
                        if var in vars_dyn:
                            var_idx = vars_dyn.index(var)
                            time_idx = t + lag
                            if 0 <= time_idx < T:
                                args.append(X_dyn[time_idx, var_idx])
                            else:
                                args.append(0)  # Out of bounds, use 0
                        elif exog_path is not None and var in model_funcs.get('vars_exo', []):
                            var_idx = model_funcs['vars_exo'].index(var)
                            time_idx = t + lag
                            if 0 <= time_idx < T:
                                args.append(exog_path[time_idx, var_idx])
                            else:
                                args.append(0)
                        elif sym in params_dict:
                            args.append(params_dict[sym])
                        else:
                            # Try to find parameter by name
                            found = False
                            for param_sym, param_val in params_dict.items():
                                if hasattr(param_sym, 'name') and param_sym.name == var:
                                    args.append(param_val)
                                    found = True
                                    break
                            if not found:
                                args.append(0)  # Unknown, use 0

                    # Compute auxiliary variable value
                    X_aux[t, i] = func(*args)

        return X_aux

    else:
        # Unknown method
        return None

# ============================================================
# 10. Perfect foresight solver
# ============================================================

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
        Solver method: 'hybr' (default), 'lm', 'krylov', etc.
    use_terminal_conditions : bool
        Whether to enforce terminal steady-state conditions (default: True)
    solver_options : dict
        Additional options to pass to scipy.optimize.root

    Returns:
    --------
    OptimizeResult : Solution from scipy.optimize.root, with full path including X[0]
    """
    from scipy.optimize import root

    all_syms = model_funcs['all_syms']
    residual_funcs = model_funcs['residual_funcs']
    block_funcs = model_funcs['block_funcs']
    dynamic_eqs = model_funcs['dynamic_eqs']
    vars_exo = model_funcs.get('vars_exo', [])
    n = len(vars_dyn)

    if solver_options is None:
        solver_options = {}

    if ss_initial is None:
        ss_initial = ss  # Default: initial SS same as terminal SS

    # Handle stock vs jump variables
    if stock_var_indices is not None and initial_state is not None:
        # New behavior: distinguish stock (predetermined) and jump variables
        # Stock variables: fixed at t=0, free at t=T-1
        # Jump variables: free at t=0, fixed at t=T-1
        stock_indices = set(stock_var_indices)
        jump_indices = set(range(n)) - stock_indices

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
                    X_full[0, i] = initial_state[list(stock_indices).index(i)]
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

        def J_full(x):
            X = reconstruct_full_path(x)
            J_all = sparse_jacobian(X, params_dict, all_syms, block_funcs, vars_dyn, dynamic_eqs, vars_exo, exog_path)

            # J_all has shape ((T-1)*n, T*n)
            # We need to extract columns corresponding to unknowns only
            # Build column mask for unknowns in VARIABLE-MAJOR order (same as build_solution_vector)
            col_indices = []
            for i in range(n):
                for t in range(T):
                    # Is X[t,i] an unknown?
                    if i in stock_indices:
                        # Stock: unknown if t > 0
                        is_unknown = (t > 0)
                    else:
                        # Jump: unknown if t < T-1
                        is_unknown = (t < T-1)

                    if is_unknown:
                        # Column index in J_all is t*n + i
                        col_indices.append(t * n + i)

            J = J_all[:, col_indices]
            return J.toarray()

        # Initial guess
        x0_solve = build_solution_vector(X0)

        sol = root(
            F_full,
            x0_solve,
            jac=J_full,
            method=method,
            options=solver_options
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

            return J.toarray()

        # Initial guess for X[1:T-1]
        x0_solve = X0[1:-1, :].ravel()

        sol = root(
            F_full,
            x0_solve,
            jac=J_full,
            method=method,
            options=solver_options
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

            # Add initial conditions
            for i in range(n):
                F = np.append(F, X[0, i] - ss_initial[i])

            if use_terminal_conditions:
                J = sparse_jacobian(X, params_dict, all_syms, block_funcs, vars_dyn, dynamic_eqs, vars_exo, exog_path)
                F, J = append_terminal_conditions(F, J, X, ss)
            return F

        def J_full(x):
            from scipy.sparse import lil_matrix, vstack
            X = x.reshape(T, -1)
            J = sparse_jacobian(X, params_dict, all_syms, block_funcs, vars_dyn, dynamic_eqs, vars_exo, exog_path)

            # Add initial condition Jacobian rows: d(X[0,i] - ss_initial[i])/dX
            initial_rows = lil_matrix((n, n*T))
            for i in range(n):
                initial_rows[i, i] = 1.0  # Derivative of X[0,i] w.r.t. X[0,i]
            J = vstack([J, initial_rows.tocsr()])

            if use_terminal_conditions:
                F = residual(X, params_dict, all_syms, residual_funcs, vars_dyn, dynamic_eqs, vars_exo, exog_path)
                for i in range(n):
                    F = np.append(F, X[0, i] - ss_initial[i])
                F, J = append_terminal_conditions(F, J, X, ss)
            return J.toarray()

        sol = root(
            F_full,
            X0.ravel(),
            jac=J_full,
            method=method,
            options=solver_options
        )

    # Compute auxiliary variables if they exist
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
