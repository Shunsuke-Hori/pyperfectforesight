"""
rbc_with_investment.py
RBC model with investment as an auxiliary (static) variable

This demo shows:
1. Defining auxiliary variables that are derived from dynamic and exogenous variables
2. Using process_model with vars_aux to enable automatic auxiliary-variable handling
3. Solving the transition path and accessing the computed auxiliary-variable path
4. Visualizing the response to a government spending shock including investment
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from dynare_python import v, process_model, solve_perfect_foresight, compute_steady_state_numerical

# ============================================================
# 1. Model declaration (RBC with government spending)
# ============================================================

print("Setting up RBC model with government spending...")

# Parameters
beta, delta, alpha = sp.symbols("beta delta alpha")

# Endogenous variables (dynamic)
vars_dyn = ["c", "k"]  # consumption, capital

# Auxiliary variables (static, derived from dynamic and exogenous)
vars_aux = ["i"]  # investment

# Exogenous variables
vars_exo = ["g"]  # government spending

# Time-indexed symbols - endogenous
c_0, c_p = v("c", 0), v("c", 1)
k_0, k_p = v("k", 0), v("k", 1)

# Time-indexed symbols - auxiliary
i_0 = v("i", 0)

# Time-indexed symbols - exogenous
g_0 = v("g", 0)

# Output (determined by production function)
y_0 = k_0**alpha

# Equations
eq_euler = 1/c_0 - beta*(alpha*k_p**(alpha-1) + (1-delta))/c_p
eq_kacc = k_p - (1-delta)*k_0 - y_0 + c_0 + g_0  # Resource constraint with government
eq_i = y_0 - c_0 - i_0 - g_0  # Investment definition (auxiliary equation)

equations = [eq_euler, eq_kacc, eq_i]

# ============================================================
# 2. Process model equations
# ============================================================

model_funcs = process_model(equations, vars_dyn, vars_exo=vars_exo, vars_aux=vars_aux)  # Uses 'auto' - tries analytical first

print(f"Endogenous variables: {vars_dyn}")
print(f"Auxiliary variables: {vars_aux}")
print(f"Auxiliary method used: {model_funcs['aux_method']}")
print(f"Exogenous variables: {vars_exo}")
print(f"Number of dynamic equations: {len(model_funcs['dynamic_eqs'])}")

# ============================================================
# 3. Steady state computation
# ============================================================

def compute_steady_state_analytical(params_dict, g_ss):
    """
    Compute analytical steady state for the RBC model with government spending

    Returns
    -------
    ss_dyn : array
        Steady state values for dynamic variables [c, k]
    ss_aux : array
        Steady state values for auxiliary variables [i]
    y_ss : float
        Steady state output
    """
    beta_val = params_dict[beta]
    delta_val = params_dict[delta]
    alpha_val = params_dict[alpha]

    # Compute steady-state capital
    k_ss = ((1/beta_val - (1-delta_val)) / alpha_val) ** (1/(alpha_val-1))

    # Output
    y_ss = k_ss**alpha_val

    # Consumption (output minus depreciation and government spending)
    c_ss = y_ss - delta_val*k_ss - g_ss

    # Investment in steady state equals depreciation (auxiliary variable)
    i_ss = delta_val * k_ss

    ss_dyn = np.array([c_ss, k_ss])
    ss_aux = np.array([i_ss])

    return ss_dyn, ss_aux, y_ss

# ============================================================
# 4. DEMO: Government spending shock
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("RBC Model with Government Spending Shock")
    print("=" * 60)

    # Parameters
    params = {
        beta: 0.96,
        delta: 0.08,
        alpha: 0.36
    }

    print("\nParameters:")
    print(f"  beta (discount factor):  {params[beta]:.3f}")
    print(f"  delta (depreciation):    {params[delta]:.3f}")
    print(f"  alpha (capital share):   {params[alpha]:.3f}")

    # Steady state with baseline government spending
    g_baseline = 0.2
    ss_dyn, ss_aux, y_ss = compute_steady_state_analytical(params, g_baseline)

    print("\nSteady State (baseline):")
    print(f"  Consumption (c): {ss_dyn[0]:.4f}")
    print(f"  Capital (k):     {ss_dyn[1]:.4f}")
    print(f"  Investment (i):  {ss_aux[0]:.4f}")
    print(f"  Output (y):      {y_ss:.4f}")
    print(f"  Government (g):  {g_baseline:.4f}")

    # Setup perfect foresight problem
    T = 100
    print(f"\nSolving transition path over {T} periods...")

    # Initial condition: start at steady state for stock variable (capital)
    # vars_dyn = ["c", "k"], stock_var_indices = [1] means k (index 1) is stock
    initial_stock = np.array([ss_dyn[1]])  # k starts at steady state

    # Initial guess for path (dynamic variables only)
    X0 = np.tile(ss_dyn, (T, 1))

    # Define exogenous path: government spending shock
    # Shock: increase government spending by 1% for 20 periods, then return to baseline
    exog_path = np.zeros((T, 1))  # 1 exogenous variable (g)
    exog_path[:, 0] = g_baseline  # Baseline

    # Temporary increase in government spending
    shock_start = 10
    shock_duration = 20
    shock_size = 0.002  # Small shock: from 0.2 to 0.202

    exog_path[shock_start:shock_start+shock_duration, 0] = g_baseline + shock_size

    print(f"\nGovernment spending shock:")
    print(f"  Baseline g: {g_baseline:.4f}")
    print(f"  Shock size: +{shock_size:.4f} ({100*shock_size/g_baseline:.1f}%)")
    print(f"  Shock periods: {shock_start} to {shock_start+shock_duration-1}")

    # Solve with stock/jump variable distinction
    sol = solve_perfect_foresight(T, X0, params, ss_dyn, model_funcs, vars_dyn,
                                  exog_path=exog_path,
                                  initial_state=initial_stock,
                                  stock_var_indices=[1],
                                  method='hybr')

    print(f"\nSolver Status:")
    print(f"  Converged: {sol.success}")
    print(f"  Message: {sol.message}")
    print(f"  Function evals: {sol.nfev}")

    # Extract solution for dynamic variables
    X_sol = sol.x.reshape(T, -1)
    c_path = X_sol[:, 0]
    k_path = X_sol[:, 1]

    # Extract auxiliary variables (investment)
    X_aux = sol.x_aux  # This is computed automatically by the solver
    i_path = X_aux[:, 0]  # Investment path

    # Compute output from capital
    y_path = k_path**params[alpha]
    g_path = exog_path[:, 0]

    # ============================================================
    # 5. Visualization
    # ============================================================

    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('RBC Model: Response to Government Spending Shock', fontsize=14, fontweight='bold')

    periods = np.arange(T)

    # Government spending (exogenous)
    axes[0, 0].plot(periods, g_path, 'r-', linewidth=2, label='Gov. Spending')
    axes[0, 0].axhline(y=g_baseline, color='k', linestyle='--', alpha=0.5, label='Baseline')
    axes[0, 0].axvspan(shock_start, shock_start+shock_duration, alpha=0.2, color='red')
    axes[0, 0].set_xlabel('Period')
    axes[0, 0].set_ylabel('Government Spending (g)')
    axes[0, 0].set_title('Government Spending (Exogenous)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Output
    axes[0, 1].plot(periods, y_path, 'm-', linewidth=2, label='Output')
    axes[0, 1].axhline(y=y_ss, color='k', linestyle='--', label='Steady State')
    axes[0, 1].axvspan(shock_start, shock_start+shock_duration, alpha=0.2, color='red')
    axes[0, 1].set_xlabel('Period')
    axes[0, 1].set_ylabel('Output (y)')
    axes[0, 1].set_title('Output')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Consumption
    axes[0, 2].plot(periods, c_path, 'g-', linewidth=2, label='Consumption')
    axes[0, 2].axhline(y=ss_dyn[0], color='k', linestyle='--', label='Steady State')
    axes[0, 2].axvspan(shock_start, shock_start+shock_duration, alpha=0.2, color='red')
    axes[0, 2].set_xlabel('Period')
    axes[0, 2].set_ylabel('Consumption (c)')
    axes[0, 2].set_title('Consumption')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Capital
    axes[1, 0].plot(periods, k_path, 'b-', linewidth=2, label='Capital')
    axes[1, 0].axhline(y=ss_dyn[1], color='k', linestyle='--', label='Steady State')
    axes[1, 0].axvspan(shock_start, shock_start+shock_duration, alpha=0.2, color='red')
    axes[1, 0].set_xlabel('Period')
    axes[1, 0].set_ylabel('Capital (k)')
    axes[1, 0].set_title('Capital Stock')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Investment
    axes[1, 1].plot(periods, i_path, 'c-', linewidth=2, label='Investment')
    axes[1, 1].axhline(y=ss_aux[0], color='k', linestyle='--', label='Steady State')
    axes[1, 1].axvspan(shock_start, shock_start+shock_duration, alpha=0.2, color='red')
    axes[1, 1].set_xlabel('Period')
    axes[1, 1].set_ylabel('Investment (i)')
    axes[1, 1].set_title('Investment')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Total absorption (C + I + G)
    absorption = c_path + i_path + g_path
    axes[1, 2].plot(periods, absorption, 'orange', linewidth=2, label='C + I + G')
    axes[1, 2].plot(periods, y_path, 'm--', linewidth=1.5, alpha=0.7, label='Output')
    axes[1, 2].axvspan(shock_start, shock_start+shock_duration, alpha=0.2, color='red')
    axes[1, 2].set_xlabel('Period')
    axes[1, 2].set_ylabel('Total Absorption')
    axes[1, 2].set_title('Resource Constraint: Y = C + I + G')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = 'rbc_with_investment.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
