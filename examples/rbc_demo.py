"""
demo.py
Demonstration of the Dynare-style perfect foresight solver

This demo shows:
1. Defining an RBC model using the framework
2. Computing the steady state analytically
3. Setting up an initial shock to capital
4. Solving the transition path
5. Visualizing the results
"""

import os
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Import framework functions from dynare_python package
from dynare_python import v, process_model, solve_perfect_foresight, compute_steady_state_numerical

# ============================================================
# 1. Model declaration (RBC model)
# ============================================================

print("Setting up RBC model...")

# Parameters
beta, delta, alpha = sp.symbols("beta delta alpha")

# Dynamic variables
vars_dyn = ["c", "k"]

# Time-indexed symbols
c_m, c_0, c_p = v("c",-1), v("c",0), v("c",1)
k_m, k_0, k_p = v("k",-1), v("k",0), v("k",1)

# Output
y_0 = k_0**alpha

# Equations (corrected Euler equation)
eq_euler = 1/c_0 - beta*(alpha*k_p**(alpha-1) + (1-delta))/c_p
eq_kacc  = k_p - (1-delta)*k_0 - y_0 + c_0

equations = [eq_euler, eq_kacc]

# ============================================================
# 2. Process model equations
# ============================================================

model_funcs = process_model(equations, vars_dyn)

print(f"Lead/lag incidence: {model_funcs['incidence']}")
print(f"Number of dynamic equations: {len(model_funcs['dynamic_eqs'])}")
print(f"Jacobian blocks at lags: {list(model_funcs['blocks'].keys())}")

# ============================================================
# 3. Steady state computation
# ============================================================

def compute_steady_state_analytical(params_dict):
    """
    Compute analytical steady state for the RBC model (model-specific)

    From the Euler equation at steady state:
    1/c = beta * (alpha * k^(alpha-1) + (1-delta)) / c
    => 1 = beta * (alpha * k^(alpha-1) + (1-delta))
    => k^(alpha-1) = (1/beta - (1-delta)) / alpha

    From capital accumulation:
    0 = y - delta*k - c
    => c = k^alpha - delta*k
    """
    beta_val = params_dict[beta]
    delta_val = params_dict[delta]
    alpha_val = params_dict[alpha]

    # Compute steady-state capital
    k_ss = ((1/beta_val - (1-delta_val)) / alpha_val) ** (1/(alpha_val-1))

    # Compute steady-state consumption
    c_ss = k_ss**alpha_val - delta_val*k_ss

    return np.array([c_ss, k_ss])

# ============================================================
# 4. DEMO: Transition from capital shock
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("RBC Model Perfect Foresight Demonstration")
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

    # Compute steady state (showing both methods)
    print("\nComputing steady state...")

    # Method 1: Analytical (model-specific, fast, exact)
    ss_analytical = compute_steady_state_analytical(params)
    print("  Method 1 - Analytical (model-specific):")
    print(f"    Consumption (c): {ss_analytical[0]:.4f}")
    print(f"    Capital (k):     {ss_analytical[1]:.4f}")

    # Method 2: Numerical (generic, works for any model)
    ss_numerical = compute_steady_state_numerical(equations, vars_dyn, params,
                                                  initial_guess=np.array([1.0, 5.0]))
    print("  Method 2 - Numerical (generic, from main.py):")
    print(f"    Consumption (c): {ss_numerical[0]:.4f}")
    print(f"    Capital (k):     {ss_numerical[1]:.4f}")

    # Use analytical for the rest of demo (it's more precise)
    ss = ss_analytical

    print(f"\nUsing analytical steady state for simulation:")
    print(f"  Consumption (c): {ss[0]:.4f}")
    print(f"  Capital (k):     {ss[1]:.4f}")
    print(f"  Output (y):      {ss[1]**params[alpha]:.4f}")

    # Setup perfect foresight problem
    T = 100
    print(f"\nSolving transition path over {T} periods...")

    # Initial condition: 10% below steady-state capital (stock variable)
    # c is jump variable (index 0), k is stock variable (index 1)
    k0 = 0.9 * ss[1]  # Shock to initial capital
    initial_stock = np.array([k0])  # Only stock variable values

    print(f"  Initial capital: {k0:.4f} (10% below steady state)")
    print(f"  Initial consumption: solver will determine optimal c[0]")

    # Initial guess for path
    X0 = np.tile(ss, (T, 1))  # Start with steady state guess
    X0[:, 1] = np.linspace(k0, ss[1], T)  # Linear path for capital

    # Solve with stock/jump variable distinction
    # vars_dyn = ["c", "k"], stock_var_indices = [1] means k is stock
    sol = solve_perfect_foresight(T, X0, params, ss, model_funcs, vars_dyn,
                                  initial_state=initial_stock,
                                  stock_var_indices=[1],
                                  method='hybr')

    print(f"\nSolver Status:")
    print(f"  Converged: {sol.success}")
    print(f"  Message: {sol.message}")
    print(f"  Function evals: {sol.nfev}")

    # Extract solution
    X_sol = sol.x.reshape(T, -1)
    c_path = X_sol[:, 0]
    k_path = X_sol[:, 1]
    y_path = k_path**params[alpha]
    i_path = k_path[1:] - (1-params[delta])*k_path[:-1]

    # Print some results
    print(f"\nFinal values (period {T}):")
    print(f"  Consumption: {c_path[-1]:.4f} (steady state: {ss[0]:.4f})")
    print(f"  Capital:     {k_path[-1]:.4f} (steady state: {ss[1]:.4f})")

    # ============================================================
    # 5. Visualization
    # ============================================================

    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('RBC Model: Transition After 10% Capital Shock', fontsize=14, fontweight='bold')

    periods = np.arange(T)

    # Capital
    axes[0, 0].plot(periods, k_path, 'b-', linewidth=2, label='Capital')
    axes[0, 0].axhline(y=ss[1], color='k', linestyle='--', label='Steady State')
    axes[0, 0].set_xlabel('Period')
    axes[0, 0].set_ylabel('Capital (k)')
    axes[0, 0].set_title('Capital Stock')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Consumption
    axes[0, 1].plot(periods, c_path, 'g-', linewidth=2, label='Consumption')
    axes[0, 1].axhline(y=ss[0], color='k', linestyle='--', label='Steady State')
    axes[0, 1].set_xlabel('Period')
    axes[0, 1].set_ylabel('Consumption (c)')
    axes[0, 1].set_title('Consumption')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Output
    axes[1, 0].plot(periods, y_path, 'm-', linewidth=2, label='Output')
    axes[1, 0].axhline(y=ss[1]**params[alpha], color='k', linestyle='--', label='Steady State')
    axes[1, 0].set_xlabel('Period')
    axes[1, 0].set_ylabel('Output (y)')
    axes[1, 0].set_title('Output')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Investment
    axes[1, 1].plot(periods[:-1], i_path, 'c-', linewidth=2, label='Investment')
    axes[1, 1].axhline(y=params[delta]*ss[1], color='k', linestyle='--', label='Steady State')
    axes[1, 1].set_xlabel('Period')
    axes[1, 1].set_ylabel('Investment (i)')
    axes[1, 1].set_title('Investment')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(os.path.dirname(__file__), 'rbc_transition.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
