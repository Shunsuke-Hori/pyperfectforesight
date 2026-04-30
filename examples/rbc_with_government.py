"""
rbc_with_government.py
RBC model with government spending as an exogenous variable

This demo shows:
1. Defining a model with both endogenous and exogenous variables using Model
2. Specifying a path for exogenous variables (government spending shock)
3. Solving the transition path with model.solve()
4. Visualizing the response to government spending shocks
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from pyperfectforesight import p, v, Model

# ============================================================
# 1. Model declaration (RBC with government spending)
# ============================================================

print("Setting up RBC model with government spending...")

# Parameters
beta, delta, alpha = p("beta"), p("delta"), p("alpha")
vars_params = ["beta", "delta", "alpha"]

# Endogenous variables
vars_dyn = ["c", "k"]

# Exogenous variables
vars_exo = ["g"]

# Time-indexed symbols
c_0, c_p = v("c", 0), v("c", 1)
k_0, k_p = v("k", 0), v("k", 1)
g_0      = v("g", 0)

# Output
y_0 = k_0**alpha

# Equations
eq_euler = 1/c_0 - beta*(alpha*k_p**(alpha-1) + (1-delta))/c_p
eq_kacc  = k_p - (1-delta)*k_0 - y_0 + c_0 + g_0

# ============================================================
# 2. Build model
# ============================================================

model = Model([eq_euler, eq_kacc], vars_dyn,
              vars_exo=vars_exo, vars_params=vars_params)

print(f"Endogenous variables: {model.vars_dyn}")
print(f"Exogenous variables:  {model.vars_exo}")

# ============================================================
# 3. Steady state computation
# ============================================================

def compute_steady_state_analytical(params_dict, g_ss):
    """Analytical steady state for the RBC model with government spending."""
    beta_val  = params_dict[beta]
    delta_val = params_dict[delta]
    alpha_val = params_dict[alpha]

    k_ss = ((1/beta_val - (1-delta_val)) / alpha_val) ** (1/(alpha_val - 1))
    y_ss = k_ss**alpha_val
    c_ss = y_ss - delta_val*k_ss - g_ss
    return np.array([c_ss, k_ss]), y_ss


# ============================================================
# 4. DEMO: Government spending shock
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("RBC Model with Government Spending Shock")
    print("=" * 60)

    params = {
        beta:  0.96,
        delta: 0.08,
        alpha: 0.36,
    }

    print("\nParameters:")
    print(f"  beta (discount factor):  {params[beta]:.3f}")
    print(f"  delta (depreciation):    {params[delta]:.3f}")
    print(f"  alpha (capital share):   {params[alpha]:.3f}")

    g_baseline = 0.2
    ss, y_ss = compute_steady_state_analytical(params, g_baseline)

    print("\nSteady State (baseline):")
    print(f"  Consumption (c): {ss[0]:.4f}")
    print(f"  Capital (k):     {ss[1]:.4f}")
    print(f"  Output (y):      {y_ss:.4f}")
    print(f"  Government (g):  {g_baseline:.4f}")

    T = 100
    print(f"\nSolving transition path over {T} periods...")

    initial_stock = np.array([ss[1]])
    X0 = np.tile(ss, (T, 1))

    # Government spending shock: +1% for 20 periods
    exog_path = np.full((T, 1), g_baseline)
    shock_start    = 10
    shock_duration = 20
    shock_size     = 0.002
    exog_path[shock_start:shock_start + shock_duration, 0] = g_baseline + shock_size

    print(f"\nGovernment spending shock:")
    print(f"  Baseline g: {g_baseline:.4f}")
    print(f"  Shock size: +{shock_size:.4f} ({100*shock_size/g_baseline:.1f}%)")
    print(f"  Shock periods: {shock_start} to {shock_start + shock_duration - 1}")

    sol = model.solve(T, params,
                      endval=ss, X0=X0,
                      exog_path=exog_path,
                      initial_state=initial_stock,
                      stock_var_indices=[1])

    print(f"\nSolver Status:")
    print(f"  Converged: {sol.success}")
    print(f"  Message: {sol.message}")
    print(f"  Function evals: {sol.nfev}")

    X_sol  = sol.x.reshape(T, -1)
    c_path = X_sol[:, 0]
    k_path = X_sol[:, 1]
    y_path = k_path**params[alpha]
    g_path = exog_path[:, 0]
    i_path = k_path[1:] - (1 - params[delta])*k_path[:-1]

    # ============================================================
    # 5. Visualization
    # ============================================================

    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('RBC Model: Response to Government Spending Shock',
                 fontsize=14, fontweight='bold')

    periods = np.arange(T)

    def _shade(ax):
        ax.axvspan(shock_start, shock_start + shock_duration, alpha=0.2, color='red')

    axes[0, 0].plot(periods, g_path, 'r-', linewidth=2, label='Gov. Spending')
    axes[0, 0].axhline(y=g_baseline, color='k', linestyle='--', alpha=0.5, label='Baseline')
    _shade(axes[0, 0])
    axes[0, 0].set_xlabel('Period')
    axes[0, 0].set_ylabel('Government Spending (g)')
    axes[0, 0].set_title('Government Spending (Exogenous)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(periods, y_path, 'm-', linewidth=2, label='Output')
    axes[0, 1].axhline(y=y_ss, color='k', linestyle='--', label='Steady State')
    _shade(axes[0, 1])
    axes[0, 1].set_xlabel('Period')
    axes[0, 1].set_ylabel('Output (y)')
    axes[0, 1].set_title('Output')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(periods, c_path, 'g-', linewidth=2, label='Consumption')
    axes[0, 2].axhline(y=ss[0], color='k', linestyle='--', label='Steady State')
    _shade(axes[0, 2])
    axes[0, 2].set_xlabel('Period')
    axes[0, 2].set_ylabel('Consumption (c)')
    axes[0, 2].set_title('Consumption')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(periods, k_path, 'b-', linewidth=2, label='Capital')
    axes[1, 0].axhline(y=ss[1], color='k', linestyle='--', label='Steady State')
    _shade(axes[1, 0])
    axes[1, 0].set_xlabel('Period')
    axes[1, 0].set_ylabel('Capital (k)')
    axes[1, 0].set_title('Capital Stock')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(periods[:-1], i_path, 'c-', linewidth=2, label='Investment')
    axes[1, 1].axhline(y=params[delta]*ss[1], color='k', linestyle='--', label='Steady State')
    _shade(axes[1, 1])
    axes[1, 1].set_xlabel('Period')
    axes[1, 1].set_ylabel('Investment (i)')
    axes[1, 1].set_title('Investment')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    absorption = np.append(c_path[:-1] + i_path + g_path[:-1],
                           c_path[-1] + g_path[-1])
    axes[1, 2].plot(periods, absorption, color='orange', linewidth=2, label='C + I + G')
    axes[1, 2].plot(periods, y_path, 'm--', linewidth=1.5, alpha=0.7, label='Output')
    _shade(axes[1, 2])
    axes[1, 2].set_xlabel('Period')
    axes[1, 2].set_ylabel('Total Absorption')
    axes[1, 2].set_title('Resource Constraint: Y = C + I + G')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(os.path.dirname(__file__), 'rbc_government_shock.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
