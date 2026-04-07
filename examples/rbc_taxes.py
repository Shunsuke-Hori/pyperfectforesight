"""
rbc_taxes.py
RBC model with labor, government spending, and AR(1) TFP

Python replication of the 'taxes' model in rbc_taxes.yaml (dolo format).

Variables
---------
States  : z (TFP, AR(1)), k (capital)
Controls: i (investment), n (labor)
Exog.   : e_g → g (government spending)

The three block structure follows the YAML exactly:
  arbitrage : Euler equation + labor-leisure optimality
  transition: AR(1) for z, capital accumulation, g = e_g

Demo: impulse response to a 5% TFP shock (z_{-1} = 1.05).
"""

import os
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from pyperfectforesight import v, process_model, solve_perfect_foresight

# ============================================================
# 1. Parameters
# ============================================================

beta_s, delta_s, alpha_s = sp.symbols("beta delta alpha")
rho_s, sigma_s, eta_s, chi_s, zbar_s = sp.symbols("rho sigma eta chi zbar")

# ============================================================
# 2. Dynamic variables and time-indexed symbols
# ============================================================

# Order matches the YAML: states first, then controls
vars_dyn = ["z", "k", "g", "i", "n"]
vars_exo = ["e_g"]

# Current period
z_0, k_0, g_0, i_0, n_0 = [v(x, 0) for x in vars_dyn]
# Lagged (for transition equations)
z_m, k_m, i_m = v("z", -1), v("k", -1), v("i", -1)
# Lead (for Euler equation)
z_p, k_p, g_p, i_p, n_p = [v(x, 1) for x in vars_dyn]
# Exogenous
e_g_0 = v("e_g", 0)

# ============================================================
# 3. Auxiliary definitions (replicated from YAML 'definitions')
# ============================================================

def rk(z, k, n):
    return alpha_s * z * (n / k) ** (1 - alpha_s)

def w(z, k, n):
    return (1 - alpha_s) * z * (k / n) ** alpha_s

def y(z, k, n):
    return z * k**alpha_s * n**(1 - alpha_s)

def c(z, k, g, i, n):
    return y(z, k, n) - i - g

# Current and lead period values
rk_0 = rk(z_0, k_0, n_0)
rk_p = rk(z_p, k_p, n_p)
w_0  = w(z_0, k_0, n_0)
c_0  = c(z_0, k_0, g_0, i_0, n_0)
c_p  = c(z_p, k_p, g_p, i_p, n_p)

# ============================================================
# 4. Equations
# ============================================================

# Arbitrage (equilibrium conditions)
eq_euler = 1 - beta_s * (c_0 / c_p)**sigma_s * (1 - delta_s + rk_p)
eq_labor = chi_s * n_0**eta_s * c_0**sigma_s - w_0

# Transition (laws of motion)
eq_z = z_0 - (1 - rho_s) * zbar_s - rho_s * z_m
eq_k = k_0 - (1 - delta_s) * k_m - i_m
eq_g = g_0 - e_g_0

equations = [eq_euler, eq_labor, eq_z, eq_k, eq_g]

# ============================================================
# 5. Process model
# ============================================================

model_funcs = process_model(equations, vars_dyn, vars_exo=vars_exo)

# ============================================================
# 6. Steady-state computation (analytical, from YAML calibration)
# ============================================================

def compute_steady_state(p):
    """Analytical steady state from the YAML calibration block."""
    beta_v  = p["beta"]
    delta_v = p["delta"]
    alpha_v = p["alpha"]
    sigma_v = p["sigma"]
    eta_v   = p["eta"]
    zbar_v  = p["zbar"]

    z_ss  = zbar_v
    rk_ss = 1 / beta_v - 1 + delta_v
    n_ss  = 0.33                                          # calibrated target
    k_ss  = n_ss / (rk_ss / alpha_v) ** (1 / (1 - alpha_v))
    y_ss  = z_ss * k_ss**alpha_v * n_ss**(1 - alpha_v)
    i_ss  = delta_v * k_ss
    g_ss  = 0.0
    c_ss  = y_ss - i_ss - g_ss
    w_ss  = (1 - alpha_v) * z_ss * (k_ss / n_ss)**alpha_v
    chi_v = w_ss / (c_ss**sigma_v * n_ss**eta_v)

    # ss order matches vars_dyn: [z, k, g, i, n]
    ss_vals = np.array([z_ss, k_ss, g_ss, i_ss, n_ss])
    return ss_vals, chi_v, {"y": y_ss, "c": c_ss, "w": w_ss, "rk": rk_ss}


# ============================================================
# 7. Demo: impulse response to a 5% TFP shock
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RBC Taxes Model: TFP Shock")
    print("=" * 60)

    # Calibration (from YAML)
    param_vals = {
        "beta": 0.99,
        "delta": 0.025,
        "alpha": 0.33,
        "rho": 0.80,
        "sigma": 1.0,
        "eta": 1.0,
        "zbar": 1.0,
    }

    ss, chi_val, aux_ss = compute_steady_state(param_vals)
    param_vals["chi"] = chi_val

    # Build the sympy-keyed dict that solve_perfect_foresight expects
    params = {
        beta_s: param_vals["beta"],
        delta_s: param_vals["delta"],
        alpha_s: param_vals["alpha"],
        rho_s: param_vals["rho"],
        sigma_s: param_vals["sigma"],
        eta_s: param_vals["eta"],
        chi_s: chi_val,
        zbar_s: param_vals["zbar"],
    }

    print("\nSteady state:")
    for name, val in zip(vars_dyn, ss):
        print(f"  {name}: {val:.6f}")
    print(f"  y:  {aux_ss['y']:.6f}")
    print(f"  c:  {aux_ss['c']:.6f}")
    print(f"  chi: {chi_val:.6f}")

    # ----------------------------------------------------------
    # Shock: z_{-1} = 1.05  (5% above steady-state TFP)
    # k_{-1} = k_ss         (capital starts at steady state)
    # stock vars: z (index 0) and k (index 1)
    # ----------------------------------------------------------
    T = 100
    z_ss, k_ss, g_ss, i_ss, n_ss = ss
    shock_size = 0.05

    # initial_state contains the pre-period-0 values of stock variables.
    # Inferred stocks: z (idx 0), k (idx 1), i (idx 3) — i appears at lag -1 in eq_k.
    initial_state = np.array([z_ss * (1 + shock_size), k_ss, i_ss])

    # Flat exogenous path (no gov spending)
    exog_path = np.zeros((T, 1))  # e_g = 0 throughout

    print(f"\nShock: z_{{-1}} = {initial_state[0]:.4f} ({100*shock_size:.0f}% above SS)")
    print(f"Solving over T = {T} periods...")

    sol = solve_perfect_foresight(
        T, params, ss, model_funcs, vars_dyn,
        exog_path=exog_path,
        initial_state=initial_state,
    )

    print(f"\nConverged: {sol.success}  |  {sol.message}")

    # Extract paths
    X = sol.x.reshape(T, len(vars_dyn))
    z_path = X[:, 0]
    k_path = X[:, 1]
    g_path = X[:, 2]
    i_path = X[:, 3]
    n_path = X[:, 4]

    # Compute auxiliary paths
    y_path = z_path * k_path**param_vals["alpha"] * n_path**(1 - param_vals["alpha"])
    c_path = y_path - i_path - g_path

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"RBC Taxes Model: Response to {100*shock_size:.0f}% TFP Shock",
        fontsize=14, fontweight="bold",
    )

    periods = np.arange(T)

    def _plot(ax, path, ss_val, label, ylabel):
        ax.plot(periods, path, linewidth=2, label=label)
        ax.axhline(ss_val, color="k", linestyle="--", alpha=0.6, label="SS")
        ax.set_xlabel("Period")
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], z_path, z_ss, "TFP (z)", "z")
    _plot(axes[0, 1], y_path, aux_ss["y"], "Output (y)", "y")
    _plot(axes[0, 2], c_path, aux_ss["c"], "Consumption (c)", "c")
    _plot(axes[1, 0], k_path, k_ss, "Capital (k)", "k")
    _plot(axes[1, 1], i_path, i_ss, "Investment (i)", "i")
    _plot(axes[1, 2], n_path, n_ss, "Labor (n)", "n")

    plt.tight_layout()

    output_file = os.path.join(os.path.dirname(__file__), "rbc_taxes_irf.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")
    print("=" * 60)
