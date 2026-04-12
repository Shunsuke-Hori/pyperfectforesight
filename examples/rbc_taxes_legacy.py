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

Demo: impulse response to a decline in government spending.
"""

import os
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from pyperfectforesight import v, p, process_model, solve_perfect_foresight

# ============================================================
# 1. Parameters
# ============================================================

beta_s  = p("beta")
delta_s = p("delta")
alpha_s = p("alpha")
rho_s   = p("rho")
sigma_s = p("sigma")
eta_s   = p("eta")
chi_s   = p("chi")
zbar_s  = p("zbar")

VARS_PARAMS = ["beta", "delta", "alpha", "rho", "sigma", "eta", "chi", "zbar"]

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

model_funcs = process_model(equations, vars_dyn, vars_exo=vars_exo, vars_params=VARS_PARAMS)

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

    # ss order matches vars_dyn: [z, k, g, i, n]
    ss_vals = np.array([z_ss, k_ss, g_ss, i_ss, n_ss])
    return ss_vals, {"y": y_ss, "c": c_ss, "w": w_ss, "rk": rk_ss}


def compute_distorted_ss(p, g_val):
    """Analytical steady state with exogenous government spending g_val (sigma = eta = 1)."""
    rk_ss   = 1 / p["beta"] - 1 + p["delta"]
    h       = (rk_ss / p["alpha"]) ** (1 / (1 - p["alpha"]))  # n/k at SS
    A       = p["zbar"] / h ** p["alpha"]   # output per unit of n
    B       = p["delta"] / h                # investment per unit of n
    chi_v   = p["chi"]
    alpha_v = p["alpha"]
    # Solve chi*(A-B)*n^2 - chi*g*n - (1-alpha)*A = 0
    a = chi_v * (A - B)
    b = -chi_v * g_val
    c = -(1 - alpha_v) * A
    n_ss = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    k_ss = n_ss / h
    i_ss = p["delta"] * k_ss
    return np.array([p["zbar"], k_ss, g_val, i_ss, n_ss])


# ============================================================
# 7. Demo: government spending declines from 10% to 0%
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RBC Taxes Model: Government Spending Shock")
    print("=" * 60)

    # Calibration (from YAML)
    beta  = 0.99
    delta = 0.025
    alpha = 0.33
    rho   = 0.8
    sigma = 1.0
    eta   = 1.0
    zbar  = 1.0

    # Baseline (g=0) steady state — also used to calibrate chi
    param_vals_base = {
        "beta": beta, "delta": delta, "alpha": alpha,
        "rho": rho, "sigma": sigma, "eta": eta, "zbar": zbar,
    }
    ss_base, aux_base = compute_steady_state(param_vals_base)
    z_ss, k_ss, g_ss, i_ss, n_ss = ss_base
    # chi calibrated so labor FOC holds at baseline SS
    chi = aux_base["w"] / (aux_base["c"] ** sigma * n_ss ** eta)

    param_vals = {**param_vals_base, "chi": chi}

    params = {
        beta_s: beta, delta_s: delta, alpha_s: alpha,
        rho_s: rho, sigma_s: sigma, eta_s: eta,
        chi_s: chi, zbar_s: zbar,
    }

    print("\nBaseline steady state (g=0):")
    for name, val in zip(vars_dyn, ss_base):
        print(f"  {name}: {val:.6f}")
    print(f"  y:  {aux_base['y']:.6f}")
    print(f"  c:  {aux_base['c']:.6f}")

    # ----------------------------------------------------------
    # Scenario: government spending declines from 10% to 0%
    # over 10 periods (mirrors rbc_perfect_foresight.ipynb).
    # Economy starts at the distorted SS (g=0.1).
    # ----------------------------------------------------------
    T = 50
    G_INIT = 0.1

    ss_initial = compute_distorted_ss(param_vals, G_INIT)
    print(f"\nDistorted steady state (g={G_INIT}):")
    for name, val in zip(vars_dyn, ss_initial):
        print(f"  {name}: {val:.6f}")

    # Direct timing: exog_path[t] = g at period t.
    # Period 0 starts at G_INIT=0.1 and linearly declines to 0 by period 9.
    exog_path = np.concatenate([np.linspace(G_INIT, 0.0, 10), np.zeros(T - 10)]).reshape(T, 1)

    # stock vars: z (idx 0), k (idx 1), i (idx 3) — i appears at lag -1 in eq_k
    z0, k0, g0, i0, n0 = ss_initial
    initial_state = np.array([z0, k0, i0])

    print(f"\nSolving over T = {T} periods...")

    sol = solve_perfect_foresight(
        T, params, ss_base, model_funcs, vars_dyn,
        exog_path=exog_path,
        initial_state=initial_state,
        ss_initial=ss_initial,
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
    y_path = z_path * k_path**alpha * n_path**(1 - alpha)
    c_path = y_path - i_path - g_path

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "RBC Taxes Model: Gov. Spending Declining from 10% to 0%",
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
    _plot(axes[0, 1], y_path, aux_base["y"], "Output (y)", "y")
    _plot(axes[0, 2], c_path, aux_base["c"], "Consumption (c)", "c")
    _plot(axes[1, 0], k_path, k_ss, "Capital (k)", "k")
    _plot(axes[1, 1], i_path, i_ss, "Investment (i)", "i")
    _plot(axes[1, 2], n_path, n_ss, "Labor (n)", "n")

    plt.tight_layout()

    output_file = os.path.join(os.path.dirname(__file__), "rbc_taxes_irf.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")
    print("=" * 60)
