"""
gali_2015_zlb.py
Optimal Monetary Policy at the ZLB under Commitment
Gali (2015): Monetary Policy, Inflation, and the Business Cycle,
             Princeton University Press, 2nd ed., Chapter 5.4.2

Python replication of the Dynare mod-file by J. Pfeifer.

Model
-----
Variables (deviations from SS except interest rates):
  pi     inflation
  x      welfare-relevant output gap
  i      nominal interest rate (quarterly, in %)
  p      price level (deviations from SS)
  xi_1   Lagrange multiplier on NKPC
  xi_2   Lagrange multiplier on IS curve

Exogenous:
  r_nat  natural rate of interest (quarterly, in %)

The zero lower bound on i is handled via the NCP (Fischer–min) condition:
  min(xi_2 / sigma, i) = 0
which replaces Dynare's mcp='i>0' tag on the FOC w.r.t. i.
This encodes:
  xi_2 / sigma >= 0,  i >= 0,  (xi_2 / sigma) * i = 0.

Note: sp.Min introduces a non-analytic kink; process_model automatically falls
back to no static elimination when sp.solve raises NotImplementedError.

Replicates Figure 5.3 (commitment IRFs) from Gali (2015).
"""

import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from pyperfectforesight import p, v, process_model, solve_perfect_foresight

# ============================================================
# 1. Parameters
# ============================================================

alppha_s  = p("alppha")
betta_s   = p("betta")
siggma_s  = p("siggma")
varphi_s  = p("varphi")
epsilon_s = p("epsilon")
theta_s   = p("theta")

VARS_PARAMS = ["alppha", "betta", "siggma", "varphi", "epsilon", "theta"]

PARAMS = {
    alppha_s:  0.25,
    betta_s:   0.99,
    siggma_s:  1.0,
    varphi_s:  5.0,
    epsilon_s: 9.0,
    theta_s:   0.75,
}

# ============================================================
# 2. Model-local definitions (Gali 2015, pp. 60-63, 128)
# ============================================================

Omega    = (1 - alppha_s) / (1 - alppha_s + alppha_s * epsilon_s)
lambda_e = (1 - theta_s) * (1 - betta_s * theta_s) / theta_s * Omega
kappa    = lambda_e * (siggma_s + (varphi_s + alppha_s) / (1 - alppha_s))
vartheta = kappa / epsilon_s

# ============================================================
# 3. Variables
# ============================================================

VARS_DYN = ['pi', 'x', 'i', 'p', 'xi_1', 'xi_2']
VARS_EXO = ['r_nat']

pi_0,   pi_p   = v('pi',   0), v('pi',   1)
x_0,    x_p    = v('x',    0), v('x',    1)
i_0            = v('i',    0)
p_0,    p_m    = v('p',    0), v('p',   -1)
xi_1_0, xi_1_m = v('xi_1', 0), v('xi_1', -1)
xi_2_0, xi_2_m = v('xi_2', 0), v('xi_2', -1)
r_nat_0        = v('r_nat', 0)

# ============================================================
# 4. Equations
# ============================================================

# New Keynesian Phillips Curve (eq. 29)
eq_nkpc  = pi_0 - betta_s * pi_p - kappa * x_0

# Dynamic IS Curve (eq. 30)
eq_is    = x_0 - x_p + 1/siggma_s * (i_0 - pi_p - r_nat_0)

# FOC w.r.t. pi (eq. 35)
eq_foc1  = pi_0 + xi_1_0 - xi_1_m - 1/(betta_s * siggma_s) * xi_2_m

# FOC w.r.t. x (eq. 36)
eq_foc2  = vartheta * x_0 - kappa * xi_1_0 + xi_2_0 - 1/betta_s * xi_2_m

# ZLB — NCP formulation replacing Dynare's mcp='i>0' on FOC w.r.t. i:
#   min(xi_2/sigma, i) = 0  encodes  xi_2/sigma >= 0,  i >= 0,  product = 0
eq_zlb   = sp.Min(xi_2_0 / siggma_s, i_0)

# Price level (definition: pi = p - p(-1))
eq_price = p_0 - p_m - pi_0

EQUATIONS = [eq_nkpc, eq_is, eq_foc1, eq_foc2, eq_zlb, eq_price]

# ============================================================
# 5. Process model
# ============================================================

model = process_model(
    EQUATIONS, VARS_DYN,
    vars_exo=VARS_EXO,
    vars_params=VARS_PARAMS,
)

# ============================================================
# 6. Steady state
# ============================================================
# All model variables in deviations from SS except interest rates.
# SS: pi=0, x=0, i=r_nat=1 (quarterly %), p=0, xi_1=0, xi_2=0

SS = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])   # [pi, x, i, p, xi_1, xi_2]

# ============================================================
# 7. Simulation
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Gali (2015) Ch.5.4.2 — Optimal MP at ZLB, Commitment")
    print("=" * 60)

    T = 50
    R_NAT_SS = 1.0   # quarterly natural rate at non-ZLB steady state (in %)

    # Natural rate shock: r_nat drops from 1 to -1 for periods 0-5
    # (mirrors Dynare: shocks periods 1:6 values -1)
    exog_path = np.full((T, 1), R_NAT_SS)
    exog_path[0:6, 0] = -1.0

    # Stock variables: p (idx 3), xi_1 (idx 4), xi_2 (idx 5)
    # All at SS values pre-period-0
    initial_state = SS[[3, 4, 5]]   # [0, 0, 0]

    print(f"\nSolving T={T} periods (homotopy fallback enabled)...")
    sol = solve_perfect_foresight(
        T, PARAMS, SS, model, VARS_DYN,
        exog_path=exog_path,
        initial_state=initial_state,
    )

    print(f"Converged: {sol.success}  |  {sol.message}")
    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")

    X = sol.x.reshape(T, len(VARS_DYN))
    pi_path   = X[:, 0]
    x_path    = X[:, 1]
    i_path    = X[:, 2]
    p_path    = X[:, 3]
    xi_1_path = X[:, 4]
    xi_2_path = X[:, 5]

    # Annualized quantities (matching Dynare definitions)
    pi_ann_path    = 4.0 * pi_path
    i_ann_path     = 4.0 * np.maximum(i_path, 0.0)   # i_ann = 4*max(i,0)
    r_nat_ann_path = 4.0 * exog_path[:, 0]

    # Print first 8 periods
    print(f"\n{'Period':>6}  {'x':>8}  {'pi_ann':>8}  {'i_ann':>8}  {'r_nat_ann':>10}")
    print("-" * 50)
    for t in range(8):
        print(f"{t:>6}  {x_path[t]:>8.4f}  {pi_ann_path[t]:>8.4f}"
              f"  {i_ann_path[t]:>8.4f}  {r_nat_ann_path[t]:>10.4f}")

    # ============================================================
    # 8. Plot (mirrors Dynare figure layout)
    # ============================================================
    t_plot = np.arange(13)   # periods 0-12

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(
        "Gali (2015) Ch.5.4.2: Optimal MP at ZLB under Commitment\n"
        "Natural rate shock: r_nat → −1 for 6 periods (SS = 1%/quarter)",
        fontsize=11,
    )

    def _plot(ax, path, ylabel, ylim):
        ax.plot(t_plot, path[:13], '-x', linewidth=1.5, markersize=5)
        ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlim(0, 12)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Period")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], x_path,        "Output gap x",                  (-15, 5))
    _plot(axes[0, 1], pi_ann_path,   "Inflation (ann.) π_ann",         (-25, 5))
    _plot(axes[1, 0], i_ann_path,    "Nom. rate (ann.) i_ann",         (-2, 6))
    _plot(axes[1, 1], r_nat_ann_path,"Natural rate (ann.) r_nat_ann",  (-6, 6))

    axes[0, 0].set_title("Output gap")
    axes[0, 1].set_title("Annualized inflation")
    axes[1, 0].set_title("Annualized nominal rate")
    axes[1, 1].set_title("Annualized natural rate")

    plt.tight_layout()
    output_file = os.path.join(os.path.dirname(__file__), "gali_2015_zlb_irf.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")
    print("=" * 60)
