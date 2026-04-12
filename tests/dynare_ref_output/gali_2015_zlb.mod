/*
 * gali_2015_zlb.mod
 * Optimal Monetary Policy at the ZLB under Commitment
 * Gali (2015): Monetary Policy, Inflation, and the Business Cycle,
 *              Princeton University Press, 2nd ed., Chapter 5.4.2
 *
 * Original mod-file by J. Pfeifer. See
 *   https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Gali_2015/
 *
 * Simulation: natural rate drops from 1 to -1 (quarterly %) for 6 periods,
 * then returns to 1. T=50 periods.
 *
 * To generate the pyperfectforesight reference CSV, run this file in Dynare
 * and then execute the export block below (or let Dynare run it automatically).
 *
 * Usage (Dynare 6.x or 7.x with lmmcp support):
 *   dynare gali_2015_zlb
 */

var pi x i p xi_1 xi_2;
varexo r_nat;

parameters alppha betta siggma varphi epsilon theta;

alppha  = 0.25;
betta   = 0.99;
siggma  = 1.0;
varphi  = 5.0;
epsilon = 9.0;
theta   = 0.75;

model;
    #Omega    = (1 - alppha) / (1 - alppha + alppha * epsilon);
    #lambda_e = (1 - theta) * (1 - betta * theta) / theta * Omega;
    #kappa    = lambda_e * (siggma + (varphi + alppha) / (1 - alppha));
    #vartheta = kappa / epsilon;

    // New Keynesian Phillips Curve
    pi = betta * pi(+1) + kappa * x;

    // Dynamic IS Curve
    x = x(+1) - 1/siggma * (i - pi(+1) - r_nat);

    // Optimal policy FOC w.r.t. pi
    pi + xi_1 - xi_1(-1) - 1/(betta*siggma) * xi_2(-1) = 0;

    // Optimal policy FOC w.r.t. x
    vartheta * x - kappa * xi_1 + xi_2 - 1/betta * xi_2(-1) = 0;

    // Optimal policy FOC w.r.t. i — ZLB complementarity
    // Dynare interprets mcp='i > 0' as: i >= 0, eq >= 0, i * eq = 0
    [mcp = 'i > 0']
    xi_2 / siggma = 0;

    // Price level accumulation
    p = p(-1) + pi;
end;

initval;
    r_nat = 1;
    pi    = 0;
    x     = 0;
    i     = 1;
    p     = 0;
    xi_1  = 0;
    xi_2  = 0;
end;

steady;

shocks;
    var r_nat;
    periods 1:6;
    values -1;
end;

perfect_foresight_setup(periods=50);
perfect_foresight_solver(lmmcp);

// ── Export reference data ─────────────────────────────────────────────────────
// Writes gali_2015_zlb_output.csv to the current working directory.
// Columns (declaration order): pi, x, i, p, xi_1, xi_2
// Rows: simulation periods 1..50 (Dynare 1-indexed = pypf 0-indexed)

T_sim = options_.periods;
sim_data = oo_.endo_simul(:, 2:T_sim+1)';   // (50 x 6)
dlmwrite('gali_2015_zlb_output.csv', sim_data, 'delimiter', ',', 'precision', 15);
disp('Exported: gali_2015_zlb_output.csv');
