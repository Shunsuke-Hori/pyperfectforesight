/*
 * Benchmark timing for Dynare 6.2 perfect foresight solver.
 * Matches the RBC model in tests/dynare_ref_output/perfect_foresight_rbc.mod.
 *
 * Run from MATLAB (from this script's directory):
 *   addpath('C:\dynare\6.2\matlab');
 *   dynare benchmark_dynare nointeractive nolog
 *
 * Results are written to benchmark_dynare_times.csv in the same directory.
 */

var c k;
varexo z;

parameters alpha sigma delta beta;
alpha = 0.5;
sigma = 0.5;
delta = 0.02;
beta  = 1/1.05;

model;
[name='Resource constraint']
c + k = z*k(-1)^alpha + (1-delta)*k(-1);
[name='Euler equation']
c^(-sigma) = beta*(alpha*z(+1)*k^(alpha-1) + 1-delta)*c(+1)^(-sigma);
end;

steady_state_model;
k = ((1/beta-(1-delta))/(z*alpha))^(1/(alpha-1));
c = z*k^alpha-delta*k;
end;

initval;
z = 1;
end;
steady;
check;

shocks;
var z;
periods 1;
values 1.2;
end;

% Initial setup at T=200 to populate M_, oo_, options_ fully.
perfect_foresight_setup(periods=200);

%% -----------------------------------------------------------------------
%% Benchmark: time perfect_foresight_solver at several horizon lengths.
%% -----------------------------------------------------------------------
T_vals    = [50, 100, 200, 500, 1000];
N_reps    = 20;
n_T       = length(T_vals);
all_times = zeros(n_T, N_reps);

for ti = 1:n_T
    T = T_vals(ti);

    % Re-initialise setup for this horizon (sets oo_.endo_simul, oo_.exo_simul).
    options_.periods = T;
    perfect_foresight_setup;

    for ri = 1:N_reps
        % Reset endogenous path to steady state so each run starts fresh.
        oo_.endo_simul = repmat(oo_.steady_state, 1, T + 2);

        t0 = tic;
        perfect_foresight_solver;
        all_times(ti, ri) = toc(t0);
    end

    med_ms = median(all_times(ti,:)) * 1000;
    q25_ms = interp_quantile(all_times(ti,:), 0.25) * 1000;
    q75_ms = interp_quantile(all_times(ti,:), 0.75) * 1000;
    fprintf('Dynare T=%4d: %7.2f ms  [IQR %.2f-%.2f]  (median over %d runs)\n', ...
            T, med_ms, q25_ms, q75_ms, N_reps);
end

% Save [T, median_s, q25_s, q75_s] to CSV so benchmark.py can read it.
q25_all = arrayfun(@(i) interp_quantile(all_times(i,:), 0.25), 1:n_T)';
q75_all = arrayfun(@(i) interp_quantile(all_times(i,:), 0.75), 1:n_T)';
output = [T_vals(:), median(all_times, 2), q25_all, q75_all];
writematrix(output, 'benchmark_dynare_times.csv');
fprintf('Saved to benchmark_dynare_times.csv\n');

%% -----------------------------------------------------------------------
%% Helper: linear-interpolation quantile (no Statistics Toolbox required).
%% Matches the behaviour of quantile(x, p) for a vector x.
%% Local functions must appear after all executable script code in MATLAB.
%% -----------------------------------------------------------------------
function q = interp_quantile(x, p)
    x    = sort(x(:))';           % sort into ascending row vector
    n    = length(x);
    idx  = 1 + p * (n - 1);      % 1-based linear interpolation index
    lo   = floor(idx);
    hi   = ceil(idx);
    frac = idx - lo;
    q    = x(lo) * (1 - frac) + x(hi) * frac;
end
